"""Checks for the MPLight/FRAP movement-pressure state (fidings sec 101).

These run without SUMO: they verify the CONFIG side (coverage, index mapping,
action vectors), which is where a silent mistake would quietly invalidate the
FRAP baseline. The traci-dependent numbers are exercised by the smoke run.
"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.movement_pressure import DEFAULT_CONFIG, MovementPressureExtractor

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_CONFIG),
    reason="FRAP config not built (diagnostics/build_frap_config.py)")


@pytest.fixture(scope="module")
def ex():
    return MovementPressureExtractor(env=None)


def test_config_loads(ex):
    assert ex.available
    assert ex.n_movements == 12, "RESCO's movement table is 12 entries"
    assert len(ex.union_pairs) == 11, "union of the 3 training cities' pairs"


def test_every_signal_of_every_city_is_covered(ex):
    """A signal with no authored config cannot be controlled by FRAP at all --
    that is the §101 point, and it must not happen silently on our rosters."""
    import json
    cfg = json.load(open(DEFAULT_CONFIG))
    for city, blk in cfg["scenarios"].items():
        for sid in blk["act_to_union"]:
            assert ex.covers(sid), f"{city}: {sid} has no lane_sets"
        assert ex.missing(list(blk["act_to_union"])) == []


def test_action_vector_matches_config(ex):
    import json
    cfg = json.load(open(DEFAULT_CONFIG))
    blk = cfg["scenarios"]["city_5_holdout"]
    sid = sorted(blk["act_to_union"])[0]
    expected = {int(k): int(v) for k, v in blk["act_to_union"][sid].items()}
    _p, _c, vec = ex.extract(sid, current_phase=0, action_dim=8)
    for local_act, union_phase in expected.items():
        assert vec[local_act] == union_phase
    # every other slot is padding
    for i in range(8):
        if i not in expected:
            assert vec[i] == -1


def test_current_union_phase_resolves(ex):
    import json
    cfg = json.load(open(DEFAULT_CONFIG))
    blk = cfg["scenarios"]["city_5_holdout"]
    sid = sorted(blk["act_to_union"])[0]
    amap = {int(k): int(v) for k, v in blk["act_to_union"][sid].items()}
    a_valid = sorted(amap)[0]
    _p, cur, _v = ex.extract(sid, current_phase=a_valid, action_dim=8)
    assert cur == amap[a_valid]
    # a phase index the signal does not have resolves to -1, not a wrong row
    _p, cur, _v = ex.extract(sid, current_phase=99, action_dim=8)
    assert cur == -1


def test_unknown_signal_degrades_safely(ex):
    p, cur, vec = ex.extract("not_a_real_signal", current_phase=0, action_dim=8)
    assert p.shape == (12,) and not p.any()
    assert cur == -1
    assert (vec == -1).all()
    assert ex.missing(["not_a_real_signal"]) == ["not_a_real_signal"]


def test_union_indices_in_range(ex):
    for sid, amap in ex.act_to_union.items():
        for union_phase in amap.values():
            assert 0 <= union_phase < len(ex.union_pairs), \
                f"{sid}: union index {union_phase} out of range"


def test_outbound_is_populated_for_interior_signals(ex):
    """Pressure needs the downstream half; if lane_sets_outbound came back empty
    everywhere, pressure would silently degenerate to plain queue length."""
    with_outbound = sum(
        1 for sid, moves in ex.lane_sets_outbound.items()
        if any(moves.values()))
    assert with_outbound > 0.5 * len(ex.lane_sets_outbound), \
        "most signals should have some downstream lanes"
