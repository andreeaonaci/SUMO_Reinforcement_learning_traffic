"""Checks for the FRAP baseline head (agents/frap_head.py, fidings sec 101).

The point of this arm is to be a FAITHFUL published baseline, so these tests
check it against RESCO's own arithmetic rather than merely checking it runs.
A FRAP arm that quietly differs from FRAP would make the comparison worthless.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.frap_head import (
    FRAPHead, N_MOVEMENTS, act_to_union_vector, build_competition_mask,
)

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "resco_frap", "phase_pairs.json")


def resco_build_comp_mask(phase_pairs):
    """Verbatim from RESCO's mplight.py, as the reference implementation."""
    comp_mask = []
    for i in range(len(phase_pairs)):
        zeros = np.zeros(len(phase_pairs) - 1, dtype=int)
        cnt = 0
        for j in range(len(phase_pairs)):
            if i == j:
                continue
            pair_a = phase_pairs[i]
            pair_b = phase_pairs[j]
            if len(list(set(pair_a + pair_b))) == 3:
                zeros[cnt] = 1
            cnt += 1
        comp_mask.append(zeros)
    return torch.from_numpy(np.asarray(comp_mask))


# grid4x4's own phase_pairs as movement indices -- RESCO's canonical 8-phase set.
GRID4X4_PAIRS = [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5]]


def test_competition_mask_matches_resco():
    ours = build_competition_mask(GRID4X4_PAIRS)
    theirs = resco_build_comp_mask(GRID4X4_PAIRS)
    assert torch.equal(ours, theirs), "competition mask diverges from RESCO's"


def test_competition_mask_semantics():
    """Phases sharing exactly one movement compete; disjoint ones do not."""
    mask = build_competition_mask(GRID4X4_PAIRS)
    assert mask.shape == (8, 7)
    # [1,7] vs [1,2] share movement 1 -> compete. [1,7] is index 0, [1,2] index 2,
    # which after dropping self sits at column 1.
    assert mask[0, 1] == 1
    # [1,7] vs [4,10] are disjoint -> no competition (index 4 -> column 3).
    assert mask[0, 3] == 0


def test_output_shape_and_padding_masked():
    pairs = GRID4X4_PAIRS
    action_dim = 8
    head = FRAPHead(pairs, action_dim=action_dim)
    B = 4
    demand = torch.randn(B, N_MOVEMENTS)
    current = torch.tensor([0, 3, -1, 7])
    a2u = torch.full((B, action_dim), -1, dtype=torch.long)
    a2u[:, :3] = torch.tensor([0, 1, 2])          # only 3 real actions
    q = head(demand, current, a2u)
    assert q.shape == (B, action_dim)
    assert torch.isfinite(q[:, :3]).all(), "real action slots must be finite"
    assert torch.isinf(q[:, 3:]).all() and (q[:, 3:] < 0).all(), \
        "padded slots must be -inf so argmax can never select them"


def test_gather_selects_the_right_union_rows():
    """Two intersections with different local->union maps must read the same
    union phase identically -- that is the whole point of a shared head."""
    head = FRAPHead(GRID4X4_PAIRS, action_dim=8).eval()
    demand = torch.randn(1, N_MOVEMENTS).repeat(2, 1)
    current = torch.tensor([-1, -1])
    a2u = torch.full((2, 8), -1, dtype=torch.long)
    a2u[0, 0] = 5      # intersection A calls union phase 5 its action 0
    a2u[1, 3] = 5      # intersection B calls the same union phase its action 3
    with torch.no_grad():
        q = head(demand, current, a2u)
    assert torch.allclose(q[0, 0], q[1, 3], atol=1e-6)


def test_current_phase_flag_changes_output():
    head = FRAPHead(GRID4X4_PAIRS, action_dim=8).eval()
    demand = torch.randn(1, N_MOVEMENTS)
    a2u = torch.arange(8, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        q_none = head(demand, torch.tensor([-1]), a2u)
        q_zero = head(demand, torch.tensor([0]), a2u)
    assert not torch.allclose(q_none, q_zero), \
        "the current-phase flag must actually reach the network"


def test_parameter_count_independent_of_action_dim():
    a = FRAPHead(GRID4X4_PAIRS, action_dim=8)
    b = FRAPHead(GRID4X4_PAIRS, action_dim=32)
    assert sum(p.numel() for p in a.parameters()) == \
           sum(p.numel() for p in b.parameters())


def test_act_to_union_vector():
    vec = act_to_union_vector({0: 4, 1: 5, 2: 0}, action_dim=6)
    assert vec.tolist() == [4, 5, 0, -1, -1, -1]
    # out-of-range local actions are dropped rather than corrupting the vector
    vec = act_to_union_vector({0: 1, 9: 2}, action_dim=3)
    assert vec.tolist() == [1, -1, -1]


@pytest.mark.skipif(not os.path.exists(CONFIG), reason="FRAP config not built")
def test_real_config_is_usable_by_the_head():
    """The extracted RESCO config must actually drive the head for every city."""
    cfg = json.load(open(CONFIG))
    directions = cfg["directions"]
    pairs = [[directions.index(a), directions.index(b)]
             for a, b in cfg["union_pairs"]]
    action_dim = 8
    head = FRAPHead(pairs, action_dim=action_dim).eval()

    for city, blk in cfg["scenarios"].items():
        maps = blk["act_to_union"]
        vecs = np.stack([act_to_union_vector(m, action_dim) for m in maps.values()])
        assert (vecs.max(axis=1) >= 0).all(), f"{city}: a signal has no valid action"
        assert vecs.max() < len(pairs), f"{city}: union index out of range"
        demand = torch.randn(len(vecs), N_MOVEMENTS)
        current = torch.full((len(vecs),), -1, dtype=torch.long)
        with torch.no_grad():
            q = head(demand, current, torch.from_numpy(vecs))
        assert torch.isfinite(q[torch.from_numpy(vecs) >= 0]).all(), \
            f"{city}: real actions produced non-finite Q"
