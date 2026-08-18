"""Regression tests for the "flag/attribute silently doesn't reach where it
needs to" bug class that has bitten this project twice:

  1. ``--disable_head_fix`` never gated aggregation under ``--parallel``
     (fidings/divergence_investigation.md sec 10) -- every "fix off" ablation
     ever run through the parallel path was actually still running
     masked-head aggregation, silently.
  2. ``fixed_ts`` never reached the raw SUMO env through the wrapper chain
     (sec 24) -- the ``fixed_time`` rule-based baseline degenerated into a
     "never switch off the first phase" policy for this project's entire
     history, invisibly.

Both were caught by luck/persistence, not tests -- a cheap assertion of the
shape below would have caught each in seconds instead of days of misleading
ablation runs. These tests target the two invariants directly rather than
re-testing SUMO/training behavior:

  - a wrapper-chain attribute write actually reaches the innermost object it
    claims to control, through every layer;
  - an aggregation-mode flag actually changes which aggregation function
    gets called, not just which local network architecture gets built.

No SUMO/torch-heavy environment required -- everything here uses tiny fake
stand-ins so this file can run in any environment that has the repo on
PYTHONPATH.
"""
import inspect
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.federated_env import (
    FixedTsForwardingMixin,
    MultiAgentFederatedWrapper,
    RewardShapingWrapper,
    ActionMaskPadder,
)
from federated.comm_dropout import CommDropoutWrapper
from federated.aggregation import aggregate_round


# ---------------------------------------------------------------------------
# 1. fixed_ts write-forwarding
# ---------------------------------------------------------------------------

class _FakeRawEnv:
    """Stand-in for the innermost SumoEnvironment: just needs `fixed_ts`."""

    def __init__(self):
        self.fixed_ts = False


class _FakeFederatedEnv(_FakeRawEnv):
    """Stand-in for MultiAgentFederatedWrapper's expected inner env: also
    needs `max_action_dim` for ActionMaskPadder's constructor guard."""

    def __init__(self, max_action_dim=4):
        super().__init__()
        self.max_action_dim = max_action_dim


ALL_ENV_WRAPPER_CLASSES = [
    MultiAgentFederatedWrapper,
    RewardShapingWrapper,
    ActionMaskPadder,
    CommDropoutWrapper,
]


def test_all_env_wrappers_use_fixed_ts_mixin():
    """Every wrapper that sits in the real production chain
    (`CommDropoutWrapper(ActionMaskPadder(MultiAgentFederatedWrapper(raw_env)))`,
    per CLAUDE.md) must inherit `FixedTsForwardingMixin`, or `fixed_ts`
    writes silently stop at whichever layer skips it -- exactly the sec 24
    bug. A 5th wrapper class added later that forgets this inheritance
    should fail this test, not a multi-day training run.
    """
    for cls in ALL_ENV_WRAPPER_CLASSES:
        assert issubclass(cls, FixedTsForwardingMixin), (
            f"{cls.__name__} does not inherit FixedTsForwardingMixin -- "
            f"`wrapper.fixed_ts = True` on this class (or anything wrapping "
            f"it) will silently no-op instead of reaching the raw env."
        )


def test_fixed_ts_forwards_through_two_layer_wrapper_chain():
    """Behavioral check, not just an inheritance check: writing `fixed_ts`
    on the OUTERMOST wrapper of a chain must actually flip it on the
    INNERMOST raw env, and reading it back from the outside must reflect
    that. Uses RewardShapingWrapper and CommDropoutWrapper directly since
    both accept a bare fake env (MultiAgentFederatedWrapper needs a full set
    of real SUMO-derived collaborators to construct, covered separately by
    the inheritance check above).
    """
    raw_env = _FakeRawEnv()
    inner = RewardShapingWrapper(raw_env)
    outer = CommDropoutWrapper(inner, p_link=0.0, p_isolate=0.0, p_hop_cutoff=0.0)

    assert outer.fixed_ts is False

    outer.fixed_ts = True

    assert raw_env.fixed_ts is True, (
        "Setting fixed_ts on the outer wrapper did not reach the raw env "
        "two layers down -- this is exactly the sec 24 failure mode."
    )
    assert inner.fixed_ts is True
    assert outer.fixed_ts is True

    outer.fixed_ts = False
    assert raw_env.fixed_ts is False


def test_action_mask_padder_forwards_fixed_ts():
    """ActionMaskPadder specifically, since it's the layer that sits
    directly around MultiAgentFederatedWrapper in the real chain and has
    its own constructor guard (target_action_dim >= env.max_action_dim)."""
    inner = _FakeFederatedEnv(max_action_dim=4)
    padder = ActionMaskPadder(inner, target_action_dim=6)

    padder.fixed_ts = True
    assert inner.fixed_ts is True
    assert padder.fixed_ts is True


# ---------------------------------------------------------------------------
# 2. aggregate_round's use_masked_head flag actually changes the result
# ---------------------------------------------------------------------------

def _toy_state_dict(head_row_values):
    """3-row head, everything else a single shared scalar param so the
    "every other parameter" branch of masked_head_weighted_average has
    something to average uniformly too."""
    return {
        "shared.weight": torch.tensor([1.0]),
        "head.4.weight": torch.tensor(
            [[float(v)] for v in head_row_values], dtype=torch.float32
        ),
        "head.4.bias": torch.tensor(head_row_values, dtype=torch.float32),
    }


def test_aggregate_round_masked_head_true_differs_from_false():
    """Two clients with disjoint action coverage: client A only ever
    touches row 0, client B only ever touches row 1. Plain averaging
    (use_masked_head=False) blends both clients into every row regardless
    of who touched it; masked-head aggregation (True) should let each
    client's own row through undiluted. If these two results are ever
    equal, the flag has stopped doing anything -- the sec 10 failure mode.
    """
    client_a = _toy_state_dict([10.0, 0.0, 5.0])
    client_b = _toy_state_dict([0.0, 20.0, 5.0])
    action_counts = [{0: 100}, {1: 100}]  # A only touched row 0, B only row 1

    plain = aggregate_round(
        state_dicts=[client_a, client_b],
        base_weights=[1.0, 1.0],
        action_counts=action_counts,
        use_masked_head=False,
    )
    masked = aggregate_round(
        state_dicts=[client_a, client_b],
        base_weights=[1.0, 1.0],
        action_counts=action_counts,
        use_masked_head=True,
    )

    # Plain averaging always blends 50/50 regardless of who touched a row.
    assert plain["head.4.bias"].tolist() == pytest.approx([5.0, 10.0, 5.0])

    # Masked-head: row 0 comes from A alone (10.0), row 1 from B alone (20.0).
    assert masked["head.4.bias"].tolist() == pytest.approx([10.0, 20.0, 5.0])

    assert not torch.equal(plain["head.4.bias"], masked["head.4.bias"]), (
        "use_masked_head=True and False produced identical head weights -- "
        "the flag is not actually changing aggregation behavior."
    )

    # Non-head parameters are unaffected by the flag either way.
    assert plain["shared.weight"].tolist() == masked["shared.weight"].tolist()


def test_masked_head_leaves_untouched_row_at_previous_global_value():
    """If NO client touched a given action index this round, masked-head
    aggregation must leave that row exactly as it was in the previous
    global state -- not reset it, not silently uniform-average it (see
    aggregate.py's own docstring for why: there's no new information for
    an untouched row, so it shouldn't move)."""
    client_a = _toy_state_dict([10.0, 99.0, 5.0])
    client_b = _toy_state_dict([0.0, 99.0, 5.0])
    previous_global = _toy_state_dict([1.0, 2.0, 3.0])
    action_counts = [{0: 100}, None]  # row 1 touched by nobody this round

    result = aggregate_round(
        state_dicts=[client_a, client_b],
        base_weights=[1.0, 1.0],
        action_counts=action_counts,
        use_masked_head=True,
        previous_global_state=previous_global,
    )

    assert result["head.4.bias"][1].item() == pytest.approx(2.0), (
        "Row untouched by any client this round should be frozen at the "
        "previous global value, not reset or averaged from stale client "
        "copies."
    )


# ---------------------------------------------------------------------------
# 3. Server classes must dynamically thread their head-fix flag into
#    aggregate_round's use_masked_head kwarg, not hardcode it.
#
# ParallelFederatedServer/FederatedServer need real multiprocessing/SUMO
# environments to instantiate, so a full behavioral test here would just be
# a slow duplicate of the smoke test. What actually broke (sec 10) was
# structural: the parallel path called masked_head_weighted_average
# unconditionally with no gate at all, not a logic bug inside a testable
# pure function. Source-inspection is the right tool for that specific
# shape of regression -- it fails immediately if someone reintroduces a
# hardcoded call instead of threading the instance flag through.
# ---------------------------------------------------------------------------

def test_parallel_server_threads_head_fix_into_aggregate_round():
    from federated.parallel_server import ParallelFederatedServer

    source = inspect.getsource(ParallelFederatedServer)
    assert "aggregate_round(" in source, (
        "ParallelFederatedServer no longer calls aggregate_round() -- if "
        "it now calls masked_head_weighted_average/weighted_average "
        "directly again, confirm there's still an explicit self.head_fix "
        "gate (this is exactly how the flag broke before, sec 10)."
    )
    assert "use_masked_head=self.head_fix" in source, (
        "ParallelFederatedServer's aggregate_round() call no longer passes "
        "use_masked_head=self.head_fix -- --disable_head_fix would silently "
        "stop working under --parallel again (sec 10)."
    )


def test_sequential_server_threads_head_fix_into_aggregate_round():
    from federated.server import FederatedServer

    source = inspect.getsource(FederatedServer)
    assert "use_masked_head=self.use_masked_head" in source, (
        "FederatedServer's aggregate_round() call no longer passes "
        "use_masked_head=self.use_masked_head -- --disable_head_fix would "
        "silently stop working under the sequential path."
    )


# ---------------------------------------------------------------------------
# 4. eval_city_name / is_true_holdout must survive end-to-end from
#    HoldoutEvaluator through to federated_history.json, not just exist on
#    the evaluator object. The sec 25/29 incident (every 2-/3-city result in
#    this project's history silently evaluated in-distribution on a training
#    city instead of the true holdout) was invisible specifically because
#    nothing about *which* city was used ever made it into the persisted
#    output -- only a log line, easy to miss and not present in the JSON
#    artifact people actually read later.
# ---------------------------------------------------------------------------

def test_holdout_evaluator_stores_eval_city_metadata():
    from federated.evaluator import HoldoutEvaluator

    evaluator = HoldoutEvaluator(
        env_builder=lambda: None,
        eval_city_name="city_1",
        is_true_holdout=False,
    )
    assert evaluator.eval_city_name == "city_1"
    assert evaluator.is_true_holdout is False

    true_holdout_evaluator = HoldoutEvaluator(
        env_builder=lambda: None,
        eval_city_name="city_5_holdout",
        is_true_holdout=True,
    )
    assert true_holdout_evaluator.is_true_holdout is True


def test_holdout_evaluator_stamps_metadata_into_results():
    """Source-inspection, not a full SUMO-backed behavioral run (evaluate()
    steps a real env internally) -- checks that evaluate()/evaluate_controller()
    still inject eval_city_name/is_true_holdout into the dict they return,
    which is what ends up in federated_history.json."""
    from federated.evaluator import HoldoutEvaluator

    source = inspect.getsource(HoldoutEvaluator)
    for method_marker in ["def evaluate(", "def evaluate_controller("]:
        method_start = source.index(method_marker)
        next_def = source.find("\n    def ", method_start + 1)
        method_source = source[method_start: next_def if next_def != -1 else None]
        assert "eval_city_name" in method_source and "is_true_holdout" in method_source, (
            f"{method_marker} no longer stamps eval_city_name/is_true_holdout "
            f"into its result dict -- federated_history.json would silently "
            f"stop recording which city an eval actually ran on."
        )


def test_make_holdout_evaluator_passes_metadata_through():
    from experiments.federated_training import make_holdout_evaluator

    source = inspect.getsource(make_holdout_evaluator)
    assert "eval_city_name=selected_name" in source
    assert "is_true_holdout=is_true_holdout" in source


def test_servers_persist_eval_city_metadata_in_history():
    """Both the parallel and sequential round loops, and both servers'
    _evaluate_multiple_models (the --no_federation / clustered_fedavg path),
    must read eval_city_name/is_true_holdout off the metrics dict into
    `history` -- easy to add the field to HoldoutEvaluator and forget to
    thread it through the two separate history-building call sites."""
    from federated.parallel_server import ParallelFederatedServer
    from federated.server import FederatedServer

    for cls in (ParallelFederatedServer, FederatedServer):
        source = inspect.getsource(cls)
        # setdefault(...), not direct indexing, deliberately: a --resume run
        # loads an old federated_history.json that predates these two keys,
        # and history["eval_city_name"].append(...) would KeyError on it.
        assert 'history.setdefault("eval_city_name", []).append(' in source, (
            f"{cls.__name__} no longer appends eval_city_name to history "
            f"resume-safely."
        )
        assert 'history.setdefault("is_true_holdout", []).append(' in source, (
            f"{cls.__name__} no longer appends is_true_holdout to history "
            f"resume-safely."
        )
        assert '"eval_city_name": self.evaluator.eval_city_name' in source, (
            f"{cls.__name__}._evaluate_multiple_models (the --no_federation "
            f"path) no longer carries eval_city_name into its aggregate dict."
        )


# ---------------------------------------------------------------------------
# 5. RewardShapingWrapper: a third instance of the same bug class. wait_weight
# looked up an info key ("{ts}_waiting_time") that has never existed (the
# real key is "{ts}_accumulated_waiting_time") -- silently a no-op for
# anyone who set it, indistinguishable from it working correctly on a city
# with genuinely zero waiting time. Nobody was actually depending on the old
# silent-zero behavior (no city config sets reward_shaping today), so this
# is a correctness fix, not a breaking change.
# ---------------------------------------------------------------------------

def test_reward_shaping_wait_weight_uses_a_real_info_key():
    wrapper = RewardShapingWrapper.__new__(RewardShapingWrapper)
    wrapper.wait_weight = 1.0
    wrapper.stopped_weight = 0.0
    wrapper.queue_weight = 0.0
    wrapper.raw_weight = 1.0

    real_info = {"nt1_stopped": 3, "nt1_accumulated_waiting_time": 42.0, "nt1_average_speed": 5.0}
    shaped = wrapper._shape_reward(reward=0.0, info=real_info, ts_id="nt1")
    assert shaped == pytest.approx(-42.0), (
        "wait_weight did not apply the real per-intersection waiting-time "
        "value -- it's still reading a key that doesn't exist in the real "
        "env's info dict and silently defaulting to 0."
    )


def test_reward_shaping_raises_loudly_on_missing_key_instead_of_silently_defaulting():
    """A non-zero weight whose key genuinely isn't in `info` (e.g.
    queue_weight, which has no matching per-ts metric in sumo_rl today)
    must fail loudly, not silently contribute 0 to the shaped reward --
    that silent-0 path is exactly how wait_weight's bug went unnoticed."""
    wrapper = RewardShapingWrapper.__new__(RewardShapingWrapper)
    wrapper.wait_weight = 0.0
    wrapper.stopped_weight = 0.0
    wrapper.queue_weight = 1.0
    wrapper.raw_weight = 1.0

    real_info = {"nt1_stopped": 3, "nt1_accumulated_waiting_time": 42.0}
    with pytest.raises(KeyError):
        wrapper._shape_reward(reward=0.0, info=real_info, ts_id="nt1")
