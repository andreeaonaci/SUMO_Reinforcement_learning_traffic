"""Aggregation mechanics: turning (state_dicts, weights) into one global
state_dict. Deliberately knows nothing about *how* weights are chosen --
that policy lives in ``aggregation_strategies.py``. Keeping the two apart
means adding a new weighting scheme never touches this file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def weighted_average(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """Weighted average of a list of state_dicts.

    Weights need not sum to 1 (renormalized internally). Negative weights
    are clamped to 0 defensively. Falls back to a uniform average if every
    weight is ~0, so a degenerate scoring round never produces a
    divide-by-zero or an all-zero global model.
    """
    if not state_dicts:
        raise ValueError("weighted_average called with no state_dicts.")
    if len(state_dicts) != len(weights):
        raise ValueError(
            f"state_dicts ({len(state_dicts)}) and weights ({len(weights)}) "
            "must be the same length."
        )

    clamped = [max(float(w), 0.0) for w in weights]
    total = sum(clamped)
    if total <= 1e-12:
        norm_weights = [1.0 / len(clamped)] * len(clamped)
    else:
        norm_weights = [w / total for w in clamped]

    keys = state_dicts[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = torch.zeros_like(state_dicts[0][k], dtype=torch.float32)
        for sd, w in zip(state_dicts, norm_weights):
            acc += w * sd[k].float()
        out[k] = acc
    return out


def aggregate_round(
    state_dicts: List[Dict[str, torch.Tensor]],
    base_weights: List[float],
    action_counts: List[Optional[Dict[int, int]]],
    use_masked_head: bool = True,
    head_weight_key: str = "head.4.weight",
    head_bias_key: str = "head.4.bias",
    previous_global_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Aggregate one federated round.

    If use_masked_head=True, use per-action aggregation for the final Q-head.
    Otherwise perform ordinary weighted averaging on every parameter,
    reproducing the pre-fix behavior.
    """
    if use_masked_head:
        return masked_head_weighted_average(
            state_dicts,
            base_weights,
            action_counts,
            head_weight_key=head_weight_key,
            head_bias_key=head_bias_key,
            previous_global_state=previous_global_state,
        )

    # original FedAvg on every parameter, including the head.
    return weighted_average(state_dicts, base_weights)

def masked_head_weighted_average(
    state_dicts: List[Dict[str, torch.Tensor]],
    base_weights: List[float],
    action_counts: List[Optional[Dict[int, int]]],
    head_weight_key: str = "head.4.weight",
    head_bias_key: str = "head.4.bias",
    previous_global_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Weighted average where the final Q-head is aggregated ROW-BY-ROW,
    weighted by how many samples actually trained each action index --
    instead of uniformly like every other parameter.

    Why this exists
    ----------------
    A shared network's final Linear(d_model, action_dim) layer only
    receives gradient in the output row for whichever action was actually
    taken (see DQNAgent.optimize: only q_taken = q.gather(1, actions) is
    backpropped, the Bellman target side is no_grad). If client A's
    environment can only ever produce actions 0-3 and client B's can
    produce actions 0-7, rows 4-7 get a real update from B only -- every
    other client's copy of those rows is untouched. Averaging all clients
    uniformly then dilutes B's real update with N-1 stale copies, which
    is an N-times effective learning-rate cut on exactly the rows a
    higher-action-count holdout city depends on most.

    This function keeps ordinary ``base_weights`` (e.g. from a
    BaseAggregationStrategy) for every parameter EXCEPT the head weight/
    bias, where each output row ``i`` is instead averaged using weights
    proportional to ``action_counts[client][i]`` -- clients that never
    touched row ``i`` this round contribute 0 to that row, so a single
    active client's update passes through undiluted.

    If NO client touched a given row this round, that row is left
    UNCHANGED from the previous global state (rather than reset to 0 or
    silently uniform-averaged) -- there's no new information for it,
    so it shouldn't move.

    Args:
        state_dicts:      one state_dict per client, from this round.
        base_weights:     one scalar weight per client (from the
                          aggregation strategy), used for every parameter
                          except the head.
        action_counts:    one dict per client, {action_index: count},
                          or None if that client didn't report counts
                          (falls back to base_weights for the head too).
        head_weight_key:  state_dict key of the final Linear's weight.
        head_bias_key:    state_dict key of the final Linear's bias.
        previous_global_state: the global state_dict this round started
                          from. Used (when given) as the source for a row
                          no client touched this round, since that is the
                          actual "don't move this row" value. Without it,
                          falls back to state_dicts[0]'s row, which is only
                          an approximation -- a client's local copy of an
                          untouched row still drifts round to round (optimizer
                          weight decay, floating point), and "client 0" is an
                          arbitrary pick with no guarantee its drift is
                          smallest.

    Returns:
        Aggregated state_dict, same keys as the inputs.
    """
    if not state_dicts:
        raise ValueError("masked_head_weighted_average called with no state_dicts.")

    # 1. Ordinary uniform-per-client aggregation for every parameter.
    agg = weighted_average(state_dicts, base_weights)

    if head_weight_key not in state_dicts[0] or head_bias_key not in state_dicts[0]:
        return agg  # architecture doesn't match -- nothing special to do

    # If nobody reported action_counts, we have no per-row signal --
    # keep the ordinary uniform result for the head too.
    if all(c is None for c in action_counts):
        return agg

    action_dim = state_dicts[0][head_weight_key].shape[0]
    prev_head_w = state_dicts[0][head_weight_key]  # placeholder shape/dtype reference
    new_head_w = torch.zeros_like(prev_head_w)
    new_head_b = torch.zeros_like(state_dicts[0][head_bias_key])

    fallback_w = (
        previous_global_state[head_weight_key]
        if previous_global_state is not None and head_weight_key in previous_global_state
        else state_dicts[0][head_weight_key]
    )
    fallback_b = (
        previous_global_state[head_bias_key]
        if previous_global_state is not None and head_bias_key in previous_global_state
        else state_dicts[0][head_bias_key]
    )

    for row in range(action_dim):
        row_weights = []
        for client_idx, counts in enumerate(action_counts):
            n = 0 if counts is None else counts.get(row, 0)
            row_weights.append(float(n) * max(base_weights[client_idx], 0.0))

        total = sum(row_weights)
        if total <= 1e-12:
            # No client touched this action index this round -- leave the
            # row exactly as it was, don't invent an update from nothing.
            new_head_w[row] = fallback_w[row]
            new_head_b[row] = fallback_b[row]
            continue

        norm_weights = [w / total for w in row_weights]
        for client_idx, w in enumerate(norm_weights):
            if w <= 0.0:
                continue
            new_head_w[row] += w * state_dicts[client_idx][head_weight_key][row].float()
            new_head_b[row] += w * state_dicts[client_idx][head_bias_key][row].float()

    agg[head_weight_key] = new_head_w
    agg[head_bias_key] = new_head_b
    return agg


def _state_delta(
    a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """``a - b``, elementwise, float-cast -- shared by every server-side
    update-shaping helper below (momentum, gradient clipping), all of which
    reason about ``agg_state - global_state_before`` as "this round's
    pseudo-gradient"."""
    return {k: a[k].float() - b[k].float() for k in a}


def head_key_names(dueling: bool) -> Tuple[str, str]:
    """State-dict key names for the action-indexed Q-head that
    ``masked_head_weighted_average`` should target: ``"advantage_head.*"``
    under the dueling architecture (``agents/networks.py`` splits the final
    layer into ``value_head``/``advantage_head``), ``"head.4.*"`` for the
    plain single-Linear head otherwise. Getting this wrong makes
    masked-head aggregation silently no-op (falls through to plain
    averaging) because the configured key doesn't exist in the state dict.
    """
    if dueling:
        return "advantage_head.weight", "advantage_head.bias"
    return "head.4.weight", "head.4.bias"


def shape_server_update(
    agg_state: Dict[str, torch.Tensor],
    global_state_before: Dict[str, torch.Tensor],
    pseudo_grad_clip: float,
    server_momentum: float,
    momentum_buffer: Optional[Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """Apply the optional server-side update-shaping pipeline in the
    documented order: pseudo-gradient clipping first, then FedAvgM-style
    momentum, so the velocity buffer only ever accumulates already-bounded
    deltas rather than an occasional huge spike. Both stages are
    individually exact no-ops at their default (0) value -- see
    ``clip_pseudo_gradient``/``apply_server_momentum``. Returns
    ``(new_agg_state, new_momentum_buffer)``; thread the buffer back in as
    ``momentum_buffer`` next round.
    """
    if pseudo_grad_clip > 0.0:
        agg_state = clip_pseudo_gradient(agg_state, global_state_before, pseudo_grad_clip)
    if server_momentum > 0.0:
        agg_state, momentum_buffer = apply_server_momentum(
            agg_state, global_state_before, momentum_buffer, server_momentum
        )
    return agg_state, momentum_buffer


def update_eval_ema(
    eval_ema_state: Optional[Dict[str, torch.Tensor]],
    agg_state: Dict[str, torch.Tensor],
    eval_ema_decay: float,
) -> Dict[str, torch.Tensor]:
    """EMA-update the eval-only snapshot used by ``--eval_ema_decay``:
    ``None`` (cold start, round 1) just clones this round's aggregate;
    afterward, standard exponential smoothing. Only meaningful to call when
    ``eval_ema_decay > 0`` -- callers should skip calling this entirely
    when it's disabled, matching every other optional mechanism's
    0-is-off convention, since there'd be nothing meaningful to return.
    """
    if eval_ema_state is None:
        return {k: v.clone() for k, v in agg_state.items()}
    return {
        k: eval_ema_decay * eval_ema_state[k] + (1.0 - eval_ema_decay) * agg_state[k].float()
        for k in agg_state
    }


def evaluate_with_optional_ema(
    evaluator: Any,
    global_model: Any,
    eval_ema_decay: float,
    eval_ema_state: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """Evaluate ``global_model``, transparently swapping in the slowly-
    averaged ``--eval_ema_decay`` snapshot first if one is available --
    purely a reporting-side smoothing; the real weights are restored
    immediately after, so this never touches what gets broadcast to
    clients next round. Falls back to a plain evaluation when EMA is
    disabled or not yet warmed up (``eval_ema_state is None`` on round 1).
    """
    if eval_ema_decay > 0.0 and eval_ema_state is not None:
        real_state = global_model.state_dict()
        global_model.load_state_dict(eval_ema_state)
        metrics = evaluator.evaluate(global_model)
        global_model.load_state_dict(real_state)
        return metrics
    return evaluator.evaluate(global_model)


def apply_server_momentum(
    agg_state: Dict[str, torch.Tensor],
    global_state_before: Dict[str, torch.Tensor],
    momentum_buffer: Optional[Dict[str, torch.Tensor]],
    beta: float,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """FedAvgM-style server-side momentum on this round's aggregated update.

    Treats ``agg_state - global_state_before`` as this round's pseudo-
    gradient and applies it through an exponentially-weighted velocity
    buffer instead of directly:

        velocity  = beta * velocity_prev + (agg_state - global_state_before)
        new_state = global_state_before + velocity

    Targets a different symptom than the masked-head fix or FedProx: the
    *aggregated* global model swinging sharply round to round even when no
    individual client's local loss shows distress (see
    fidings/divergence_investigation.md sec 9) -- damping the applied update
    itself, at the server, rather than anything client-side.

    ``beta<=0`` is an exact no-op (returns ``agg_state`` unchanged, buffer
    reset to None) -- matches every other optional mechanism in this
    codebase (``mu``, ``dueling``, ``head_fix``) defaulting to "recovers
    plain FedAvg" at its off-value.

    Returns ``(new_state, new_momentum_buffer)`` -- the buffer must be
    threaded back in as ``momentum_buffer`` on the next round's call so
    velocity actually persists/accumulates across rounds.
    """
    if beta <= 0.0:
        return agg_state, None
    delta = _state_delta(agg_state, global_state_before)
    new_buffer = {
        k: beta * momentum_buffer[k] + v if momentum_buffer is not None and k in momentum_buffer else v
        for k, v in delta.items()
    }
    new_state = {k: global_state_before[k].float() + new_buffer[k] for k in agg_state}
    return new_state, new_buffer


def clip_pseudo_gradient(
    agg_state: Dict[str, torch.Tensor],
    global_state_before: Dict[str, torch.Tensor],
    max_norm: float,
) -> Dict[str, torch.Tensor]:
    """Cap the total L2 norm of this round's applied update
    (``agg_state - global_state_before``) at ``max_norm``, rescaling the
    whole delta uniformly (not per-tensor) if it's over the cap -- same
    style as ``torch.nn.utils.clip_grad_norm_`` but applied to the
    server-side aggregated update instead of a client's local gradient.

    Cheap insurance against one bad round moving the global model an
    outsized amount, independent of whatever's happening client-side.
    Composable with ``apply_server_momentum`` -- apply this first so the
    velocity buffer only ever accumulates already-bounded deltas, not an
    occasional huge spike.

    ``max_norm<=0`` is an exact no-op (returns ``agg_state`` unchanged) --
    matches every other optional mechanism in this codebase (``mu``,
    ``dueling``, ``server_momentum``) defaulting to "recovers plain FedAvg"
    at its off-value.
    """
    if max_norm <= 0.0:
        return agg_state
    delta = _state_delta(agg_state, global_state_before)
    total_norm = torch.sqrt(sum((v ** 2).sum() for v in delta.values()))
    if total_norm <= max_norm:
        return agg_state
    scale = max_norm / (total_norm + 1e-12)
    return {k: global_state_before[k].float() + v * scale for k, v in delta.items()}


def fed_avg(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Backward-compatible sample-weighted FedAvg.

    Equivalent to ``weighted_average`` with weights = num_samples --
    kept as a thin wrapper so any existing call sites that import
    ``fed_avg`` directly keep working unchanged.
    """
    state_dicts = [sd for sd, _ in updates]
    weights = [float(n) for _, n in updates]
    return weighted_average(state_dicts, weights)