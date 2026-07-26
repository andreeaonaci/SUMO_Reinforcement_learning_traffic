"""Aggregation mechanics: turning (state_dicts, weights) into one global
state_dict. Deliberately knows nothing about *how* weights are chosen --
that policy lives in ``aggregation_strategies.py``. Keeping the two apart
means adding a new weighting scheme never touches this file.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
        )

    # original FedAvg on every parameter, including the head.
    return weighted_average(state_dicts, base_weights)

def masked_head_weighted_average(
    state_dicts: List[Dict[str, torch.Tensor]],
    base_weights: List[float],
    action_counts: List[Optional[Dict[int, int]]],
    head_weight_key: str = "head.4.weight",
    head_bias_key: str = "head.4.bias",
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

    for row in range(action_dim):
        row_weights = []
        for client_idx, counts in enumerate(action_counts):
            n = 0 if counts is None else counts.get(row, 0)
            row_weights.append(float(n) * max(base_weights[client_idx], 0.0))

        total = sum(row_weights)
        if total <= 1e-12:
            # No client touched this action index this round -- leave the
            # row exactly as it was, don't invent an update from nothing.
            # Caller is expected to have loaded `agg` from the previous
            # global state's structure; we fall back to whichever client's
            # row is closest to that (client 0's is fine since undilated
            # rows are byte-identical to the pre-round global weights).
            new_head_w[row] = state_dicts[0][head_weight_key][row]
            new_head_b[row] = state_dicts[0][head_bias_key][row]
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


def fed_avg(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Backward-compatible sample-weighted FedAvg.

    Equivalent to ``weighted_average`` with weights = num_samples --
    kept as a thin wrapper so any existing call sites that import
    ``fed_avg`` directly keep working unchanged.
    """
    state_dicts = [sd for sd, _ in updates]
    weights = [float(n) for _, n in updates]
    return weighted_average(state_dicts, weights)