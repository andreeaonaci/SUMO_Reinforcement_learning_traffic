"""Proper (second-order) MAML meta-learning aggregation, item 4 of the four
"significantly improve" candidates (fidings/divergence_investigation.md
sec 91, per direct user request). Distinct from item 24's `--fedavg_blend`
(sec 83, confirmed null): that is Reptile, a FIRST-ORDER proxy that treats
"the parameter delta after k local steps" as if it were a meta-gradient.
Real MAML (Finn et al. 2017) instead differentiates THROUGH the k inner
adaptation steps themselves, so the meta-gradient accounts for how each
inner step's direction depends on the starting point -- information Reptile
discards. Whether that extra fidelity matters enough to move this project's
numbers is exactly the open question item 4 exists to answer; nothing here
should be read as assuming the answer in advance.

Design, adapted to this project's DQN (not MAML's original supervised/
few-shot setting):
  - "Support" batches = several differentiable SGD steps on a city's own
    replay data, using ``torch.func.functional_call`` so autograd can track
    how each step's result depends on the STARTING (global) parameters --
    ordinary ``optimizer.step()`` mutates parameters in place and breaks
    that dependency, which is why this needs its own hand-rolled inner loop.
  - "Query" batch = a held-out batch from the SAME city, evaluated at the
    ADAPTED parameters -- its gradient w.r.t. the ORIGINAL global parameters
    (via the chain of inner steps) is that city's meta-gradient contribution.
  - The double-DQN bootstrap target within both support and query losses
    uses a frozen target-network snapshot (no grad), exactly the standard
    DQN target-network convention -- only the ONLINE network's parameters
    are ever part of the differentiable inner-loop chain.
  - Cities' meta-gradients are combined by a sample-count-weighted average
    (same convention as ``federated/aggregation.py``'s FedAvg) into one
    meta-gradient, applied as a single global optimizer step. This REPLACES
    weight-averaging entirely -- there is no "aggregate client state dicts"
    step in MAML, the meta-gradient IS the update.

Kept intentionally standalone (not wired into ``federated/aggregation_strategies.py``,
which only supports reweighting already-locally-trained client STATE DICTS,
not this second-order gradient-through-adaptation scheme) -- see
``diagnostics/maml_fedavg.py`` for the driver loop that uses this module.
"""
from typing import Dict, List, Tuple

try:
    import torch
    from torch.func import functional_call

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from agents.dqn import _collate, _mask_q


def _q_loss_functional(network, params: Dict[str, "torch.Tensor"], batch, target_params: Dict[str, "torch.Tensor"],
                        gamma: float, device) -> "torch.Tensor":
    """Ordinary Double-DQN Huber TD loss, expressed functionally: ``network``
    is called with an explicit ``params`` dict (via ``functional_call``)
    rather than its own internal (in-place, non-differentiable-through-time)
    parameters, so this can be invoked with the ADAPTED (inner-loop) params
    while still tracking the graph back to the original leaves. ``target_params``
    plays the exact role of ``self.q_target`` in ``agents/dqn.py::DQNAgent.optimize()``
    -- frozen, no gradient, same bootstrap semantics."""
    obs, actions, rewards, next_obs, dones, ns = batch
    own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs, device)
    n_own, n_neighbors, n_neighbor_mask, n_hop_dist, n_action_mask = _collate(next_obs, device)

    B = len(actions)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
    ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
    discount_t = torch.full_like(ns_t, gamma).pow(ns_t)

    q_values = functional_call(network, params, (own, neighbors, neighbor_mask, hop_dist))
    q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_online = _mask_q(
            functional_call(network, params, (n_own, n_neighbors, n_neighbor_mask, n_hop_dist)),
            n_action_mask,
        )
        next_actions = next_q_online.argmax(dim=1)
        next_q_target = functional_call(
            network, target_params, (n_own, n_neighbors, n_neighbor_mask, n_hop_dist)
        )
        next_q_taken = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards_t + (1.0 - dones_t) * discount_t * next_q_taken

    diff = q_taken - target
    huber = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
    return huber.mean()


def inner_adapt(network, theta: Dict[str, "torch.Tensor"], target_theta: Dict[str, "torch.Tensor"],
                 support_batches: List[tuple], inner_lr: float, gamma: float, device) -> Dict[str, "torch.Tensor"]:
    """``len(support_batches)`` differentiable SGD steps starting from
    ``theta``. ``create_graph=True`` is what makes this real MAML rather
    than Reptile: it keeps every intermediate step differentiable so the
    final adapted params are still a function of ``theta``'s leaves, not a
    detached numeric result."""
    params = theta
    for batch in support_batches:
        loss = _q_loss_functional(network, params, batch, target_theta, gamma, device)
        names = list(params.keys())
        grads = torch.autograd.grad(loss, list(params.values()), create_graph=True, allow_unused=True)
        params = {
            name: (p - inner_lr * g if g is not None else p)
            for name, p, g in zip(names, params.values(), grads)
        }
    return params


def maml_client_grad(network, global_state: Dict[str, "torch.Tensor"], target_state: Dict[str, "torch.Tensor"],
                      support_batches: List[tuple], query_batch, inner_lr: float, gamma: float,
                      device) -> Tuple[Dict[str, "torch.Tensor"], float]:
    """One client's (city's) contribution to the meta-gradient. Returns
    ``(grad_dict, query_loss_value)`` where ``grad_dict`` has the same keys
    as ``global_state`` and is the gradient of the post-adaptation query
    loss w.r.t. the PRE-adaptation (global) parameters."""
    theta = {k: v.detach().clone().requires_grad_(True) for k, v in global_state.items()}
    target_theta = {k: v.detach().clone() for k, v in target_state.items()}

    adapted = inner_adapt(network, theta, target_theta, support_batches, inner_lr, gamma, device)
    query_loss = _q_loss_functional(network, adapted, query_batch, target_theta, gamma, device)

    leaves = list(theta.values())
    grads = torch.autograd.grad(query_loss, leaves, allow_unused=True)
    grad_dict = {
        name: (g if g is not None else torch.zeros_like(p))
        for name, p, g in zip(theta.keys(), theta.values(), grads)
    }
    return grad_dict, float(query_loss.item())
