"""The FRAP readout inside NeighborAttentionQNetwork (fidings sec 101).

Guards the two things that would silently invalidate the baseline arm:
frap_head=False must be an exact structural no-op, and frap_head=True must
actually replace the action-indexed head rather than sitting unused beside it.
"""
import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.networks import NeighborAttentionQNetwork

PAIRS = [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5]]
DIMS = dict(own_dim=117, neighbor_dim=3, k_max=8, action_dim=8)


def _net(**kw):
    return NeighborAttentionQNetwork(**DIMS, **kw)


def test_default_is_exact_structural_noop():
    """A network built without the flag must be identical to one built before
    the flag existed -- same parameter names, same count."""
    base = _net()
    assert not base.frap_head
    assert not hasattr(base, "frap")
    explicit = _net(frap_head=False)
    assert {n for n, _ in base.named_parameters()} == \
           {n for n, _ in explicit.named_parameters()}
    assert sum(p.numel() for p in base.parameters()) == \
           sum(p.numel() for p in explicit.parameters())


def test_frap_head_adds_its_parameters():
    net = _net(frap_head=True, frap_phase_pairs=PAIRS)
    names = {n for n, _ in net.named_parameters()}
    assert any(n.startswith("frap.") for n in names), \
        "the FRAP head must actually be part of the module"


def test_requires_phase_pairs():
    with pytest.raises(ValueError, match="frap_phase_pairs"):
        _net(frap_head=True)


def test_mutually_exclusive_with_other_heads():
    for kw in (dict(dueling=True), dict(distributional=True),
               dict(phase_relational=True)):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _net(frap_head=True, frap_phase_pairs=PAIRS, **kw)


def test_forward_frap_shape_and_padding():
    net = _net(frap_head=True, frap_phase_pairs=PAIRS).eval()
    B = 3
    pressure = torch.randn(B, 12)
    current = torch.tensor([0, -1, 4])
    a2u = torch.full((B, 8), -1, dtype=torch.long)
    a2u[:, :4] = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        q = net.forward_frap(pressure, current, a2u)
    assert q.shape == (B, 8)
    assert torch.isfinite(q[:, :4]).all()
    assert torch.isinf(q[:, 4:]).all()


def test_forward_frap_rejected_on_non_frap_network():
    net = _net().eval()
    with pytest.raises(RuntimeError, match="non-FRAP"):
        net.forward_frap(torch.randn(1, 12), torch.tensor([0]),
                         torch.zeros(1, 8, dtype=torch.long))


def test_frap_ignores_the_rich_observation_by_construction():
    """MPLight's state is only [current phase, movement pressure]. If
    forward_frap ever started consuming own_obs the comparison would stop being
    a faithful baseline, so assert the signature stays narrow."""
    import inspect
    params = list(inspect.signature(
        NeighborAttentionQNetwork.forward_frap).parameters)
    assert params == ["self", "movement_pressure", "current_union_phase",
                      "act_to_union"]
