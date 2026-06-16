"""Placeholders for federated strategies beyond FedAvg."""

from typing import List, Tuple, Dict
import torch


def fed_prox(updates: List[Tuple[Dict[str, torch.Tensor], int]], mu: float = 0.1):
    """Stub for FedProx-like strategy. Currently delegates to FedAvg."""
    from federated.aggregation import fed_avg

    return fed_avg(updates)
