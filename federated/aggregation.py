from typing import Dict, List, Tuple
import copy
import torch


def fed_avg(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Perform Federated Averaging (FedAvg) on torch state_dicts.

    Args:
        updates: list of tuples (state_dict, n_samples)

    Returns:
        aggregated_state_dict
    """
    if not updates:
        raise ValueError("No updates to aggregate")

    total_samples = sum(n for _, n in updates)
    if total_samples <= 0:
        raise ValueError("Total samples must be positive")

    # start from zeros of first state
    base_state = copy.deepcopy(updates[0][0])
    for k in base_state.keys():
        base_state[k] = torch.zeros_like(base_state[k])

    for state, n in updates:
        weight = float(n) / float(total_samples)
        for k, v in state.items():
            if k not in base_state:
                raise KeyError(f"State key mismatch during aggregation: {k}")
            if base_state[k].shape != v.shape:
                raise ValueError(f"Shape mismatch for key {k}: {base_state[k].shape} vs {v.shape}")
            base_state[k] += v.to(base_state[k].dtype) * weight

    return base_state
