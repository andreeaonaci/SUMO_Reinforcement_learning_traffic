from typing import Dict, List, Tuple
import copy
from numbers import Number
import numpy as np


def _is_torch_tensor(x):
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def fed_avg(updates: List[Tuple[Dict, int]]) -> Dict:
    """Perform Federated Averaging on a list of (state_dict, sample_count).

    Args:
        updates: list of tuples (state_dict, n_samples)

    Returns:
        aggregated_state_dict
    """
    if not updates:
        raise ValueError("No updates to aggregate")

    # per-key aggregation only among matching shapes/types
    base_state = copy.deepcopy(updates[0][0])
    for k in list(base_state.keys()):
        # collect compatible tensors/arrays
        compatible = []
        for state, n in updates:
            if k not in state:
                continue
            v = state[k]
            # compare shapes/types with base
            v0 = base_state[k]
            try:
                if _is_torch_tensor(v0) and _is_torch_tensor(v):
                    compatible.append((v, n))
                else:
                    a0 = np.array(v0)
                    a = np.array(v)
                    if a.shape == a0.shape:
                        compatible.append((a, n))
            except Exception:
                continue

        if not compatible:
            # nothing compatible; keep original
            continue
        # weighted average among compatibles
        total = sum(n for _, n in compatible)
        if _is_torch_tensor(compatible[0][0]):
            import torch

            acc = torch.zeros_like(compatible[0][0])
            for v, n in compatible:
                acc += v * (n / total)
            base_state[k] = acc
        else:
            acc = np.zeros_like(np.array(compatible[0][0]))
            for v, n in compatible:
                acc = acc + (np.array(v) * (n / total))
            base_state[k] = acc

    return base_state
