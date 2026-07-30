"""City clustering helpers for federated aggregation."""

from __future__ import annotations

from typing import Dict


def cluster_cities(city_stats: Dict[str, dict], n_clusters: int = 2) -> Dict[str, int]:
    """Cluster cities by action_dim using deterministic sorted bucketing.

    Args:
        city_stats: mapping city -> stats dict; expects key ``action_dim``.
        n_clusters: desired number of clusters.

    Returns:
        Mapping ``{city_name: cluster_id}`` with cluster ids in ``[0, n_clusters-1]``.
    """
    if not city_stats:
        return {}

    n = max(1, int(n_clusters))
    ranked = sorted(
        ((name, int(stats.get("action_dim", 1))) for name, stats in city_stats.items()),
        key=lambda x: (x[1], x[0]),
    )

    # If we have fewer cities than clusters, collapse to one city per cluster.
    n_effective = min(n, len(ranked))
    bucket_size = (len(ranked) + n_effective - 1) // n_effective

    out: Dict[str, int] = {}
    for idx, (name, _action_dim) in enumerate(ranked):
        cluster_id = min(idx // bucket_size, n_effective - 1)
        out[name] = int(cluster_id)
    return out
