"""Pluggable client-weighting strategies for federated aggregation.

These are NOT training losses (MSE/Huber/PPO-loss/etc). They decide how much
each client's update should influence the aggregated global model each
round -- i.e. federated *aggregation* strategies (weighting/trust schemes),
analogous to FedAvg, FedProx, q-FedAvg, TrustFed, etc.

Usage
-----
    strategy = build_aggregation_strategy(cfg["aggregation_strategy"], cfg)
    infos    = [ClientRoundInfo(...), ...]        # one per client, this round
    weights  = strategy.compute_weights(infos)     # {client_id: weight}
    global_state = weighted_average(client_states, [weights[cid] for cid in ids])

Adding a new strategy
----------------------
Subclass ``BaseAggregationStrategy``, implement ``compute_client_score``,
then add one line to ``STRATEGY_REGISTRY`` at the bottom of this file.
Nothing else in the training pipeline needs to change.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Sequence, Type

import torch

from federated.clustering import cluster_cities
from federated.aggregation import weighted_average

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data carried into a strategy each round, per client
# ---------------------------------------------------------------------------

@dataclass
class ClientRoundInfo:
    """Everything a strategy might want to know about one client's round.

    Only ``client_id`` and ``num_samples`` are guaranteed populated -- every
    other field is optional and strategies must degrade gracefully when a
    field is missing (first round, gradient-less agent, comm dropout, etc).
    Don't assume any particular field is present; use the base class's
    helper methods, which already handle ``None`` safely.
    """
    client_id: str
    num_samples: int

    # Model weights
    client_state: Optional[Dict[str, torch.Tensor]] = None
    previous_client_state: Optional[Dict[str, torch.Tensor]] = None
    global_state: Optional[Dict[str, torch.Tensor]] = None
    previous_global_state: Optional[Dict[str, torch.Tensor]] = None

    # Pseudo-gradients. A federated client doesn't expose a real autograd
    # gradient to the server -- the natural stand-in is the parameter delta
    # produced by local training (client_state - state_it_started_from).
    # This is exactly how FedOpt / FedAvgM treat client updates. The
    # aggregator computes these once per round and passes them in; strategies
    # never need to know how they were derived.
    client_gradient: Optional[Dict[str, torch.Tensor]] = None
    previous_gradient: Optional[Dict[str, torch.Tensor]] = None
    global_gradient: Optional[Dict[str, torch.Tensor]] = None  # federation's own last move

    # Training signal
    local_loss: Optional[float] = None
    previous_loss: Optional[float] = None
    reward: Optional[float] = None
    previous_reward: Optional[float] = None

    # Bookkeeping
    episode: Optional[int] = None
    local_epochs: Optional[int] = None
    round_num: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseAggregationStrategy(ABC):
    """Common interface + shared math utilities for all weighting strategies.

    Subclasses implement only ``compute_client_score``. Score normalization,
    per-client scratch state (EMA values, gradient history, ...), cosine
    similarity, pseudo-gradient computation, and diagnostic logging all live
    here so concrete strategies stay short and strategy-specific.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}
        self.epsilon: float = float(self.config.get("epsilon", 1e-8))
        self.normalize_scores: bool = bool(self.config.get("normalize_scores", True))
        # Per-client scratch state (EMA values, gradient history, trust
        # scores, ...). Strategies read/write this instead of module-level
        # globals -- keeps a strategy instance self-contained and safe to
        # reuse across multiple concurrent training runs.
        self._client_state: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_client_score(self, info: ClientRoundInfo) -> float:
        """Return a scalar "how much should this client count" score.

        Higher is better; negative contributions should be clamped to 0 by
        the implementation (see the EMA strategies below). Scores are
        normalized into weights by ``compute_weights`` -- implementations
        don't need to normalize themselves.
        """
        raise NotImplementedError

    def compute_weights(self, infos: Sequence[ClientRoundInfo]) -> Dict[str, float]:
        """Score every client, then normalize into aggregation weights.

        Never raises just because one client's inputs were incomplete: a
        scoring failure or non-finite score is logged and clamped to 0
        rather than propagated. If every client ends up at 0 (fully
        degenerate round -- e.g. everyone's first round), falls back to
        uniform 1/N rather than producing an all-zero aggregation.
        """
        raw_scores: Dict[str, float] = {}
        for info in infos:
            try:
                score = float(self.compute_client_score(info))
            except Exception:
                logger.exception(
                    "Strategy '%s' failed to score client '%s'; using 0.",
                    type(self).__name__, info.client_id,
                )
                score = 0.0
            if not math.isfinite(score):
                logger.warning(
                    "Strategy '%s' produced non-finite score for client '%s' "
                    "(%r); clamping to 0.",
                    type(self).__name__, info.client_id, score,
                )
                score = 0.0
            raw_scores[info.client_id] = max(score, 0.0)

        weights = self._normalize(raw_scores) if self.normalize_scores else dict(raw_scores)
        self._log_round(infos, raw_scores, weights)
        return weights

    # ------------------------------------------------------------------
    # Shared math helpers -- available to every subclass
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(state: Optional[Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        """Flatten a state_dict / gradient dict into one 1-D tensor."""
        if not state:
            return None
        try:
            return torch.cat([v.detach().reshape(-1).float() for v in state.values()])
        except Exception:
            return None

    def cosine_similarity(
        self,
        a: Optional[Dict[str, torch.Tensor]],
        b: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[float]:
        """Cosine similarity between two flattened gradient/state dicts.

        Returns ``None`` (not 0.0) when either input is missing, shapes
        mismatch, or either vector has ~zero norm -- callers can then
        distinguish "no signal available" from "signal available and
        orthogonal" and choose an appropriate fallback.
        """
        fa, fb = self._flatten(a), self._flatten(b)
        if fa is None or fb is None or fa.shape != fb.shape:
            return None
        na, nb = fa.norm().item(), fb.norm().item()
        if na < self.epsilon or nb < self.epsilon:
            return None
        return float(torch.dot(fa, fb).item() / (na * nb))

    @staticmethod
    def compute_pseudo_gradient(
        end_state: Optional[Dict[str, torch.Tensor]],
        start_state: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """delta = end_state - start_state.

        Used as a stand-in gradient direction (client update direction, or
        the federation's own last move) when the agent doesn't expose real
        accumulated gradients to the server.
        """
        if not end_state or not start_state:
            return None
        try:
            return {
                k: end_state[k].detach().float() - start_state[k].detach().float()
                for k in end_state
                if k in start_state
            }
        except Exception:
            return None

    def ema_update(self, client_id: str, key: str, value: float, beta: float) -> float:
        """EMA = beta * EMA_prev + (1 - beta) * value, with lazy init."""
        state = self._client_state.setdefault(client_id, {})
        prev = state.get(key)
        new_val = value if prev is None else beta * prev + (1.0 - beta) * value
        state[key] = new_val
        return new_val

    def get_state(self, client_id: str) -> Dict[str, Any]:
        """Per-client scratch dict -- lazily created, persists across rounds."""
        return self._client_state.setdefault(client_id, {})

    def _compute_ema_delta_loss(self, info: ClientRoundInfo, beta: float) -> float:
        """Shared EMA(delta_loss) computation used by several strategies.

        delta_loss = previous_loss - current_loss  (positive == improving).
        Returns 0.0 (not None) when there's no loss history yet, so callers
        can use it directly in arithmetic without extra None-checks; a
        whole-round-of-zeros triggers the uniform-weight fallback in
        ``compute_weights`` automatically, which is the correct behavior
        for a cold start.
        """
        if info.local_loss is None:
            return 0.0
        if info.previous_loss is None:
            self.ema_update(info.client_id, "ema_delta_loss", 0.0, beta)
            return 0.0
        delta_loss = float(info.previous_loss) - float(info.local_loss)
        return self.ema_update(info.client_id, "ema_delta_loss", delta_loss, beta)

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        total = sum(scores.values())
        n = len(scores) or 1
        if total <= self.epsilon:
            logger.warning(
                "All client scores were ~0 this round; falling back to "
                "uniform weighting for %d client(s).", n,
            )
            return {cid: 1.0 / n for cid in scores}
        return {cid: s / total for cid, s in scores.items()}

    def _log_round(
        self,
        infos: Sequence[ClientRoundInfo],
        scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> None:
        for info in infos:
            cid = info.client_id
            # Only surface scalar diagnostics (ema, alignment, novelty,
            # survival, ...). Scratch state can also hold large blobs --
            # gradient_history (deque of state_dicts) and
            # _last_client_gradient (a full state_dict of tensors) -- which
            # would flood the log with raw tensor dumps if included.
            extra = {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.get_state(cid).items()
                if isinstance(v, (int, float, bool, str))
            }
            delta_loss = (
                f"{(info.previous_loss - info.local_loss):.6f}"
                if info.local_loss is not None and info.previous_loss is not None
                else "n/a"
            )
            logger.info(
                "[%s] client=%s  loss=%s  delta_loss=%s  score=%.6f  weight=%.4f  %s",
                type(self).__name__,
                cid,
                f"{info.local_loss:.6f}" if info.local_loss is not None else "n/a",
                delta_loss,
                scores.get(cid, 0.0),
                weights.get(cid, 0.0),
                extra,
            )


# ---------------------------------------------------------------------------
# 1. FedAvg -- baseline
# ---------------------------------------------------------------------------

class FedAvgStrategy(BaseAggregationStrategy):
    """Classic FedAvg: weight purely by how many samples the client trained on.

    Intuition: more environment steps -> lower-variance local gradient
    estimate -> should count for more. Ignores loss/gradient quality
    entirely; this is the baseline every other strategy is compared against.
    """
    def compute_client_score(self, info: ClientRoundInfo) -> float:
        return float(max(info.num_samples, 0))


# ---------------------------------------------------------------------------
# 2. EMA Loss Improvement
# ---------------------------------------------------------------------------

class EMALossImprovementStrategy(BaseAggregationStrategy):
    """Weight clients by a smoothed rate of loss improvement.

        delta_loss = previous_loss - current_loss
        EMA_t      = beta * EMA_{t-1} + (1 - beta) * delta_loss
        score      = max(EMA, 0)

    Intuition: a client whose loss keeps dropping is still discovering
    useful structure and should have more say in the global model. A client
    whose loss has flatlined (the "dead gradient" case) contributes ~0
    improvement and its score decays toward 0 -- exactly the behavior
    needed to stop a converged/stuck city from dragging the global average
    toward a stale local optimum.

    Config: ``ema_beta`` (default 0.9).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.beta = float(self.config.get("ema_beta", 0.9))

    def compute_client_score(self, info: ClientRoundInfo) -> float:
        ema = self._compute_ema_delta_loss(info, self.beta)
        return max(ema, 0.0)


# ---------------------------------------------------------------------------
# 3. EMA x Gradient Alignment
# ---------------------------------------------------------------------------

class EMAGradientAlignmentStrategy(BaseAggregationStrategy):
    """EMA(delta_loss) scaled by how aligned the client's update is with the
    federation's own recent direction.

        alignment = max(cos_sim(client_gradient, global_gradient), 0)
        score     = max(EMA(delta_loss), 0) * alignment

    Intuition: reward clients that are BOTH still learning AND pushing the
    model in a direction consistent with where the federation is already
    heading. Filters out clients that are improving locally but pulling the
    shared model somewhere incompatible with everyone else (client drift).
    Falls back to alignment=1 (i.e. degrades to strategy #2) when gradients
    aren't available, so a comm-dropout tick or a gradient-less agent never
    crashes the round.

    Config: ``ema_beta`` (default 0.9).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.beta = float(self.config.get("ema_beta", 0.9))

    def compute_client_score(self, info: ClientRoundInfo) -> float:
        ema = self._compute_ema_delta_loss(info, self.beta)

        alignment = self.cosine_similarity(info.client_gradient, info.global_gradient)
        if alignment is None:
            alignment = 1.0  # no signal available -- don't penalize
        alignment = max(alignment, 0.0)

        self.get_state(info.client_id)["alignment"] = alignment
        return max(ema, 0.0) * alignment


# ---------------------------------------------------------------------------
# 4. Learning Velocity x Novelty
# ---------------------------------------------------------------------------

class LearningVelocityNoveltyStrategy(BaseAggregationStrategy):
    """Reward clients that are learning fast AND pushing the model somewhere
    the federation hasn't already explored.

        velocity = max(EMA(delta_loss), 0)
        novelty  = max(1 - cos_sim(client_gradient, global_gradient), 0)
        score    = velocity * novelty

    Intuition: the mirror image of strategy #3. Alignment rewards "agreeing
    with the crowd"; novelty rewards clients learning something the current
    global model doesn't already reflect -- useful for surfacing an
    under-represented topology (e.g. one odd intersection shape) instead of
    always deferring to whichever city has the most samples. When no
    gradient signal is available, novelty falls back to a neutral 0.5
    rather than 0, so a client isn't silently zeroed out just because
    gradients were dropped for one round.

    Config: ``ema_beta`` (default 0.9).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.beta = float(self.config.get("ema_beta", 0.9))

    def compute_client_score(self, info: ClientRoundInfo) -> float:
        velocity = max(self._compute_ema_delta_loss(info, self.beta), 0.0)

        alignment = self.cosine_similarity(info.client_gradient, info.global_gradient)
        novelty = (1.0 - alignment) if alignment is not None else 0.5
        novelty = max(novelty, 0.0)

        self.get_state(info.client_id)["novelty"] = novelty
        return velocity * novelty


# ---------------------------------------------------------------------------
# 5. Gradient Survival Score
# ---------------------------------------------------------------------------

class GradientSurvivalStrategy(BaseAggregationStrategy):
    """Reward clients whose past updates keep "surviving" into the
    federation's later trajectory.

    Maintains a rolling window of each client's past gradients and a shared
    rolling window of the federation's own past global-gradients. Estimates:

        survival = mean over the window of
                   cos_sim(client_gradient(t), global_gradient(t'))
                   for t' in the last `survival_window` rounds (both
                   directions: this client's past vs. current global, and
                   this client's current vs. past global)

    i.e. does this client's direction keep resembling where the global
    model actually ends up moving? A client whose updates get consistently
    overwritten/undone by the rest of the federation scores low; a client
    that's steering consensus scores high. This is a coarse approximation
    (a rigorous version would need influence-function-style causal
    attribution) but is enough to separate "steering" clients from
    "overruled" ones, and the window is small and cheap to maintain.

    Config: ``survival_window`` (default 3).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.window = int(self.config.get("survival_window", 3))
        self._global_grad_history: Deque[Dict[str, torch.Tensor]] = deque(maxlen=self.window)

    def record_global_gradient(self, global_gradient: Optional[Dict[str, torch.Tensor]]) -> None:
        """Call once per round, after the new global pseudo-gradient is
        known, so next round's scoring has fresh history. The aggregator
        is responsible for calling this -- see the server integration.
        """
        if global_gradient is not None:
            self._global_grad_history.append(global_gradient)

    def compute_client_score(self, info: ClientRoundInfo) -> float:
        state = self.get_state(info.client_id)
        history: Deque[Dict[str, torch.Tensor]] = state.setdefault(
            "gradient_history", deque(maxlen=self.window)
        )

        similarities = []
        if info.client_gradient is not None:
            for past_global_grad in self._global_grad_history:
                sim = self.cosine_similarity(info.client_gradient, past_global_grad)
                if sim is not None:
                    similarities.append(sim)
        for past_client_grad in history:
            sim = self.cosine_similarity(past_client_grad, info.global_gradient)
            if sim is not None:
                similarities.append(sim)

        if info.client_gradient is not None:
            history.append(info.client_gradient)

        if not similarities:
            return 0.0  # no history yet -- uniform fallback handles cold start

        survival = float(sum(similarities) / len(similarities))
        state["survival"] = survival
        return max(survival, 0.0)


# ---------------------------------------------------------------------------
# 6. Clustered FedAvg
# ---------------------------------------------------------------------------

class ClusteredFedAvgStrategy(BaseAggregationStrategy):
    """Cluster clients by action_dim, then run FedAvg inside each cluster.

    This strategy intentionally returns one aggregated model per cluster.
    Server code must broadcast the cluster-specific model back to only the
    clients in that cluster.

    Config:
        n_clusters (int): number of clusters (default 2)
        cluster_weighting (str): 'samples' (default) or 'uniform'
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_clusters = int(self.config.get("n_clusters", 2))
        self.cluster_weighting = str(self.config.get("cluster_weighting", "samples")).lower()
        self._last_cluster_map: Dict[str, int] = {}

    def compute_client_score(self, info: ClientRoundInfo) -> float:
        return float(max(info.num_samples, 0))

    def assign_clusters(self, infos: Sequence[ClientRoundInfo]) -> Dict[str, int]:
        city_stats: Dict[str, Dict[str, int]] = {}
        for info in infos:
            action_dim = None
            if info.metadata:
                action_dim = info.metadata.get("action_dim")
            if action_dim is None and info.client_state is not None:
                head = info.client_state.get("head.4.bias")
                if head is not None:
                    action_dim = int(head.shape[0])
            if action_dim is None and info.global_state is not None:
                head = info.global_state.get("head.4.bias")
                if head is not None:
                    action_dim = int(head.shape[0])
            city_stats[info.client_id] = {"action_dim": int(action_dim or 1)}

        self._last_cluster_map = cluster_cities(city_stats, n_clusters=self.n_clusters)
        return dict(self._last_cluster_map)

    def aggregate_by_cluster(
        self,
        infos: Sequence[ClientRoundInfo],
        cluster_map: Dict[str, int],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        by_cluster: Dict[int, list[ClientRoundInfo]] = {}
        for info in infos:
            cluster_id = int(cluster_map.get(info.client_id, 0))
            by_cluster.setdefault(cluster_id, []).append(info)

        cluster_models: Dict[int, Dict[str, torch.Tensor]] = {}
        for cluster_id, members in by_cluster.items():
            member_states = [m.client_state for m in members if m.client_state is not None]
            if not member_states:
                continue

            if self.cluster_weighting == "uniform":
                weights = [1.0 for _ in members]
            else:
                weights = [float(max(m.num_samples, 0)) for m in members]
            cluster_models[cluster_id] = weighted_average(member_states, weights)

        return cluster_models

    def get_last_cluster_map(self) -> Dict[str, int]:
        return dict(self._last_cluster_map)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, Type[BaseAggregationStrategy]] = {
    "fedavg": FedAvgStrategy,
    "ema_loss": EMALossImprovementStrategy,
    "ema_alignment": EMAGradientAlignmentStrategy,
    "velocity_novelty": LearningVelocityNoveltyStrategy,
    "gradient_survival": GradientSurvivalStrategy,
    "clustered_fedavg": ClusteredFedAvgStrategy,
}


def build_aggregation_strategy(
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> BaseAggregationStrategy:
    """Factory: ``aggregation_strategy: "ema_alignment"`` -> instance.

    Raises a clear error listing valid names on typo/unknown strategy,
    rather than silently defaulting -- picking the wrong strategy silently
    would be a bad failure mode when comparing experiments.
    """
    key = (name or "fedavg").strip().lower()
    cls = STRATEGY_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown aggregation_strategy '{name}'. "
            f"Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return cls(config)


__all__ = [
    "ClientRoundInfo",
    "BaseAggregationStrategy",
    "FedAvgStrategy",
    "EMALossImprovementStrategy",
    "EMAGradientAlignmentStrategy",
    "LearningVelocityNoveltyStrategy",
    "GradientSurvivalStrategy",
    "ClusteredFedAvgStrategy",
    "STRATEGY_REGISTRY",
    "build_aggregation_strategy",
]