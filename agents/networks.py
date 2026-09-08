"""Neural network architectures for intersection agents.

Foundation-model design
------------------------
A single shared architecture must work for ANY intersection topology
(3/4/5-way, protected lefts, pedestrian phases, ...) and for ANY amount of
available neighbor information -- 0 neighbors up to K_MAX neighbors, any
subset of which may be dropped out to simulate a communication failure.

Observation contract (per intersection, per tick)
--------------------------------------------------
    own_obs        (D_own,)          fixed-size own-intersection features
    neighbor_obs   (K_MAX, D_nbr)    fixed-size per-neighbor features,
                                     zero-padded for missing/dropped slots
    neighbor_mask  (K_MAX,)          1.0 = valid neighbor this tick,
                                     0.0 = padded OR comm-dropped
    hop_dist       (K_MAX,)          integer hop distance (1..K) of each
                                     neighbor slot; 0 for padding slots
    action_mask    (A_MAX,)          1.0 = this Q-slot is a real action for
                                     THIS intersection, 0.0 = doesn't exist
                                     (replaces manual phase_mapping)

The network never sees which city or topology an observation came from --
that's the whole point. Everything topology-specific is expressed purely
through the masks.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn


class _FlattenBatchNorm1d(nn.Module):
    """BatchNorm1d over the last dim of an arbitrarily-shaped (..., C) input.

    Plain ``nn.BatchNorm1d`` only accepts ``(N, C)`` or ``(N, C, L)`` --
    this project's own-intersection tensors are ``(B, C)`` but neighbor
    tensors are ``(B, K, C)`` (channel last, not matching either accepted
    layout). Flattening every leading dim into one batch dim before
    normalizing, then restoring the original shape, lets the SAME encoder
    code path batch-norm both without a special case.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return self.bn(x.reshape(-1, shape[-1])).reshape(shape)


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "relu6":
        return nn.ReLU6()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    raise ValueError(f"Unknown activation '{name}' -- expected relu/relu6/leaky_relu.")


def _mlp_block(dims: list, activation: str, use_batchnorm: bool, final_activation: bool) -> nn.Sequential:
    """Build a Linear-BN-activation stack matching whatever the pre-upgrade
    hardcoded nn.Sequential blocks looked like when activation='relu' and
    use_batchnorm=False (byte-identical -- this is a strict superset, not a
    behavior change, at those defaults). ``final_activation=False`` omits
    BN+activation after the LAST Linear (used for own_encoder/
    neighbor_encoder, whose output feeds attention as a raw embedding, not
    a hidden layer); True includes it (used for ``head``, whose final ReLU
    was already part of the original design)."""
    layers: list = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last_linear = i == len(dims) - 2
        if not is_last_linear or final_activation:
            if use_batchnorm:
                layers.append(_FlattenBatchNorm1d(dims[i + 1]))
            layers.append(_make_activation(activation))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Kept for backward compatibility / simple non-federated baselines."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeighborAttentionQNetwork(nn.Module):
    """Permutation-invariant, topology-agnostic Q-network.

    Own observation is the attention query; neighbor observations (up to
    K_MAX hops away, zero-padded, individually maskable) are the keys/
    values. Because masked attention naturally handles a variable number
    of *valid* neighbors, the exact same weights work for:

      - an isolated intersection (mask is all zero)
      - an intersection with 1 live neighbor out of K_MAX slots
      - an intersection with a full K_MAX-neighbor, multi-hop neighborhood

    No per-city or per-topology code is needed anywhere in this class.
    """

    def __init__(
        self,
        own_dim: int,
        neighbor_dim: int,
        action_dim: int,
        k_max: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_hops: int = 4,
        head_fix: bool = True,
        dueling: bool = False,
        actor_critic: bool = False,
        use_batchnorm: bool = False,
        activation: str = "relu",
        encoder_depth: int = 2,
        n_attn_layers: int = 1,
        recurrent: bool = False,
        topology_conditioned: bool = False,
        distributional: bool = False,
        n_quantiles: int = 21,
        bounded_q: bool = False,
        q_bound_scale: float = 5.0,
        lora_adapter: bool = False,
        lora_rank: int = 8,
        boot_heads: int = 1,
        phase_relational: bool = False,
        phase_dim: int = 10,
    ):
        super().__init__()
        if dueling and actor_critic:
            raise ValueError("dueling and actor_critic are mutually exclusive head types.")
        if boot_heads < 1:
            raise ValueError(f"boot_heads must be >= 1, got {boot_heads}.")
        if phase_relational and (dueling or actor_critic or distributional):
            raise ValueError(
                "phase_relational replaces the action-indexed Q-head entirely and is "
                "mutually exclusive with dueling/actor_critic/distributional."
            )
        if boot_heads > 1 and (dueling or actor_critic or distributional):
            raise ValueError(
                "boot_heads > 1 is mutually exclusive with dueling/actor_critic/distributional -- "
                "each of those replaces the plain action-indexed Q-head with a different head "
                "structure, and the bootstrapped vote is defined over plain per-action Q-heads. "
                "Combining them is possible in principle but untested; keep the pilot on the "
                "one variable."
            )
        if distributional and (dueling or actor_critic):
            raise ValueError("distributional is mutually exclusive with dueling/actor_critic.")
        if bounded_q and distributional:
            raise ValueError(
                "bounded_q is mutually exclusive with distributional -- bounding a point-Q "
                "spread doesn't apply to a quantile distribution, which already resists "
                "collapsing to an overconfident point estimate by construction (see "
                "distributional's own docstring above)."
            )
        self.k_max = k_max
        self.d_model = d_model
        self.n_hops = n_hops
        self.head_fix = head_fix
        self.dueling = dueling
        self.actor_critic = actor_critic
        self.action_dim = action_dim
        # Distributional RL (QR-DQN, Dabney et al. 2017): learn a distribution over
        # returns per action instead of a scalar Q-value -- structurally resists the
        # confident-lock-in pathology (§32-34) this project has characterized
        # extensively, since collapsing to an overconfident POINT estimate is exactly
        # what a distributional value function has to avoid representing (it always
        # keeps a spread across n_quantiles, even for a confidently-preferred
        # action). A genuinely different mechanism from every fix attempted so far
        # (none of which changed what KIND of value function is being learned).
        # distributional=False (default) leaves forward()/_q_from_features's return
        # shape and the head structure completely unchanged.
        self.distributional = distributional
        self.n_quantiles = n_quantiles
        # Bounded Q-head ("architecture-level retention" proposal, per direct user
        # request 2026-09-07): every previous confident-lock-in fix (q_entropy_weight,
        # CQL, anchor-revert) worked at the LOSS level -- discouraging an overconfident
        # Q-gap on average, while leaving the network fully capable of representing an
        # arbitrarily large one. Averages hide the outlier lock-in; a std=0.00 round
        # only needs the network to find that extreme ONCE and get stuck there. This
        # instead removes the capacity structurally: cap how far any action's Q-value
        # can deviate from that state's own mean Q via a tanh squash, so the top1-top2
        # gap has a hard ceiling (~2*q_bound_scale) no amount of training can exceed --
        # a capacity constraint, not a training-signal preference. Absolute Q magnitude
        # (the mean term) is left completely unconstrained; only the SPREAD across
        # actions for a given state is bounded, since that spread is what the
        # confident-lock-in mechanism (fidings sec 32-34) actually measures.
        # bounded_q=False (default) leaves _q_from_features's output byte-identical.
        self.bounded_q = bounded_q
        self.q_bound_scale = q_bound_scale
        # Number of independent action-indexed Q-heads on the shared trunk
        # (fidings sec 94). 1 = the original single head, an exact structural and
        # behavioral no-op: `self.boot_q` is never created and every code path
        # below falls through to the pre-existing `self.head` final Linear.
        self.boot_heads = int(boot_heads)
        self.phase_relational = bool(phase_relational)
        self.phase_dim = int(phase_dim)
        # "Upgraded DQN" (fidings/divergence_investigation.md, 2026-09-05):
        # BatchNorm1d + relu6/leaky_relu in place of the original plain-ReLU
        # design, tested against the overnight algorithm-swap campaign's
        # DQN+q_entropy result. use_batchnorm=False, activation="relu" is an
        # EXACT behavioral no-op -- _mlp_block reproduces the original
        # hardcoded Sequential blocks byte-for-byte at those defaults.
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        # "Deeper DQN" (fidings sec 75): more Linear layers in the own/
        # neighbor feature encoders -- deliberately NOT applied to `head`
        # (see below), whose structure federated/aggregation.py's masked-
        # head aggregation depends on by fixed index ("head.4.weight" is
        # looked up by name, not derived from depth -- changing head's
        # layer count would silently break that lookup, a confound this
        # experiment specifically avoids by only deepening the encoders).
        # encoder_depth=2 (default) reproduces the original 2-Linear
        # own_encoder/neighbor_encoder exactly.
        self.encoder_depth = encoder_depth

        self.own_encoder = _mlp_block(
            [own_dim] + [d_model] * encoder_depth, activation, use_batchnorm, final_activation=False
        )

        # +1 so "padding" (hop 0) gets its own embedding, distinct from a
        # real hop-1 neighbor.
        self.hop_embedding = nn.Embedding(n_hops + 1, d_model)

        self.neighbor_encoder = _mlp_block(
            [neighbor_dim] + [d_model] * encoder_depth, activation, use_batchnorm, final_activation=False
        )

        # "Stacked attention" (fidings sec 76): n_attn_layers=1 (default)
        # reproduces the original single-attention-pass design exactly (one
        # nn.MultiheadAttention + one LayerNorm, both indexed [0] in the
        # ModuleLists below). n_attn_layers>1 gives each intersection's own
        # embedding multiple independent (separately-weighted, not shared)
        # rounds of attention over its neighbors before the head trunk sees
        # it -- capacity added to the part of the network that actually
        # sees neighbor information, unlike encoder_depth (sec 75, which
        # added capacity to the raw-feature encoders instead and hurt
        # monotonically). The neighbor keys/values (kv) stay fixed across
        # rounds; only the query (the running own-representation) is
        # iteratively refined -- simpler than a full stacked Transformer
        # encoder (which would also update kv each layer) but still a
        # genuinely different architecture, not just deeper MLPs.
        self.n_attn_layers = n_attn_layers
        attn_layers = []
        attn_norms = []
        attn_batch_first = True
        for _ in range(n_attn_layers):
            try:
                attn_layers.append(nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=n_heads, batch_first=True
                ))
                attn_batch_first = True
            except TypeError:
                attn_layers.append(nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=n_heads
                ))
                attn_batch_first = False
            attn_norms.append(nn.LayerNorm(d_model))
        self.attn_layers = nn.ModuleList(attn_layers)
        self.attn_norms = nn.ModuleList(attn_norms)
        self._attn_batch_first = attn_batch_first

        # Learnable fallback so a fully isolated intersection (mask all
        # zero) still attends to a well-defined value instead of a
        # degenerate all-masked softmax.
        self.no_neighbor_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Trunk shared by both the plain and dueling heads. Kept as a
        # 4-element Sequential (Linear, ReLU, Linear, ReLU) so the plain
        # (non-dueling) path can still append a single final Linear at
        # index 4 and keep the "head.4.weight"/"head.4.bias" key names
        # `federated/aggregation.py::masked_head_weighted_average` already
        # looks for by default -- no aggregation-side change needed unless
        # dueling is actually turned on.
        self.head = _mlp_block(
            [d_model * 2, d_model, d_model], activation, use_batchnorm, final_activation=True
        )

        if self.dueling:
            # V(s): one scalar per intersection, no action_mask involved --
            # every client updates every element of this every step, so
            # (unlike the fully action-indexed plain head) it aggregates
            # cleanly with an ordinary weighted average across cities of
            # any action_dim. A(s,a): the actual action-indexed stream,
            # still exactly action_dim wide -- masked-head aggregation
            # (see federated/aggregation.py) still applies to this one.
            self.value_head = nn.Linear(d_model, 1)
            self.advantage_head = nn.Linear(d_model, action_dim)
        elif self.actor_critic:
            # Same shared trunk as the dueling head, split into a policy
            # (action logits, masked+softmaxed by the caller -- this class
            # never applies action_mask itself, matching the plain/dueling
            # Q-head convention) and a state-value scalar. Kept as two
            # separate Linears (not fused) so PPOAgent can read raw logits
            # and value independently without slicing one tensor apart.
            self.policy_head = nn.Linear(d_model, action_dim)
            self.ac_value_head = nn.Linear(d_model, 1)
        elif self.distributional:
            # action_dim * n_quantiles outputs, reshaped to (B, action_dim,
            # n_quantiles) in _q_from_features -- QRDQNAgent reads the full
            # per-action quantile set via forward_quantiles(); every other
            # caller (action selection, _mask_q, the evaluator) sees ordinary
            # (B, action_dim) mean-Q values via the unchanged forward() path.
            self.head.append(nn.Linear(d_model, action_dim * n_quantiles))
        elif self.phase_relational:
            # Phase-relational Q-head (fidings sec 96). There is NO per-action row
            # here at all: one shared scorer maps [state features, phase features]
            # -> a scalar Q, and is applied to every candidate phase. That single
            # change removes three separate structural problems this project has
            # measured:
            #
            #   * untrained action rows (sec 95b) -- a 3-phase city trains exactly
            #     the same parameters an 8-phase holdout uses, so no row can be
            #     left at initialization the way rows 5-7 were;
            #   * index semantics (sec 95c) -- there are no indices to carry
            #     conflicting meanings across cities;
            #   * max_pressure being outside the hypothesis space -- phase feature
            #     0 IS that phase's pressure, so this head can express
            #     "Q(s,a) = pressure(a)" directly and match the baseline as a
            #     FLOOR rather than having to rediscover it from scratch.
            #
            # Width is action_dim-independent, which is what makes one policy over
            # arbitrary topologies actually representable rather than merely masked.
            self.phase_scorer = _mlp_block(
                [d_model + phase_dim, d_model, d_model], activation, use_batchnorm,
                final_activation=True,
            )
            self.phase_scorer.append(nn.Linear(d_model, 1))
        elif self.boot_heads > 1:
            # Bootstrapped multi-head (fidings sec 94), built directly on sec 93's
            # measured result: a majority VOTE across independently-trained models
            # escaped the confident lock-in that every individual member was in
            # (ensemble episode-std 422 vs members' 24-204), while a weight-space
            # average of the same members did not (std 40.49 -- it blends the locked
            # members in rather than outvoting them). This makes that structure
            # internal to one network: K action-indexed heads on the shared trunk,
            # combined by vote at action-selection time.
            #
            # Deliberately K SEPARATE Linear(d_model, action_dim) modules rather than
            # one fused Linear(d_model, action_dim * K): each stays exactly
            # one-row-per-action, so masked-head aggregation still applies to every
            # head (see federated/aggregation.py, which now takes a LIST of head
            # keys). A fused head would have action_dim*K rows and silently break
            # that per-action row indexing -- the trap `distributional` avoids only
            # by being excluded from masked-head aggregation entirely.
            self.boot_q = nn.ModuleList(
                [nn.Linear(d_model, action_dim) for _ in range(self.boot_heads)]
            )
        else:
            self.head.append(nn.Linear(d_model, action_dim))

        if not self.head_fix:
            self.pool_head = _mlp_block(
                [d_model] + [d_model] * encoder_depth, activation, use_batchnorm, final_activation=False
            )

        # Recurrent policy (item 23, fidings/divergence_investigation.md): every
        # architecture tried before this (wider/deeper/more-attention) was still a
        # purely reactive function of one tick's snapshot. A GRUCell sits between
        # `_combined_features` and `self.head`, same width in and out (d_model*2)
        # so `self.head`'s structure -- and the "head.4.weight" key masked-head
        # aggregation depends on by name -- is completely untouched regardless of
        # this flag. recurrent=False (default) never constructs `self.gru` at all,
        # an exact no-op matching every other architecture knob in this class.
        self.recurrent = recurrent
        if self.recurrent:
            self.gru = nn.GRUCell(d_model * 2, d_model * 2)

        # Topology-Conditioned FedAvg (item "TC-FedAvg", fidings/divergence_
        # investigation.md): every aggregation-strategy tweak tried in this
        # project (EMA-loss/-alignment weighting, clustered-by-action-dim,
        # gradient-survival, velocity-novelty) came back null, and federation
        # vs. no-federation makes no measurable difference either (sec 49/50/
        # 64) -- evidence the problem was never in HOW weights get combined
        # across cities. What's missing is a way for the ONE shared function
        # being averaged to behave differently for a 3-way vs. a 5-way
        # intersection in the first place; right now only the final Q-head
        # (via action_mask) gets that treatment. A small shared hypernetwork
        # maps a per-intersection topology descriptor -- valid-action
        # fraction, valid-neighbor fraction, mean/max hop distance of live
        # neighbors, ALL computable for any intersection including one never
        # trained on -- to a FiLM (Perez et al. 2018) scale/shift applied to
        # `combined`, same injection point as `recurrent` above. FedAvg
        # aggregation itself is completely unchanged: `topo_hyper`'s weights
        # are shared and averaged exactly like every other layer -- there is
        # nothing city-specific to carve out, since the conditioning comes
        # from the INPUT (the descriptor), not from per-city parameters.
        # Zero-initializing the last layer makes this an exact identity
        # transform at the start of training (gamma=0, beta=0 ->
        # combined*(1+0)+0 == combined), standard practice for adapter-style
        # layers so they don't destabilize early training.
        self.topology_conditioned = topology_conditioned
        if self.topology_conditioned:
            self._topo_dim = 4
            self.topo_hyper = nn.Sequential(
                nn.Linear(self._topo_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 2 * d_model * 2),
            )
            nn.init.zeros_(self.topo_hyper[-1].weight)
            nn.init.zeros_(self.topo_hyper[-1].bias)

        # LoRA-style low-rank adapter ("architecture-level retention" proposal,
        # per direct user request 2026-09-07, tried after --trunk_lr_scale's negative
        # result showed that RESTRICTING the trunk's own learning starves it (it's
        # still random-init early in training and needs to adapt fast, not slowly).
        # This instead ADDS a small low-rank residual correction on top of `combined`
        # -- the trunk keeps training completely normally, at full LR, with zero
        # restriction; the adapter is pure extra capacity, not a reallocation of
        # existing capacity. down: (d_model*2 -> lora_rank), up: (lora_rank ->
        # d_model*2), up-projection zero-initialized so lora_adapter=True starts as
        # an EXACT identity (combined + up(down(combined)) == combined at init,
        # since up's weights are all zero) -- same zero-init-adapter convention as
        # `topo_hyper` above, so this can never destabilize early training the way
        # an arbitrarily-initialized residual branch could.
        self.lora_adapter = lora_adapter
        self.lora_rank = lora_rank
        if self.lora_adapter:
            self.lora_down = nn.Linear(d_model * 2, lora_rank, bias=False)
            self.lora_up = nn.Linear(lora_rank, d_model * 2, bias=False)
            nn.init.zeros_(self.lora_up.weight)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Backward-compat shim: checkpoints saved before the "stacked
        attention" refactor (fidings sec 76, 2026-09-05) used flat
        `attn.*`/`attn_norm.*` keys for what is now `attn_layers.0.*`/
        `attn_norms.0.*` (the n_attn_layers=1 case, which is architecturally
        identical -- this is a pure key rename, not a shape change).
        Without this, every checkpoint from before that commit -- most of
        this project's saved results -- fails to load with a cryptic
        missing/unexpected-key error. Newer checkpoints (already using
        `attn_layers.*`) pass through unchanged."""
        if any(k == "attn.in_proj_weight" or k == "attn_norm.weight" for k in state_dict):
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("attn."):
                    k = "attn_layers.0." + k[len("attn."):]
                elif k.startswith("attn_norm."):
                    k = "attn_norms.0." + k[len("attn_norm."):]
                remapped[k] = v
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def forward_phase(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        phase_feats: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Phase-relational entry point: -> (B, A) Q-values.

        ``phase_feats`` is (B, A, phase_dim) -- what each candidate phase would
        actually do. The state representation is computed ONCE and broadcast
        against every phase, so cost is one trunk pass plus A cheap scorer
        passes, and the parameter count is independent of A.

        Unmasked, matching every other forward here -- callers apply
        ``action_mask`` themselves (this project's standing convention).
        """
        if not self.phase_relational:
            raise RuntimeError("forward_phase() called on a non-phase-relational network.")
        combined = self._combined_features(own_obs, neighbor_obs, neighbor_mask, hop_dist)
        state_feat = self.head(combined)                      # (B, d_model)
        b, a, _ = phase_feats.shape
        state_rep = state_feat.unsqueeze(1).expand(b, a, state_feat.shape[-1])
        pair = torch.cat([state_rep, phase_feats], dim=-1)    # (B, A, d_model+phase_dim)
        return self.phase_scorer(pair).squeeze(-1)            # (B, A)

    def q_per_head(self, combined: torch.Tensor) -> torch.Tensor:
        """(B, boot_heads, action_dim) -- every head's own Q-values.

        Only valid when ``boot_heads > 1``. The shared trunk runs ONCE and each
        head is a single Linear on top, so K heads cost K*d_model*action_dim
        extra multiply-adds per forward, not K full forward passes -- the
        practical reason this is worth trying as an architecture rather than
        just running sec 93's 6-model ensemble in production.

        Used by the training loss (each head gets its own TD target, bootstrapped
        off its own target-network head -- that self-referential difference is
        what keeps the heads from converging to one function) and by the voting
        action selection in DQNAgent.act().
        """
        if self.boot_heads <= 1:
            raise RuntimeError("q_per_head() requires boot_heads > 1.")
        feat = self.head(combined)
        return torch.stack([h(feat) for h in self.boot_q], dim=1)

    def _q_from_features(self, combined: torch.Tensor) -> torch.Tensor:
        """Shared trunk -> Q-values, either straight through the plain head,
        combined dueling-style (Q = V + A - mean(A)) if ``dueling``, or the
        per-quantile mean if ``distributional`` -- every caller of this
        method (action selection, ``_mask_q``, the evaluator) sees ordinary
        (B, action_dim) values regardless of which head type is active.
        If ``bounded_q``, the plain/dueling result is additionally squashed
        (see __init__'s comment) so its cross-action SPREAD can't exceed a
        hard ceiling -- absolute Q magnitude is untouched."""
        feat = self.head(combined)
        if self.dueling:
            value = self.value_head(feat)
            advantage = self.advantage_head(feat)
            raw_q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        elif self.distributional:
            quantiles = feat.view(feat.shape[0], self.action_dim, self.n_quantiles)
            return quantiles.mean(dim=-1)
        elif self.boot_heads > 1:
            # MEAN across heads, so every existing caller (Q-gap diagnostics,
            # _mask_q, the evaluator's q_values) keeps seeing ordinary
            # (B, action_dim) values. Action SELECTION does not go through here
            # when voting is enabled -- DQNAgent.act() calls q_per_head() and
            # votes, which is the whole point (sec 93: the vote beat the average).
            raw_q = self.q_per_head(combined).mean(dim=1)
        else:
            raw_q = feat

        if self.bounded_q:
            mean_q = raw_q.mean(dim=-1, keepdim=True)
            centered = raw_q - mean_q
            bounded_centered = self.q_bound_scale * torch.tanh(centered / self.q_bound_scale)
            return mean_q + bounded_centered
        return raw_q

    def _combined_features(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Own-obs + attended (or mean-pooled, if not head_fix) neighbor
        summary, concatenated -- everything upstream of ``self.head``.
        Factored out of ``forward`` so ``forward_actor_critic`` can reuse
        the exact same trunk instead of duplicating this logic."""
        B, K, _ = neighbor_obs.shape

        own_emb = self.own_encoder(own_obs)  # (B, d_model)
        nbr_emb = self.neighbor_encoder(neighbor_obs)  # (B, K, d_model)

        if not self.head_fix:
            valid = neighbor_mask.clamp(min=0.0, max=1.0)
            nbr_pool = torch.sum(nbr_emb * valid.unsqueeze(-1), dim=1)
            nbr_count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            nbr_pool = nbr_pool / nbr_count
            nbr_pool = self.pool_head(nbr_pool)
            return torch.cat([own_emb, nbr_pool], dim=-1)

        if hop_dist is not None:
            hop_dist = hop_dist.clamp(0, self.n_hops)
            nbr_emb = nbr_emb + self.hop_embedding(hop_dist.long())

        fallback = self.no_neighbor_token.expand(B, 1, -1)
        kv = torch.cat([nbr_emb, fallback], dim=1)  # (B, K+1, d_model)
        fallback_mask = torch.ones(B, 1, device=neighbor_mask.device)
        full_mask = torch.cat([neighbor_mask, fallback_mask], dim=1)  # (B, K+1)

        key_padding_mask = full_mask < 0.5  # True = ignore this position

        # Iteratively refine the own-representation across n_attn_layers
        # rounds of attention over the SAME (fixed) neighbor kv -- see the
        # __init__ comment above for why this differs from a full stacked
        # Transformer encoder. own_repr is the running query; own_emb
        # itself never changes and is what gets concatenated at the end,
        # matching the original single-layer design's semantics exactly
        # when n_attn_layers=1 (own_repr after one round == that design's
        # attn_out).
        own_repr = own_emb
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            query = own_repr.unsqueeze(1)  # (B, 1, d_model)
            if self._attn_batch_first:
                attn_out, _ = attn(query, kv, kv, key_padding_mask=key_padding_mask)
                attn_out = attn_out.squeeze(1)
            else:
                # Older torch versions only support (seq, batch, embed).
                query_t = query.transpose(0, 1)
                kv_t = kv.transpose(0, 1)
                attn_out, _ = attn(query_t, kv_t, kv_t, key_padding_mask=key_padding_mask)
                attn_out = attn_out.transpose(0, 1).squeeze(1)
            own_repr = norm(attn_out + own_repr)  # residual

        return torch.cat([own_emb, own_repr], dim=-1)

    def _topology_descriptor(
        self,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor],
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """4-dim per-intersection structural descriptor: valid-action
        fraction, valid-neighbor fraction, mean/max hop distance of live
        neighbors (0 if isolated). Every component is computable for ANY
        intersection -- including one never trained on -- from tensors
        already in the observation contract; nothing here is a city
        identity or anything that requires having seen this topology before.
        """
        valid_action_frac = (action_mask > 0.5).float().mean(dim=1, keepdim=True)
        valid_nbr_mask = (neighbor_mask > 0.5).float()
        valid_nbr_frac = valid_nbr_mask.mean(dim=1, keepdim=True)
        if hop_dist is None:
            hop_dist = torch.zeros_like(neighbor_mask)
        hop_f = hop_dist.float()
        denom = valid_nbr_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        norm = max(self.n_hops, 1)
        mean_hop = (hop_f * valid_nbr_mask).sum(dim=1, keepdim=True) / denom / norm
        # No valid neighbors -> masked hop values are all 0 -> max is 0, correctly
        # signaling "isolated" rather than an arbitrary padding hop value.
        max_hop = (hop_f * valid_nbr_mask).max(dim=1, keepdim=True).values / norm
        return torch.cat([valid_action_frac, valid_nbr_frac, mean_hop, max_hop], dim=1)

    def forward(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DQN entry point: own+neighbor obs -> masked-ready Q-values
        (plain or dueling-combined depending on ``self.dueling``).
        ``action_mask`` is only used (and only required) when
        ``self.topology_conditioned`` -- see that flag's __init__ comment --
        to build the per-intersection topology descriptor; it plays no part
        in masking this method's OUTPUT, which callers still do themselves
        via ``_mask_q``, exactly as before."""
        if self.recurrent:
            # Loud, not silent (project convention, e.g. RewardShapingWrapper's
            # _extract_local_metric): a recurrent net's Q-values depend on a hidden
            # state the caller must manage explicitly (per intersection, reset every
            # episode) -- silently starting from zero hidden state on every call
            # would look like it works while quietly discarding all temporal memory.
            raise RuntimeError(
                "This network was built with recurrent=True -- call forward_recurrent("
                "..., hidden) instead of forward(), which has no hidden state to work with."
            )
        combined = self._combined_features(own_obs, neighbor_obs, neighbor_mask, hop_dist)
        if self.lora_adapter:
            combined = combined + self.lora_up(self.lora_down(combined))
        if self.topology_conditioned:
            if action_mask is None:
                raise RuntimeError(
                    "This network was built with topology_conditioned=True -- forward() "
                    "requires action_mask (used to build the topology descriptor, not to "
                    "mask output -- callers still do that separately via _mask_q)."
                )
            t = self._topology_descriptor(neighbor_mask, hop_dist, action_mask)
            gamma, beta = self.topo_hyper(t).chunk(2, dim=-1)
            combined = combined * (1.0 + gamma) + beta
        return self._q_from_features(combined)

    def forward_recurrent(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent entry point: own+neighbor obs + previous hidden state ->
        (masked-ready Q-values, new hidden state). ``hidden`` is (B, d_model*2);
        None means "start of episode" (fed as zeros) -- callers (RecurrentDQNAgent)
        are responsible for carrying the returned hidden state to the next tick and
        resetting it to None/zeros at every episode boundary."""
        if not self.recurrent:
            raise RuntimeError(
                "This network was built with recurrent=False -- call forward(...) "
                "instead, or construct with recurrent=True to use forward_recurrent."
            )
        combined = self._combined_features(own_obs, neighbor_obs, neighbor_mask, hop_dist)
        if hidden is None:
            hidden = torch.zeros_like(combined)
        new_hidden = self.gru(combined, hidden)
        return self._q_from_features(new_hidden), new_hidden

    def forward_actor_critic(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PPO entry point: own+neighbor obs -> (action logits, state
        value). Only valid when ``self.actor_critic=True``. Logits are
        raw (not masked/softmaxed) -- the caller applies action_mask,
        matching the convention ``forward``'s Q-values follow with
        ``_mask_q`` in agents/dqn.py."""
        if not self.actor_critic:
            raise RuntimeError("forward_actor_critic() called on a non-actor-critic network.")
        combined = self._combined_features(own_obs, neighbor_obs, neighbor_mask, hop_dist)
        feat = self.head(combined)
        logits = self.policy_head(feat)
        value = self.ac_value_head(feat).squeeze(-1)
        return logits, value

    def forward_quantiles(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """QRDQNAgent entry point: own+neighbor obs -> (B, action_dim,
        n_quantiles) full per-action return distributions. Only valid when
        ``self.distributional=True``. Unmasked -- QRDQNAgent's optimize()
        masks invalid actions itself before any argmax/loss computation,
        matching this project's standing convention (``_mask_q`` is applied
        by callers, not by this network)."""
        if not self.distributional:
            raise RuntimeError("forward_quantiles() called on a non-distributional network.")
        combined = self._combined_features(own_obs, neighbor_obs, neighbor_mask, hop_dist)
        feat = self.head(combined)
        return feat.view(feat.shape[0], self.action_dim, self.n_quantiles)

    def forward_boot(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bootstrapped multi-head entry point: own+neighbor obs -> (B,
        boot_heads, action_dim). Only valid when ``boot_heads > 1``. Unmasked,
        matching ``forward_quantiles``/``forward`` -- callers apply
        ``action_mask`` themselves (this project's standing convention)."""
        combined = self._combined_features(own_obs, neighbor_obs, neighbor_mask, hop_dist)
        return self.q_per_head(combined)
