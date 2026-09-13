"""FRAP -- MPLight's Q-head (Chen et al., AAAI 2020; Zheng et al., CIKM 2019),
ported into this project's pipeline as a BASELINE readout.

Why this exists (fidings sec 101). §96-§100 show a phase-relational readout
transfers across topologies where a positionally-indexed one does not. FRAP is
the closest published phase-invariant readout, and MPLight is FRAP plus pressure,
so a faithful FRAP arm is the baseline a reviewer will ask for. The comparison is
deliberately GENEROUS to FRAP: it is handed RESCO's own hand-authored
`phase_pairs` / `pair_to_act_map` / `lane_sets` for every city INCLUDING the
unseen holdout, while the phase-relational arm derives everything from the
simulator. The claim under test is therefore parity-without-configuration, not
"we beat FRAP".

Ported from RESCO's `resco_benchmark/agents/action_value/mplight.py::FRAP`:
per-movement demand embedding, phase = a PAIR of movements, all ordered pairs of
phases compared through 1x1 convolutions gated by a competition mask, summed to
one score per phase. The arithmetic here is theirs; what differs is stated in
ADAPTATIONS below.

ADAPTATIONS, all forced by this project's multi-city setting and all recorded:

1. RESCO trains MPLight on ONE scenario at a time and sizes FRAP's output to that
   scenario's `phase_pairs` (arterial4x4=5, grid4x4=8, cologne3=9,
   ingolstadt7=11). A shared cross-city head cannot do that, so this head is
   sized to the UNION of the training cities' pairs (11) and each intersection's
   local actions are gathered out of it via `act_to_union`. sec 101 measured that
   all 8 of grid4x4's pairs already appear in that union, so the holdout is fully
   covered and this costs FRAP nothing.
2. `act_to_union` travels in the OBSERVATION, not in code, so the network still
   never learns which city it is looking at -- the same discipline `action_mask`
   follows. Slots with -1 are padding and are masked to -inf.
3. Demand is per-movement PRESSURE (inbound queue minus downstream queue), which
   is exactly RESCO's `mplight` state with `demand_shape=1`.
"""
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# RESCO's canonical movement table (docs/Environment Configuration.md).
N_MOVEMENTS = 12


def build_competition_mask(phase_pairs: Sequence[Sequence[int]]) -> torch.Tensor:
    """(P, P-1) 0/1 mask: do phases i and j share a movement direction?

    RESCO's `build_comp_mask`. Two phases compete when the union of their two
    movement pairs has exactly 3 distinct members, i.e. they share exactly one
    movement and therefore cannot both run.
    """
    mask = []
    for i in range(len(phase_pairs)):
        row = np.zeros(len(phase_pairs) - 1, dtype=np.int64)
        col = 0
        for j in range(len(phase_pairs)):
            if i == j:
                continue
            if len(set(list(phase_pairs[i]) + list(phase_pairs[j]))) == 3:
                row[col] = 1
            col += 1
        mask.append(row)
    return torch.from_numpy(np.asarray(mask))


class FRAPHead(nn.Module):
    """Q over a fixed set of movement-pair phases, gathered to local actions.

    Parameter count is independent of how many actions any particular
    intersection has -- what varies is only which rows of the union table that
    intersection's `act_to_union` selects.
    """

    def __init__(self, phase_pairs: Sequence[Sequence[int]], action_dim: int,
                 demand_shape: int = 1, lane_embed_units: int = 16,
                 relation_embed_size: int = 4, conv_units: int = 20):
        super().__init__()
        self.phase_pairs = [list(p) for p in phase_pairs]
        self.n_phases = len(self.phase_pairs)
        self.action_dim = action_dim
        self.demand_shape = demand_shape
        self.lane_embed_units = lane_embed_units

        self.d_out = 4      # demand embedding width   (RESCO's value)
        self.p_out = 4      # phase-flag embedding width (RESCO's value)

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(demand_shape, self.d_out)
        self.lane_embedding = nn.Linear(self.p_out + self.d_out, lane_embed_units)
        self.lane_conv = nn.Conv2d(2 * lane_embed_units, conv_units, kernel_size=(1, 1))
        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, conv_units, kernel_size=(1, 1))
        self.hidden_layer = nn.Conv2d(conv_units, conv_units, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(conv_units, 1, kernel_size=(1, 1))

        self.register_buffer("comp_mask", build_competition_mask(self.phase_pairs))
        # (P, 2) movement indices per phase, for the current-phase flag.
        self.register_buffer(
            "pair_index", torch.tensor(self.phase_pairs, dtype=torch.long))

    def forward(self, movement_demand: torch.Tensor, current_union_phase: torch.Tensor,
                act_to_union: torch.Tensor) -> torch.Tensor:
        """-> (B, action_dim) Q-values, unmasked padding set to -inf.

        movement_demand      (B, N_MOVEMENTS * demand_shape)
        current_union_phase  (B,) index into the union phase table, or -1 when the
                             current phase has no union entry
        act_to_union         (B, action_dim) union-phase index per local action
                             slot, -1 for padding
        """
        B = movement_demand.shape[0]
        dev = movement_demand.device

        # --- current-phase flag per movement (RESCO's `extended_acts`) ---
        flags = torch.zeros(B, N_MOVEMENTS, dtype=torch.long, device=dev)
        valid = current_union_phase >= 0
        if valid.any():
            pairs = self.pair_index[current_union_phase.clamp(min=0)]   # (B, 2)
            idx = torch.arange(B, device=dev)[valid]
            flags[idx.unsqueeze(1), pairs[valid]] = 1
        phase_embeds = torch.sigmoid(self.p(flags))                     # (B, M, p_out)

        # --- per-movement demand embedding ---
        dem = movement_demand.view(B, N_MOVEMENTS, self.demand_shape).float()
        dem = torch.sigmoid(self.d(dem))                                # (B, M, d_out)
        phase_demands = F.relu(
            self.lane_embedding(torch.cat((phase_embeds, dem), dim=-1)))  # (B, M, L)

        # --- phase = sum of its two movements ---
        pairs = [phase_demands[:, a] + phase_demands[:, b]
                 for a, b in self.phase_pairs]                          # P x (B, L)

        # --- all ordered phase pairs, competition-gated ---
        rotated = []
        for i in range(self.n_phases):
            for j in range(self.n_phases):
                if i != j:
                    rotated.append(torch.cat((pairs[i], pairs[j]), dim=-1))
        rotated = torch.stack(rotated, dim=1).reshape(
            B, self.n_phases, self.n_phases - 1, 2 * self.lane_embed_units)
        rotated = rotated.permute(0, 3, 1, 2)
        rotated = F.relu(self.lane_conv(rotated))

        relations = self.comp_mask.unsqueeze(0).expand(B, -1, -1)
        relations = F.relu(self.relation_embedding(relations)).permute(0, 3, 1, 2)
        relations = F.relu(self.relation_conv(relations))

        combined = F.relu(self.hidden_layer(rotated * relations))
        combined = self.before_merge(combined).reshape(
            B, self.n_phases, self.n_phases - 1)
        q_union = combined.sum(dim=-1)                                   # (B, P)

        # --- gather union phases into this intersection's local action slots ---
        safe = act_to_union.clamp(min=0)
        q_local = q_union.gather(1, safe)
        return q_local.masked_fill(act_to_union < 0, float("-inf"))


def act_to_union_vector(act_map: dict, action_dim: int) -> np.ndarray:
    """{local_act: union_phase} -> (action_dim,) int64 vector, -1 where unused."""
    vec = np.full(action_dim, -1, dtype=np.int64)
    for local_act, union_phase in act_map.items():
        a = int(local_act)
        if 0 <= a < action_dim:
            vec[a] = int(union_phase)
    return vec
