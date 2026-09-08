"""Did the bootstrapped Q-heads stay diverse, or collapse onto one function?

The documented kill condition for --boot_heads (fidings sec 94): if every head
learns the same function, the majority vote is decorative and the whole
architecture is a no-op wearing a flag. sec 93's ensemble members were
INDEPENDENTLY trained (different trunks, different data order); heads sharing
one trunk have far less to keep them apart, so this must be measured rather
than assumed.

Two measures, cheapest first:

  weight-space   pairwise mean |w_i - w_j| between heads, normalised by the
                 heads' own mean |w|. Needs nothing but the checkpoint. Cheap,
                 but weight distance is not behaviour: two different heads can
                 induce the same policy.

  action-space   fraction of (observation, head-pair) cases where two heads
                 pick different greedy actions on REAL holdout observations.
                 This is the number that actually matters -- it is exactly what
                 the vote does or does not have to work with. Needs SUMO.

Usage:
    python diagnostics/boot_head_diversity.py results/run_*/global_round_005.pth
    python diagnostics/boot_head_diversity.py results/run_*/global_round_*.pth \
        --base_dir environments_c1_4_6 --pad_to_true_holdout --steps 200
"""
import argparse
import itertools
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def boot_head_weights(state):
    """{head_index: weight_tensor} for a bootstrapped checkpoint, or {}."""
    out = {}
    for k, v in state.items():
        if k.startswith("boot_q.") and k.endswith(".weight"):
            out[int(k.split(".")[1])] = v.float()
    return dict(sorted(out.items()))


def weight_space_report(state) -> bool:
    heads = boot_head_weights(state)
    if len(heads) < 2:
        print("  not a bootstrapped checkpoint (no boot_q.* heads found)")
        return False
    scale = float(torch.stack([w.abs().mean() for w in heads.values()]).mean())
    dists = [
        float((heads[i] - heads[j]).abs().mean())
        for i, j in itertools.combinations(heads, 2)
    ]
    rel = [d / scale for d in dists] if scale > 0 else [0.0] * len(dists)
    print(f"  heads={len(heads)}  mean|w|={scale:.5f}")
    print(f"  pairwise mean|w_i - w_j|: min={min(dists):.5f} max={max(dists):.5f}")
    print(f"  relative to mean|w|:      min={min(rel):.3f} max={max(rel):.3f}")
    if max(rel) < 0.02:
        print("  COLLAPSE WARNING: heads are near-identical in weight space.")
    return True


def action_space_report(ckpt_path, state, args) -> None:
    """Greedy-action disagreement between heads on real holdout observations."""
    from agents.dqn import DQNAgent
    from diagnostics.finetune_on_holdout import infer_arch_from_checkpoint
    from experiments.federated_training import (
        resolve_city_configs_and_dims,
        make_holdout_evaluator,
        maybe_pad_action_dim_to_true_holdout,
    )

    arch = infer_arch_from_checkpoint(state)
    n_heads = len(boot_head_weights(state))
    _, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    agent = DQNAgent(
        own_dim=arch["own_dim"], neighbor_dim=arch["neighbor_dim"], k_max=k_max,
        action_dim=arch["action_dim"], head_fix=arch["head_fix"],
        encoder_depth=arch["encoder_depth"], n_attn_layers=arch["n_attn_layers"],
        boot_heads=n_heads,
    )
    agent.load_state_dict(state)

    evaluator = make_holdout_evaluator(
        args.base_dir, (own_dim, neighbor_dim, k_max), action_dim,
        episodes=1, eval_sumo_seed=args.eval_sumo_seed,
    )
    if evaluator is None:
        print("  could not build holdout evaluator; skipping action-space measure")
        return

    env = evaluator.env
    obs, _ = env.reset()
    disagree_num = disagree_den = 0
    per_head_actions = [[] for _ in range(n_heads)]
    agent.q.eval()
    with torch.no_grad():
        for _ in range(args.steps):
            ts_ids = list(obs.keys())
            if not ts_ids:
                break
            from agents.dqn import _collate
            own, nbr, msk, hop, amask = _collate([obs[t] for t in ts_ids], agent.device)
            q_all = agent.q.forward_boot(own, nbr, msk, hop)
            q_all = q_all.masked_fill(amask.unsqueeze(1).expand_as(q_all) <= 0, float("-inf"))
            greedy = q_all.argmax(dim=2).cpu().numpy()          # (B, heads)
            for i, j in itertools.combinations(range(n_heads), 2):
                disagree_num += int((greedy[:, i] != greedy[:, j]).sum())
                disagree_den += greedy.shape[0]
            for h in range(n_heads):
                per_head_actions[h].extend(greedy[:, h].tolist())
            actions = agent.act_batch(obs, explore=False)
            obs, _, term, trunc, _ = env.step(actions)
            if (isinstance(term, dict) and all(term.values())) or term is True:
                break
            if (isinstance(trunc, dict) and all(trunc.values())) or trunc is True:
                break
    evaluator.close()

    if disagree_den == 0:
        print("  no observations collected")
        return
    rate = disagree_num / disagree_den
    print(f"  greedy-action disagreement between head pairs: {rate:.4f} "
          f"({disagree_num}/{disagree_den} cases)")
    for h, acts in enumerate(per_head_actions):
        counts = np.bincount(acts, minlength=action_dim)
        top = counts.max() / max(counts.sum(), 1)
        print(f"    head {h}: dominant-action fraction {top:.3f}  counts={counts.tolist()}")
    if rate < 0.01:
        print("  COLLAPSE: heads almost never disagree -- the vote is decorative.")
    elif rate < 0.05:
        print("  WEAK: heads rarely disagree; the vote has little to work with.")
    else:
        print("  Heads are behaviourally diverse -- the vote is doing real work.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--base_dir", default=None,
                    help="Enable the action-space measure (needs SUMO), e.g. environments_c1_4_6")
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval_sumo_seed", type=int, default=12345)
    args = ap.parse_args()

    for path in args.checkpoints:
        print(f"\n{path}")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not weight_space_report(state):
            continue
        if args.base_dir:
            action_space_report(path, state, args)


if __name__ == "__main__":
    main()
