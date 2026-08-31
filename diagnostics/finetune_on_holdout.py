"""Fine-tune a trained checkpoint on a short burst of RANDOMIZED traffic on
the true holdout city's own topology (grid4x4), then evaluate on the REAL
holdout traffic -- isolates "does letting the model see this topology's
*structure* at all help" from "did it just memorize the real eval traffic":
the fine-tuning phase never touches grid4x4_1.rou.xml (the file every
eval_reward number in this project is measured against), only synthetic
randomTrips.py traffic on the same net (diagnostics/generate_random_routes.py).

This is "prio next step #2" from the 2026-08-29/31 discussion (queued ahead
of #1, a more diverse training roster) -- the cheapest lever that directly
tests whether a few-shot look at the target topology closes some of the
cross-topology generalization gap characterized in
fidings/divergence_investigation.md sec 43-61, without touching the
already-exhausted list of training-time levers (federation strategy,
architecture, extra features, more budget -- sec 49/50/58-64).

Protocol:
  1. Load --checkpoint's weights into a DQNAgent. own_dim/neighbor_dim/
     action_dim/dueling/head_fix are auto-detected directly from the
     checkpoint's own tensor shapes (robust to whichever project-code
     version trained it -- e.g. the own_dim 115->117 pressure-feature
     change, sec 62 -- so this script never needs to be told which
     architecture variant produced a given checkpoint).
  2. Zero-shot baseline: evaluate the UNMODIFIED loaded weights on the real
     holdout config (environments/city_5_holdout, grid4x4_1.rou.xml) --
     the same evaluator every other eval_reward number in this project uses.
  3. Fine-tune for --rounds short rounds against --n_variants independent
     randomized route files fed through the exact same ParallelFederatedServer
     path used for real training (fedavg across the variants, so the model
     sees varied random traffic instances of this topology, not one fixed
     synthetic pattern). Route files are generated on first use if missing.
  4. Re-evaluate on the SAME real holdout config every round -- reports
     zero-shot vs. each round's fine-tuned result side by side, plus the
     final round's numbers for a decision against this project's standing
     |diff|/SE >= 2 bar (single-seed pilot -- read with the same standing
     caution as every other single-seed result in this document, sec
     11->12/30->31/46->47/62->63).

Usage:
    python diagnostics/finetune_on_holdout.py results/run_.../global_round_063.pth \\
        --rounds 5 --local_episodes 2 --n_variants 5 --seed 3
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import yaml
import torch

from agents.dqn import DQNAgent
from federated.parallel_server import ParallelFederatedServer
from federated.utils import compute_eps_decay
from experiments.federated_training import make_holdout_evaluator
from environments.federated_env import build_federated_env
from diagnostics.generate_random_routes import generate_variants

REAL_HOLDOUT_CFG_PATH = os.path.join("environments", "city_5_holdout", "config.yaml")


def infer_arch_from_checkpoint(state: dict) -> dict:
    """Recover own_dim/neighbor_dim/action_dim/dueling/head_fix straight from
    tensor shapes so this script doesn't need to be told which architecture
    variant produced a given checkpoint (see module docstring point 1)."""
    own_dim = state["own_encoder.0.weight"].shape[1]
    neighbor_dim = state["neighbor_encoder.0.weight"].shape[1]
    dueling = "advantage_head.weight" in state
    action_dim = (state["advantage_head.weight"].shape[0] if dueling
                  else state["head.4.weight"].shape[0])
    head_fix = "pool_head.0.weight" not in state
    return dict(own_dim=own_dim, neighbor_dim=neighbor_dim,
                action_dim=action_dim, dueling=dueling, head_fix=head_fix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Path to a global_round_NNN.pth to fine-tune from.")
    ap.add_argument("--rounds", type=int, default=5, help="Length of the fine-tune burst.")
    ap.add_argument("--local_episodes", type=int, default=2)
    ap.add_argument("--n_variants", type=int, default=5,
                     help="Independent randomized-traffic route files to federate the "
                          "fine-tune across (see diagnostics/generate_random_routes.py).")
    ap.add_argument("--regenerate_routes", action="store_true",
                     help="Force-regenerate the random route variants even if already present.")
    ap.add_argument("--explore_fraction", type=float, default=0.3,
                     help="Fraction of the (short) fine-tune burst over which epsilon decays "
                          "from 1.0 to the floor. Lower than recovery_finetune.py's 0.5 default: "
                          "this is calibration, not an escape-a-lock-in exploration burst.")
    ap.add_argument("--n_step", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5,
                     help="Deliberately gentler than the 3e-4 used for training from scratch -- "
                          "this is a SMALL finetune, not a full re-train.")
    ap.add_argument("--lr_decay", type=float, default=0.97)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval_episodes", type=int, default=5,
                     help="Matches this project's standard per-round eval_episodes. Use a "
                          "larger value (e.g. 30) for a final confirmatory check, matching "
                          "diagnostics/reeval_checkpoint.py's convention.")
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--target_update", type=int, default=200)
    ap.add_argument("--checkpoint_dir", default=None,
                     help="Default: results/finetune_holdout_<basename of input checkpoint>")
    args = ap.parse_args()

    # --- 1. Load checkpoint, auto-detect architecture -----------------
    state = torch.load(args.checkpoint, map_location="cpu")
    arch = infer_arch_from_checkpoint(state)
    print(f"Auto-detected architecture from checkpoint: {arch}")

    with open(REAL_HOLDOUT_CFG_PATH) as f:
        real_cfg = yaml.safe_load(f)
    k_max = int(real_cfg.get("k_max", 8))
    num_seconds = int(real_cfg.get("num_seconds", 3600))

    # Fail loudly and clearly here rather than deep inside a multiprocess
    # worker: a checkpoint trained under a different observation-space
    # version of the code (e.g. before/after the own_dim 115->117
    # pressure-feature change, sec 62) can't be loaded against the
    # CURRENT environment's obs shape.
    probe_env = build_federated_env(real_cfg)
    live_own_dim, live_neighbor_dim = probe_env.own_dim, probe_env.neighbor_dim
    probe_env.close()
    if arch["own_dim"] != live_own_dim or arch["neighbor_dim"] != live_neighbor_dim:
        raise RuntimeError(
            f"Checkpoint's obs dims (own_dim={arch['own_dim']}, neighbor_dim="
            f"{arch['neighbor_dim']}) do not match what the CURRENT code produces "
            f"for city_5_holdout (own_dim={live_own_dim}, neighbor_dim={live_neighbor_dim}). "
            "This checkpoint was likely trained under a different observation-space "
            "code version (e.g. before/after the pressure-feature own_dim 115->117 "
            "change) -- pick a checkpoint trained under the currently checked-out code, "
            "or check out the matching commit before running this script."
        )

    global_model = DQNAgent(
        own_dim=arch["own_dim"], neighbor_dim=arch["neighbor_dim"], k_max=k_max,
        action_dim=arch["action_dim"], dueling=arch["dueling"], head_fix=arch["head_fix"],
        n_step=args.n_step,
    )
    global_model.load_state_dict(state)
    print(f"Loaded weights from {args.checkpoint}")

    # --- 2. Zero-shot baseline on the REAL holdout traffic -------------
    real_evaluator = make_holdout_evaluator(
        "environments", (arch["own_dim"], arch["neighbor_dim"], k_max), arch["action_dim"],
        episodes=args.eval_episodes,
    )
    if real_evaluator is None:
        raise RuntimeError("Could not construct the real-holdout evaluator.")

    zero_shot = real_evaluator.evaluate(global_model)
    print(f"\n=== Zero-shot (no fine-tune), real holdout traffic ===")
    print(f"mean_reward={zero_shot['mean_reward']:.2f}  std={zero_shot['std_reward']:.2f}  "
          f"mean_waiting_time={zero_shot['mean_waiting_time']:.2f}")

    # --- 3. Generate/reuse randomized route variants, build finetune roster --
    route_paths = generate_variants(
        net_file=real_cfg["net_file"],
        out_dir=os.path.join(os.path.dirname(real_cfg["net_file"]), "generated_random"),
        n_variants=args.n_variants,
        duration=num_seconds,
        force=args.regenerate_routes,
    )

    city_configs = []
    for i, route_path in enumerate(route_paths):
        variant_cfg = dict(real_cfg)
        variant_cfg["route_file"] = route_path
        city_configs.append((f"holdout_random_{i}", variant_cfg))

    steps_per_ep = num_seconds // int(real_cfg.get("delta_time", 5))
    eps_decay = compute_eps_decay(
        rounds=args.rounds, local_episodes=args.local_episodes,
        steps_per_episode=steps_per_ep, explore_fraction=args.explore_fraction,
    )
    print(f"\nFine-tune eps_decay={eps_decay:.1f} (epsilon restarts at 1.0 for this "
          f"burst, reaches floor by ~{args.explore_fraction*100:.0f}% of {args.rounds} rounds)")

    checkpoint_dir = args.checkpoint_dir or os.path.join(
        "results", f"finetune_holdout_{os.path.basename(args.checkpoint).replace('.pth', '')}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Fresh agent for the server: same loaded weights, but eps_decay sized
    # to the short finetune burst rather than inherited from the checkpoint's
    # original (already-decayed) schedule -- same reasoning as
    # recovery_finetune.py's init_steps_done=0.
    server_model = DQNAgent(
        own_dim=arch["own_dim"], neighbor_dim=arch["neighbor_dim"], k_max=k_max,
        action_dim=arch["action_dim"], dueling=arch["dueling"], head_fix=arch["head_fix"],
        n_step=args.n_step, eps_decay=eps_decay,
    )
    server_model.load_state_dict(state)

    server = ParallelFederatedServer(
        global_model=server_model,
        city_configs=city_configs,
        own_dim=arch["own_dim"], neighbor_dim=arch["neighbor_dim"], k_max=k_max,
        action_dim=arch["action_dim"],
        comm_dropout_cfg={"p_link": 0.0, "p_isolate": 0.0, "p_hop_cutoff": 0.0},
        local_episodes=args.local_episodes,
        eps_decay=eps_decay,
        evaluator=real_evaluator,
        checkpoint_dir=checkpoint_dir,
        default_lr=args.lr, lr_decay=args.lr_decay, min_lr=args.min_lr,
        head_fix=True,
        neighbor_attention=arch["head_fix"],
        tau=args.tau, target_update=args.target_update,
        seed=args.seed,
        dueling=arch["dueling"],
        n_step=args.n_step,
        init_steps_done=0,
    )
    try:
        history = server.run(rounds=args.rounds, eval_every=1)
    finally:
        server.close()
        real_evaluator.close()

    print(f"\ncheckpoint_dir={checkpoint_dir}")
    print(f"zero-shot   mean_reward={zero_shot['mean_reward']:.2f}  "
          f"mean_waiting_time={zero_shot['mean_waiting_time']:.2f}")
    print("round  eval_reward  eval_waiting_time")
    for r, rew, wt in zip(history["round"], history["eval_reward"], history["eval_waiting_time"]):
        print(f"{r:5d}  {rew:11.2f}  {wt:17.2f}")

    final_reward = history["eval_reward"][-1]
    delta = final_reward - zero_shot["mean_reward"]
    print(f"\nfinal round reward {final_reward:.2f} vs zero-shot {zero_shot['mean_reward']:.2f} "
          f"(delta={delta:+.2f})")


if __name__ == "__main__":
    main()
