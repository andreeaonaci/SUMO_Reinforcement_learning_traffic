#!/bin/bash
# Train Braun's scratch PPO (train.sh's iteration-85 architecture flags) on one arm.
#   baselines/braun/train_braun.sh ARM SEED [ITERATIONS] [DEVICE] [STAGES]
# ARM: synthfb | native   (configs/rescofull_<ARM>.yaml)
#
# Training runs in stages (default "1 10 ITERATIONS"); after each stage the
# policy-only checkpoint is copied to snapshots/movement_policy_iter_XXXX.pt so
# budget points can be evaluated.  Iteration 1 = 30 rollouts x 3600 s =
# 108,000 simulated seconds = exactly our method's whole training budget
# (5 rounds x 2 local episodes x 3 cities x 3600 s).  Stages resume through
# Braun's own --resume-checkpoint (model, optimizer, RNG and iteration restored;
# --iterations is the FINAL target).  Skip-or-resume: rerunning continues.
set -u
W=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/.claude/worktrees/rescofull-writeup
ARM=$1; SEED=$2; ITERS=${3:-85}; DEVICE=${4:-cuda}; STAGES=${5:-"1 10 $ITERS"}
RUN=$W/results/braun/train/${ARM}_s${SEED}
CKPT=$RUN/ckpt
SNAP=$RUN/snapshots
mkdir -p "$RUN" "$SNAP"
for STAGE in $STAGES; do
  TARGET=$(printf '%s/movement_policy_iter_%04d.pt' "$SNAP" "$STAGE")
  if [ -f "$TARGET" ]; then continue; fi
  CUR=0
  if [ -f "$CKPT/movement_ppo_latest.pt" ]; then
    CUR=$("$W/baselines/braun/bpy.sh" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).iteration)" "$CKPT/movement_ppo_latest.pt" 2>/dev/null)
  fi
  if [ "$CUR" = "$STAGE" ]; then cp "$CKPT/movement_policy_latest.pt" "$TARGET"; continue; fi
  if [ "$CUR" -gt "$STAGE" ]; then
    echo "[$(date -Is)] WARN latest is at iteration $CUR > stage $STAGE; snapshot $STAGE unavailable" >> "$RUN/launcher.log"
    continue
  fi
  if [ -f "$CKPT/movement_ppo_latest.pt" ]; then
    INIT=(--resume-checkpoint "$CKPT/movement_ppo_latest.pt")
  else
    INIT=(--scratch-random --scratch-lane-feature-dim 29 --scratch-movement-feature-dim 4
          --scratch-hidden-dim 64 --scratch-num-hops 1)
  fi
  echo "[$(date -Is)] start arm=$ARM seed=$SEED stage=$STAGE device=$DEVICE init=${INIT[0]}" >> "$RUN/launcher.log"
  "$W/baselines/braun/bpy.sh" scripts/train_rl.py \
    --experiment-config "$W/baselines/braun/configs/rescofull_${ARM}.yaml" \
    "${INIT[@]}" --iterations "$STAGE" --seed "$SEED" --device "$DEVICE" \
    --ckpt-dir "$CKPT" --log-dir "$RUN/tb" >> "$RUN/train_stdout.log" 2>&1
  RC=$?
  echo "[$(date -Is)] stage=$STAGE exit=$RC" >> "$RUN/launcher.log"
  if [ $RC -ne 0 ]; then exit $RC; fi
  DONE_ITER=$("$W/baselines/braun/bpy.sh" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).iteration)" "$CKPT/movement_ppo_latest.pt" 2>/dev/null)
  if [ "$DONE_ITER" != "$STAGE" ]; then
    echo "[$(date -Is)] ERROR latest checkpoint reports iteration '$DONE_ITER', expected $STAGE" >> "$RUN/launcher.log"
    exit 3
  fi
  cp "$CKPT/movement_policy_latest.pt" "$TARGET"
done
echo "[$(date -Is)] all stages done" >> "$RUN/launcher.log"
