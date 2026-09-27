#!/bin/bash
# Evaluate ONLY the final (iteration 85) Braun checkpoints, one scenario per call,
# skip-or-resume. Same protocol as run_eval_braun.sh (sampled: 5 episodes, sample
# seeds 1000..1004; greedy: 1 episode; SUMO seed 12345 on grid4x4, 42 elsewhere),
# without re-running the iteration 1 / 10 snapshots the summary tables do not use.
#   baselines/braun/run_eval_it85.sh <scenario> [seed...]
W=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/.claude/worktrees/rescofull-writeup
OUT=$W/results/braun/eval
s=$1; shift
SEEDS=${*:-"3 7 11 17 21 25"}
case $s in grid4x4) SEED=12345;; *) SEED=42;; esac
run() {  # out args...
  local out=$1; shift
  [ -f "$out" ] && return 0
  "$W/baselines/braun/bpy.sh" "$W/baselines/braun/eval_braun.py" "$@" --out "$out" 2>&1 | grep -E "wrote|Error|Traceback"
}
for arm in synthfb native; do
  for seed in $SEEDS; do
    rd=${arm}_s${seed}
    snap=$W/results/braun/train/$rd/snapshots/movement_policy_iter_0085.pt
    [ -f "$snap" ] || { echo "missing $snap"; continue; }
    run "$OUT/braun_${s}_${rd}_it0085_sample.json" --scenario "$s" --arm "$arm" --policy learned-sample \
        --checkpoint "$snap" --episodes 5 --sumo-seed $SEED --tag "_${rd}_it0085"
    run "$OUT/braun_${s}_${rd}_it0085_greedy.json" --scenario "$s" --arm "$arm" --policy learned-greedy \
        --checkpoint "$snap" --episodes 1 --sumo-seed $SEED --tag "_${rd}_it0085"
  done
done
echo "it85 pass done: $s"
