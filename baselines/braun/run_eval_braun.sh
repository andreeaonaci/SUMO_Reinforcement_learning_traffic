#!/bin/bash
# Evaluate every finished Braun snapshot + Braun's rule-based controls, skip-or-resume.
#   baselines/braun/run_eval_braun.sh [SCENARIO...]   (default: grid4x4 cologne3 ingolstadt7)
# Learned: sampled (his reference protocol, 5 episodes, sample seeds 1000..1004) and
# greedy (1 episode; deterministic given the SUMO seed).  SUMO seed per scenario = the
# one our own evaluation uses (holdout 12345, in-distribution cities 42), so every
# controller sees the same traffic realisation.
W=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/.claude/worktrees/rescofull-writeup
EV=$W/results/braun/eval
mkdir -p "$EV"
SCEN=${*:-"grid4x4 cologne3 ingolstadt7"}
run() {  # out args...
  local out=$1; shift
  [ -f "$out" ] && return 0
  "$W/baselines/braun/bpy.sh" "$W/baselines/braun/eval_braun.py" "$@" --out "$out" 2>&1 | grep -E "ep[0-9]+\]|wrote|Error|Traceback"
}
for s in $SCEN; do
  case $s in grid4x4) SEED=12345;; *) SEED=42;; esac
  # Braun's own rule-based controllers on each action space (deterministic -> 1 episode)
  for arm in synthfb native synth; do
    [ "$arm" = synth ] && [ "$s" != ingolstadt7 ] && continue   # synth == synthfb elsewhere
    for pol in max-pressure queue fixed-time; do
      run "$EV/braun_${s}_${arm}_${pol}.json" --scenario "$s" --arm "$arm" --policy "$pol" --sumo-seed $SEED
    done
  done
  # Learned snapshots
  for snap in "$W"/results/braun/train/*_s*/snapshots/movement_policy_iter_*.pt; do
    [ -f "$snap" ] || continue
    rd=$(basename "$(dirname "$(dirname "$snap")")")        # e.g. synthfb_s3
    arm=${rd%_s*}
    it=$(basename "$snap" .pt); it=${it##*_iter_}
    run "$EV/braun_${s}_${rd}_it${it}_sample.json" --scenario "$s" --arm "$arm" --policy learned-sample \
        --checkpoint "$snap" --episodes 5 --sumo-seed $SEED --tag "_${rd}_it${it}"
    run "$EV/braun_${s}_${rd}_it${it}_greedy.json" --scenario "$s" --arm "$arm" --policy learned-greedy \
        --checkpoint "$snap" --episodes 1 --sumo-seed $SEED --tag "_${rd}_it${it}"
  done
done
echo "eval pass done: $SCEN"
