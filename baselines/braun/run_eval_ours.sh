#!/bin/bash
# Our 6 phase-relational rescofull checkpoints (global_round_005, seeds 3/7/11/17/21/25)
# + our max_pressure / fixed_time, through baselines/braun/eval_ours.py.
#   baselines/braun/run_eval_ours.sh holdout|city_4|city_6
W=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/.claude/worktrees/rescofull-writeup
MAIN=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/results
CK=""
for d in run_2026_09_09-01_29_50_1483406 run_2026_09_09-01_29_50_1483410 run_2026_09_09-01_29_50_1483409 \
         run_2026_09_09-02_09_28_1491576 run_2026_09_09-02_09_32_1491628 run_2026_09_09-02_09_34_1491714; do
  CK="$CK $MAIN/$d/global_round_005.pth"
done
cd "$W" || exit 1
WHERE=$1
if [ "$WHERE" = holdout ]; then CITY=(); else CITY=(--city "$WHERE"); fi
/home/deea/miniconda3/bin/python baselines/braun/eval_ours.py $CK max_pressure fixed_time \
  --base_dir environments_rescofull --pad_to_true_holdout --episodes 5 "${CITY[@]}" \
  --out_dir results/braun/eval
