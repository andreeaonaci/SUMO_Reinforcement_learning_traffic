#!/bin/bash
# Sequential training queue (one run at a time, 10 rollout workers each).
# Skip-or-resume: every entry is resumable, so relaunching continues.
#   nohup baselines/braun/run_training_queue.sh > results/braun/train/queue.log 2>&1 &
# QUEUE env var overrides the default order ("arm:seed ...").
W=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/.claude/worktrees/rescofull-writeup
QUEUE=${QUEUE:-"synthfb:3 synthfb:7 synthfb:11 native:3 native:7 native:11 synthfb:17 synthfb:21 synthfb:25 native:17 native:21 native:25"}
for item in $QUEUE; do
  arm=${item%%:*}; seed=${item##*:}
  echo "[$(date -Is)] queue: $arm seed $seed"
  "$W/baselines/braun/train_braun.sh" "$arm" "$seed" 85 cuda
  echo "[$(date -Is)] queue: $arm seed $seed exit=$?"
done
echo "[$(date -Is)] queue finished"
