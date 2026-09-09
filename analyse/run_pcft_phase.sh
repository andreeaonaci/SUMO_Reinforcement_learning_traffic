#!/bin/bash
# PCFT x phase-relational screen (fidings sec 102).
#
# Three arms, all --phase_relational, environments_c1_4_6, seeds 3/7/11:
#   pcftC  PCFT, complexity order (simplest first)  -- PCFT as proposed
#   pcftR  PCFT, reverse order (most complex first) -- ORDERING CONTROL:
#          identical budget, identical focus fine-tune phases, only order differs
#   fedavg plain FedAvg, budget-matched at 8 rounds x 2 local episodes x 3 cities
#          = 48 episodes vs PCFT's 10 + 5 + 12 + 5 + 18 = 50
#
# pcftC vs pcftR isolates the CURRICULUM. fedavg vs pcftR isolates the
# focus-fine-tune/extra-budget effect. sec 87 confirmed PCFT at 6 seeds but never
# separated the two, and sec 101 found curriculum-over-clients is already
# published (Vahidian et al. ICCV 2023), so ordering is now the whole claim.
set -u
cd /mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/.claude/worktrees/rescofull-writeup
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

OUT=results/pcft_phase
mkdir -p $OUT
MAX_CONCURRENT=3
DRIVER=$OUT/driver.log
: > $DRIVER

launch() {   # tag, command...
  local tag=$1; shift
  echo "=== [$(date '+%F %T')] starting $tag ===" >> $DRIVER
  ( "$@" > "$OUT/$tag.log" 2>&1; \
    echo "=== [$(date '+%F %T')] finished $tag exit=$? ===" >> $DRIVER ) &
}

throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 20; done
}

PCFT_ARGS="--base_dir environments_c1_4_6 --pad_to_true_holdout --phase_relational \
  --warmup_episodes 10 --focus_episodes 5 --fedavg_rounds 3 --local_episodes 2 --eval_episodes 5"

for SEED in 3 7 11; do
  throttle
  launch "pcftC_s$SEED" python diagnostics/progressive_curriculum_fedavg.py \
      $PCFT_ARGS --city_order complexity --seed $SEED
  throttle
  launch "pcftR_s$SEED" python diagnostics/progressive_curriculum_fedavg.py \
      $PCFT_ARGS --city_order reverse --seed $SEED
  throttle
  launch "fedavg_s$SEED" python -m experiments.federated_training --parallel \
      --base_dir environments_c1_4_6 --pad_to_true_holdout --phase_relational \
      --rounds 8 --local_episodes 2 --eval_every 1 --eval_episodes 5 \
      --lr 3e-4 --q_entropy_weight 0.05 --seed $SEED
done

wait
echo "PCFT PHASE SCREEN ALL DONE $(date '+%F %T')" >> $DRIVER
