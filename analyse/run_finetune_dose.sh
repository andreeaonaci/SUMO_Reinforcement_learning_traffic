#!/bin/bash
# How much does ONE or TWO rounds of adaptation on the target topology buy?
# (fidings sec 105)
#
# Starts from the sec 100 phase-relational checkpoints -- the best model, trained
# on environments_rescofull at the benchmark's 3 s yellow -- and adapts on
# SYNTHETIC randomised demand over the holdout topology, never the evaluation
# route file. Evaluation is the real holdout traffic as usual.
#
# TIMING MATCHED ON PURPOSE: --holdout_config points at the rescofull holdout, so
# adaptation and evaluation happen at the same 3 s yellow the checkpoint was
# trained under. The script's default holdout is the 2 s roster, which would have
# measured adaptation against a different signal timing than training used.
#
# Two durations, three seeds each, against the known zero-shot baseline. Short
# runs are single-phase (--phase1_rounds >= --rounds) so the two-phase learning
# rate schedule does not silently apply to a 1-round burst.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

OUT=results/finetune_dose
mkdir -p $OUT
DRIVER=$OUT/driver.log
MAX_CONCURRENT=2
HOLDOUT=environments_rescofull/city_5_holdout/config.yaml

# sec 100 phase-relational runs, one per seed (final-round checkpoints).
MAIN=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/results
declare -A CKPT=(
  [3]="$MAIN/run_2026_09_09-01_29_50_1483406/global_round_005.pth"
  [7]="$MAIN/run_2026_09_09-01_29_50_1483410/global_round_005.pth"
  [11]="$MAIN/run_2026_09_09-01_29_50_1483409/global_round_005.pth"
)

log() { echo "=== [$(date '+%F %T')] $* ===" >> $DRIVER; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 20; done; }
drain() {
  wait 2>/dev/null || true
  while pgrep -f "finetune_on_holdout.py" > /dev/null; do sleep 20; done
}

run_one() {   # rounds, seed
  local rounds=$1
  local seed=$2
  local tag="ft${rounds}_s${seed}"
  local ck="${CKPT[$seed]}"
  if [ ! -f "$ck" ]; then log "SKIP $tag (missing $ck)"; return; fi
  if grep -q "FINETUNE DONE" "$OUT/$tag.log" 2>/dev/null; then
    log "SKIP $tag (already complete)"; return
  fi
  log "starting $tag (rounds=$rounds)"
  ( python diagnostics/finetune_on_holdout.py "$ck" \
      --holdout_config "$HOLDOUT" \
      --rounds "$rounds" --phase1_rounds "$rounds" \
      --local_episodes 2 --n_variants 5 --eval_episodes 10 \
      --seed "$seed" > "$OUT/$tag.log" 2>&1
    local rc=$?
    echo "FINETUNE DONE rc=$rc" >> "$OUT/$tag.log"
    log "finished $tag exit=$rc" ) &
}

log "finetune dose batch: rounds={1,2} x seeds={3,7,11}, holdout=$HOLDOUT"
for SEED in 3 7 11; do
  throttle; run_one 1 "$SEED"
  throttle; run_one 2 "$SEED"
done
drain
log "FINETUNE DOSE BATCH DONE"
