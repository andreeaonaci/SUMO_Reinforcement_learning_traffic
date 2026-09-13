#!/bin/bash
# Budget sensitivity AND the FRAP baseline, in one batch (fidings sec 103).
#
# Three arms at a MATCHED 20-round budget, so one batch answers two questions:
#   indexed  vs phase  -> does the sec 96-100 gap survive 2.5-4x the budget?
#   frap     vs phase  -> does a published phase-invariant readout, handed
#                         RESCO's own hand-authored per-signal configuration,
#                         match ours which is handed none? (sec 101)
#
# THE QUESTION A REVIEWER WILL ASK: every indexed-vs-phase-relational comparison
# in this document (sec 96-100, sec 102) runs 5-8 rounds. "The indexed head just
# needs more training" is the obvious objection and nothing here answers it.
# This runs both heads at 20 rounds -- 2.5-4x the standard budget -- on the same
# roster, same seeds, same everything else.
#
# Outcomes:
#   gap persists or widens -> the objection is answered; representation, not budget
#   gap closes             -> sec 96-100 are a budget artifact and must be requalified
#
# Uses only existing code paths (federated_training --phase_relational), so there
# is no new-implementation risk. Safe to stop and relaunch: each job records its
# run_dir and is resumed via federated_training's own --resume, or skipped when
# it already has --rounds rounds.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

OUT=results/budget_sens
RUNS=results/budget_runs
mkdir -p $OUT $RUNS
DRIVER=$OUT/driver.log
MAX_CONCURRENT=3
SEEDS="${SEEDS:-3 7 11}"
ROUNDS="${ROUNDS:-20}"

log() { echo "=== [$(date '+%F %T')] $* ===" >> $DRIVER; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 30; done; }

run_arm() {   # arm(phase|indexed), seed
  local arm=$1
  local seed=$2
  local tag="${arm}_s${seed}"
  local marker="$RUNS/${tag}.rundir"
  local resume=""
  local extra=""
  [ "$arm" = "phase" ] && extra="--phase_relational"
  [ "$arm" = "frap" ]  && extra="--frap_head"

  if [ -f "$marker" ]; then
    local rd
    rd=$(cat "$marker")
    local n
    n=$(python - "$rd" <<'PY'
import json, os, sys
f = os.path.join(sys.argv[1], "federated_history.json")
print(len(json.load(open(f))["round"]) if os.path.exists(f) else 0)
PY
)
    if [ "$n" -ge "$ROUNDS" ]; then log "SKIP $tag (already $n rounds)"; return; fi
    if [ -d "$rd" ] && [ "$n" -gt 0 ]; then
      resume="--resume $rd"
      log "$tag resuming from $rd (had $n rounds)"
    fi
  fi

  log "starting $tag"
  ( python -m experiments.federated_training --parallel \
      --base_dir environments_c1_4_6 --pad_to_true_holdout $extra \
      --rounds "$ROUNDS" --local_episodes 2 --eval_every 1 --eval_episodes 5 \
      --lr 3e-4 --lr_decay 0.97 --min_lr 1e-5 --q_entropy_weight 0.05 \
      --seed "$seed" $resume >> "$OUT/$tag.log" 2>&1
    local rc=$?
    grep -oE "results/run_[0-9_-]+_[0-9]+" "$OUT/$tag.log" | head -1 > "$marker"
    log "finished $tag exit=$rc" ) &
}

log "budget + FRAP batch starting: seeds=$SEEDS rounds=$ROUNDS arms=phase,indexed,frap"
for SEED in $SEEDS; do
  throttle; run_arm phase   "$SEED"
  throttle; run_arm indexed "$SEED"
  throttle; run_arm frap    "$SEED"
done
wait
log "BUDGET + FRAP BATCH DONE"
