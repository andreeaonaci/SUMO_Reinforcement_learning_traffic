#!/bin/bash
# Rule-based references on the BENCHMARK-TIMED (3 s yellow) zero-shot holdouts.
# (fidings sec 109)
#
# WHY: the zero-shot table's 3 s block (y3, rescofull) compares the readouts
# against each other but carries no rule-based reference at all -- every
# max_pressure / fixed_time holdout number in the study so far was measured on
# the 2 s holdout. The claim "phase-relational beats max_pressure zero-shot" was
# therefore unmeasured at the benchmark's own signal timing. This measures it,
# with the same evaluator, episode count and eval_sumo_seed the RL arms used.
#
# Eval-only: no training. Sequential on purpose -- it runs beside other batches.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

OUT=results/holdout_baselines
mkdir -p $OUT
DRIVER=$OUT/driver.log
log() { echo "=== [$(date '+%F %T')] $* ===" >> $DRIVER; }

for BASE in environments_rescofull environments_y3; do
  for CTRL in max_pressure fixed_time; do
    tag="$(basename $BASE | sed 's/^environments_//')_$CTRL"
    if grep -q "BASELINE DONE rc=0" "$OUT/$tag.log" 2>/dev/null; then
      log "SKIP $tag (already complete)"; continue
    fi
    log "starting $tag"
    python -m experiments.federated_training \
      --base_dir "$BASE" --pad_to_true_holdout \
      --baseline_controller "$CTRL" --eval_episodes 5 > "$OUT/$tag.log" 2>&1
    rc=$?
    echo "BASELINE DONE rc=$rc" >> "$OUT/$tag.log"
    log "finished $tag exit=$rc"
  done
done
log "HOLDOUT BASELINES DONE"
