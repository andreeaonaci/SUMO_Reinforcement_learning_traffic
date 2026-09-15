#!/bin/bash
# FRAP arm on the RESCO-EXACT roster (fidings sec 103b).
#
# WHY THIS EXISTS RATHER THAN REUSING THE sec 103 CHECKPOINTS: that batch ran on
# environments_c1_4_6, which carries sec 99's three mismatches (shifted route
# files, wrong evaluation windows, yellow_time 2 vs RESCO's 3). Its numbers are
# internally valid -- all arms share the configuration, so the comparison holds
# -- but they CANNOT be quoted against RESCO's published figures. Doing so would
# repeat the exact error sec 99 retracted sec 59 for.
#
# environments_rescofull is RESCO-exact, and sec 100 already ran phase +
# indexed on it at 5 rounds x 6 seeds. This adds the matching FRAP arm so all
# three readouts can be evaluated in the literature's own metrics on the same
# footing.
#
# Protocol matched to sec 100 EXACTLY: 5 rounds, local_episodes 2, lr 3e-4,
# seeds 3/7/11/17/21/25, --pad_to_true_holdout. Resumable and skip-if-complete.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

OUT=results/rescofull_frap
RUNS=results/rescofull_frap_runs
mkdir -p $OUT $RUNS
DRIVER=$OUT/driver.log
MAX_CONCURRENT=3
SEEDS="${SEEDS:-3 7 11 17 21 25}"
ROUNDS=5

log() { echo "=== [$(date '+%F %T')] $* ===" >> $DRIVER; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 20; done; }
drain() {
  wait 2>/dev/null || true
  while pgrep -f "experiments.federated_training" > /dev/null; do sleep 30; done
}

run_seed() {
  local seed=$1
  local tag="frap_s${seed}"
  local marker="$RUNS/${tag}.rundir"
  local resume=""
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
      resume="--resume $rd"; log "$tag resuming from $rd (had $n rounds)"
    fi
  fi
  log "starting $tag"
  ( python -m experiments.federated_training --parallel \
      --base_dir environments_rescofull --pad_to_true_holdout --frap_head \
      --rounds $ROUNDS --local_episodes 2 --eval_every 1 --eval_episodes 5 \
      --lr 3e-4 --seed "$seed" $resume >> "$OUT/$tag.log" 2>&1
    local rc=$?
    grep -oE "results/run_[0-9_-]+_[0-9]+" "$OUT/$tag.log" | tail -1 > "$marker"
    log "finished $tag exit=$rc" ) &
}

log "rescofull FRAP arm starting: seeds=$SEEDS rounds=$ROUNDS"
for SEED in $SEEDS; do throttle; run_seed "$SEED"; done
drain
log "RESCOFULL FRAP ARM DONE"
