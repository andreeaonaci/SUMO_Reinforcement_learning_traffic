#!/bin/bash
# PCFT x phase-relational, 6-seed escalation (fidings sec 102).
#
# SAFE TO STOP AND RELAUNCH. Every job either resumes from its own checkpoint or
# is skipped if already complete:
#   PCFT    always passed --resume; it no-ops when there is no checkpoint and
#           skips completed steps when there is one. Steps are the 9 holdout-eval
#           boundaries, so at most one step's work is ever lost.
#   fedavg  its run_dir is recorded in a sidecar; a partial run is continued with
#           federated_training's own --resume, a complete one is skipped.
#
# Caveat carried from both --resume implementations: replay buffers, optimizer
# momentum and epsilon step counters are NOT restored, so a resumed run is not
# bit-identical to an uninterrupted one. Interrupting arms unevenly therefore
# introduces a (small) confound -- prefer letting a seed finish, and note in the
# write-up which runs were resumed.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

OUT=results/pcft_phase
RUNS=results/pcft_runs
mkdir -p $OUT $RUNS
DRIVER=$OUT/driver6.log
MAX_CONCURRENT=3
SEEDS="${SEEDS:-3 7 11 17 21 25}"
FED_ROUNDS=8

PCFT_ARGS="--base_dir environments_c1_4_6 --pad_to_true_holdout --phase_relational \
  --warmup_episodes 10 --focus_episodes 5 --fedavg_rounds 3 --local_episodes 2 --eval_episodes 5"

log() { echo "=== [$(date '+%F %T')] $* ===" >> $DRIVER; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 20; done; }

pcft_done() {   # order, seed -> 0 if all steps complete
  python - "$1" "$2" <<'PY'
import os, sys, torch
order, seed = sys.argv[1], sys.argv[2]
p = f"results/pcft_runs/{order}_s{seed}_phase/pcft_state.pt"
if not os.path.exists(p):
    sys.exit(1)
try:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sys.exit(0 if ck["done_steps"] >= len(ck["plan"]) else 1)
except Exception:
    sys.exit(1)
PY
}

run_pcft() {   # order, seed
  local order=$1 seed=$2 tag
  case "$order" in
    complexity) tag="pcftC_s$2" ;;
    reverse)    tag="pcftR_s$2" ;;
    *)          tag="pcftX_s$2" ;;
  esac
  if pcft_done "$order" "$seed"; then log "SKIP $tag (already complete)"; return; fi
  log "starting $tag"
  ( python diagnostics/progressive_curriculum_fedavg.py $PCFT_ARGS \
      --city_order "$order" --seed "$seed" \
      --run_dir "$RUNS/${order}_s${seed}_phase" --resume \
      >> "$OUT/$tag.log" 2>&1
    log "finished $tag exit=$?" ) &
}

run_fedavg() {   # seed
  # NB: bash expands every word of a `local` before assigning any of them, so
  # these must be separate statements -- a single `local a=$1 b=$a` breaks under
  # `set -u`.
  local seed=$1
  local tag="fedavg_s${seed}"
  local marker="$RUNS/fedavg_s${seed}.rundir"
  local resume=""
  if [ -f "$marker" ]; then
    local rd; rd=$(cat "$marker")
    local n; n=$(python - "$rd" "$FED_ROUNDS" <<'PY'
import json, os, sys
rd, want = sys.argv[1], int(sys.argv[2])
f = os.path.join(rd, "federated_history.json")
print(len(json.load(open(f))["round"]) if os.path.exists(f) else 0)
PY
)
    if [ "$n" -ge "$FED_ROUNDS" ]; then log "SKIP $tag (already $n rounds)"; return; fi
    [ -d "$rd" ] && resume="--resume $rd" && log "$tag resuming from $rd (had $n rounds)"
  fi
  log "starting $tag"
  ( python -m experiments.federated_training --parallel \
      --base_dir environments_c1_4_6 --pad_to_true_holdout --phase_relational \
      --rounds $FED_ROUNDS --local_episodes 2 --eval_every 1 --eval_episodes 5 \
      --lr 3e-4 --q_entropy_weight 0.05 --seed "$seed" $resume \
      >> "$OUT/$tag.log" 2>&1
    grep -oE "results/run_[0-9_-]+_[0-9]+" "$OUT/$tag.log" | head -1 > "$marker"
    log "finished $tag exit=$?" ) &
}

log "6-seed escalation starting, seeds: $SEEDS"
for SEED in $SEEDS; do
  throttle; run_pcft complexity "$SEED"
  throttle; run_pcft reverse    "$SEED"
  throttle; run_fedavg "$SEED"
done
wait
log "PCFT PHASE 6-SEED BATCH DONE"
