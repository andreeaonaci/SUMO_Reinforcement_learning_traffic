#!/bin/bash
# Does TRAINING-TOPOLOGY DIVERSITY help the phase-relational head? (fidings sec 104)
#
# WHY NOW, AND WHY THIS IS NOT A REPEAT OF sec 71:
#
# Domain randomization over varied synthetic networks is the mechanism the
# 2024-2026 generalization literature actually leans on -- TransferLight
# (arXiv:2412.09719) attributes its zero-shot transfer to it, and the
# domain-randomization/meta-learning line (arXiv:2307.11357) to robustness. This
# project has never tested it properly.
#
# sec 71 looked like it had: 14 training cities, null result. That conclusion is
# uninterpretable for TWO independent reasons discovered since:
#   1. It ran on the INDEXED readout, which sec 102 showed produces floor effects
#      -- interventions cannot show benefit through a readout that cannot express
#      a transferable policy.
#   2. Its roster (environments_wide) contains city_7, whose net_file IS
#      grid4x4.net.xml -- the holdout's own network (sec 95a). `is_true_holdout`
#      does not catch this: it checks the eval city's NAME, not whether a training
#      city shares its topology.
#
# environments_divwide fixes both: phase-relational head, and city_7 excluded
# (verified: no training city shares grid4x4.net.xml).
#
# DESIGN. Both arms identical except the roster, so the ONLY variable is added
# topological diversity -- the three real cities are shared, not swapped:
#   base  environments_c1_4_6   3 real cities
#   div   environments_divwide  the same 3 + 4 synthetic irregular grids
#                               (3x3/4x4/5x5/6x6, 20-30% of interior signals
#                               deleted, so junction degree varies within a city)
#
# Confound stated up front, NOT controlled: more cities also means more gradient
# steps per round, so `div` sees more data as well as more diverse data. That is
# inherent to the intervention and sec 71 did not control it either. If div wins,
# the follow-up is a data-matched arm; if it is null, the confound does not matter.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

OUT=results/diversity
RUNS=results/diversity_runs
mkdir -p $OUT $RUNS
DRIVER=$OUT/driver.log
# 7-city runs need ~6GB each (~1.5 + 0.65/city); 2 concurrent fits 23GB with the
# 3-city arm alongside. Do NOT raise this unattended.
MAX_CONCURRENT=2
SEEDS="${SEEDS:-3 7 11}"
ROUNDS="${ROUNDS:-10}"

log() { echo "=== [$(date '+%F %T')] $* ===" >> $DRIVER; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 30; done; }
drain() {
  wait 2>/dev/null || true
  while pgrep -f "experiments.federated_training" > /dev/null; do sleep 30; done
}

run_arm() {   # arm(base|div), seed
  local arm=$1
  local seed=$2
  local tag="${arm}_s${seed}"
  local marker="$RUNS/${tag}.rundir"
  local resume=""
  local base_dir
  case "$arm" in
    base) base_dir="environments_c1_4_6" ;;
    div)  base_dir="environments_divwide" ;;
    *) echo "unknown arm $arm"; return 1 ;;
  esac

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

  log "starting $tag ($base_dir)"
  ( python -m experiments.federated_training --parallel \
      --base_dir "$base_dir" --pad_to_true_holdout --phase_relational \
      --rounds "$ROUNDS" --local_episodes 2 --eval_every 1 --eval_episodes 5 \
      --lr 3e-4 --lr_decay 0.97 --min_lr 1e-5 --q_entropy_weight 0.05 \
      --seed "$seed" $resume >> "$OUT/$tag.log" 2>&1
    local rc=$?
    grep -oE "results/run_[0-9_-]+_[0-9]+" "$OUT/$tag.log" | tail -1 > "$marker"
    log "finished $tag exit=$rc" ) &
}

log "diversity batch starting: seeds=$SEEDS rounds=$ROUNDS arms=base,div"
for SEED in $SEEDS; do
  throttle; run_arm base "$SEED"
  throttle; run_arm div  "$SEED"
done
drain
log "DIVERSITY BATCH DONE"
