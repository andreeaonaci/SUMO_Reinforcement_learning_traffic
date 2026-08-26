#!/bin/bash
# One-off batch driver for the §49 follow-up: 30-episode reeval confirmation of
# lowest-5-episode-std candidate rounds, matched across federated and
# no_federation runs (see fidings/divergence_investigation.md §49's open item).
# Reads /tmp/lockin_candidates.txt (group\tseed\tlabel\tcheckpoint_path per line).
set -uo pipefail

export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

export OUT_DIR="/tmp/lockin_reeval_out"
mkdir -p "$OUT_DIR"

run_one() {
  local group="$1" seed="$2" label="$3" ckpt="$4"
  local out="$OUT_DIR/${group}_s${seed}_${label}.log"
  echo "=== [$(date '+%F %T')] START $group seed=$seed label=$label ckpt=$ckpt ===" >> "$OUT_DIR/driver.log"
  python3 diagnostics/reeval_checkpoint.py "$ckpt" \
    --base_dir environments_c1_4 --episodes 30 --dueling --pad_to_true_holdout \
    > "$out" 2>&1
  local rc=$?
  echo "=== [$(date '+%F %T')] DONE $group seed=$seed label=$label exit=$rc out=$out ===" >> "$OUT_DIR/driver.log"
}
export -f run_one

cd "$(dirname "$0")/.."

cat /tmp/lockin_candidates.txt | xargs -P 3 -L 1 bash -c 'run_one "$@"' _
echo "=== ALL DONE $(date '+%F %T') ===" >> "$OUT_DIR/driver.log"
