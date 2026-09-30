#!/bin/bash
# Parallel version of run_unseen_eval.sh (fidings sec 112): one process per
# (network, controller), up to MAX_CONCURRENT at once. SUMO runs one simulation per
# core, and running a network's 14 controllers one after another left most cores idle.
# Skip-or-resume at controller level: a controller whose result JSON exists is skipped.
#
#   MAX_CONCURRENT=10 bash analyse/run_unseen_parallel.sh [network ...]
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
M=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/results
MAX_CONCURRENT=${MAX_CONCURRENT:-10}
NETS=${*:-"ingolstadt21 cologne8 grid_5x5_drop20 grid_6x6_drop20 grid_4x4_drop30 grid_3x3_drop20 cologne1"}
PHASE="1483406 1483410 1483409 1491576 1491628 1491714"      # seeds 3 7 11 17 21 25
INDEXED="1499810 1499847 1499949 1507663 1507788 1507922"    # seeds 3 7 11 17 21 25

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 10; done; }

launch() {  # net ctrl head name
  local net=$1 ctrl=$2 head=$3 name=$4
  local cfg=environments_unseen/$net/config.yaml
  local stem
  stem=$(basename "$(grep '^net_file:' "$cfg" | awk '{print $2}')" .net.xml)
  local out=results/unseen/$net
  local res=$out/ours_${stem}_${head}_${name}.json
  [ -f "$res" ] && return 0
  mkdir -p "$out"
  throttle
  ( python baselines/braun/eval_ours.py "$ctrl" --base_dir environments_rescofull --pad_to_true_holdout \
      --config "$cfg" --episodes 5 --out_dir "$out" > "$out/job_${head}_${name}.log" 2>&1
    echo "JOB DONE rc=$?" >> "$out/job_${head}_${name}.log" ) &
}

for net in $NETS; do
  launch "$net" max_pressure max_pressure max_pressure
  launch "$net" fixed_time fixed_time fixed_time
  for p in $PHASE; do
    ck=$(ls "$M"/run_2026_09_09-*_"$p"/global_round_005.pth)
    launch "$net" "$ck" phase "$(basename "$(dirname "$ck")")"
  done
  for p in $INDEXED; do
    ck=$(ls "$M"/run_2026_09_09-*_"$p"/global_round_005.pth)
    launch "$net" "$ck" indexed "$(basename "$(dirname "$ck")")"
  done
done
wait
for net in $NETS; do
  n=$(ls results/unseen/$net/ours_*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 14 ]; then echo "UNSEEN DONE $net rc=0 (parallel, $n controllers)" >> results/unseen/$net/run.log; fi
done
