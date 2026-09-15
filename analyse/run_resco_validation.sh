#!/bin/bash
# RESCO validation, three readouts, done the way sec 99 / sec 100b / sec 101b say
# it must be done (fidings sec 103b).
#
# THE CORRECT COMPARISON, and why the obvious one is wrong:
#
#  * Roster MUST be environments_rescofull. sec 99 found three mismatches in the
#    configs this project had been quoting RESCO against -- shifted route files,
#    wrong evaluation windows, and yellow_time 2 against RESCO's 3 (50% more
#    usable green per phase change, for every controller, in every experiment).
#    sec 103's batch ran on environments_c1_4_6, which still carries all three,
#    so its numbers are internally valid but MUST NOT be quoted against RESCO.
#
#  * Metrics MUST be Avg. Delay and Avg. Trip Time only. sec 99 measured that
#    this project's `wait` and `queue` do not reconcile with RESCO's definitions
#    in either direction. Reward is an internal diff-waiting-time unit.
#
#  * `arrived` MUST be reported beside every row. sec 100b: eval_paper_metrics
#    computes delay and trip time over ARRIVED vehicles only, so a controller
#    that strands traffic reports BETTER delay. The raw Cologne mean in sec 100b
#    was a survivorship artifact for exactly this reason.
#
#  * Comparison is IN-DISTRIBUTION (city_4 cologne3, city_6 ingolstadt7), because
#    RESCO's published numbers are in-distribution. Our zero-shot holdout result
#    is a different and harder claim and has no published counterpart.
#
#  * The Ingolstadt row carries sec 101b's caveat: our vendored ingolstadt7 is
#    missing a green phase RESCO's net has at one of its seven intersections.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

OUT=results/resco_validation
mkdir -p $OUT
DRIVER=$OUT/driver.log
EPISODES="${EPISODES:-5}"
log() { echo "=== [$(date '+%F %T')] $* ===" >> $DRIVER; }

# Final-round checkpoints, one per seed. global_round_005.pth, not global_fed.pth:
# they are different files and the latter would be a hindsight pick (sec 100).
#
# sec 100's runs live in the MAIN checkout's results/, not this worktree's, and
# the run dirs are listed EXPLICITLY rather than globbed -- a glob silently
# matched only 4 of the 6 phase runs on the first attempt, which would have
# produced a quietly-wrong 4-seed mean.
MAIN=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/results

PHASE_DIRS="run_2026_09_09-01_29_50_1483406 run_2026_09_09-01_29_50_1483410 \
run_2026_09_09-01_29_50_1483409 run_2026_09_09-02_09_28_1491576 \
run_2026_09_09-02_09_32_1491628 run_2026_09_09-02_09_34_1491714"

IDX_DIRS="run_2026_09_09-02_49_30_1499810 run_2026_09_09-02_49_33_1499847 \
run_2026_09_09-02_49_36_1499949 run_2026_09_09-03_27_33_1507663 \
run_2026_09_09-03_27_46_1507788 run_2026_09_09-03_27_57_1507922"

PHASE=""; for d in $PHASE_DIRS; do
  [ -f "$MAIN/$d/global_round_005.pth" ] && PHASE="$PHASE $MAIN/$d/global_round_005.pth"
done
IDX=""; for d in $IDX_DIRS; do
  [ -f "$MAIN/$d/global_round_005.pth" ] && IDX="$IDX $MAIN/$d/global_round_005.pth"
done
FRAP=""
for m in results/rescofull_frap_runs/frap_s*.rundir; do
  [ -f "$m" ] || continue
  rd=$(cat "$m")
  [ -f "$rd/global_round_005.pth" ] && FRAP="$FRAP $rd/global_round_005.pth"
done

# Refuse to produce a table from an incomplete arm -- a 4-of-6-seed mean
# presented as a 6-seed mean is exactly the class of error sec 99 retracted.
for arm in PHASE IDX FRAP; do
  eval "n=\$(echo \$$arm | wc -w)"
  if [ "$n" -ne 6 ]; then
    log "ABORT: arm $arm has $n/6 checkpoints"
    echo "ABORT: arm $arm has $n/6 checkpoints -- refusing to build a partial table"
    exit 1
  fi
done

log "phase ckpts: $(echo $PHASE | wc -w) | indexed: $(echo $IDX | wc -w) | frap: $(echo $FRAP | wc -w)"

for CITY in city_4 city_6; do
  for ARM in phase indexed frap; do
    case $ARM in
      phase)   CK="$PHASE" ;;
      indexed) CK="$IDX" ;;
      frap)    CK="$FRAP" ;;
    esac
    if [ -z "$CK" ]; then log "SKIP $ARM $CITY (no checkpoints)"; continue; fi
    log "eval $ARM on $CITY ($(echo $CK | wc -w) ckpts)"
    python diagnostics/eval_paper_metrics.py $CK \
      --base_dir environments_rescofull --pad_to_true_holdout \
      --episodes "$EPISODES" --city $CITY > "$OUT/${ARM}_${CITY}.log" 2>&1
    log "done $ARM $CITY exit=$?"
  done
  log "eval rule baselines on $CITY"
  python diagnostics/eval_paper_metrics.py max_pressure fixed_time \
    --base_dir environments_rescofull --pad_to_true_holdout \
    --episodes "$EPISODES" --city $CITY > "$OUT/rule_${CITY}.log" 2>&1
  log "done rule $CITY exit=$?"
done
log "RESCO VALIDATION DONE"
