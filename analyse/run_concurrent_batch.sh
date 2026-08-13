#!/usr/bin/env bash
# Run a batch of federated_training experiments with bounded concurrency
# instead of strictly sequentially. Default entry point for any multi-run
# experiment batch (seed sweeps, flag ablations, etc.) -- prefer this over a
# fresh one-off sequential script.
#
# Why concurrency still helps despite real contention: with only ONE job
# running, city workers use just ~13-15% of one core each (SUMO/libsumo
# per-tick stepping is the bottleneck, not CPU/PyTorch compute) -- but that
# measurement understated real contention at MAX_CONCURRENT=3. Measured
# end-to-end (2026-08-10, 12-core/15GB-RAM dev machine, 4-job batch, 3
# concurrent + 1 trailing solo): each of the 3 concurrent runs slowed from
# the ~5 min/round solo baseline to ~8 min/round (~60% slower per run, not
# the ~20% a short early sample suggested -- contention worsens as a run
# progresses, plausibly growing replay-buffer memory pressure). Net result
# over the whole batch: ~1.5x faster wall-clock than running the same jobs
# fully sequentially (259 min actual vs ~400 min sequential estimate) --
# a real, worthwhile win, just not the near-linear speedup the idle-capacity
# framing alone would suggest. RAM remains the hard ceiling on how many can
# run at all (~2.5GB per 2-city run, ~3.5GB per 3-city run); MAX_CONCURRENT=3
# is a conservative default balancing that against the per-run slowdown
# above -- don't assume raising it keeps paying off linearly.
#
# Usage:
#   bash analyse/run_concurrent_batch.sh <log_file> <max_concurrent> \
#       "TAG1|BASE_DIR1|EXTRA_FLAGS1" "TAG2|BASE_DIR2|EXTRA_FLAGS2" ...
#
# Each job spec is "TAG|BASE_DIR|EXTRA_FLAGS" (pipe-delimited, 3 fields).
# EXTRA_FLAGS is appended after BASE_COMMON, so it can override any
# BASE_COMMON default (e.g. a per-job --seed) since argparse keeps the last
# occurrence of a repeated flag.
#
# Override the shared default flags by exporting BASE_COMMON before calling,
# e.g.:
#   BASE_COMMON="--parallel --rounds 10 --local_episodes 2 --aggregation_strategy fedavg --lr 3e-4" \
#       bash analyse/run_concurrent_batch.sh ...
#
# Each job's own run_dir (results/run_<timestamp>_<pid>) is captured via the
# exact PID of the python process that ran it (not "most recently modified
# directory", which is unreliable once multiple jobs are finishing near-
# simultaneously) -- safe to read run_dir out of the finish marker line.
#
# Prerequisites: SUMO_HOME set, run from a shell that can resolve `python`
# to the project's env. Uses the run_dir PID-suffix fix in
# experiments/federated_training.py (main()) -- concurrent launches within
# the same wall-clock second get distinct directories, verified 2026-08-10.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

LOG="$1"          # shared log file (each job's stdout/stderr appended, tag-prefixed per line)
MAX_CONCURRENT="${2:-3}"
shift 2
JOBS=("$@")        # each element: "TAG|BASE_DIR|EXTRA_FLAGS"

BASE_COMMON="${BASE_COMMON:-"--parallel --rounds 20 --local_episodes 2 --aggregation_strategy fedavg --lr 3e-4 --lr_decay 0.97 --min_lr 1e-5"}"

run_one () {
    local SPEC=$1
    local TAG BASE_DIR EXTRA_FLAGS
    IFS='|' read -r TAG BASE_DIR EXTRA_FLAGS <<< "$SPEC"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $TAG base_dir=$BASE_DIR flags='$EXTRA_FLAGS' ===" >> "$LOG"
    python -m experiments.federated_training $BASE_COMMON --base_dir "$BASE_DIR" $EXTRA_FLAGS \
        > >(sed "s/^/[$TAG:$BASE_DIR] /" >> "$LOG") 2>&1 &
    local PYPID=$!
    wait "$PYPID"
    local STATUS=$?
    local RUN_DIR
    RUN_DIR=$(ls -1d results/run_*_"$PYPID" 2>/dev/null | head -1)
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] finished $TAG base_dir=$BASE_DIR exit=$STATUS run_dir=$RUN_DIR ===" >> "$LOG"
}

running=0
for spec in "${JOBS[@]}"; do
    run_one "$spec" &
    running=$((running+1))
    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running-1))
    fi
done
wait
echo "=== BATCH ALL DONE ===" >> "$LOG"
