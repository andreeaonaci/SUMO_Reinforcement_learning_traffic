#!/usr/bin/env bash
# Phase 0 ablation: fedavg strategy, masked-head fix ON vs OFF, 3 seeds,
# cheap 3-city subset. Run overnight; safe to re-run (skips runs already
# moved into results/phase0/<run_name>/).
#
# Prereq (one-time, run before the loop):
#   1. Add --base_dir CLI arg to federated_training.py's main() (see
#      phase0_patch_notes.md, Option A) -- default it to your normal
#      "environments" dir so nothing else breaks.
#   2. Symlink the 3-city subset:
#        mkdir -p environments_phase0
#        for c in city_1 city_2 city_7; do
#          ln -s "$(pwd)/environments/$c" "environments_phase0/$c"
#        done
#      (city_5_holdout is excluded by name inside load_clients regardless
#      of base_dir, so no need to symlink or exclude it manually.)

set -euo pipefail

SEEDS=(1 2 3)
CONDITIONS=("with_fix" "without_fix")
ROUNDS=10
LOCAL_EPISODES=2
STRATEGY=fedavg
BASE_DIR="environments_phase0"                      #"environments_phase0"   "environments_city1"
RESULTS_DIR="results"                # where run_<timestamp> dirs land
PHASE0_DIR="results/phase0"          # where we file them afterwards

mkdir -p "$PHASE0_DIR"

for seed in "${SEEDS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    run_name="${STRATEGY}_${cond}_seed${seed}"
    final_dir="${PHASE0_DIR}/${run_name}"

    if [[ -f "${final_dir}/federated_history.json" ]]; then
      echo "[skip] ${run_name} already has results"
      continue
    fi

    extra_flag=""
    if [[ "$cond" == "without_fix" ]]; then
      extra_flag="--disable_head_fix"
    fi

    # snapshot existing run_* dirs so we can spot the new one afterwards
    before=$(ls -1d "${RESULTS_DIR}"/run_* 2>/dev/null || true)

    echo "=== Running ${run_name} ==="
    python -m experiments.federated_training \
      --aggregation_strategy "$STRATEGY" \
      --seed "$seed" \
      --rounds "$ROUNDS" \
      --local_episodes "$LOCAL_EPISODES" \
      --base_dir "$BASE_DIR" \
      --parallel \
      $extra_flag \
      2>&1 | tee "/tmp/${run_name}.log"

    after=$(ls -1d "${RESULTS_DIR}"/run_* 2>/dev/null || true)
    new_dir=$(comm -13 <(echo "$before" | sort) <(echo "$after" | sort) | tail -n 1)

    if [[ -z "$new_dir" ]]; then
      echo "[error] couldn't detect new results dir for ${run_name} -- check /tmp/${run_name}.log"
      exit 1
    fi

    mkdir -p "$final_dir"
    mv "$new_dir"/* "$final_dir"/
    rmdir "$new_dir"
    mv "/tmp/${run_name}.log" "${final_dir}/run.log"

    echo "=== Finished ${run_name} -> ${final_dir} ==="
  done
done

echo "All Phase 0 runs complete. Results under ${PHASE0_DIR}/"
echo "Next: /usr/bin/python3 experiments/analyze_phase0.py --results_root ${PHASE0_DIR}"