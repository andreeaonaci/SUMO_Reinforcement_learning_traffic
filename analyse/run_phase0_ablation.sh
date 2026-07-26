#!/usr/bin/env bash
# Phase 0 ablation: fedavg-strategy, masked-head-fix ON vs OFF, 3 seeds,
# cheap 3-city subset. Run overnight; safe to Ctrl-C and resume (skips
# runs whose output dir already has a federated_history.json).
#
# Assumes main.py exposes: --aggregation_strategy, --disable_head_fix,
# --seed, --parallel, --rounds, --local_episodes, --output_dir (adjust
# flag names to match your real main.py / experiments/federated_training.py).

set -euo pipefail

SEEDS=(1 2 3)
CONDITIONS=("with_fix" "without_fix")   # maps to --disable_head_fix flag below
ROUNDS=10
LOCAL_EPISODES=2
STRATEGY=fedavg                          # isolate the head-fix effect; don't
                                          # mix in ema_loss etc. yet
RESULTS_ROOT="results/phase0"
CITIES_DIR="configs/phase0"              # the 3-city subset + fixed holdout

mkdir -p "$RESULTS_ROOT"

for seed in "${SEEDS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    run_name="${STRATEGY}_${cond}_seed${seed}"
    out_dir="${RESULTS_ROOT}/${run_name}"

    if [[ -f "${out_dir}/federated_history.json" ]]; then
      echo "[skip] ${run_name} already has results"
      continue
    fi

    mkdir -p "$out_dir"

    extra_flag=""
    if [[ "$cond" == "without_fix" ]]; then
      extra_flag="--disable_head_fix"
    fi

    echo "=== Running ${run_name} ==="
    python main.py \
      --aggregation_strategy "$STRATEGY" \
      --seed "$seed" \
      --rounds "$ROUNDS" \
      --local_episodes "$LOCAL_EPISODES" \
      --cities_dir "$CITIES_DIR" \
      --output_dir "$out_dir" \
      --parallel \
      $extra_flag \
      2>&1 | tee "${out_dir}/run.log"

    echo "=== Finished ${run_name} ==="
  done
done

echo "All Phase 0 runs complete. Results under ${RESULTS_ROOT}/"
echo "Next: python analyze_phase0.py --results_root ${RESULTS_ROOT}"
