#!/usr/bin/env bash
# Phase 1 ablation: full 7-city roster, 5 seeds, multiple strategies.
#
# Runs compared:
#   1. fedavg_with_fix        (5 seeds)  — main trained baseline
#   2. fedavg_without_fix     (5 seeds)  — ablation: no masked-head aggregation
#   3. ema_alignment_with_fix (5 seeds)  — best adaptive strategy from Phase 0
#   4. no_federation          (5 seeds)  — independent per-city DQN (no FedAvg)
#   5. clustered_fedavg_k2    (5 seeds)  — Phase 3 strategy, action_dim clusters
#   6. baseline_fixed_time    (1 run)    — rule-based, always phase 0
#   7. baseline_max_pressure  (1 run)    — rule-based, max-pressure per phase
#
# Prerequisites (run once before this script):
#   - All 7 city environments must exist under environments/
#   - city_5_holdout is auto-excluded from training and used for evaluation
#   - SUMO_HOME must be set and the sumo-fl conda env active
#
# Usage:
#   conda activate sumo-fl
#   bash analyse/run_phase1_ablation.sh
#
# Safe to re-run: skips any run whose final_dir already has federated_history.json.
# Logs for each run are saved to results/phase1/<run_name>/run.log

set -euo pipefail

SEEDS=(1 2 3 4 5)
ROUNDS=10
LOCAL_EPISODES=2
BASE_DIR="environments"
RESULTS_DIR="results"
PHASE1_DIR="results/phase1"

mkdir -p "$PHASE1_DIR"

# ── Helper: run one training job and file the results ─────────────────────
run_training() {
  local run_name="$1"; shift          # remaining args passed to python
  local final_dir="${PHASE1_DIR}/${run_name}"

  if [[ -f "${final_dir}/federated_history.json" ]]; then
    echo "[skip] ${run_name} already complete"
    return 0
  fi

  before=$(ls -1d "${RESULTS_DIR}"/run_* 2>/dev/null | sort || true)

  echo "=== Running ${run_name} ==="
  python -m experiments.federated_training \
    --base_dir "$BASE_DIR" \
    --rounds "$ROUNDS" \
    --local_episodes "$LOCAL_EPISODES" \
    --eval_episodes 2 \
    --parallel \
    "$@" \
    2>&1 | tee "/tmp/${run_name}.log"

  after=$(ls -1d "${RESULTS_DIR}"/run_* 2>/dev/null | sort || true)
  new_dir=$(comm -13 <(echo "$before") <(echo "$after") | tail -n 1)

  if [[ -z "$new_dir" ]]; then
    echo "[error] couldn't detect results dir for ${run_name}. Check /tmp/${run_name}.log"
    exit 1
  fi

  mkdir -p "$final_dir"
  mv "$new_dir"/* "$final_dir"/
  rmdir "$new_dir"
  mv "/tmp/${run_name}.log" "${final_dir}/run.log"
  echo "=== Finished ${run_name} -> ${final_dir} ==="
}

# ── Helper: run a baseline-only evaluation ────────────────────────────────
run_baseline() {
  local controller="$1"
  local run_name="baseline_${controller}"
  local final_dir="${PHASE1_DIR}/${run_name}"

  if [[ -f "${final_dir}/federated_history.json" ]]; then
    echo "[skip] ${run_name} already complete"
    return 0
  fi

  before=$(ls -1d "${RESULTS_DIR}"/run_* 2>/dev/null | sort || true)

  echo "=== Running ${run_name} ==="
  python -m experiments.federated_training \
    --base_dir "$BASE_DIR" \
    --baseline_controller "$controller" \
    --eval_episodes 5 \
    2>&1 | tee "/tmp/${run_name}.log"

  after=$(ls -1d "${RESULTS_DIR}"/run_* 2>/dev/null | sort || true)
  new_dir=$(comm -13 <(echo "$before") <(echo "$after") | tail -n 1)

  if [[ -z "$new_dir" ]]; then
    echo "[error] couldn't detect results dir for ${run_name}. Check /tmp/${run_name}.log"
    exit 1
  fi

  mkdir -p "$final_dir"
  mv "$new_dir"/* "$final_dir"/
  rmdir "$new_dir"
  mv "/tmp/${run_name}.log" "${final_dir}/run.log"
  echo "=== Finished ${run_name} -> ${final_dir} ==="
}

# ==========================================================================
# 1. Baselines (single deterministic run each, no seeds needed)
# ==========================================================================
run_baseline fixed_time
run_baseline max_pressure

# ==========================================================================
# 2. FedAvg ablation: masked-head fix ON vs OFF (5 seeds)
# ==========================================================================
for seed in "${SEEDS[@]}"; do
  run_training "fedavg_with_fix_seed${seed}" \
    --aggregation_strategy fedavg \
    --seed "$seed"

  run_training "fedavg_without_fix_seed${seed}" \
    --aggregation_strategy fedavg \
    --seed "$seed" \
    --disable_head_fix
done

# ==========================================================================
# 3. EMA Gradient Alignment (5 seeds, with_fix)
# ==========================================================================
for seed in "${SEEDS[@]}"; do
  run_training "ema_alignment_with_fix_seed${seed}" \
    --aggregation_strategy ema_alignment \
    --seed "$seed"
done

# ==========================================================================
# 4. No-federation independent DQN (5 seeds)
# ==========================================================================
for seed in "${SEEDS[@]}"; do
  run_training "no_federation_seed${seed}" \
    --aggregation_strategy fedavg \
    --no_federation \
    --seed "$seed"
done

# ==========================================================================
# 5. Clustered FedAvg k=2 (5 seeds)
# ==========================================================================
for seed in "${SEEDS[@]}"; do
  run_training "clustered_fedavg_k2_seed${seed}" \
    --aggregation_strategy clustered_fedavg \
    --n_clusters 2 \
    --seed "$seed"
done

# ==========================================================================
echo ""
echo "All Phase 1 runs complete. Results under ${PHASE1_DIR}/"
echo ""
echo "Next — summary table:"
echo "  python experiments/analyze_phase1.py --results_root ${PHASE1_DIR}"
echo ""
echo "Next — learning curves CSV:"
echo "  python experiments/analyze_phase1.py --results_root ${PHASE1_DIR} --csv results/phase1/curves.csv"
