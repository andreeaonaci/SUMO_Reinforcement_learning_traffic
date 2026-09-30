#!/bin/bash
# Zero-shot evaluation of the ALREADY TRAINED environments_rescofull checkpoints on
# networks none of them trained on (fidings sec 112). Evaluation only.
#
#   bash analyse/run_unseen_eval.sh <network> [...]      e.g. cologne8 grid_4x4_drop30
#   SMOKE=1 bash analyse/run_unseen_eval.sh cologne8     one phase + one indexed ckpt, 1 episode
#
# Networks: environments_unseen/<network>/config.yaml (RESCO window, 3 s yellow).
#   cologne1, cologne8           real, no signal within 685 m / 4.4 km of cologne3
#   ingolstadt21                 real, 14 of 21 signals new, 7 coincide with ingolstadt7
#   grid_*_drop*                 synthetic irregular grids (junctions removed)
#   ingolstadt1 is NOT included: its only signal is an ingolstadt7 training signal.
# Controllers: the six sec-100 phase-relational checkpoints and the six indexed ones
# (final round), plus max pressure and fixed time, all through eval_ours.py (the same
# trip-level pipeline as every trip number in the paper). Skip-or-resume per network.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
M=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/results
PHASE="1483406 1483410 1483409 1491576 1491628 1491714"      # seeds 3 7 11 17 21 25
INDEXED="1499810 1499847 1499949 1507663 1507788 1507922"    # seeds 3 7 11 17 21 25

ckpt() { ls "$M"/run_2026_09_09-*_"$1"/global_round_005.pth; }

for net in "$@"; do
  cfg=environments_unseen/$net/config.yaml
  [ -f "$cfg" ] || { echo "no config for $net"; continue; }
  if [ "${SMOKE:-0}" = 1 ]; then
    out=results/unseen/smoke; eps=1
    ctrls="$(ckpt 1483406) $(ckpt 1499810)"
  else
    out=results/unseen/$net; eps=5
    ctrls="max_pressure fixed_time"
    for p in $PHASE $INDEXED; do ctrls="$ctrls $(ckpt $p)"; done
  fi
  mkdir -p "$out"
  if grep -q "UNSEEN DONE $net" "$out/run.log" 2>/dev/null; then echo "skip $net"; continue; fi
  python baselines/braun/eval_ours.py $ctrls --base_dir environments_rescofull --pad_to_true_holdout \
      --config "$cfg" --episodes $eps --out_dir "$out" >> "$out/run.log" 2>&1
  echo "UNSEEN DONE $net rc=$?" >> "$out/run.log"
done
