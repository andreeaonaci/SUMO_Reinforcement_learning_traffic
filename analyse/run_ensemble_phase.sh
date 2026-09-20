#!/bin/bash
# Does the sec 93 ensemble still win once the readout is fixed? (fidings sec 106)
#
# Sec 93 found a majority vote over six independently trained checkpoints beat
# every member and their weight-space average -- but on the INDEXED head, where
# members were gridlocked and locked-in. The phase-relational head's remaining
# weakness is volatility, which is exactly what vote-space combination targets.
# The six sec 100 checkpoints already exist, so this costs evaluation only.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

MAIN=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/results
OUT=results/ensemble_phase
mkdir -p $OUT

# sec 100 phase-relational, rescofull, seeds 3/7/11/17/21/25.
CK=""
for D in 01_29_50_1483406 01_29_50_1483410 01_29_50_1483409 \
         02_09_28_1491576 02_09_32_1491628 02_09_34_1491714; do
  CK="$CK $MAIN/run_2026_09_09-$D/global_round_005.pth"
done

python diagnostics/swa_reeval.py $CK \
  --base_dir environments_rescofull --pad_to_true_holdout \
  --mode both --episodes 30 > $OUT/ens_phase.log 2>&1
echo "ENSEMBLE DONE rc=$?" >> $OUT/ens_phase.log
