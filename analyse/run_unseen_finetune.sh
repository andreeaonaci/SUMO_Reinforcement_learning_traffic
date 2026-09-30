#!/bin/bash
# One-episode fine-tune of the six phase-relational rescofull checkpoints on each
# unseen real network, then the same 5-episode trip evaluation as the zero-shot
# table (fidings sec 113). Skip-or-resume per (network, seed).
#
#   MAX_CONCURRENT=6 bash analyse/run_unseen_finetune.sh [network ...]
#
# Fine-tuning uses SYNTHETIC demand (randomTrips over the network, total rate matched
# to the real hour's departures), never the evaluation route file. Exactly one
# training episode: --rounds 1 --local_episodes 1 --n_variants 1.
set -u
cd "$(dirname "$0")/.."
export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
M=/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/results
MAX_CONCURRENT=${MAX_CONCURRENT:-6}
NETS=${*:-"ingolstadt21 cologne8 cologne1"}
declare -A CK=([3]=1483406 [7]=1483410 [11]=1483409 [17]=1491576 [21]=1491628 [25]=1491714)

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 15; done; }

for net in $NETS; do
  cfg=environments_unseen/$net/config.yaml
  stem=$(basename "$(grep '^net_file:' "$cfg" | awk '{print $2}')" .net.xml)
  out=results/unseen_ft/$net
  mkdir -p "$out"
  for seed in 3 7 11 17 21 25; do
    ck=$(ls "$M"/run_2026_09_09-*_"${CK[$seed]}"/global_round_005.pth)
    ckdir=$out/ckpt_s$seed
    res=$out/ours_${stem}_phase_ckpt_s$seed.json
    [ -f "$res" ] && continue
    throttle
    (
      if [ ! -f "$ckdir/global_round_001.pth" ]; then
        python diagnostics/finetune_on_holdout.py "$ck" --holdout_config "$cfg" \
          --rounds 1 --phase1_rounds 1 --local_episodes 1 --n_variants 1 --eval_episodes 1 \
          --match_real_demand --seed "$seed" --checkpoint_dir "$ckdir" > "$out/ft_s$seed.log" 2>&1
        echo "FINETUNE DONE rc=$?" >> "$out/ft_s$seed.log"
      fi
      python baselines/braun/eval_ours.py "$ckdir/global_round_001.pth" --base_dir environments_rescofull \
        --pad_to_true_holdout --config "$cfg" --episodes 5 --out_dir "$out" > "$out/eval_s$seed.log" 2>&1
      echo "EVAL DONE rc=$?" >> "$out/eval_s$seed.log"
    ) &
  done
done
wait
echo "UNSEEN FINETUNE BATCH DONE"
