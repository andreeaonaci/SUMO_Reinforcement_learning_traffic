#!/usr/bin/env bash
# Capacity + hygiene check before launching a training batch.
#
# RAM is the binding constraint on this machine, not CPU (sec 22/23: city
# workers are bursty at ~13-15% of one core each; SUMO/libsumo per-tick
# stepping is the real limit). And daemon workers from a CRASHED parent do
# not get cleaned up -- ~11 orphans once sat on ~9GB for four days before
# anyone noticed (2026-08-31).
#
# Usage: bash .claude/skills/launch/preflight.sh [n_cities_per_run] [n_concurrent]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CITIES="${1:-3}"
CONCURRENT="${2:-3}"

echo "=============================================================="
echo "PREFLIGHT  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=============================================================="

# ---- environment ----
if [ -z "${SUMO_HOME:-}" ]; then
    echo "FAIL  SUMO_HOME is unset -- export SUMO_HOME=/usr/share/sumo"
else
    echo "ok    SUMO_HOME=$SUMO_HOME"
fi
case ":${PYTHONPATH:-}:" in
    *":${SUMO_HOME:-/nonexistent}/tools:"*) echo "ok    SUMO tools on PYTHONPATH" ;;
    # Not a problem for the documented launch path: run_concurrent_batch.sh
    # exports both itself. Only matters if you invoke the trainer directly.
    *) echo "info  \$SUMO_HOME/tools not on PYTHONPATH here (run_concurrent_batch.sh sets it)" ;;
esac

# ---- live runs ----
# pgrep -c prints 0 but exits 1 when nothing matches, so count lines instead.
LIVE=$(pgrep -f "experiments\.federated_training" 2>/dev/null | wc -l)
echo "info  $LIVE live federated_training process(es)"
if [ "$LIVE" -gt 0 ]; then
    pgrep -af "experiments\.federated_training" 2>/dev/null | cut -c1-150 | sed 's/^/      /'
fi

# ---- orphaned workers ----
# A spawn_main worker whose parent is no longer a live federated_training
# process is an orphan: its parent died without cleaning up its daemon children.
#
# Do NOT test for ppid==1. Reparenting does not always go to PID 1 -- under WSL
# these land on the session init (observed: PID 465), so a ppid==1 test reports
# "no orphans" while 18 of them sit on ~10GB. Test against the set of live
# parents instead, which is correct wherever reparenting points.
LIVE_PARENTS=$(pgrep -f "experiments\.federated_training" 2>/dev/null | tr '\n' ' ')
ORPHANS=$(ps -eo pid,ppid,rss,args --no-headers 2>/dev/null \
    | awk -v live=" $LIVE_PARENTS " '
        /spawn_main/ && index(live, " " $2 " ") == 0 {n++; s+=$3; pids=pids $1 " "}
        END {printf "%d %.1f %s", n, s/1048576, pids}')
read -r ORPHAN_N ORPHAN_GB ORPHAN_PIDS <<<"$ORPHANS"
if [ "${ORPHAN_N:-0}" -gt 0 ]; then
    echo "WARN  $ORPHAN_N orphaned spawn_main worker(s) holding ~${ORPHAN_GB}GB"
    echo "      these are from a crashed/killed parent and will never exit on their own."
    echo "      Results are checkpointed every round, so SIGTERM is safe:"
    echo "        kill $ORPHAN_PIDS"
else
    echo "ok    no orphaned spawn_main workers"
fi

# ---- memory ----
# ~2.5GB per 2-city run, ~3.5GB per 3-city run (sec 22), roughly linear per city.
AVAIL_MB=$(free -m | awk 'NR==2 {print $7}')
read -r AVAIL_GB PER_RUN NEED <<<"$(awk -v a="$AVAIL_MB" -v c="$CITIES" -v n="$CONCURRENT" \
    'BEGIN {p = 1.5 + 0.65*c; printf "%.1f %.1f %.1f", a/1024, p, p*n}')"
echo "info  RAM available ${AVAIL_GB}GB; estimate ${PER_RUN}GB/run x ${CONCURRENT} concurrent = ${NEED}GB"
if awk -v a="$AVAIL_GB" -v n="$NEED" 'BEGIN {exit !(a > n*1.25)}'; then
    echo "ok    enough headroom for MAX_CONCURRENT=$CONCURRENT"
else
    echo "WARN  headroom is tight -- lower MAX_CONCURRENT or wait for running jobs"
fi

# ---- cores ----
echo "info  $(nproc) cores. CPU is not the binding constraint; do not raise"
echo "      MAX_CONCURRENT on core count alone (each concurrent run slows ~60%, sec 22)."

# ---- disk ----
df -h . | awk 'END {print "info  disk " $4 " free (" $5 " used) on " $6}'

echo "=============================================================="
echo "Reminders before you launch:"
echo "  - matched baseline arm in the SAME batch (not a historical number)"
echo "  - identical --rounds on both arms (compute_eps_decay scales with it)"
echo "  - --pad_to_true_holdout on reduced rosters, else eval silently falls"
echo "    back to a TRAINING city (sec 25); verify is_true_holdout afterwards"
echo "  - seeds 3/7/11 for a screen; add 17/21/25 for confirmation"
echo "=============================================================="
