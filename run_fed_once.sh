#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${SUMO_HOME:-}" ]]; then
  if [[ -d /usr/share/sumo ]]; then
    export SUMO_HOME=/usr/share/sumo
  elif [[ -d /usr/local/share/sumo ]]; then
    export SUMO_HOME=/usr/local/share/sumo
  else
    echo "SUMO_HOME is not set and no default SUMO install was found." >&2
    exit 1
  fi
fi

export PYTHONPATH="${SUMO_HOME}/tools${PYTHONPATH:+:$PYTHONPATH}"

python3 experiments/run_fed_once.py "$@"
