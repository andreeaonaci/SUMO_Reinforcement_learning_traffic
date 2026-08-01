#!/usr/bin/env bash
# One-time setup for WSL/Ubuntu. Run from repo root:
#   bash setup_wsl.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y python3-pip python3-torch sumo sumo-tools

python3 -m pip install --user gymnasium pyyaml

echo
echo "Setup complete. Run:"
echo "  export SUMO_HOME=/usr/share/sumo"
echo "  export PYTHONPATH=\"\$SUMO_HOME/tools:\$PYTHONPATH\""
echo "  python3 experiments/run_fed_once.py 2>&1 | head -400"
