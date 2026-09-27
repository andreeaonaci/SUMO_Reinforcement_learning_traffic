#!/bin/bash
# Run a Python command inside Braun's isolated uv venv (never our miniconda env).
# Clears PYTHONPATH so ROS / SUMO_HOME/tools packages cannot shadow the venv's
# pinned traci/sumolib/libsumo (1.27.x from uv.lock).
BRAUN_ROOT=/home/deea/external/GNN-Traffic-Signal-Control-ea47985ccba2bbca273eb08139645399cf53ef23
unset PYTHONPATH
export SUMO_HOME=/usr/share/sumo
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$BRAUN_ROOT" || exit 1
exec "$BRAUN_ROOT/.venv/bin/python" "$@"
