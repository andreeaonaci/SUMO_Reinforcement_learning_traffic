# Federated RL for Traffic Signal Control

This repository implements a research-grade federated reinforcement learning pipeline for adaptive traffic signal control.

Quick start (mock, no SUMO required):

```bash
 python -m experiments.federated_training --parallel --rounds 10 --local_episodes 2 --aggregation_strategy fedavg --lr 3e-4 --lr_decay 0.97 --min_lr 1e-5
python experiments/evaluate.py --model results/global_fed.pth --episodes 5
```

See `experiments/` for scenario scripts.
# SUMO Reinforcement Learning Traffic

`sumo-rl` provides reinforcement learning environments and training utilities for traffic signal control using SUMO.

## Overview

- Gymnasium-compatible SUMO traffic signal control environments
- PettingZoo-compatible multi-agent interfaces
- Example and research-ready scenarios for 2-way, grid, and arterial intersections
- Support for SUMO network and route files packaged with the repository

## Installation

1. Install SUMO and set `SUMO_HOME` in your environment.
2. Install the package and required Python dependencies:

```bash
pip install .
```

3. For optional rendering support:

```bash
pip install .[rendering]
```

4. For optional training and NACRL PyTorch models, make sure `torch` is installed. If `pip install .` does not install PyTorch on your platform, install it separately:

```bash
pip install torch
```

## Requirements

- Python 3.9+
- `gymnasium>=0.28`
- `pettingzoo>=1.24.3`
- `numpy`
- `pandas`
- `pillow`
- `sumolib>=1.14.0`
- `traci>=1.14.0`
- `torch>=1.13.1`

## Usage

```python
from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
    route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    use_gui=False,
    num_seconds=1000,
)

obs = env.reset()
print(obs)

state, reward, done, info = env.step(env.action_space.sample())
print(reward, done)
```

## Package Contents

- `sumo_rl/environment/` – SUMO environment wrappers and observation utilities
- `sumo_rl/agents/` – learning agents and policies
- `sumo_rl/nets/` – sample SUMO network and route files
- `experiments/` – example training scripts

## License

MIT License
