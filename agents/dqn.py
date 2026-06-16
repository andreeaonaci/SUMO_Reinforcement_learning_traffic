from typing import Any, Dict, Tuple
import random
import logging
import math
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from agents.networks import MLP

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: float) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


if TORCH_AVAILABLE:

    class DQNAgent:
        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            lr: float = 1e-3,
            device: str = "cpu",
            hidden: int = 128,
            gamma: float = 0.99,
            batch_size: int = 64,
            buffer_size: int = 100000,
            target_update: int = 1000,
        ) -> None:
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.device = torch.device(device)
            self.q = MLP(self.obs_dim, self.action_dim, hidden).to(self.device)
            self.q_target = MLP(self.obs_dim, self.action_dim, hidden).to(self.device)
            self.q_target.load_state_dict(self.q.state_dict())
            self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
            self.replay = ReplayBuffer(buffer_size)
            self.batch_size = batch_size
            self.gamma = gamma
            self.eps_start = 1.0
            self.eps_end = 0.05
            self.eps_decay = 50000.0
            self.steps_done = 0
            self.target_update = target_update

        def select_action(self, state: np.ndarray) -> int:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if random.random() < eps_threshold:
                return random.randrange(self.action_dim)
            with torch.no_grad():
                s = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
                qv = self.q(s)
                return int(qv.argmax(dim=1).item())

        def optimize(self) -> None:
            if len(self.replay) < max(4, self.batch_size):
                return
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            q_values = self.q(states).gather(1, actions)
            with torch.no_grad():
                next_q = self.q_target(next_states).max(1)[0].unsqueeze(1)
                expected = rewards + (1.0 - dones) * self.gamma * next_q

            loss = nn.functional.mse_loss(q_values, expected)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.optimizer.step()

        def train(self, env: Any, episodes: int = 5) -> Tuple[Dict[str, torch.Tensor], int]:
            total_steps = 0
            for _ in range(episodes):
                reset_ret = env.reset()
                if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
                    state = reset_ret[0]
                else:
                    state = reset_ret
                done = False
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    self.replay.add(state, int(action), float(reward), next_state, float(done))
                    self.optimize()
                    state = next_state
                    total_steps += 1
                    if total_steps % self.target_update == 0:
                        self.q_target.load_state_dict(self.q.state_dict())

            return self.state_dict(), total_steps

        def state_dict(self) -> Dict[str, torch.Tensor]:
            return {k: v.cpu() for k, v in self.q.state_dict().items()}

        def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
            self.q.load_state_dict(state)
            self.q_target.load_state_dict(state)

        def save(self, path: str) -> None:
            torch.save(self.q.state_dict(), path)

        def load(self, path: str) -> None:
            st = torch.load(path, map_location=self.device)
            self.load_state_dict(st)

else:
    class DQNAgent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for DQNAgent. Install it with: "
                "sudo apt install python3-torch  (WSL/Ubuntu) or pip install torch"
            )
