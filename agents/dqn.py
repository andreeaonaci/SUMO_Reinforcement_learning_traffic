from typing import Dict, Optional, Tuple
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
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:

    class DQNAgent:
        """Torch-based DQN agent."""

        def __init__(self, obs_dim: int = 4, action_dim: int = 2, lr: float = 1e-3, device: str = "cpu"):
            self.device = device
            self.q = MLP(obs_dim, action_dim).to(self.device)
            self.q_target = MLP(obs_dim, action_dim).to(self.device)
            self.q_target.load_state_dict(self.q.state_dict())
            self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
            self.replay = ReplayBuffer(10000)
            self.batch_size = 64
            self.gamma = 0.99
            self.eps_start = 1.0
            self.eps_end = 0.05
            self.eps_decay = 500
            self.steps_done = 0

        def select_action(self, state: np.ndarray) -> int:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if random.random() < eps_threshold:
                return random.randrange(self.q.net[-1].out_features)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    return int(self.q(s).argmax().item())

        def optimize(self):
            if len(self.replay) < 4:
                return
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            q_values = self.q(states).gather(1, actions)
            next_q = self.q_target(next_states).max(1)[0].detach().unsqueeze(1)
            expected = rewards + (1 - dones) * self.gamma * next_q
            loss = nn.functional.mse_loss(q_values, expected)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def train(self, env, episodes: int = 5) -> Tuple[Dict[str, torch.Tensor], int]:
            total_steps = 0
            for ep in range(episodes):
                state = env.reset()
                done = False
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    self.replay.add((state, action, reward, next_state, float(done)))
                    self.optimize()
                    state = next_state
                    total_steps += 1
                # update target
                if ep % 2 == 0:
                    self.q_target.load_state_dict(self.q.state_dict())

            return self.state_dict(), total_steps

        def state_dict(self) -> Dict[str, torch.Tensor]:
            return {k: v.cpu() for k, v in self.q.state_dict().items()}

        def load_state_dict(self, state: Dict[str, torch.Tensor]):
            self.q.load_state_dict(state)
            self.q_target.load_state_dict(state)

        def save(self, path: str):
            torch.save(self.q.state_dict(), path)

        def load(self, path: str):
            st = torch.load(path, map_location=self.device)
            self.load_state_dict(st)

else:

    class DQNAgent:
        """Numpy-based lightweight DQN agent fallback when torch is unavailable.

        This agent uses simple linear weights and random updates; intended for smoke tests only.
        """

        def __init__(self, obs_dim: int = 4, action_dim: int = 2, lr: float = 1e-3, device: str = "cpu"):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.w = np.zeros((obs_dim, action_dim), dtype=float)
            self.b = np.zeros((action_dim,), dtype=float)
            self.replay = ReplayBuffer(10000)
            self.batch_size = 32
            self.steps_done = 0

        def select_action(self, state: np.ndarray) -> int:
            # epsilon-greedy with simple decay
            eps = max(0.05, 1.0 - self.steps_done * 1e-3)
            self.steps_done += 1
            if random.random() < eps:
                return random.randrange(self.action_dim)
            logits = state.dot(self.w) + self.b
            return int(np.argmax(logits))

        def train(self, env, episodes: int = 5) -> Tuple[Dict[str, np.ndarray], int]:
            total_steps = 0
            for ep in range(episodes):
                state = env.reset()
                done = False
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    self.replay.add((state, action, reward, next_state, float(done)))
                    # tiny random update to weights (mock learning)
                    self.w += 1e-4 * (np.random.randn(*self.w.shape))
                    self.b += 1e-4 * (np.random.randn(*self.b.shape))
                    state = next_state
                    total_steps += 1

            return self.state_dict(), total_steps

        def state_dict(self) -> Dict[str, np.ndarray]:
            return {"w": np.array(self.w), "b": np.array(self.b)}

        def load_state_dict(self, state: Dict[str, np.ndarray]):
            if "w" in state:
                w = np.array(state["w"])
                if w.shape == self.w.shape:
                    self.w = w
                else:
                    logger.warning("Skipping load of weights with incompatible shape %s -> %s", w.shape, self.w.shape)
            if "b" in state:
                b = np.array(state["b"])
                if b.shape == self.b.shape:
                    self.b = b
                else:
                    logger.warning("Skipping load of bias with incompatible shape %s -> %s", b.shape, self.b.shape)

        def save(self, path: str):
            np.savez(path, w=self.w, b=self.b)

        def load(self, path: str):
            try:
                d = np.load(path)
            except FileNotFoundError:
                # try common alternative extensions
                if not path.endswith(".npz"):
                    d = np.load(path + ".npz")
                else:
                    raise
            self.w = d["w"]
            self.b = d["b"]

