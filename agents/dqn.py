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

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:

    class DQNAgent:
        """Torch-based DQN agent."""

        def __init__(
            self,
            obs_dim: int = 4,
            action_dim: int = 2,
            lr: float = 1e-3,
            device: str = "cpu",
            buffer_size: int = 10000,
            batch_size: int = 64,
            gamma: float = 0.99,
            target_update: int = 100,
        ):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.device = torch.device(device)
            self.q = MLP(obs_dim, action_dim).to(self.device)
            self.q_target = MLP(obs_dim, action_dim).to(self.device)
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
            self.learn_steps = 0

        def _greedy_action(self, state: np.ndarray) -> int:
            with torch.no_grad():
                s = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q(s)
                return int(q_values.argmax(dim=1).item())

        def _epsilon_action(self, state: np.ndarray) -> int:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1.0 * self.steps_done / self.eps_decay
            )
            self.steps_done += 1
            if random.random() < eps_threshold:
                return random.randrange(self.action_dim)
            return self._greedy_action(state)

        def act(self, state: np.ndarray, explore: bool = True) -> int:
            if explore:
                return self._epsilon_action(state)
            return self._greedy_action(state)

        def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
            self.replay.add(state, int(action), float(reward), next_state, float(done))

        def train_step(self) -> None:
            self.optimize()

        def select_action(self, state: np.ndarray) -> int:
            return self.act(state, explore=True)

        def optimize(self) -> None:
            if len(self.replay) < max(4, self.batch_size):
                return
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            q_values = self.q(states).gather(1, actions)
            with torch.no_grad():
                next_q = self.q_target(next_states).max(1)[0].unsqueeze(1)
                expected = rewards + (1.0 - dones) * self.gamma * next_q

            loss = nn.functional.mse_loss(q_values, expected)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.optimizer.step()
            self.learn_steps += 1
            if self.learn_steps % self.target_update == 0:
                self.q_target.load_state_dict(self.q.state_dict())

        def save(self, path: str) -> None:
            torch.save(self.q.state_dict(), path)

        def load(self, path: str) -> None:
            state = torch.load(path, map_location=self.device)
            self.q.load_state_dict(state)
            self.q_target.load_state_dict(state)

        def state_dict(self) -> Dict[str, torch.Tensor]:
            return {k: v.cpu() for k, v in self.q.state_dict().items()}

        def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
            self.q.load_state_dict(state)
            self.q_target.load_state_dict(state)

        def train(self, env, episodes: int = 5) -> Tuple[Dict[str, torch.Tensor], int]:
            total_steps = 0
            for ep in range(episodes):
                reset_ret = env.reset()
                if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
                    state = reset_ret[0]
                else:
                    state = reset_ret
                done = False
                while not done:
                    action = self.act(state, explore=True)
                    next_state, reward, done, info = env.step(action)
                    self.remember(state, action, reward, next_state, done)
                    self.train_step()
                    state = next_state
                    total_steps += 1
            return self.state_dict(), total_steps

else:

    class DQNAgent:
        """Fallback numpy-based DQN agent."""

        def __init__(
            self,
            obs_dim: int = 4,
            action_dim: int = 2,
            lr: float = 1e-3,
            device: str = "cpu",
            buffer_size: int = 10000,
            batch_size: int = 64,
            gamma: float = 0.99,
        ):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.lr = lr
            self.replay = ReplayBuffer(buffer_size)
            self.batch_size = batch_size
            self.gamma = gamma
            self.eps_start = 1.0
            self.eps_end = 0.05
            self.eps_decay = 50000.0
            self.steps_done = 0
            self.w = np.zeros((obs_dim, action_dim), dtype=np.float32)
            self.b = np.zeros((action_dim,), dtype=np.float32)

        def _greedy_action(self, state: np.ndarray) -> int:
            q_values = np.dot(np.array(state, dtype=np.float32), self.w) + self.b
            return int(np.argmax(q_values))

        def _epsilon_action(self, state: np.ndarray) -> int:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1.0 * self.steps_done / self.eps_decay
            )
            self.steps_done += 1
            if random.random() < eps_threshold:
                return random.randrange(self.action_dim)
            return self._greedy_action(state)

        def act(self, state: np.ndarray, explore: bool = True) -> int:
            if explore:
                return self._epsilon_action(state)
            return self._greedy_action(state)

        def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
            self.replay.add(state, int(action), float(reward), next_state, float(done))

        def train_step(self) -> None:
            if len(self.replay) < 4:
                return
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            # Simple gradient-like weight update for fallback
            for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                td_target = r + self.gamma * np.max(np.dot(ns, self.w) + self.b) * (1.0 - d)
                td_error = td_target - (np.dot(s, self.w[:, a]) + self.b[a])
                self.w[:, a] += self.lr * td_error * s
                self.b[a] += self.lr * td_error

        def save(self, path: str) -> None:
            np.savez(path, w=self.w, b=self.b)

        def load(self, path: str) -> None:
            data = np.load(path)
            self.w = data["w"]
            self.b = data["b"]

        def state_dict(self):
            return {"w": self.w.copy(), "b": self.b.copy()}

        def load_state_dict(self, state):
            self.w = np.array(state["w"], dtype=np.float32)
            self.b = np.array(state["b"], dtype=np.float32)

        def select_action(self, state: np.ndarray) -> int:
            return self.act(state, explore=True)

        def train(self, env, episodes: int = 5) -> Tuple[Dict[str, np.ndarray], int]:
            total_steps = 0
            for ep in range(episodes):
                reset_ret = env.reset()
                if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
                    state = reset_ret[0]
                else:
                    state = reset_ret
                done = False
                while not done:
                    action = self.act(state, explore=True)
                    next_state, reward, done, info = env.step(action)
                    self.remember(state, action, reward, next_state, done)
                    self.train_step()
                    state = next_state
                    total_steps += 1
            return self.state_dict(), total_steps

