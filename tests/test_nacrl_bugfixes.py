"""Regression tests for sumo_rl/nacrl/'s real bugs, fixed in this session:

  1. The buffer could never fill within one episode under the original
     run_nacrl.py config (max_size=10000, T=200, 1 agent, buffer.clear()
     every episode reset) -- agent.update() was never called across 100
     "training" episodes. Nothing ever trained.
  2. Transitions were stored as raw numpy arrays / python scalars, not
     tensors -- update()'s torch.stack()/torch.tensor() calls would have
     crashed the moment the buffer above ever actually filled.
  3. update() looped over EVERY agent_id in the batch and ran a full
     actor/critic gradient step for each one using the calling agent's own
     networks -- every agent's policy got trained to imitate every other
     agent's (state, action) pairs, not just its own.
  4. Attention compared an agent's CURRENT-state embedding against other
     agents' NEXT-state embeddings (temporal mismatch).
  5. The shared embedding network had no optimizer anywhere -- its
     gradients were computed (via the critic loss) but never applied, so it
     stayed a random, untrained projection for the agent's entire lifetime.
  6. `dones.get("__all__", False)` is always False under PettingZoo's
     parallel API (no such key exists), so an episode never ended early
     even once every agent was done.

Uses a tiny fake 2-agent env (no SUMO) so this runs anywhere torch is
installed.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_rl.nacrl.nacrl_agent import NACRLAgent
from sumo_rl.nacrl.nacrl_runner import NACRLRunner
from sumo_rl.nacrl.embedding import StateEmbedding
from sumo_rl.nacrl.attention import AgentAttention
from sumo_rl.nacrl.replay_buffer import MultiAgentReplayBuffer
from sumo_rl.nacrl.actor import Actor
from sumo_rl.nacrl.critic import Critic


OBS_DIM = 6
ACTION_DIM = 3
EMB_DIM = 8


class _FakeMultiAgentEnv:
    """Minimal PettingZoo-parallel-shaped fake env: fixed agent set,
    deterministic obs/reward, 5-tuple step() return (matching the branch of
    NACRLRunner.train() real SUMO envs take)."""

    def __init__(self, agent_ids, episode_len=50, done_after=None):
        self.agents = list(agent_ids)
        self.episode_len = episode_len
        self.done_after = done_after  # if set, terminate all agents after this many steps
        self._t = 0

    def reset(self):
        self._t = 0
        obs = {a: torch.rand(OBS_DIM) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions):
        self._t += 1
        obs = {a: torch.rand(OBS_DIM) for a in self.agents}
        rewards = {a: float(actions[a]) * 0.1 for a in self.agents}
        done = self.done_after is not None and self._t >= self.done_after
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, rewards, terminations, truncations, infos


def _build_agents(agent_ids, device="cpu"):
    embedding = StateEmbedding(input_dim=OBS_DIM, hidden_dim=16, output_dim=EMB_DIM).to(device)
    attention = AgentAttention(embed_dim=EMB_DIM).to(device)
    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-2)

    agents = {}
    for aid in agent_ids:
        actor = Actor(state_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=16).to(device)
        critic = Critic(embedding_dim=EMB_DIM).to(device)
        agents[aid] = NACRLAgent(
            actor=actor,
            critic=critic,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-2),
            critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-2),
        )
    return agents, embedding, attention, embedding_optimizer


def test_buffer_fills_and_update_actually_fires_within_one_episode():
    """The original bug: max_size=10000 could never be reached within one
    episode's step budget, so is_full() was never true and update() was
    dead code. Here buffer_size is sized to the episode so it must fire."""
    agent_ids = ["A", "B"]
    env = _FakeMultiAgentEnv(agent_ids, episode_len=50)
    agents, embedding, attention, embedding_optimizer = _build_agents(agent_ids)
    buffer = MultiAgentReplayBuffer(max_size=16)  # 8 steps * 2 agents = 16

    update_calls = {"count": 0}
    real_update = NACRLAgent.update

    def counting_update(self, *a, **kw):
        update_calls["count"] += 1
        return real_update(self, *a, **kw)

    NACRLAgent.update = counting_update
    try:
        runner = NACRLRunner(
            env=env, agents=agents, buffer=buffer, embedding=embedding,
            attention=attention, embedding_optimizer=embedding_optimizer,
            gamma=0.99, T=20, buffer_size=16,
        )
        runner.train(M=1)
    finally:
        NACRLAgent.update = real_update

    assert update_calls["count"] > 0, (
        "agent.update() was never called -- the buffer never filled, "
        "reproducing the original dead-training bug."
    )


def test_transitions_stored_as_tensors_not_raw_types():
    agent_ids = ["A"]
    env = _FakeMultiAgentEnv(agent_ids, episode_len=10)
    agents, embedding, attention, embedding_optimizer = _build_agents(agent_ids)
    buffer = MultiAgentReplayBuffer(max_size=1000)  # large enough to never trigger update
    runner = NACRLRunner(
        env=env, agents=agents, buffer=buffer, embedding=embedding,
        attention=attention, embedding_optimizer=embedding_optimizer,
        gamma=0.99, T=5, buffer_size=1000,
    )
    runner.train(M=1)

    transitions = buffer.sample()["A"]
    assert len(transitions) > 0
    s, a, r, s_next = transitions[0]
    assert isinstance(s, torch.Tensor)
    assert isinstance(r, torch.Tensor)
    assert isinstance(s_next, torch.Tensor)
    # Directly reproduce what update() does with a sampled batch -- this
    # crashed with a TypeError before the fix.
    torch.stack([t[0] for t in transitions])
    torch.stack([t[2] for t in transitions])


def test_update_does_not_evaluate_actor_on_other_agents_states():
    """Direct unit test of NACRLAgent.update() in isolation: agent A's
    actor must only ever be called with A's OWN states (batch["A"]'s s_i),
    never B's -- checked by spying on Actor.forward. Note this deliberately
    does NOT assert A's resulting parameters are identical with/without B's
    data present: the advantage signal legitimately incorporates B's reward
    through the attention-based coordination (r_i_new), so A's gradient
    magnitude is expected to shift slightly depending on B's presence --
    that's the intended cross-agent reward-coordination mechanism, not a
    bug. What must NOT happen (the original bug) is A's actor being
    evaluated on B's (state, action) pairs as if they were its own
    experience.
    """
    torch.manual_seed(42)
    B = 4
    a_data = [
        (torch.rand(OBS_DIM), 1, torch.tensor([0.5]), torch.rand(OBS_DIM))
        for _ in range(B)
    ]
    b_data = [
        (torch.rand(OBS_DIM), 2, torch.tensor([-0.5]), torch.rand(OBS_DIM))
        for _ in range(B)
    ]
    a_states_stacked = torch.stack([t[0] for t in a_data])
    a_actions = torch.tensor([t[1] for t in a_data]).long()

    agents, embedding, attention, embedding_optimizer = _build_agents(["A", "B"])
    batch = {"A": a_data, "B": b_data}

    seen_forward_inputs = []
    real_forward = agents["A"].actor.forward

    def spying_forward(x):
        seen_forward_inputs.append(x.clone())
        return real_forward(x)

    agents["A"].actor.forward = spying_forward
    try:
        agents["A"].update("A", batch, embedding, attention, embedding_optimizer, gamma=0.99)
    finally:
        agents["A"].actor.forward = real_forward

    assert len(seen_forward_inputs) == 1, (
        f"Actor.forward was called {len(seen_forward_inputs)} times -- "
        f"expected exactly 1 (the original bug called it once per agent_id "
        f"in the batch, so 2 calls here would mean B's data reached the actor)."
    )
    assert torch.allclose(seen_forward_inputs[0], a_states_stacked), (
        "Actor was evaluated on states that don't match agent A's own "
        "batch -- it's seeing B's states, not just its own."
    )


def test_embedding_receives_gradient_updates():
    """embedding previously had no optimizer anywhere -- its parameters
    were frozen at their random init for the agent's entire lifetime even
    though the critic loss backpropagates through it."""
    torch.manual_seed(0)
    agent_ids = ["A", "B"]
    agents, embedding, attention, embedding_optimizer = _build_agents(agent_ids)
    before = [p.clone() for p in embedding.parameters()]

    batch = {
        aid: [(torch.rand(OBS_DIM), i % ACTION_DIM, torch.rand(1), torch.rand(OBS_DIM)) for i in range(6)]
        for aid in agent_ids
    }
    agents["A"].update("A", batch, embedding, attention, embedding_optimizer, gamma=0.99)

    after = list(embedding.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed, "embedding's parameters did not change after update() -- it's still untrained."


def test_attention_compares_same_timestep_not_current_vs_next():
    """h_i (current-state query) must be compared against OTHER agents'
    CURRENT-state embeddings, not their next-state embeddings -- the
    original bug mixed timesteps in the reward-coordination attention call.
    Verified by spying on what tensors `embedding` gets called with for the
    "current state" attention pass; the h_others input must not be built
    from the other agent's s_next batch column.
    """
    torch.manual_seed(0)
    agent_ids = ["A", "B"]
    agents, embedding, attention, embedding_optimizer = _build_agents(agent_ids)

    a_state = torch.rand(OBS_DIM)
    a_state_next = torch.rand(OBS_DIM)
    b_state = torch.rand(OBS_DIM)
    b_state_next = torch.rand(OBS_DIM)
    assert not torch.allclose(b_state, b_state_next)

    batch = {
        "A": [(a_state, 0, torch.rand(1), a_state_next)],
        "B": [(b_state, 0, torch.rand(1), b_state_next)],
    }

    seen_inputs = []
    real_forward = embedding.forward

    def spying_forward(x):
        seen_inputs.append(x.clone())
        return real_forward(x)

    embedding.forward = spying_forward
    try:
        agents["A"].update("A", batch, embedding, attention, embedding_optimizer, gamma=0.99)
    finally:
        embedding.forward = real_forward

    # Call order in NACRLAgent.update(): h_i (s_i), h_i_next (s_i_next),
    # then _embed_others for h_others (use_next_state=False -> b_state),
    # then _embed_others for h_others_next (use_next_state=True -> b_state_next).
    assert torch.allclose(seen_inputs[0][0], a_state)
    assert torch.allclose(seen_inputs[1][0], a_state_next)
    assert torch.allclose(seen_inputs[2][0], b_state), (
        "The current-state attention pass (h_i vs h_others) embedded B's "
        "NEXT state instead of B's current state -- temporal mismatch bug."
    )
    assert torch.allclose(seen_inputs[3][0], b_state_next)


def test_runner_stops_episode_early_once_all_agents_done():
    """dones.get('__all__', False) is always False under the PettingZoo
    parallel API (no such key exists in terminations/truncations) -- an
    episode never broke early even once every agent was actually done."""
    agent_ids = ["A"]
    env = _FakeMultiAgentEnv(agent_ids, episode_len=100, done_after=3)
    agents, embedding, attention, embedding_optimizer = _build_agents(agent_ids)
    buffer = MultiAgentReplayBuffer(max_size=10000)  # never triggers update, isolates the done check

    real_step = env.step
    step_calls = {"count": 0}

    def counting_step(actions):
        step_calls["count"] += 1
        return real_step(actions)

    env.step = counting_step

    runner = NACRLRunner(
        env=env, agents=agents, buffer=buffer, embedding=embedding,
        attention=attention, embedding_optimizer=embedding_optimizer,
        gamma=0.99, T=100, buffer_size=10000,
    )
    runner.train(M=1)

    assert step_calls["count"] == 3, (
        f"Expected the episode to stop after 3 steps (done_after=3), but "
        f"env.step() was called {step_calls['count']} times -- the "
        f"all-agents-done check isn't stopping the episode early."
    )
