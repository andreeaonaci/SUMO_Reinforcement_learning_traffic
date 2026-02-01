import torch

class NACRLRunner:
    def __init__(self, env, agents, buffer, embedding, attention, gamma, T, buffer_size):
        """Initialize the NACRL training runner.

        Args:
            env: The environment instance.
            agents (dict): Dictionary of agent_id to NACRLAgent instances.
            buffer (MultiAgentReplayBuffer): The replay buffer.
            embedding (StateEmbedding): The state embedding module.
            attention (AgentAttention): The attention mechanism.
            gamma (float): Discount factor.
            T (int): Maximum number of timesteps per episode.
            buffer_size (int): Size of the replay buffer.
        """
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.embedding = embedding
        self.attention = attention
        self.gamma = gamma
        self.T = T
        self.buffer_size = buffer_size

    def train(self, M):
        for episode in range(M):

            # --- reset ---
            states, infos = self.env.reset()
            self.buffer.clear()

            for t in range(self.T):
                actions = {}

                # --- action selection ---
                for agent_id, agent in self.agents.items():
                    s_i = torch.tensor(states[agent_id], dtype=torch.float32)
                    actions[agent_id] = agent.act(s_i)

                # --- env step ---
                step_out = self.env.step(actions)

                # PettingZoo-style return handling
                if len(step_out) == 5:
                    next_states, rewards, terminations, truncations, infos = step_out
                    dones = {
                        a: terminations[a] or truncations[a]
                        for a in terminations
                    }
                else:
                    next_states, rewards, dones, infos = step_out

                # --- store transitions ---
                for agent_id in self.agents:
                    self.buffer.add(
                        agent_id,
                        states[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_states[agent_id],
                    )

                # --- update ---
                if self.buffer.is_full():
                    batch = self.buffer.sample()
                    for agent in self.agents.values():
                        agent.update(
                            batch,
                            self.embedding,
                            self.attention,
                            self.gamma,
                        )
                    self.buffer.clear()

                states = next_states

                if dones.get("__all__", False):
                    break
