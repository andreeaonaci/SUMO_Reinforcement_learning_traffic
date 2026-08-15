import torch

class NACRLRunner:
    def __init__(self, env, agents, buffer, embedding, attention, embedding_optimizer, gamma, T, buffer_size):
        """Initialize the NACRL training runner.

        Args:
            env: The environment instance.
            agents (dict): Dictionary of agent_id to NACRLAgent instances.
            buffer (MultiAgentReplayBuffer): The replay buffer.
            embedding (StateEmbedding): The state embedding module.
            attention (AgentAttention): The attention mechanism.
            embedding_optimizer (torch.optim.Optimizer): Optimizer for the
                shared embedding network -- see NACRLAgent.update() for why
                this is required (the embedding previously never received a
                gradient step at all).
            gamma (float): Discount factor.
            T (int): Maximum number of timesteps per episode.
            buffer_size (int): Size of the replay buffer.
        """
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.embedding = embedding
        self.attention = attention
        self.embedding_optimizer = embedding_optimizer
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
                state_tensors = {}

                # --- action selection ---
                for agent_id, agent in self.agents.items():
                    s_i = torch.tensor(states[agent_id], dtype=torch.float32)
                    state_tensors[agent_id] = s_i
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
                # Stored as tensors, not raw numpy arrays / python scalars --
                # NACRLAgent.update() does torch.stack()/torch.tensor() on
                # whatever comes out of the buffer, which crashes on
                # non-tensor elements (the original bug: buffer.add() was
                # passed states[agent_id]/rewards[agent_id] straight from
                # the env, never converted).
                for agent_id in self.agents:
                    self.buffer.add(
                        agent_id,
                        state_tensors[agent_id],
                        int(actions[agent_id]),
                        torch.tensor([float(rewards[agent_id])], dtype=torch.float32),
                        torch.tensor(next_states[agent_id], dtype=torch.float32),
                    )

                # --- update ---
                if self.buffer.is_full():
                    batch = self.buffer.sample()
                    for agent_id, agent in self.agents.items():
                        agent.update(
                            agent_id,
                            batch,
                            self.embedding,
                            self.attention,
                            self.embedding_optimizer,
                            self.gamma,
                        )
                    self.buffer.clear()

                states = next_states

                # PettingZoo's parallel API has no "__all__" key -- dones is
                # a plain {agent_id: bool} dict, so dones.get("__all__",
                # False) was always False and an episode never terminated
                # early even once every agent was actually done.
                if dones and all(dones.values()):
                    break
