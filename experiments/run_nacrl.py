from sumo_rl.nacrl.nacrl_runner import NACRLRunner
import torch

from sumo_rl.environment.env import parallel_env
from sumo_rl.nacrl.nacrl_agent import NACRLAgent
from sumo_rl.nacrl.embedding import StateEmbedding
from sumo_rl.nacrl.attention import AgentAttention
from sumo_rl.nacrl.replay_buffer import MultiAgentReplayBuffer
from sumo_rl.nacrl.actor import Actor
from sumo_rl.nacrl.critic import Critic


def main():
    # -----------------------------
    # SUMO configuration (PARALLEL ENV)
    # -----------------------------
    # 2x2grid, not 2way-single-intersection: NACRL's whole point is
    # attention-based reward/embedding coordination ACROSS agents. A net
    # with a single traffic light has only one agent, so that coordination
    # path (NACRLAgent.update()'s `if h_others is not None:` branch) never
    # ran under the old single-intersection config -- the experiment never
    # actually exercised the mechanism it was built to test.
    env = parallel_env(
        net_file="sumo_rl/nets/2x2grid/2x2.net.xml",
        route_file="sumo_rl/nets/2x2grid/2x2.rou.xml",
        out_csv_name="outputs/nacrl/2x2grid",
        use_gui=False,
        num_seconds=1000,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_green=50,
        reward_fn="diff-waiting-time",
        sumo_seed=42,
        single_agent=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Infer dimensions
    # -----------------------------
    env.reset()
    sample_agent = env.agents[0]

    state_dim = env.observation_space(sample_agent).shape[0]
    action_dim = env.action_space(sample_agent).n
    n_agents = len(env.agents)

    # -----------------------------
    # Shared NACRL modules
    # -----------------------------
    embedding = StateEmbedding(
        input_dim=state_dim,
        hidden_dim=64,
        output_dim=32
    ).to(device)

    attention = AgentAttention(
        embed_dim=32
    ).to(device)

    # embedding has learnable parameters (it's an MLP) but previously had no
    # optimizer anywhere in this module -- its gradients were computed
    # (critic_loss.backward() flows into it through h_i_new/h_i_next_new)
    # but never applied, so it stayed a random, untrained projection for the
    # agent's entire lifetime. See NACRLAgent.update().
    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-3)

    # -----------------------------
    # Create agents
    # -----------------------------
    agents = {}

    for agent_id in env.agents:
        actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64
        ).to(device)

        critic = Critic(
            embedding_dim=32
        ).to(device)

        agents[agent_id] = NACRLAgent(
            actor=actor,
            critic=critic,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        )

    # -----------------------------
    # Replay buffer
    # -----------------------------
    # max_size=10000 previously meant the buffer could never fill within one
    # episode (buffer.clear() runs every episode reset, and one step only
    # adds n_agents entries -- with T=200 and 1 agent that's a maximum of
    # 200 entries/episode, so buffer.is_full() was never true and
    # agent.update() was never called: this "trained" for 100 episodes
    # without a single gradient step). Sized here relative to n_agents and T
    # so the buffer fills (and an update happens) multiple times per
    # episode, matching the on-policy-batch design the update() code
    # actually implements.
    T = 200
    buffer_size = min(64, max(8, n_agents * 8))
    buffer = MultiAgentReplayBuffer(max_size=buffer_size)

    # -----------------------------
    # NACRL runner
    # -----------------------------
    runner = NACRLRunner(
        env=env,
        agents=agents,
        buffer=buffer,
        embedding=embedding,
        attention=attention,
        embedding_optimizer=embedding_optimizer,
        gamma=0.99,
        T=T,
        buffer_size=buffer_size,
    )

    # -----------------------------
    # Train
    # -----------------------------
    runner.train(M=100)


if __name__ == "__main__":
    main()
