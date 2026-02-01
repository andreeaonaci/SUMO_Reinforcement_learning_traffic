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
    env = parallel_env(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/nacrl/2way-single-intersection",
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
    buffer = MultiAgentReplayBuffer(max_size=10000)

    # -----------------------------
    # NACRL runner
    # -----------------------------
    runner = NACRLRunner(
        env=env,
        agents=agents,
        buffer=buffer,
        embedding=embedding,
        attention=attention,
        gamma=0.99,
        T=200,
        buffer_size=10000,
    )

    # -----------------------------
    # Train
    # -----------------------------
    runner.train(M=100)


if __name__ == "__main__":
    main()
