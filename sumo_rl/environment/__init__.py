"""SUMO Environment for Traffic Signal Control."""

# Note: gymnasium.envs.registration is not available in all gymnasium versions.
# Direct instantiation of SumoEnvironment is used instead in experiments.
# Commented out for compatibility:
# from gymnasium.envs.registration import register
# register(
#     id="sumo-rl-v0",
#     entry_point="sumo_rl.environment.env:SumoEnvironment",
#     kwargs={"single_agent": True},
# )
