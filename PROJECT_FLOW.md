# PROJECT_FLOW

## High-Level Overview

This project trains a shared reinforcement-learning policy for multiple traffic-signal intersections across different city configurations. The parallel entry point starts a separate worker process per city, each worker builds its own SUMO-backed environment and trains the same DQN agent locally, and the main process periodically aggregates the resulting model weights with Federated Averaging. The pipeline is designed to make one shared policy learn a topology-agnostic control strategy while the environment wrapper supplies a fixed observation contract for every intersection.

## Execution Flow

1. The entry point in experiments/federated_training.py parses the command line and enters the parallel branch because --parallel is supplied.
2. The script calls resolve_city_configs_and_dims to discover every city config under environments, build each environment once just to extract its observation/action dimensions, and close it again.
3. It constructs a DQNAgent sized to the shared observation/action dimensions and creates a HoldoutEvaluator for the holdout city if that configuration exists.
4. It instantiates ParallelFederatedServer with the global model, the city configs, the shared dims, the communication-dropout settings, training parameters, and the evaluator.
5. ParallelFederatedServer starts one persistent worker process per city using multiprocessing with a spawn context.
6. Each worker process builds its own federated environment, wraps it with communication dropout simulation, and creates its own DQNAgent instance.
7. The main server sends the current global model state to every worker each round.
8. Each worker loads the incoming state into its local agent, trains locally for the requested number of episodes, and sends its updated state dict and sample count back through the result queue.
9. The server collects all updates, aggregates them with fed_avg, loads the aggregated weights into the global model, and optionally evaluates the resulting global model on the holdout city.
10. The server saves a checkpoint for the global model and the training history, then closes the worker processes.
11. The script saves the final global model and the history JSON.

## Module Dependency Tree

```text
experiments/federated_training.py
├── federated/parallel_server.py
│   ├── federated/aggregation.py
│   ├── federated/comm_dropout.py
│   ├── agents/dqn.py
│   │   ├── agents/networks.py
│   │   └── federated/comm_dropout.py (via wrapped env)
│   └── environments/federated_env.py
│       └── agents/dqn.py (via environment interaction contract)
└── federated/evaluator.py
    └── environments/federated_env.py
```

## Call Hierarchy

```text
main()
    ↓
resolve_city_configs_and_dims()
    ↓
build_federated_env()
    ↓
ParallelFederatedServer(...)
    ↓
_start worker processes_
    ↓
_client_worker()
    ↓
build_federated_env()
    ↓
DQNAgent(...)
    ↓
agent.train()
    ↓
env.reset()/env.step()
    ↓
agent.remember()/agent.optimize()
    ↓
out_queue.put(...)
    ↓
ParallelFederatedServer.run()
    ↓
fed_avg()
    ↓
HoldoutEvaluator.evaluate()
    ↓
model.act()/model.q_values()
```

## Used Classes

### ParallelFederatedServer
- Purpose: Coordinates the distributed federated training loop across multiple city worker processes.
- Responsibilities: Starts workers, dispatches model state, collects updates, aggregates them, evaluates the global model, and manages checkpoints.
- Important attributes: global_model, evaluator, checkpoint_dir, in_queues, out_queue, processes, names.
- Interaction: It is the orchestrator for the parallel path and is instantiated directly by the entry point.

### HoldoutEvaluator
- Purpose: Runs evaluation episodes on the holdout city using the current global model.
- Responsibilities: Builds or reuses an environment, steps through episodes, collects reward and waiting-time metrics, and reports action distribution and Q-gap diagnostics.
- Important attributes: env_builder, episodes, _env.
- Interaction: It is given to the parallel server so that each round can evaluate the aggregated global model.

### DQNAgent
- Purpose: Implements the shared Q-learning agent used by both the main process and each worker.
- Responsibilities: Builds the shared neural network, selects actions, stores transitions, optimizes the Q-network, saves/loads state, and trains over episodes in the environment.
- Important attributes: q, q_target, optimizer, replay, steps_done, target_update, reward_clip.
- Interaction: Each worker owns one DQNAgent and trains it against its local city environment; the main process uses the same agent class for the global model.

### ReplayBuffer
- Purpose: Stores transitions collected by the agent.
- Responsibilities: Adds experience tuples and samples batches for optimization.
- Important attributes: buffer.
- Interaction: Used by DQNAgent for learning from the local environment.

### CommDropoutWrapper
- Purpose: Simulates unreliable communication between intersections during training.
- Responsibilities: Corrupts neighbor-mask and neighbor features in the observation stream according to configurable dropout probabilities.
- Important attributes: env, p_link, p_isolate, p_hop_cutoff, max_hop.
- Interaction: Wrapped around the federated environment in both training and evaluation setup.

### MultiAgentFederatedWrapper
- Purpose: Exposes a city-agnostic, multi-agent observation/action interface for a SUMO city.
- Responsibilities: Builds observations for every traffic signal, exposes reset/step semantics, and applies action clipping to the underlying SUMO environment.
- Important attributes: env, lane_extractor, sorter, encoder, neighbor_graph, neighbor_summary, action_inspector, k_max, own_dim, neighbor_dim, max_action_dim.
- Interaction: This is the environment object used by the DQNAgent during local training.

### ActionMaskPadder
- Purpose: Pads each city's action masks to a shared global width so all cities can use a common action head.
- Responsibilities: Extends the observation action mask to match the largest action space seen across the federation.
- Important attributes: env, target_action_dim, max_action_dim.
- Interaction: Wrapped around each city's environment before training so the shared model has a consistent output dimension.

### ActionSpaceInspector
- Purpose: Discovers each intersection's valid action count directly from the environment instead of using handwritten mappings.
- Responsibilities: Computes action masks and clips out-of-range actions.
- Important attributes: ts_ids, action_counts, max_action_dim.
- Interaction: Used by MultiAgentFederatedWrapper to build per-intersection action masks.

### NeighborGraphBuilder
- Purpose: Builds a graph of neighboring traffic signals for each city.
- Responsibilities: Uses the SUMO network topology to discover K-hop neighbors and return deterministic neighbor lists.
- Important attributes: graph, ts_ids, max_hops.
- Interaction: Used by MultiAgentFederatedWrapper to populate the neighbor slots in each observation.

### NeighborSummaryExtractor
- Purpose: Converts each neighbor's lane-level state into a compact fixed-size feature vector.
- Responsibilities: Summarizes queue, waiting time, and phase information for each neighbor.
- Important attributes: lane_extractor, max_queue, max_wait.
- Interaction: Used by MultiAgentFederatedWrapper to build the neighbor feature tensor.

### TopKEncoder, LaneNormalizer, LaneSorter, LaneExtractor, SumoLaneExtractor
- Purpose: Provide the fixed-size own-observation representation for each traffic signal.
- Responsibilities: Extract lane features, normalize them, sort lanes, and encode them into a consistent vector.
- Important attributes: vary by class, but each contributes to the shared observation vector used by the agent.
- Interaction: They are composed by MultiAgentFederatedWrapper to build each intersection's own observation.

## Data Flow

1. The entry point reads city configuration files and builds federated environments to discover their dimensions.
2. The environment wrapper extracts per-intersection lane data, neighbor relationships, action-space information, and constructs a fixed observation dictionary for each traffic signal.
3. The DQNAgent consumes those observations to select actions and stores experience tuples in the replay buffer.
4. The environment steps forward and returns rewards and done flags for each traffic signal.
5. The agent optimizes from the replay buffer and returns its updated state dict to the parallel server.
6. The server aggregates the per-city state dicts into one global state dict and loads it back into the global model.
7. Evaluation reuses the same environment contract to measure reward, waiting time, and action distribution metrics for the holdout city.

## Important Design Decisions

- The parallel path is separated from the single-process path so each city can keep a warm-running SUMO environment and replay buffer across rounds without paying repeated startup costs.
- A single shared DQNAgent is used across all intersections and cities, which is why the environment wrapper exposes a fixed-size observation space and action masks instead of per-topology custom code.
- Communication dropout is modeled in the environment wrapper rather than in the learner, so the learning process naturally experiences noisy neighbor information without changing the agent interface.
- The federated server aggregates by sample count rather than by equal weighting, which reflects the amount of local training each city performed.
- The action space is made uniform across cities through action-mask padding rather than by forcing each city to share the same number of actions.

## Files Traversed

- experiments/federated_training.py
- federated/parallel_server.py
- federated/aggregation.py
- federated/comm_dropout.py
- federated/evaluator.py
- agents/dqn.py
- agents/networks.py
- environments/federated_env.py

## Files Intentionally Ignored

The following were not analyzed because they are not part of the reachable execution path for the requested --parallel invocation:

- federated/server.py
- federated/client.py
- experiments/local_training.py
- experiments/centralized.py
- experiments/evaluate.py
- other experiment scripts and result files

## Dynamic Behavior

- The entry point chooses between the parallel and non-parallel branches based on the parsed --parallel flag.
- The environment construction path can switch between a real SUMO-backed environment and a mock environment based on the config mode and the presence of net/route files.
- The worker processes are launched with multiprocessing spawn, which affects how state and environment objects are initialized.
- The environment wrapper uses runtime inspection of traffic-signal properties and SUMO APIs rather than a statically known interface, so some behavior depends on the loaded city configuration and the installed SUMO tooling.
