"""SUMO Environment for Traffic Signal Control."""

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import time

try:
    import torch
except ImportError:
    torch = None


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import sumolib
import traci

try:
    from pettingzoo.utils.env import AECEnv
    from pettingzoo.utils import wrappers
    from pettingzoo.utils import seeding
except ImportError:
    class AECEnv:
        def __init__(self):
            self.agents = []
            self.possible_agents = []
            self.agent_selection = None
            self.rewards = {}
            self.terminations = {}
            self.truncations = {}
            self.infos = {}
            self._cumulative_rewards = {}

        def _was_dead_step(self, action):
            return None

        def _clear_rewards(self):
            pass

        def _accumulate_rewards(self):
            pass

    class _PassthroughWrapper:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)

    class _WrappersModule:
        AssertOutOfBoundsWrapper = _PassthroughWrapper
        OrderEnforcingWrapper = _PassthroughWrapper

    wrappers = _WrappersModule()

    class _SeedingModule:
        @staticmethod
        def np_random(seed=None):
            import random

            rng = random.Random(seed)
            return rng, seed

    seeding = _SeedingModule()

# Minimal stubs for gymnasium.utils compatibility
class EzPickle:
    def __init__(self, *args, **kwargs):
        pass

try:
    # pettingzoo 1.25+
    from pettingzoo.utils import AgentSelector
except ImportError:
    # pettingzoo 1.24 or earlier
    try:
        from pettingzoo.utils import agent_selector as AgentSelector
    except ImportError:
        # fallback: simple round-robin selector
        class AgentSelector:
            def __init__(self, agents):
                self.agents = list(agents)
                self._idx = 0
            def reset(self):
                self._idx = 0
                return self.agents[self._idx] if self.agents else None
            def next(self):
                if not self.agents:
                    return None
                self._idx = (self._idx + 1) % len(self.agents)
                return self.agents[self._idx]
            def is_last(self):
                return self._idx == len(self.agents) - 1

# Import or fallback for parallel_wrapper_fn
try:
    from pettingzoo.utils.conversions import parallel_wrapper_fn
except ImportError:
    # Simple fallback: identity wrapper
    def parallel_wrapper_fn(fn):
        return fn

from .observations import DefaultObservationFunction, ObservationFunction
from .traffic_signal import TrafficSignal


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEnvironment(gym.Env):
    """SUMO Environment for Traffic Signal Control.

    Class that implements a gym.Env interface for traffic signal control using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The duration in seconds of the simulation. Default: 20000
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        enforce_max_green (bool): If true, it enforces the max green time and selects the next green phase when the max green time is reached. Default: False
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict/List): String with the name of the reward function used by the agents, a reward function, dictionary with reward functions assigned to individual traffic lights by their keys, or a List of reward functions.
        reward_weights (List[float]/np.ndarray): Weights for linearly combining the reward functions, in case reward_fn is a list. If it is None, the reward returned will be a np.ndarray. Default: None
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        ts_ids (Optional[List[str]]): List of traffic light IDs to be controlled by SUMO-RL. If None, all traffic lights in the simulation are controlled.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        enforce_max_green: bool = False,
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict, List] = "diff-waiting-time",
        reward_weights: Optional[List[float]] = None,
        observation_class: type[ObservationFunction] = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        ts_ids: Optional[List[str]] = None,
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."
        assert max_green > min_green, "Max green time must be greater than min green time."

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.sumo_seed = sumo_seed
        if isinstance(self.sumo_seed, int):
            self.sumo_seed &= 0x7FFFFFFF  # Ensure 32 bit non-negative seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        init_label = "init_connection" + self.label
        if LIBSUMO:
            try:
                self._close_traci(label=None)
            except Exception:
                pass
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net, "-r", self._route])
            conn = traci
        else:
            try:
                self._close_traci(label=init_label)
            except Exception:
                pass
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net, "-r", self._route], label=init_label)
            conn = traci.getConnection(init_label)

        if ts_ids is None:
            self.ts_ids = list(conn.trafficlight.getIDList())
        else:
            self.ts_ids = ts_ids
        self.observation_class = observation_class
        self.traffic_signals = {}

        try:
            conn.close()
        except Exception:
            pass

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}

    def _build_traffic_signals(self, conn):
        if not isinstance(self.reward_fn, dict):
            self.reward_fn = {ts: self.reward_fn for ts in self.ts_ids}

        self.traffic_signals = {
            ts: TrafficSignal(
                self,
                ts,
                self.delta_time,
                self.yellow_time,
                self.min_green,
                self.max_green,
                self.enforce_max_green,
                self.begin_time,
                self.reward_fn[ts],
                self.reward_weights,
                conn,
            )
            for ts in self.ts_ids
        }

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        def _start_traci_connection():
            attempts = 5
            for attempt in range(1, attempts + 1):
                try:
                    # asigura-te ca labelul nu e deja activ inainte de a porni
                    if not LIBSUMO:
                        try:
                            existing = traci.getConnection(self.label)
                            existing.close()
                        except Exception:
                            pass

                    if LIBSUMO:
                        traci.start(sumo_cmd)
                    else:
                        traci.start(sumo_cmd, label=self.label)
                    return
                except Exception as e:
                    print(f"traci.start failed (attempt {attempt}/{attempts}): {e}")
                    self._close_traci(label=None if LIBSUMO else self.label)
                    if attempt < attempts:
                        time.sleep(2)
                    else:
                        raise

        if self.sumo is not None:
            self.close()

        _start_traci_connection()
        if LIBSUMO:
            self.sumo = traci
        else:
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            if "DEFAULT_VIEW" not in dir(traci.gui):  # traci.gui.DEFAULT_VIEW is not defined in libsumo
                traci.gui.DEFAULT_VIEW = "View #0"
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)

        if self.episode != 0:
            try:
                self.close()
            except Exception:
                pass
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed & 0x7FFFFFFF  # Ensure 32 bit non-negative seed
        self._start_simulation()

        self._build_traffic_signals(self.sumo)

        self.vehicles = dict()
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()
    
    def step(self, action):
        if self.single_agent:
            if torch is not None and isinstance(action, torch.Tensor):
                if action.numel() != 1:
                    raise ValueError("Action must be a scalar.")
                action = int(action.item())
            else:
                action = int(action)

            if not self.action_space.contains(action):
                raise Exception(
                    f"Action must be in Discrete({self.action_space.n}). It is currently {action}"
                )

            if not self.fixed_ts:
                self._apply_actions(action)
        else:
            if not isinstance(action, dict):
                raise ValueError("Action must be a dict for multi-agent SUMO environment.")
            for ts, act in action.items():
                if ts not in self.ts_ids:
                    raise KeyError(f"Unknown traffic signal id: {ts}")
            if not self.fixed_ts:
                for ts, act in action.items():
                    if self.traffic_signals[ts].time_to_act:
                        self._apply_actions({ts: act})

        if not self.fixed_ts:
            self._run_steps()
        else:
            for _ in range(self.delta_time):
                self._sumo_step()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        info = self._compute_info()
        dones = self._compute_dones()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], dones["__all__"], info

        return observations, rewards, dones, info


    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True
            if self.sim_step >= self.sim_max_time:
                break
                    
    def _apply_actions(self, actions):
        """Set the next green phase for the traffic signals.

        Args:
            actions:
                - single-agent: int or torch scalar
                - multi-agent: dict {ts_id: int or torch scalar}
        """

        def _to_int(a):
            if torch is not None and isinstance(a, torch.Tensor):
                if a.numel() != 1:
                    raise ValueError("Action must be a scalar.")
                return int(a.item())
            return int(a)

        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                action = _to_int(actions)
                self.traffic_signals[self.ts_ids[0]].set_next_phase(action)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    action = _to_int(action)
                    self.traffic_signals[ts].set_next_phase(action)


    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info

    def _compute_observations(self):
        self.observations.update(
            {
                ts: self.traffic_signals[ts].compute_observation()
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }
        )
        return {
            ts: self.observations[ts].copy()
            for ts in self.observations.keys()
            if self.traffic_signals[ts].time_to_act or self.fixed_ts
        }

    def _compute_rewards(self):
        self.rewards.update(
            {
                ts: self.traffic_signals[ts].compute_reward()
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act or self.fixed_ts}

    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].action_space

    @property
    def reward_space(self):
        """Return the reward space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].reward_space

    @property
    def reward_dim(self):
        """Return the reward dimension of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].reward_dim

    def observation_spaces(self, ts_id: str):
        """Return the observation space of a traffic signal."""
        return self.traffic_signals[ts_id].observation_space

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal."""
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()
        self.num_arrived_vehicles += self.sumo.simulation.getArrivedNumber()
        self.num_departed_vehicles += self.sumo.simulation.getDepartedNumber()
        self.num_teleported_vehicles += self.sumo.simulation.getEndingTeleportNumber()

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        num_backlogged_vehicles = len(self.sumo.simulation.getPendingVehicles())
        return {
            "system_total_running": len(vehicles),
            "system_total_backlogged": num_backlogged_vehicles,
            "system_total_stopped": sum(
                int(speed < 0.1) for speed in speeds
            ),  # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_arrived": self.num_arrived_vehicles,
            "system_total_departed": self.num_departed_vehicles,
            "system_total_teleported": self.num_teleported_vehicles,
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids
        ]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def _close_traci(self, label: Optional[str] = None):
        if LIBSUMO:
            try:
                traci.close()
            except Exception:
                pass
            return

        if label is None:
            try:
                traci.close()
            except Exception:
                pass
            return

        try:
            conn = traci.getConnection(label)
        except Exception:
            # nu exista conexiune cu acest label, nimic de facut
            return

        try:
            conn.close()
        except Exception:
            pass

        try:
            if hasattr(conn, '_process') and conn._process is not None:
                conn._process.poll()
                if conn._process.returncode is None:
                    conn._process.kill()
                    conn._process.wait(timeout=5)
        except Exception:
            pass

    def close(self):
        # Idempotent: capture and clear self.sumo FIRST so that any __del__
        # triggered by the cyclic GC on a previously-closed SumoEnvironment
        # (the cycle is SumoEnv → traffic_signals → TrafficSignal.env →
        # SumoEnv) can never reach traci.close() and kill an *active* libsumo
        # simulation.  The cycle is then broken by clearing traffic_signals,
        # which allows CPython's reference counter to free the objects
        # immediately rather than waiting for a non-deterministic GC pass.
        sumo = self.sumo
        self.sumo = None        # mark closed before any I/O so re-entry is safe
        if sumo is None:
            return              # already closed – do NOT call traci.close() again

        # Break the reference cycle so CPython refcounts can free these objects
        # without waiting for the cyclic GC.
        self.traffic_signals = {}

        try:
            if LIBSUMO:
                traci.close()
            else:
                self._close_traci(label=self.label)
        except Exception:
            pass

        if self.disp is not None:
            try:
                self.disp.stop()
            except Exception:
                pass
            self.disp = None

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self.close()

    def render(self):
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame
        elif self.render_mode == "rgb_array":
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            import pandas as pd

            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        """Encode the state of the traffic signal into a hashable object."""
        phase = int(np.where(state[: self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1 :]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density * 10), 9)

    def get_agent_states(self):
        """Returns the current states of all agents in the environment.

        Returns:
            dict: A dictionary where keys are agent IDs and values are their respective states (observations).
        """
        return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids}

    def get_agent_rewards(self):
        """Returns the current rewards of all agents in the environment.

        Returns:
            dict: A dictionary where keys are agent IDs and values are their respective rewards.
        """
        return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids}


class SumoEnvironmentPZ(AECEnv, EzPickle):
    """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo.

    For more information, see https://pettingzoo.farama.org/api/aec/.

    The arguments are the same as for :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        """Initialize the environment."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironment(**self._kwargs)
        self.render_mode = self.env.render_mode

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        """Set the seed for the environment."""
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.compute_info()

    def compute_info(self):
        """Compute the info for the current step."""
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a) or k.startswith("system"):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        """Return the observation space for the agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for the agent."""
        return self.action_spaces[agent]

    def observe(self, agent):
        """Return the observation for the agent."""
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.env.close()

    def render(self):
        """Render the environment."""
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file."""
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        """Step the environment."""
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({})."
                "It is currently {}".format(agent, self.action_spaces[agent].n, action)
            )

        if not self.env.fixed_ts:
            self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            if not self.env.fixed_ts:
                self.env._run_steps()
            else:
                for _ in range(self.env.delta_time):
                    self.env._sumo_step()

            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
