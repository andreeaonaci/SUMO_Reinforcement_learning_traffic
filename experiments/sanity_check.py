import importlib
import inspect
import sys


REQUIRED_IMPORTS = {
    "centralized": [
        "agents.dqn",
        "environments.common"
    ],
    "local_training": [
        "agents.dqn",
        "environments.common"
    ],
    "evaluate": [
        "agents.dqn",
        "environments.common"
    ],
}


REQUIRED_FUNCTIONS = {
    "agents.dqn": ["DQNAgent"],
    "environments.common": ["build_env_from_config"],
}


def check_imports(module_name):
    try:
        module = importlib.import_module(f"experiments.{module_name}")
        print(f"[OK] {module_name} imports successfully")
        return module
    except Exception as e:
        print(f"[FAIL] {module_name} import error:")
        print(e)
        return None


def check_required_symbols():
    print("\n=== Checking core modules ===")

    for module_name, symbols in REQUIRED_FUNCTIONS.items():
        try:
            mod = importlib.import_module(module_name)
            for sym in symbols:
                if hasattr(mod, sym):
                    print(f"[OK] {module_name}.{sym}")
                else:
                    print(f"[FAIL] Missing {module_name}.{sym}")
        except Exception as e:
            print(f"[FAIL] Cannot import {module_name}: {e}")


def check_env_consistency():
    print("\n=== Checking environment consistency ===")

    try:
        from environments.common import build_env_from_config
        from configs.loader import load_cfg

        cfg = load_cfg(os.path.join(os.path.dirname(__file__), "..", "environments", "city_1", "config.yaml"))

        envs = ["city_1", "city_2", "city_3", "city_4"]

        shapes = []

        for city in envs:
            env = build_env_from_config(cfg, city)
            obs, _ = env.reset()

            shapes.append((city, len(obs) if hasattr(obs, "__len__") else None))

        print("Observation shapes per city:")
        for c, s in shapes:
            print(f"  {c}: {s}")

        if len(set([s for _, s in shapes])) == 1:
            print("[OK] All envs have consistent observation shape")
        else:
            print("[FAIL] Inconsistent observation shapes")

    except Exception as e:
        print(f"[FAIL] Env consistency check failed: {e}")


def check_dqn_instantiation():
    print("\n=== Checking DQN instantiation ===")

    try:
        from agents.dqn import DQNAgent
        from configs.loader import load_cfg

        cfg = load_cfg(os.path.join(os.path.dirname(__file__), "..", "environments", "city_1", "config.yaml"))

        agent = DQNAgent(cfg["state_dim"], cfg["action_dim"])
        print("[OK] DQNAgent initializes correctly")

    except Exception as e:
        print(f"[FAIL] DQN init failed: {e}")


def check_eval_mode():
    print("\n=== Checking evaluation mode behavior ===")

    try:
        from agents.dqn import DQNAgent

        # crude check: act() should accept explore flag
        sig = inspect.signature(DQNAgent.act)

        if "explore" in sig.parameters:
            print("[OK] DQNAgent.act supports explore flag")
        else:
            print("[WARN] DQNAgent.act missing explore flag (evaluation may be wrong)")

    except Exception as e:
        print(f"[FAIL] Eval check failed: {e}")


def main():
    print("\n===== FEDERATED RL SANITY CHECK =====\n")

    for script in ["centralized", "local_training", "evaluate"]:
        check_imports(script)

    check_required_symbols()
    check_env_consistency()
    check_dqn_instantiation()
    check_eval_mode()

    print("\n===== DONE =====\n")


if __name__ == "__main__":
    sys.path.append(".")
    main()