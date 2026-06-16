from typing import Dict
import os


def load_cfg(path: str) -> Dict:
    """Load a simple key: value config. Prefer PyYAML if available, else use a fallback parser.

    This keeps the repo runnable without installing PyYAML.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        # fallback: parse simple key: value lines
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # try to interpret ints
                    if v.isdigit():
                        cfg[k] = int(v)
                    else:
                        try:
                            cfg[k] = float(v)
                        except Exception:
                            cfg[k] = v
        return cfg
