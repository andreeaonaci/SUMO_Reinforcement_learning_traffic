from typing import List, Dict
import pandas as pd


def summarize_rewards(rewards: List[float]) -> Dict[str, float]:
    s = pd.Series(rewards)
    return {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}


def save_csv(records: List[Dict], out_path: str):
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
