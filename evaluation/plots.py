import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Sequence


def plot_curve(values: Sequence[float], out_path: str, title: str = "Curve"):
    plt.figure()
    plt.plot(values)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
