"""Zero-shot holdout cost per training round, benchmark-exact roster (fig:zeroshot).

Regenerates paper/figures/fig_zeroshot.pdf from the raw federated_history.json of
the six sec-100 runs per readout (environments_rescofull, 3 s yellow) plus the
rule-based references measured on the same holdout (fidings sec 109).

Cost = -reward, on one log axis: the readouts are four orders of magnitude apart,
so a linear axis would flatten one of them to a line on the floor.

    python paper/figures/plot_zeroshot.py [--results <main-checkout results dir>]
"""
import argparse
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_RESULTS = ("/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/"
                   "SUMO_Reinforcement_learning_traffic/results")

# sec 100 runs; seed labels verified per run from each training.log args dict
RUNS = {
    "phase": {3: "1483406", 7: "1483410", 11: "1483409", 17: "1491576", 21: "1491628", 25: "1491714"},
    "indexed": {3: "1499810", 7: "1499847", 11: "1499949", 17: "1507663", 21: "1507788", 25: "1507922"},
}
# Rule-based references on the same 3 s holdout, deterministic over 5 episodes (sec 109)
MAX_PRESSURE = 0.380
FIXED_TIME = 2.730

# Identity colours shared with the TikZ diagrams (\definecolor acc / fail in main.tex).
# Validated with the dataviz skill's checks: CVD dE 20.5, normal-vision dE 20.8.
PHASE_C = "#A85A0A"
INDEX_C = "#7A2E6E"
INK = "#222222"
MUTED = "#6B6B6B"
GRID = "#E6E6E6"


def load(results, pid):
    for d in os.listdir(results):
        if d.endswith("_" + pid):
            f = os.path.join(results, d, "federated_history.json")
            if os.path.exists(f):
                with open(f) as fh:
                    return json.load(fh)
    raise FileNotFoundError("no run dir ending in _%s under %s" % (pid, results))


def costs(results, runs):
    per_seed = []
    for _seed, pid in sorted(runs.items()):
        h = load(results, pid)
        per_seed.append([-r for r in h["eval_reward"]])
    n_rounds = {len(c) for c in per_seed}
    assert len(n_rounds) == 1, "runs disagree on round count: %s" % n_rounds
    by_round = list(zip(*per_seed))
    return ([st.fmean(r) for r in by_round], [min(r) for r in by_round], [max(r) for r in by_round])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fig_zeroshot.pdf"))
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK,
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    for key, colour, marker, label in (("indexed", INDEX_C, "s", "A. Indexed"),
                                       ("phase", PHASE_C, "o", "B. Phase-relational")):
        mean, lo, hi = costs(args.results, RUNS[key])
        x = list(range(1, len(mean) + 1))
        ax.fill_between(x, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.plot(x, mean, color=colour, linewidth=1.6, marker=marker, markersize=4.2,
                markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)

    for y, name, ls in ((FIXED_TIME, "Fixed time", (0, (1.2, 1.6))), (MAX_PRESSURE, "Max pressure", (0, (4, 2)))):
        ax.axhline(y, color=MUTED, linewidth=0.9, linestyle=ls, zorder=2)
        ax.text(5.08, y, name, color=INK, fontsize=6.5, va="center", ha="left")

    ax.set_yscale("log")
    ax.set_ylim(0.05, 3e4)
    ax.set_xlim(0.8, 5.0)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Training round")
    ax.set_ylabel(r"Holdout cost ($-$reward, log)")
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.60), frameon=False, handlelength=2.2)

    fig.subplots_adjust(left=0.15, right=0.80, bottom=0.18, top=0.97)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
