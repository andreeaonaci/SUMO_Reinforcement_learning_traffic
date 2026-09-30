"""Zero-shot vs one-episode fine-tune on the unseen real networks (fidings sec 113).

Pairs each seed's fine-tuned result (results/unseen_ft/<net>/) with the same seed's
zero-shot result (results/unseen/<net>/), both from eval_ours.py (5 episodes).
    python analyse/unseen_ft_table.py
"""
import glob
import json
import os
import statistics as st

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PID = {3: "1483406", 7: "1483410", 11: "1483409", 17: "1491576", 21: "1491628", 25: "1491714"}
NETS = ("cologne1", "cologne8", "ingolstadt21")
KEYS = ("delay", "completion", "ti_wait")


def ep_mean(path):
    eps = json.load(open(path))["episodes"]
    return {k: st.fmean(e[k] for e in eps) for k in KEYS}


def one(pattern):
    f = glob.glob(pattern)
    return ep_mean(f[0]) if f else None


out = {}
for net in NETS:
    zs, ft = {}, {}
    for seed, pid in PID.items():
        z = one(os.path.join(W, "results/unseen", net, "ours_*_phase_run_*_%s.json" % pid))
        f = one(os.path.join(W, "results/unseen_ft", net, "ours_*_phase_ckpt_s%d.json" % seed))
        if z and f:
            zs[seed], ft[seed] = z, f
    mp = one(os.path.join(W, "results/unseen", net, "ours_*_max_pressure_*.json"))
    fx = one(os.path.join(W, "results/unseen", net, "ours_*_fixed_time_*.json"))
    seeds = sorted(zs)
    row = {"n": len(seeds), "max_pressure": mp, "fixed_time": fx}
    for k in KEYS:
        a = [zs[s][k] for s in seeds]
        b = [ft[s][k] for s in seeds]
        se = (st.pstdev(a) ** 2 / len(a) + st.pstdev(b) ** 2 / len(b)) ** .5
        better = sum(1 for s in seeds if (ft[s][k] < zs[s][k]) == (k != "completion") and ft[s][k] != zs[s][k])
        row[k] = dict(zero_shot=st.fmean(a), finetuned=st.fmean(b), z=abs(st.fmean(a) - st.fmean(b)) / se if se else 0,
                      seeds_improved=better, per_seed_zs=a, per_seed_ft=b)
    out[net] = row
    print("== %s (n=%d seeds)" % (net, len(seeds)))
    for k, label in (("delay", "delay s"), ("completion", "completion"), ("ti_wait", "wait s")):
        r = row[k]
        scale = 100 if k == "completion" else 1
        print("   %-11s zero-shot %7.1f -> fine-tuned %7.1f   |d|/SE %5.2f   improved %d/%d"
              % (label, scale * r["zero_shot"], scale * r["finetuned"], r["z"], r["seeds_improved"], len(seeds)))
    print("   max pressure: delay %.1f, completion %.1f%%, wait %.1f;  fixed time: delay %.1f, completion %.1f%%"
          % (mp["delay"], 100 * mp["completion"], mp["ti_wait"], fx["delay"], 100 * fx["completion"]))
json.dump(out, open(os.path.join(W, "results", "unseen_ft", "summary.json"), "w"), indent=1)
