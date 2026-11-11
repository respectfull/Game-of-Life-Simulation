import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def best_per_core(rows, pred):
    best = {}
    for r in rows:
        if not pred(r):
            continue
        c, t = int(r["cores"]), float(r["time"])
        if c not in best or t < best[c]:
            best[c] = t
    return sorted(best.items())


def main():
    strong = load("strong.csv")
    t_seq = float(strong[0]["seq_time"])
    maxc = max(int(r["cores"]) for r in strong)

    series = {
        "pure OpenMP (1 x c)": lambda r: int(r["procs"]) == 1,
        "pure MPI (c x 1), sync": lambda r: int(r["threads"]) == 1 and r["mode"] == "sync",
        "pure MPI (c x 1), overlap": lambda r: int(r["threads"]) == 1 and r["mode"] == "overlap",
        "best hybrid": lambda r: int(r["procs"]) > 1 and int(r["threads"]) > 1,
    }

    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    cs = sorted({int(r["cores"]) for r in strong})
    ax1.plot(cs, cs, "k--", label="ideal")
    ax2.axhline(1.0, color="k", ls="--", label="ideal")
    for label, pred in series.items():
        pts = best_per_core(strong, pred)
        if len(pts) < 2:
            continue
        x = [c for c, _ in pts]
        s = [t_seq / t for _, t in pts]
        ax1.plot(x, s, "o-", label=label)
        ax2.plot(x, [si / c for si, c in zip(s, x)], "o-", label=label)
    ax1.set(xlabel="cores", ylabel="speedup  T_seq / T(c)", title="Strong scaling: speedup")
    ax2.set(xlabel="cores", ylabel="efficiency  speedup / c", title="Strong scaling: efficiency", ylim=(0, 1.15))
    for ax, fig, name in ((ax1, fig1, "speedup.png"), (ax2, fig2, "efficiency.png")):
        ax.set_xticks(cs); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(RES, name), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    rows = [r for r in strong if int(r["cores"]) == maxc]
    splits = sorted({(int(r["procs"]), int(r["threads"])) for r in rows})
    width = 0.38
    for k, mode in enumerate(("sync", "overlap")):
        xs, ys = [], []
        for i, (p, t) in enumerate(splits):
            m = [float(r["time"]) for r in rows
                 if int(r["procs"]) == p and int(r["threads"]) == t and r["mode"] == mode]
            if m:
                xs.append(i + (k - .5) * width); ys.append(t_seq / m[0])
        ax.bar(xs, ys, width, label=mode)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(["%dP x %dT" % s for s in splits])
    ax.set(ylabel="speedup", title="Processes x threads at %d cores" % maxc)
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "modes.png"), dpi=150)

    weak = load("weak.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    by = defaultdict(list)
    for r in weak:
        by[r["experiment"]].append((int(r["cores"]), float(r["time"])))
    for name, pts in by.items():
        pts.sort()
        ax.plot([c for c, _ in pts], [t for _, t in pts], "o-", label=name.replace("weak-", ""))
    ax.set(xlabel="cores", ylabel="time (s)", title="Weak scaling (work per core fixed)")
    ax.set_ylim(bottom=0); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "weak.png"), dpi=150)
    print("plots written to", RES)


if __name__ == "__main__":
    main()
