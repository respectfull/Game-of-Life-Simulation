import argparse
import csv
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(ROOT, "gol_seq")
HYB = os.path.join(ROOT, "gol_hybrid")


def mpirun_prefix(np_):
    extra = os.environ.get("MPIRUN_ARGS", "--bind-to none --oversubscribe").split()
    if os.geteuid() == 0 and "--allow-run-as-root" not in extra:
        extra.append("--allow-run-as-root")
    return ["mpirun"] + extra + ["-np", str(np_)]


def parse(stdout):
    m = re.search(r"RESULT (.*)", stdout)
    return dict(kv.split("=") for kv in m.group(1).split())


def best_of(cmd, repeats, env=None):
    best = None
    for _ in range(repeats):
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            raise RuntimeError(p.stderr)
        r = parse(p.stdout)
        if best is None or float(r["time"]) < float(best["time"]):
            best = r
    return best


def divisors_splits(c):
    return [(p, c // p) for p in range(1, c + 1) if c % p == 0]


def core_counts(maxc):
    out, c = [], 1
    while c <= maxc:
        out.append(c)
        c *= 2
    if out[-1] != maxc:
        out.append(maxc)
    return out


def run_hybrid(np_, threads, mode, rows, cols, steps, repeats):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = mpirun_prefix(np_) + [HYB, "-r", str(rows), "-c", str(cols), "-s", str(steps),
                                "-t", str(threads), "-m", mode]
    return best_of(cmd, repeats, env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=os.cpu_count())
    ap.add_argument("--size", type=int, default=4096, help="grid is size x size (strong scaling)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--weak-rows-per-core", type=int, default=1024)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    if a.quick:
        a.size, a.steps, a.repeats, a.weak_rows_per_core = 256, 20, 1, 64
    os.makedirs(a.out, exist_ok=True)
    fields = ["experiment", "rows", "cols", "steps", "cores", "procs", "threads", "mode", "time", "seq_time"]

    print("== strong scaling: %dx%d, %d steps ==" % (a.size, a.size, a.steps))
    seq = best_of([SEQ, "-r", str(a.size), "-c", str(a.size), "-s", str(a.steps)], a.repeats)
    t_seq = float(seq["time"])
    print("sequential baseline: %.3f s" % t_seq)
    rows_out = []
    for c in core_counts(a.cores):
        for p, t in divisors_splits(c):
            for mode in (("sync",) if p == 1 else ("sync", "overlap")):
                r = run_hybrid(p, t, mode, a.size, a.size, a.steps, a.repeats)
                assert r["checksum"] == seq["checksum"], "RESULT MISMATCH vs sequential!"
                print("  cores=%-3d procs=%-3d threads=%-3d %-7s %.3f s  speedup %.2f" %
                      (c, p, t, mode, float(r["time"]), t_seq / float(r["time"])))
                rows_out.append(dict(experiment="strong", rows=a.size, cols=a.size, steps=a.steps,
                                     cores=c, procs=p, threads=t, mode=mode, time=r["time"], seq_time=t_seq))
    with open(os.path.join(a.out, "strong.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows_out)

    print("== weak scaling: %d rows per core x %d cols ==" % (a.weak_rows_per_core, a.size))
    rows_out = []
    for c in core_counts(a.cores):
        rows = a.weak_rows_per_core * c
        for label, (p, t) in (("mpi", (c, 1)), ("openmp", (1, c))):
            r = run_hybrid(p, t, "sync", rows, a.size, a.steps, a.repeats)
            print("  cores=%-3d %-6s rows=%-6d %.3f s" % (c, label, rows, float(r["time"])))
            rows_out.append(dict(experiment="weak-" + label, rows=rows, cols=a.size, steps=a.steps,
                                 cores=c, procs=p, threads=t, mode="sync", time=r["time"], seq_time=""))
    with open(os.path.join(a.out, "weak.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows_out)
    print("done -> %s" % a.out)


if __name__ == "__main__":
    main()
