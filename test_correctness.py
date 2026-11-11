import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(ROOT, "gol_seq")
HYB = os.path.join(ROOT, "gol_hybrid")
MPIRUN_ARGS = os.environ.get("MPIRUN_ARGS", "--oversubscribe").split()
if os.geteuid() == 0 and "--allow-run-as-root" not in MPIRUN_ARGS:
    MPIRUN_ARGS.append("--allow-run-as-root")

PATTERN_ORIGIN = 4
GLIDER = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
BLINKER = [(0, 0), (0, 1), (0, 2)]
BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1)]

passed = failed = 0


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("command failed: %s\n%s" % (" ".join(cmd), p.stderr))
    m = re.search(r"RESULT (.*)", p.stdout)
    return dict(kv.split("=") for kv in m.group(1).split())


def run_seq(args):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        out = f.name
    info = run([SEQ] + args + ["-o", out])
    grid = open(out).read()
    os.unlink(out)
    return info, grid


def run_hyb(np_, threads, mode, args):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        out = f.name
    info = run(["mpirun"] + MPIRUN_ARGS + ["-np", str(np_), HYB] + args +
               ["-t", str(threads), "-m", mode, "-o", out])
    grid = open(out).read()
    os.unlink(out)
    return info, grid


def alive_cells(grid):
    return {(i, j) for i, line in enumerate(grid.splitlines())
            for j, ch in enumerate(line) if ch == "#"}


def shape(cells, di=0, dj=0, n=None):
    return {((PATTERN_ORIGIN + a + di) % n, (PATTERN_ORIGIN + b + dj) % n) for a, b in cells}


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print("  PASS  " + name)
    else:
        failed += 1
        print("  FAIL  " + name + (("  -> " + detail) if detail else ""))


def runners():
    yield "seq", lambda a: run_seq(a)[1]
    for np_ in (1, 2, 3, 4):
        for mode in ("sync", "overlap"):
            yield "hybrid np=%d %s" % (np_, mode), (lambda a, np_=np_, mode=mode: run_hyb(np_, 2, mode, a)[1])


def test_patterns():
    print("Known patterns")
    N = 16
    base = ["-r", str(N), "-c", str(N)]
    for label, f in runners():
        g = alive_cells(f(base + ["-p", "block", "-s", "7"]))
        check("block is still life            [%s]" % label, g == shape(BLOCK, n=N))

        g1 = alive_cells(f(base + ["-p", "blinker", "-s", "1"]))
        g2 = alive_cells(f(base + ["-p", "blinker", "-s", "2"]))
        vertical = {(PATTERN_ORIGIN - 1 + k, PATTERN_ORIGIN + 1) for k in range(3)}
        check("blinker flips, period 2        [%s]" % label,
              g1 == vertical and g2 == shape(BLINKER, n=N))

        g = alive_cells(f(base + ["-p", "glider", "-s", "4"]))
        check("glider moves (+1,+1) in 4 gens [%s]" % label, g == shape(GLIDER, 1, 1, N))

        g = alive_cells(f(base + ["-p", "glider", "-s", str(4 * N)]))
        check("glider wraps torus after 4N    [%s]" % label, g == shape(GLIDER, n=N))


def test_seq_vs_hybrid():
    print("Sequential vs hybrid on random grids")
    cases = [
        (32, 32, 20, 1),
        (37, 29, 15, 2),
        (50, 41, 25, 3),
        (8, 8, 30, 4),
        (11, 64, 12, 5),
    ]
    for rows, cols, steps, seed in cases:
        args = ["-r", str(rows), "-c", str(cols), "-s", str(steps), "-S", str(seed), "-d", "0.35"]
        ref_info, ref_grid = run_seq(args)
        for np_ in (1, 2, 3, 4, 5, 8):
            if np_ > rows:
                continue
            for threads in (1, 3):
                for mode in ("sync", "overlap"):
                    info, grid = run_hyb(np_, threads, mode, args)
                    ok = (grid == ref_grid and info["checksum"] == ref_info["checksum"]
                          and info["population"] == ref_info["population"])
                    check("%dx%d steps=%d np=%d thr=%d %-7s" % (rows, cols, steps, np_, threads, mode), ok)


def main():
    for exe in (SEQ, HYB):
        if not os.path.exists(exe):
            sys.exit("missing %s - run `make` first" % exe)
    test_patterns()
    test_seq_vs_hybrid()
    print("\n%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
