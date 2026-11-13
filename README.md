# Game of Life

Final project.
Conway's Game of Life on a torus, with a sequential baseline and a hybrid
MPI + OpenMP version, correctness tests and a scaling benchmark.

## Files

| File | What it is |
|---|---|
| `src/common.h` | Config parsing, deterministic initial state, checksum, the Life rule (shared by both programs) |
| `src/gol_seq.c` | Sequential reference (no MPI, no OpenMP) |
| `src/gol_hybrid.c` | Hybrid version: 1D row decomposition, halo exchange, OpenMP inside each strip |
| `tests/test_correctness.py` | Known-pattern tests + sequential-vs-hybrid comparison (156 checks) |
| `bench/run_bench.py` | Strong and weak scaling measurements -> `results/*.csv` |
| `bench/plot.py` | Speedup, efficiency, mode comparison, weak scaling graphs |

## Build and run

Requirements: `gcc`, an MPI implementation (OpenMPI or MPICH) with `mpicc`/`mpirun`, Python 3 with `matplotlib`.

```bash
make                    # builds gol_seq and gol_hybrid
make test               # correctness tests (must print "0 failed")

./gol_seq -r 2048 -c 2048 -s 100
mpirun -np 4 ./gol_hybrid -r 2048 -c 2048 -s 100 -t 2 -m overlap

python3 bench/run_bench.py --cores 8   # use your physical core count
python3 bench/plot.py                  # graphs appear in results/
```

Options (both programs): `-r rows -c cols -s steps -d density -S seed -p pattern -o outfile`,
hybrid only: `-t omp_threads -m sync|overlap`. Patterns: `random`, `glider`, `blinker`, `block`.
`-o file` writes the final grid as text (use only for small grids).
Every run prints one `RESULT ...` line with the time, a global checksum and the population.

On a machine with fewer cores than processes use `MPIRUN_ARGS="--oversubscribe"`.

## Execution instructions

**Requirements.** Linux (tested on Ubuntu, also works in WSL), `gcc`, an MPI implementation with `mpicc` and `mpirun` (OpenMPI or MPICH), Python 3 with `matplotlib`.

```bash
sudo apt install build-essential openmpi-bin libopenmpi-dev python3 python3-matplotlib
```

**1. Build.** From the project root run `make`. This produces two executables: `gol_seq` (sequential baseline, built with `gcc`, no MPI/OpenMP) and `gol_hybrid` (built with `mpicc -fopenmp`).

**2. Check correctness.** Run `make test`. It runs 156 checks (known patterns and sequential-vs-hybrid comparison for many process/thread combinations) and must end with `0 failed`.

**3. Run a single simulation.** Both programs print one `RESULT` line with the run parameters, the time, a global checksum and the final population. For the same `-r -c -s` (and seed), the `checksum` and `population` of both programs must be identical.

```bash
./gol_seq -r 2000 -c 2000 -s 100
mpirun -np 4 ./gol_hybrid -r 2000 -c 2000 -s 100 -t 2 -m overlap
```

| Option | Meaning | Default |
|---|---|---|
| `-r`, `-c` | grid rows / columns | 1024 |
| `-s` | number of generations | 100 |
| `-d`, `-S` | initial density / random seed | 0.3 / 12345 |
| `-p` | pattern: `random`, `glider`, `blinker`, `block` | `random` |
| `-o file` | write the final grid as text (small grids only) | off |
| `-t` | OpenMP threads per MPI process (`gol_hybrid` only) | `OMP_NUM_THREADS` |
| `-m` | halo exchange: `sync` or `overlap` (`gol_hybrid` only) | `sync` |

`mpirun -np P` starts `P` MPI processes; total cores used = `P × t`. If `mpirun` reports not enough slots, add `--oversubscribe`. The number of rows must be at least `P`.

**4. Watch the simulation (optional demo).** `python3 watch.py` animates a glider in the terminal; `python3 watch.py --np 3` computes it with 3 MPI processes; `-p random` starts from a random grid.

**5. Reproduce the performance results.** Use the number of **physical** cores of your machine (`lscpu`):

```bash
python3 bench/run_bench.py --cores 8      # writes results/strong.csv and results/weak.csv
python3 bench/plot.py                     # writes speedup, efficiency, modes and weak-scaling PNGs to results/
```

`run_bench.py` measures strong scaling (fixed 4096×4096 grid, 200 generations) and weak scaling (1024 rows per core) for pure OpenMP, pure MPI and hybrid splits, in both communication modes, repeating each run 3 times and keeping the minimum. Speedup is computed against `gol_seq`, and every measurement is checked against the sequential checksum. Use `--quick` for a short smoke test. For stable numbers, do not run other heavy programs during the benchmark.

## Design

**Problem.** Each cell is alive/dead; every generation a live cell with 2 or 3 live
neighbours survives, a dead cell with exactly 3 is born. Edges wrap (torus).
Work per step is O(R·C), communication is O(C) per process → computation dominates
for large grids, so the problem scales well.

**Data ownership.** The R×C grid is split into horizontal strips. Rank r owns
its rows and is the only one that writes them. Each rank stores two extra *halo* rows
(copies of the neighbours' border rows). Initial state is computed from global
coordinates (`init_cell(i,j)`), so each rank builds its own strip without any scatter,
and the result does not depend on the number of processes.

**Communication.** Per generation each rank sends its first row up and its last row
down (and receives the two halos). One halo row suffices because the stencil reaches 1 row.
- `sync`: two `MPI_Sendrecv` calls (deadlock-free by construction).
- `overlap`: `MPI_Irecv`/`MPI_Isend`, compute the inner rows (they need no halo) while
  messages are in flight, `MPI_Waitall`, then compute the 2 border rows.

**Shared memory.** Inside a rank, `#pragma omp parallel for schedule(static)` over rows.
Threads write different rows of the output buffer, so there are no races. MPI is used
with `MPI_THREAD_FUNNELED` (only the main thread calls MPI, outside parallel regions).

**Correctness.** (1) Known patterns: block (still life), blinker (period 2), glider
(moves (+1,+1) per 4 generations, returns after 4N steps on an N×N torus). (2) Random grids:
final grid of the hybrid program must be identical to the sequential one for many
(P, threads, mode, size) combinations, including row counts not divisible by P.
(3) The benchmark asserts that checksums match the sequential run on every measurement.

**Metrics.** speedup = T_seq / T(c) with T_seq from the separate sequential program;
efficiency = speedup / c; weak scaling keeps rows-per-core fixed and looks at time.
