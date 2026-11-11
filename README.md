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