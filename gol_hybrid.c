#include <mpi.h>
#include <omp.h>
#include "common.h"

static void decompose(int R, int P, int rank, int *local_rows, int *first_row) {
    int base = R / P, rem = R % P;
    *local_rows = base + (rank < rem ? 1 : 0);
    *first_row  = rank * base + (rank < rem ? rank : rem);
}

static void step_rows(const uint8_t *cur, uint8_t *nxt, int C, int r_begin, int r_end) {
    #pragma omp parallel for schedule(static)
    for (int i = r_begin; i < r_end; i++) {
        const uint8_t *up  = cur + (size_t)(i - 1) * C;
        const uint8_t *mid = cur + (size_t)i * C;
        const uint8_t *dn  = cur + (size_t)(i + 1) * C;
        uint8_t *out = nxt + (size_t)i * C;
        for (int j = 0; j < C; j++) {
            int l = (j == 0) ? C - 1 : j - 1;
            int r = (j == C - 1) ? 0 : j + 1;
            int n = up[l] + up[j] + up[r] + mid[l] + mid[r] + dn[l] + dn[j] + dn[r];
            out[j] = life_rule(mid[j], n);
        }
    }
}

enum { TAG_UP = 0, TAG_DOWN = 1 };

static void halo_exchange_sync(uint8_t *g, int C, int L, int up, int down, MPI_Comm comm) {
    MPI_Sendrecv(g + (size_t)1 * C,       C, MPI_UINT8_T, up,   TAG_UP,
                 g + (size_t)(L + 1) * C, C, MPI_UINT8_T, down, TAG_UP,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(g + (size_t)L * C,       C, MPI_UINT8_T, down, TAG_DOWN,
                 g + 0,                   C, MPI_UINT8_T, up,   TAG_DOWN,
                 comm, MPI_STATUS_IGNORE);
}

static void halo_exchange_start(uint8_t *g, int C, int L, int up, int down,
                                MPI_Comm comm, MPI_Request req[4]) {
    MPI_Irecv(g + (size_t)(L + 1) * C, C, MPI_UINT8_T, down, TAG_UP,   comm, &req[0]);
    MPI_Irecv(g + 0,                   C, MPI_UINT8_T, up,   TAG_DOWN, comm, &req[1]);
    MPI_Isend(g + (size_t)1 * C,       C, MPI_UINT8_T, up,   TAG_UP,   comm, &req[2]);
    MPI_Isend(g + (size_t)L * C,       C, MPI_UINT8_T, down, TAG_DOWN, comm, &req[3]);
}

static void write_grid(const Config *cfg, const uint8_t *g, int L, int rank, int P, MPI_Comm comm) {
    const int C = cfg->cols;
    int *counts = NULL, *displs = NULL;
    uint8_t *all = NULL;
    if (rank == 0) {
        counts = malloc(P * sizeof(int)); displs = malloc(P * sizeof(int));
        all = malloc((size_t)cfg->rows * C);
        for (int r = 0, off = 0; r < P; r++) {
            int lr, fr; decompose(cfg->rows, P, r, &lr, &fr);
            counts[r] = lr * C; displs[r] = off; off += lr * C;
        }
    }
    MPI_Gatherv(g + C, L * C, MPI_UINT8_T, all, counts, displs, MPI_UINT8_T, 0, comm);
    if (rank == 0) {
        FILE *f = fopen(cfg->out_file, "w");
        if (!f) { perror("fopen"); MPI_Abort(comm, 1); }
        for (int i = 0; i < cfg->rows; i++) {
            for (int j = 0; j < C; j++) fputc(all[(size_t)i * C + j] ? '#' : '.', f);
            fputc('\n', f);
        }
        fclose(f);
        free(all); free(counts); free(displs);
    }
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    Config cfg;
    if (config_parse(&cfg, argc, argv) != 0) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
    if (cfg.rows < P) {
        if (rank == 0) fprintf(stderr, "error: need rows >= number of processes\n");
        MPI_Finalize();
        return 1;
    }

    const int C = cfg.cols;
    int L, first;
    decompose(cfg.rows, P, rank, &L, &first);
    const int up = (rank - 1 + P) % P, down = (rank + 1) % P;

    size_t bytes = (size_t)(L + 2) * C;
    uint8_t *cur = calloc(bytes, 1), *nxt = calloc(bytes, 1);
    if (!cur || !nxt) { fprintf(stderr, "rank %d: out of memory\n", rank); MPI_Abort(comm, 1); }

    for (int i = 0; i < L; i++)
        for (int j = 0; j < C; j++)
            cur[(size_t)(i + 1) * C + j] = init_cell(&cfg, first + i, j);

    MPI_Request req[4];
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    for (int s = 0; s < cfg.steps; s++) {
        if (P == 1) {
            memcpy(cur, cur + (size_t)L * C, C);
            memcpy(cur + (size_t)(L + 1) * C, cur + C, C);
            step_rows(cur, nxt, C, 1, L + 1);
        } else if (!cfg.overlap) {
            halo_exchange_sync(cur, C, L, up, down, comm);
            step_rows(cur, nxt, C, 1, L + 1);
        } else {
            halo_exchange_start(cur, C, L, up, down, comm, req);
            if (L > 2) step_rows(cur, nxt, C, 2, L);
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            step_rows(cur, nxt, C, 1, 2);
            if (L > 1) step_rows(cur, nxt, C, L, L + 1);
        }
        uint8_t *tmp = cur; cur = nxt; nxt = tmp;
    }

    MPI_Barrier(comm);
    double local_t = MPI_Wtime() - t0, elapsed;
    MPI_Reduce(&local_t, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    uint64_t checksum = 0, population = 0, g_sum = 0, g_pop = 0;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < C; j++)
            if (cur[(size_t)(i + 1) * C + j]) { checksum += cell_hash(first + i, j, C); population++; }
    MPI_Reduce(&checksum,   &g_sum, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
    MPI_Reduce(&population, &g_pop, 1, MPI_UINT64_T, MPI_SUM, 0, comm);

    if (cfg.out_file) write_grid(&cfg, cur, L, rank, P, comm);

    if (rank == 0) {
        printf("RESULT impl=hybrid rows=%d cols=%d steps=%d procs=%d threads=%d mode=%s "
               "time=%.6f checksum=%llu population=%llu\n",
               cfg.rows, C, cfg.steps, P, omp_get_max_threads(),
               (P == 1) ? "sync" : (cfg.overlap ? "overlap" : "sync"), elapsed,
               (unsigned long long)g_sum, (unsigned long long)g_pop);
    }
    free(cur); free(nxt);
    MPI_Finalize();
    return 0;
}
