#include <time.h>
#include "common.h"

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    Config cfg;
    if (config_parse(&cfg, argc, argv) != 0) { print_usage(argv[0]); return 1; }
    const int R = cfg.rows, C = cfg.cols;

    uint8_t *cur = malloc((size_t)R * C), *nxt = malloc((size_t)R * C);
    if (!cur || !nxt) { fprintf(stderr, "out of memory\n"); return 1; }

    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            cur[(size_t)i * C + j] = init_cell(&cfg, i, j);

    double t0 = now();
    for (int s = 0; s < cfg.steps; s++) {
        for (int i = 0; i < R; i++) {
            const uint8_t *up  = cur + (size_t)((i + R - 1) % R) * C;
            const uint8_t *mid = cur + (size_t)i * C;
            const uint8_t *dn  = cur + (size_t)((i + 1) % R) * C;
            for (int j = 0; j < C; j++) {
                int l = (j == 0) ? C - 1 : j - 1;
                int r = (j == C - 1) ? 0 : j + 1;
                int n = up[l] + up[j] + up[r] + mid[l] + mid[r] + dn[l] + dn[j] + dn[r];
                nxt[(size_t)i * C + j] = life_rule(mid[j], n);
            }
        }
        uint8_t *tmp = cur; cur = nxt; nxt = tmp;
    }
    double elapsed = now() - t0;

    uint64_t checksum = 0, population = 0;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            if (cur[(size_t)i * C + j]) { checksum += cell_hash(i, j, C); population++; }

    if (cfg.out_file) {
        FILE *f = fopen(cfg.out_file, "w");
        if (!f) { perror("fopen"); return 1; }
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) fputc(cur[(size_t)i * C + j] ? '#' : '.', f);
            fputc('\n', f);
        }
        fclose(f);
    }

    printf("RESULT impl=seq rows=%d cols=%d steps=%d procs=1 threads=1 mode=seq "
           "time=%.6f checksum=%llu population=%llu\n",
           R, C, cfg.steps, elapsed,
           (unsigned long long)checksum, (unsigned long long)population);
    free(cur); free(nxt);
    return 0;
}
