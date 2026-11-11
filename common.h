#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int rows, cols;
    int steps;
    double density;
    uint64_t seed;
    char pattern[16];
    const char *out_file;
    int threads;
    int overlap;
} Config;

static void config_defaults(Config *cfg) {
    cfg->rows = 1024; cfg->cols = 1024; cfg->steps = 100;
    cfg->density = 0.3; cfg->seed = 12345;
    strcpy(cfg->pattern, "random");
    cfg->out_file = NULL; cfg->threads = 0; cfg->overlap = 0;
}

static int config_parse(Config *cfg, int argc, char **argv) {
    config_defaults(cfg);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (i + 1 >= argc) return -1;
        const char *v = argv[++i];
        if      (!strcmp(a, "-r")) cfg->rows = atoi(v);
        else if (!strcmp(a, "-c")) cfg->cols = atoi(v);
        else if (!strcmp(a, "-s")) cfg->steps = atoi(v);
        else if (!strcmp(a, "-d")) cfg->density = atof(v);
        else if (!strcmp(a, "-S")) cfg->seed = strtoull(v, NULL, 10);
        else if (!strcmp(a, "-p")) { strncpy(cfg->pattern, v, 15); cfg->pattern[15] = 0; }
        else if (!strcmp(a, "-o")) cfg->out_file = v;
        else if (!strcmp(a, "-t")) cfg->threads = atoi(v);
        else if (!strcmp(a, "-m")) cfg->overlap = !strcmp(v, "overlap");
        else return -1;
    }
    return (cfg->rows > 0 && cfg->cols > 0 && cfg->steps >= 0) ? 0 : -1;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [-r rows] [-c cols] [-s steps] [-d density] [-S seed]\n"
        "          [-p random|glider|blinker|block] [-o outfile]\n"
        "          [-t omp_threads] [-m sync|overlap]\n", prog);
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static const int GLIDER[5][2]  = {{0,1},{1,2},{2,0},{2,1},{2,2}};
static const int BLINKER[3][2] = {{0,0},{0,1},{0,2}};
static const int BLOCK[4][2]   = {{0,0},{0,1},{1,0},{1,1}};
#define PAT_ORIGIN 4

static inline int in_pattern(const int (*cells)[2], int n, int i, int j) {
    for (int k = 0; k < n; k++)
        if (i == PAT_ORIGIN + cells[k][0] && j == PAT_ORIGIN + cells[k][1]) return 1;
    return 0;
}

static inline uint8_t init_cell(const Config *cfg, int i, int j) {
    if (!strcmp(cfg->pattern, "glider"))  return in_pattern(GLIDER, 5, i, j);
    if (!strcmp(cfg->pattern, "blinker")) return in_pattern(BLINKER, 3, i, j);
    if (!strcmp(cfg->pattern, "block"))   return in_pattern(BLOCK, 4, i, j);
    uint64_t h = splitmix64(cfg->seed ^ ((uint64_t)i * (uint64_t)cfg->cols + (uint64_t)j));
    double u = (double)(h >> 11) * (1.0 / 9007199254740992.0);
    return u < cfg->density;
}

static inline uint64_t cell_hash(int i, int j, int cols) {
    return splitmix64(0xABCDEF ^ ((uint64_t)i * (uint64_t)cols + (uint64_t)j));
}

static inline uint8_t life_rule(uint8_t alive, int neighbours) {
    return (uint8_t)(neighbours == 3 || (alive && neighbours == 2));
}

#endif
