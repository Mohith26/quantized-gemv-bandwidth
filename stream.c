// Minimal STREAM-style memory bandwidth test (copy and triad), single and multi threaded.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define N (64L * 1024 * 1024) /* 64M doubles = 512 MiB per array */
static double *a, *b, *c;
static int nthreads;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct { long lo, hi; int op; } job_t;

static void *work(void *p) {
    job_t *j = p;
    if (j->op == 0) {
        for (long i = j->lo; i < j->hi; i++) c[i] = a[i];
    } else {
        for (long i = j->lo; i < j->hi; i++) a[i] = b[i] + 3.0 * c[i];
    }
    return NULL;
}

static double run(int op) {
    pthread_t th[64];
    job_t jobs[64];
    double best = 1e9;
    for (int rep = 0; rep < 10; rep++) {
        double t0 = now();
        for (int t = 0; t < nthreads; t++) {
            jobs[t].lo = N * t / nthreads;
            jobs[t].hi = N * (t + 1) / nthreads;
            jobs[t].op = op;
            pthread_create(&th[t], NULL, work, &jobs[t]);
        }
        for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    double bytes = (op == 0 ? 2.0 : 3.0) * N * sizeof(double);
    return bytes / best / 1e9;
}

int main(int argc, char **argv) {
    nthreads = argc > 1 ? atoi(argv[1]) : 1;
    a = malloc(N * sizeof(double));
    b = malloc(N * sizeof(double));
    c = malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.5; }
    printf("threads=%d copy=%.1f GB/s triad=%.1f GB/s\n", nthreads, run(0), run(1));
    return 0;
}
