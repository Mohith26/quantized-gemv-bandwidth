// Batch-1 decode is a stream of matrix-vector products: every weight is read once per token.
// This measures y = W x for one big weight matrix stored at different precisions, and reports
// the effective weight bandwidth (bytes of W read per second) and the implied tokens/s for a
// model with the same number of weights.
//
// Formats: fp32, fp16, int8 (per-row scale), int4 (two weights per byte, one fp32 scale per
// group of 32). Kernels are plain C, compiled with clang -O3 (NEON via auto-vectorisation);
// they are not hand-tuned like llama.cpp's.
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROWS 8192L
#define COLS 16384L /* 134M weights: 512 MiB as fp32, 64 MiB as int4 */
#define GROUP 32

static float *x, *y;
static float *w32;
static _Float16 *w16;
static int8_t *w8;
static float *s8;
static uint8_t *w4;
static float *s4;
static int nthreads;
static int8_t *xq; /* x quantised to int8, one scale per group of 32 */
static float *xs;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct { long lo, hi; int fmt; } job_t;

static void rows_fp32(long lo, long hi) {
    for (long r = lo; r < hi; r++) {
        const float *w = w32 + r * COLS;
        float acc = 0;
        for (long c = 0; c < COLS; c++) acc += w[c] * x[c];
        y[r] = acc;
    }
}

static void rows_fp16(long lo, long hi) {
    for (long r = lo; r < hi; r++) {
        const _Float16 *w = w16 + r * COLS;
        float acc = 0;
        for (long c = 0; c < COLS; c++) acc += (float)w[c] * x[c];
        y[r] = acc;
    }
}

static void rows_int8(long lo, long hi) {
    for (long r = lo; r < hi; r++) {
        const int8_t *w = w8 + r * COLS;
        float acc = 0;
        for (long c = 0; c < COLS; c++) acc += (float)w[c] * x[c];
        y[r] = acc * s8[r];
    }
}

static void rows_int4(long lo, long hi) {
    for (long r = lo; r < hi; r++) {
        const uint8_t *w = w4 + r * COLS / 2;
        const float *s = s4 + r * (COLS / GROUP);
        float acc = 0;
        for (long g = 0; g < COLS / GROUP; g++) {
            float gacc = 0;
            const uint8_t *wg = w + g * GROUP / 2;
            const float *xg = x + g * GROUP;
            for (int k = 0; k < GROUP / 2; k++) {
                int lo4 = (wg[k] & 0x0F) - 8;
                int hi4 = (wg[k] >> 4) - 8;
                gacc += lo4 * xg[2 * k] + hi4 * xg[2 * k + 1];
            }
            acc += gacc * s[g];
        }
        y[r] = acc;
    }
}

// Same int4 data, read with the split layout used by llama.cpp's Q4_0: byte k of a group holds
// weight k in its low nibble and weight k+16 in its high nibble, so both halves are contiguous.
static void rows_int4_split(long lo, long hi) {
    for (long r = lo; r < hi; r++) {
        const uint8_t *w = w4 + r * COLS / 2;
        const float *s = s4 + r * (COLS / GROUP);
        float acc = 0;
        for (long g = 0; g < COLS / GROUP; g++) {
            const uint8_t *wg = w + g * GROUP / 2;
            const float *xg = x + g * GROUP;
            float gacc = 0;
            for (int k = 0; k < GROUP / 2; k++) {
                gacc += (float)((wg[k] & 0x0F) - 8) * xg[k] + (float)((wg[k] >> 4) - 8) * xg[k + GROUP / 2];
            }
            acc += gacc * s[g];
        }
        y[r] = acc;
    }
}

// Split-layout int4 weights times int8-quantised activations: the inner loop is integer only,
// and the two scales are applied once per group.
static void rows_int4_q8(long lo, long hi) {
    for (long r = lo; r < hi; r++) {
        const uint8_t *w = w4 + r * COLS / 2;
        const float *s = s4 + r * (COLS / GROUP);
        float acc = 0;
        for (long g = 0; g < COLS / GROUP; g++) {
            const uint8_t *wg = w + g * GROUP / 2;
            const int8_t *xg = xq + g * GROUP;
            int32_t gacc = 0;
            for (int k = 0; k < GROUP / 2; k++) {
                gacc += (int16_t)((int8_t)(wg[k] & 0x0F) - 8) * xg[k] + (int16_t)((int8_t)(wg[k] >> 4) - 8) * xg[k + GROUP / 2];
            }
            acc += (float)gacc * s[g] * xs[g];
        }
        y[r] = acc;
    }
}

// int8 weights times int8-quantised activations, integer inner loop per group.
static void rows_int8_q8(long lo, long hi) {
    for (long r = lo; r < hi; r++) {
        const int8_t *w = w8 + r * COLS;
        float acc = 0;
        for (long g = 0; g < COLS / GROUP; g++) {
            const int8_t *wg = w + g * GROUP;
            const int8_t *xg = xq + g * GROUP;
            int32_t gacc = 0;
            for (int k = 0; k < GROUP; k++) gacc += (int16_t)wg[k] * xg[k];
            acc += (float)gacc * xs[g];
        }
        y[r] = acc * s8[r];
    }
}

// Same as rows_int4_q8, written with NEON intrinsics: nibbles unpacked with and/shift, and
// the int8 dot products done by SDOT (vdotq_s32), the approach llama.cpp uses on ARM.
static void rows_int4_q8_neon(long lo, long hi) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(8);
    for (long r = lo; r < hi; r++) {
        const uint8_t *w = w4 + r * COLS / 2;
        const float *s = s4 + r * (COLS / GROUP);
        float32x4_t acc = vdupq_n_f32(0);
        for (long g = 0; g < COLS / GROUP; g++) {
            uint8x16_t b = vld1q_u8(w + g * GROUP / 2);
            int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, m4)), s8b);
            int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), s8b);
            int32x4_t d = vdotq_s32(vdupq_n_s32(0), wl, vld1q_s8(xq + g * GROUP));
            d = vdotq_s32(d, wh, vld1q_s8(xq + g * GROUP + GROUP / 2));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(d), s[g] * xs[g]);
        }
        y[r] = vaddvq_f32(acc);
    }
}

static void *work(void *p) {
    job_t *j = p;
    switch (j->fmt) {
    case 0: rows_fp32(j->lo, j->hi); break;
    case 1: rows_fp16(j->lo, j->hi); break;
    case 2: rows_int8(j->lo, j->hi); break;
    case 3: rows_int4(j->lo, j->hi); break;
    case 4: rows_int4_split(j->lo, j->hi); break;
    case 5: rows_int8_q8(j->lo, j->hi); break;
    case 6: rows_int4_q8(j->lo, j->hi); break;
    case 7: rows_int4_q8_neon(j->lo, j->hi); break;
    }
    return NULL;
}

static double weight_bytes(int fmt) {
    switch (fmt) {
    case 0: return ROWS * COLS * 4.0;
    case 1: return ROWS * COLS * 2.0;
    case 2:
    case 5: return ROWS * COLS * 1.0 + ROWS * 4.0;
    default: return ROWS * COLS * 0.5 + ROWS * (COLS / GROUP) * 4.0;
    }
}

static double best_time(int fmt) {
    pthread_t th[64];
    job_t jobs[64];
    double best = 1e9;
    for (int rep = 0; rep < 15; rep++) {
        double t0 = now();
        for (int t = 0; t < nthreads; t++) {
            jobs[t] = (job_t){ROWS * t / nthreads, ROWS * (t + 1) / nthreads, fmt};
            pthread_create(&th[t], NULL, work, &jobs[t]);
        }
        for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

// Reference result in double precision from the dequantised weights, to check each kernel.
static double max_rel_err(int fmt) {
    double worst = 0;
    for (long r = 0; r < ROWS; r += 997) {
        double ref = 0;
        for (long c = 0; c < COLS; c++) {
            double wv;
            long g = c / GROUP, k = c % GROUP;
            double xv = (fmt >= 5) ? xq[c] * (double)xs[g] : x[c];
            switch (fmt) {
            case 0: wv = w32[r * COLS + c]; break;
            case 1: wv = (double)w16[r * COLS + c]; break;
            case 2:
            case 5: wv = w8[r * COLS + c] * (double)s8[r]; break;
            case 3: {
                uint8_t b = w4[(r * COLS + c) / 2];
                int q = ((c & 1) ? (b >> 4) : (b & 0x0F)) - 8;
                wv = q * (double)s4[r * (COLS / GROUP) + g];
                break;
            }
            default: {
                uint8_t b = w4[r * COLS / 2 + g * GROUP / 2 + (k % (GROUP / 2))];
                int q = (k < GROUP / 2 ? (b & 0x0F) : (b >> 4)) - 8;
                wv = q * (double)s4[r * (COLS / GROUP) + g];
            }
            }
            ref += wv * xv;
        }
        double err = fabs(y[r] - ref) / (fabs(ref) + 1e-6);
        if (err > worst) worst = err;
    }
    return worst;
}

int main(int argc, char **argv) {
    nthreads = argc > 1 ? atoi(argv[1]) : 1;
    double model_params = argc > 2 ? atof(argv[2]) : 8e9;
    srand(1);
    x = malloc(COLS * sizeof(float));
    y = malloc(ROWS * sizeof(float));
    w32 = malloc(ROWS * COLS * sizeof(float));
    w16 = malloc(ROWS * COLS * sizeof(_Float16));
    w8 = malloc(ROWS * COLS);
    s8 = malloc(ROWS * sizeof(float));
    w4 = malloc(ROWS * COLS / 2);
    s4 = malloc(ROWS * (COLS / GROUP) * sizeof(float));
    for (long c = 0; c < COLS; c++) x[c] = (float)((double)rand() / RAND_MAX) - 0.5f;
    for (long i = 0; i < ROWS * COLS; i++) {
        float v = ((float)((double)rand() / RAND_MAX) - 0.5f) * 0.1f;
        w32[i] = v;
        w16[i] = (_Float16)v;
        w8[i] = (int8_t)(rand() % 255 - 127);
        if ((i & 1) == 0) w4[i / 2] = (uint8_t)(rand() & 0xFF);
    }
    for (long r = 0; r < ROWS; r++) s8[r] = 0.05f / 127;
    xq = malloc(COLS);
    xs = malloc((COLS / GROUP) * sizeof(float));
    for (long g = 0; g < COLS / GROUP; g++) {
        float amax = 0;
        for (int k = 0; k < GROUP; k++) amax = fmaxf(amax, fabsf(x[g * GROUP + k]));
        xs[g] = amax / 127;
        for (int k = 0; k < GROUP; k++) xq[g * GROUP + k] = (int8_t)lrintf(x[g * GROUP + k] / xs[g]);
    }
    for (long g = 0; g < ROWS * (COLS / GROUP); g++) s4[g] = 0.05f / 7;

    const char *names[] = {"fp32", "fp16", "int8", "int4", "int4s", "i8q8", "i4q8", "i4q8n"};
    printf("threads=%d matrix=%ldx%ld model_params=%.2g\n", nthreads, ROWS, COLS, model_params);
    printf("%-6s %9s %10s %12s %11s %9s\n", "fmt", "ms/gemv", "MiB(W)", "W GB/s", "tok/s@model", "max_rel_err");
    for (int f = 0; f < 8; f++) {
        double t = best_time(f);
        double bytes = weight_bytes(f);
        double bytes_per_param = bytes / (ROWS * COLS);
        double gbps = bytes / t / 1e9;
        double toks = gbps * 1e9 / (bytes_per_param * model_params);
        printf("%-6s %9.2f %10.0f %12.1f %11.1f %9.1e\n", names[f], t * 1e3, bytes / (1 << 20), gbps, toks,
               max_rel_err(f));
    }
    return 0;
}
