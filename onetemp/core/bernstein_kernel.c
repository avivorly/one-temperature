/* Bernstein-basis GD kernel — single-core C implementation.
 *
 * Side-dependent CP variance (see method.md): the inner loop picks the per-
 * point weight by sign(pred - y), folds it into the residual, and accumulates
 * the gradient against B (the weight is already in the residual, so there is
 * no Bw precompute — multiplying by Bw would double-count it).
 *
 * No allocation inside the outer loop; everything uses fixed stack arrays
 * (MAX_M = 32, MAX_N = 64 cover the degree-10 / 16-temp use case with margin).
 *
 * Build (MinGW UCRT64):
 *   gcc -O3 -funroll-loops -fno-math-errno -march=native -flto \
 *       -shared -o bernstein_kernel.dll bernstein_kernel.c
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define MAX_M 32
#define MAX_N 64

#if defined(_WIN32) || defined(__WIN32__)
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT __attribute__((visibility("default")))
#endif

static void pava_nondec(const double *in, double *out, int n);
static void pava_noninc(const double *in, double *out, int n);
static void project_unimodal_nonneg(const double *b, double *best, int m);

/* ---------------------------------------------------------------------- */
/* PAVA non-decreasing projection.                                         */
/* ---------------------------------------------------------------------- */
static void pava_nondec(const double *in, double *out, int n) {
    if (n == 0) return;
    double sums[MAX_M];
    double cnts[MAX_M];
    for (int i = 0; i < n; i++) {
        sums[i] = in[i];
        cnts[i] = 1.0;
    }
    int nb = n;
    int merged = 1;
    while (merged) {
        merged = 0;
        int w = 1;
        for (int bi = 1; bi < nb; bi++) {
            double prev_mean = sums[w - 1] / cnts[w - 1];
            double cur_mean = sums[bi] / cnts[bi];
            if (prev_mean > cur_mean) {
                sums[w - 1] += sums[bi];
                cnts[w - 1] += cnts[bi];
                merged = 1;
            } else {
                sums[w] = sums[bi];
                cnts[w] = cnts[bi];
                w++;
            }
        }
        nb = w;
    }
    int idx = 0;
    for (int bi = 0; bi < nb; bi++) {
        double avg = sums[bi] / cnts[bi];
        int cnt = (int)(cnts[bi] + 0.5);
        for (int k = 0; k < cnt; k++) {
            out[idx + k] = avg;
        }
        idx += cnt;
    }
}

static void pava_noninc(const double *in, double *out, int n) {
    if (n == 0) return;
    double sums[MAX_M];
    double cnts[MAX_M];
    for (int i = 0; i < n; i++) {
        sums[i] = in[i];
        cnts[i] = 1.0;
    }
    int nb = n;
    int merged = 1;
    while (merged) {
        merged = 0;
        int w = 1;
        for (int bi = 1; bi < nb; bi++) {
            double prev_mean = sums[w - 1] / cnts[w - 1];
            double cur_mean = sums[bi] / cnts[bi];
            if (prev_mean < cur_mean) {
                sums[w - 1] += sums[bi];
                cnts[w - 1] += cnts[bi];
                merged = 1;
            } else {
                sums[w] = sums[bi];
                cnts[w] = cnts[bi];
                w++;
            }
        }
        nb = w;
    }
    int idx = 0;
    for (int bi = 0; bi < nb; bi++) {
        double avg = sums[bi] / cnts[bi];
        int cnt = (int)(cnts[bi] + 0.5);
        for (int k = 0; k < cnt; k++) {
            out[idx + k] = avg;
        }
        idx += cnt;
    }
}

/* ---------------------------------------------------------------------- */
/* Project onto unimodal+nonneg cone (closest peak position by L2 to b).   */
/* ---------------------------------------------------------------------- */
static void project_unimodal_nonneg(const double *b, double *best, int m) {
    if (m <= 1) {
        for (int i = 0; i < m; i++) {
            best[i] = (b[i] > 0.0) ? b[i] : 0.0;
        }
        return;
    }
    int d = m - 1;

    double bp[MAX_M];
    for (int i = 0; i < m; i++) {
        bp[i] = (b[i] > 0.0) ? b[i] : 0.0;
    }

    double best_dist = INFINITY;
    double cand[MAX_M];
    double left_buf[MAX_M];
    double right_buf[MAX_M];

    for (int peak = 0; peak <= d; peak++) {
        for (int i = 0; i < m; i++) cand[i] = bp[i];

        if (peak > 0) {
            pava_nondec(cand, left_buf, peak + 1);
            for (int i = 0; i <= peak; i++) cand[i] = left_buf[i];
        }
        if (peak < d) {
            pava_noninc(cand + peak, right_buf, m - peak);
            for (int i = 0; i < m - peak; i++) cand[peak + i] = right_buf[i];
        }
        for (int i = 0; i < m; i++) {
            if (cand[i] < 0.0) cand[i] = 0.0;
        }
        double dist = 0.0;
        for (int i = 0; i < m; i++) {
            double diff = cand[i] - b[i];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            for (int i = 0; i < m; i++) best[i] = cand[i];
        }
    }
}

/* ---------------------------------------------------------------------- */
/* Main GD inner loop (side-dependent weighting).                          */
/* B is row-major (n, m). Per-point weights w_lo, w_hi are passed in       */
/* directly; the inner loop picks one of them by sign(pred - y), folds it  */
/* into the residual, and accumulates the gradient against B (no Bw).      */
/* ---------------------------------------------------------------------- */
DLL_EXPORT
void bernstein_gd(
    const double *B,        /* (n, m) row-major */
    const double *w_lo,     /* (n,)  per-point lower-side weight */
    const double *w_hi,     /* (n,)  per-point upper-side weight */
    const double *y,        /* (n,)  target values (pass rates, percent) */
    int n,
    int m,
    int iterations,
    double lr,
    int ramp_iters,
    int proj_every,
    double *bc_out          /* (m,) preallocated by caller */
) {
    double bc[MAX_M];
    double pred[MAX_N];
    double residual[MAX_N];
    double grad[MAX_M];
    double bc_clamped[MAX_M];
    double bc_uni[MAX_M];

    for (int j = 0; j < m; j++) bc[j] = 0.0;

    const double inv_n_times_2 = 2.0 / (double)n;

    for (int it = 0; it < iterations; it++) {
        /* pred = B @ bc; residual = w_side * (y - pred) */
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            const double *Bi = B + (size_t)i * m;
            for (int j = 0; j < m; j++) {
                s += Bi[j] * bc[j];
            }
            pred[i] = s;
            double w = (s > y[i]) ? w_hi[i] : w_lo[i];
            residual[i] = w * (y[i] - s);
        }

        /* grad[j] = -(2/n) * sum_i B[i, j] * residual[i].
         * NOTE: this is B (not Bw) — the weight is already folded into
         * residual above, so multiplying by Bw here would double-count it. */
        for (int j = 0; j < m; j++) {
            double s = 0.0;
            for (int i = 0; i < n; i++) {
                s += B[(size_t)i * m + j] * residual[i];
            }
            grad[j] = -inv_n_times_2 * s;
        }

        for (int j = 0; j < m; j++) {
            bc[j] -= lr * grad[j];
        }

        double ramp;
        if (ramp_iters > 0) {
            ramp = (double)it / (double)ramp_iters;
            if (ramp > 1.0) ramp = 1.0;
        } else {
            ramp = 1.0;
        }

        if ((it % proj_every == 0) || (it == iterations - 1)) {
            for (int j = 0; j < m; j++) {
                bc_clamped[j] = (bc[j] > 0.0) ? bc[j] : 0.0;
            }
            project_unimodal_nonneg(bc_clamped, bc_uni, m);
            for (int j = 0; j < m; j++) {
                bc[j] = bc[j] + ramp * (bc_uni[j] - bc[j]);
            }
        }
    }

    /* Final hard projection. */
    for (int j = 0; j < m; j++) {
        bc_clamped[j] = (bc[j] > 0.0) ? bc[j] : 0.0;
    }
    project_unimodal_nonneg(bc_clamped, bc_uni, m);
    for (int j = 0; j < m; j++) bc_out[j] = bc_uni[j];
}
