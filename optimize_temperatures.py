"""Compare temperature-sampling schemes under pass@K, with held-out evaluation (paper Sec. 4.1-4.2 and
App. "Temperature Sampling Schemes Comparisons").

Given per-question single-attempt pass rates P[i, j] on a grid of temperatures gamma_1..gamma_l, and a
budget of K samples per question, the pass@K of an allocation (K_1, ..., K_l), sum_j K_j = K, is

    score(K_1..K_l) = (1/n) * sum_i [ 1 - exp( sum_j K_j * ln(1 - P[i, j]) ) ]        (paper Eq. 8)

For every K on the grid this script reports, as ratios to the uniform scheme at the same K:
  * single temperature: all K samples at one gamma_j (the best one is chosen in-sample, as in the paper);
  * uniform (Wu et al., 2025): K / l samples at every grid temperature (the baseline, ratio 1);
  * optimal allocation, in-sample: the convex relaxation of Eq. 8 solved by SLSQP with its analytic
    gradient, then an integer search around the rounded optimum;
  * optimal allocation, held-out: leave-one-out (default; the red trace of the paper's figures and
    tables) or k-fold (--folds 10). The allocation is fit on the training questions and scored on the
    held-out ones, then divided by the same in-sample uniform total.

Questions whose peak pass rate is below --min-peak (paper: 1e-3) are excluded first.

Usage:
    python optimize_temperatures.py counts.jsonl                 # leave-one-out
    python optimize_temperatures.py counts.jsonl --folds 10      # 10-fold, folds from default_rng(0)
    python optimize_temperatures.py counts.jsonl --max-k 256 --out results.json

The solver code (SLSQP setup, rounding, folds) is copied from the analysis code behind the paper's figures.
Simplified: P is the measured rate c/n; the paper uses its fitted temperature profiles (see README).
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find passk.py next to this file
from passk import K_GRID, load_counts, pass_rates  # noqa: E402


def log_fail_matrix(P):
    """L[i, j] = ln(1 - P[i, j]), clipped at 1e-15 as in the original code (so P = 1 stays finite)."""
    return np.log(np.clip(1.0 - P, 1e-15, 1.0))


def fail_sum(q, L):
    """sum_i exp(L_i . q): the number of questions expected to stay unsolved under allocation q."""
    return float(np.sum(np.exp(L @ np.asarray(q, dtype=float))))


def solve_relaxed(L, k, q_init):
    """Continuous optimum of min_q sum_i exp(L_i . q) s.t. sum(q) = k, 0 <= q_j <= k (SLSQP).

    Convex (paper App., "Convexity and uniqueness"); warm-started from q_init rescaled to sum k.
    """
    n_t = L.shape[1]
    q0 = np.maximum(np.asarray(q_init, dtype=float), 0.0)
    s = q0.sum()
    q0 = q0 * (k / s) if s > 0 else np.full(n_t, k / n_t)
    res = minimize(
        lambda q: float(np.sum(np.exp(L @ q))), q0,
        jac=lambda q: L.T @ np.exp(L @ q), method="SLSQP",
        bounds=[(0.0, float(k))] * n_t,
        constraints=[{"type": "eq", "fun": lambda q: np.sum(q) - k, "jac": lambda q: np.ones_like(q)}],
        options={"maxiter": 100, "ftol": 1e-12},
    )
    return np.asarray(res.x, dtype=float)


def round_to_sum(q_star, k):
    """Largest-remainder rounding to integers with sum == k and q >= 0 (copied)."""
    base = np.maximum(np.floor(q_star).astype(np.int64), 0)
    deficit = int(k - base.sum())
    frac = q_star - np.floor(q_star)
    if deficit > 0:
        for j in np.argsort(-frac)[:deficit]:
            base[j] += 1
    elif deficit < 0:
        removed = 0
        for j in np.argsort(frac):
            if removed >= -deficit:
                break
            if base[j] > 0:
                base[j] -= 1
                removed += 1
    return base


def greedy_1swap(q, L, max_iter=200):
    """Move one sample from temperature a to b while that lowers the failure sum (vectorized)."""
    q = q.astype(np.int64).copy()
    E, Einv = np.exp(L), np.exp(-L)
    for _ in range(max_iter):
        ev = np.exp(L @ q.astype(float))
        f0 = ev.sum()
        # moved[a, b] = sum_i ev_i * exp(L_ib - L_ia): failure sum after moving one sample a -> b
        moved = (ev[:, None] * Einv).T @ E
        moved[q < 1, :] = np.inf
        np.fill_diagonal(moved, np.inf)
        a, b = np.unravel_index(np.argmin(moved), moved.shape)
        if not moved[a, b] < f0 * (1 - 1e-12):
            break
        q[a] -= 1
        q[b] += 1
    return q


def best_integer_alloc(q_star, k, L, radius=2, max_box=20000):
    """Integer allocation (sum == k) near the continuous optimum q_star.

    Simplified: the paper's code certifies the integer optimum with a KKT-bound branch-and-bound over a
    radius-2 box around q_star. Here we (1) round by largest remainder, (2) enumerate the radius-2 box
    (radius 1 if that box is too large) over the temperatures that carry mass, and (3) polish the best
    point with 1-swap descent over all temperatures.
    """
    best = round_to_sum(q_star, k)
    best_f = fail_sum(best, L)
    active = np.flatnonzero(q_star > 1e-9)
    if len(active) >= 2:
        for r in (radius, 1):
            rng = [np.arange(max(0, math.floor(q_star[j]) - r), min(k, math.ceil(q_star[j]) + r) + 1)
                   for j in active]
            if math.prod(len(x) for x in rng[:-1]) <= max_box:
                # Enumerate all but the last active temperature; the last one takes the remainder.
                head = np.array(list(itertools.product(*rng[:-1])), dtype=np.int64)
                last = k - head.sum(axis=1)
                ok = (last >= rng[-1][0]) & (last <= rng[-1][-1])
                cand = np.zeros((int(ok.sum()), L.shape[1]), dtype=np.int64)
                cand[:, active[:-1]] = head[ok]
                cand[:, active[-1]] = last[ok]
                for s in range(0, len(cand), 4096):
                    f = np.exp(L @ cand[s:s + 4096].T.astype(float)).sum(axis=0)
                    m = int(np.argmin(f))
                    if f[m] < best_f:
                        best, best_f = cand[s + m].copy(), float(f[m])
                break
    best = greedy_1swap(best, L)
    return best, fail_sum(best, L)


def make_folds(n, folds):
    """Leave-one-out if folds <= 0 or folds >= n; else k-fold from default_rng(0) (as in the paper's code)."""
    if folds <= 0 or folds >= n:
        return [np.array([i]) for i in range(n)], "leave-one-out"
    perm = np.random.default_rng(0).permutation(n)
    return [np.asarray(f) for f in np.array_split(perm, folds)], f"{folds}-fold"


def compare_schemes(P, temps, k_values, folds=0):
    """Per K: uniform, every single temperature, optimal in-sample and held-out. Totals are in solved-question units."""
    n, n_t = P.shape
    L = log_fail_matrix(P)
    fold_list, cv_mode = make_folds(n, folds)
    rows = []
    for k in k_values:
        uniform = float(np.sum(1.0 - np.exp(L @ np.full(n_t, k / n_t))))    # uniform scheme
        single = np.sum(1.0 - np.exp(k * L), axis=0)                         # all K at one temperature
        j_best = int(np.argmax(single))                                      # best single T, in-sample
        q_full = solve_relaxed(L, k, np.full(n_t, k / n_t))
        q_int, f_full = best_integer_alloc(q_full, k, L)
        held_out = np.empty(n)                                               # held-out success probabilities
        for fold in fold_list:
            train = np.setdiff1d(np.arange(n), fold, assume_unique=True)
            q_tr = solve_relaxed(L[train], k, q_full)
            q_tr_int, _ = best_integer_alloc(q_tr, k, L[train])
            held_out[fold] = 1.0 - np.exp(L[fold] @ q_tr_int.astype(float))
        rows.append({
            "K": k,
            "uniform_pass_at_k": uniform / n,
            "single_T_ratio": {f"{t:g}": float(s / uniform) for t, s in zip(temps, single)},
            "best_single_T": float(temps[j_best]),
            "best_single_T_ratio": float(single[j_best] / uniform),
            "optimal_in_sample_ratio": float((n - f_full) / uniform),
            "optimal_held_out_ratio": float(held_out.sum() / uniform),
            "optimal_allocation": {f"{t:g}": int(c) for t, c in zip(temps, q_int) if c > 0},
        })
    return {"n_included": n, "temperatures": [float(t) for t in temps], "cv_mode": cv_mode, "per_K": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("counts", help="counts JSONL from a grader: one {id, temperature, n, c} line per cell")
    ap.add_argument("--min-peak", type=float, default=1e-3,
                    help="exclude questions whose peak pass rate over the grid is below this (paper: 1e-3)")
    ap.add_argument("--folds", type=int, default=0,
                    help="0 = leave-one-out (paper's tables and red traces); 10 = 10-fold")
    ap.add_argument("--max-k", type=int, default=1024, help="largest K of the paper's K grid to evaluate")
    ap.add_argument("--out", help="optional JSON output path")
    args = ap.parse_args()

    ids, temps, N, C = load_counts(args.counts)
    P = pass_rates(N, C)
    keep = P.max(axis=1) >= args.min_peak
    P = P[keep]
    k_values = [k for k in K_GRID if k <= args.max_k]
    print(f"{int(keep.sum())} of {len(ids)} questions kept (peak pass rate >= {args.min_peak:g}); "
          f"{len(temps)} temperatures")
    res = compare_schemes(P, temps, k_values, args.folds)
    res["n_total"] = len(ids)
    print(f"ratios to the uniform scheme; held-out = {res['cv_mode']}")
    print(f"{'K':>5} {'uniform':>8} {'best T':>7} {'ratio':>7} {'opt in':>7} {'opt out':>8}  allocation (T: samples)")
    for r in res["per_K"]:
        alloc = ", ".join(f"{t}: {c}" for t, c in r["optimal_allocation"].items())
        print(f"{r['K']:>5} {r['uniform_pass_at_k']:8.4f} {r['best_single_T']:7g} {r['best_single_T_ratio']:7.4f} "
              f"{r['optimal_in_sample_ratio']:7.4f} {r['optimal_held_out_ratio']:8.4f}  {alloc}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
