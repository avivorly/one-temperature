"""Paired bootstrap bands on the ratio to uniform (Appendix F, "Bootstrap error bars").

The question set is resampled with replacement B times. In every resample the pass@K of each
setting and of the uniform scheme are recomputed on the same resampled questions, and their ratio
is formed; the band of a curve is +/- the standard deviation of that ratio over the resamples
(ddof = 1), drawn around the full-data ratio. Pairing (one resample for every setting and the
baseline) keeps each question's difficulty on both sides of the ratio.

Two implementations draw the same resamples (one ``rng.integers(0, n, size=n)`` per replicate, in
order) and agree to rounding:

* ``method="paper"``: the paper's loop (``_compute_ratio_to_uniform``): index, exponentiate, mean;
* ``method="fast"`` (default): the count-then-matmul identity of Appendix F, sum over the n slots
  of f(question) = sum_i f_i c_{b,i} with c_{b,i} the number of draws of question i in resample b,
  one matrix product per chunk of replicates.

Seeds. The paper's LiveCodeBench figures drew from fresh OS entropy (not reproducible). Its MATH-500,
Hi-ToM and min_p figures seeded the generator as ``default_rng(SeedSequence(root).spawn(1)[0])``
with root 33273348774934605029837111157444119385 ("spawn"); ``seed_scheme="plain"`` is
``default_rng(seed)``.

Command line::

    python -m onetemp.core.bootstrap --profiles profiles.json --B 10000 --seed 0 --out rtu.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from onetemp.core.passk import k_grid

SEED_SCHEMES = ("plain", "spawn", "entropy")


def make_rng(seed: Optional[int] = 0, scheme: str = "plain") -> np.random.Generator:
    """plain: default_rng(seed); spawn: default_rng(SeedSequence(seed).spawn(1)[0]), the scheme of
    the paper's seeded figures; entropy: default_rng() (fresh entropy, as the LiveCodeBench figures)."""
    if scheme == "plain":
        return np.random.default_rng(seed)
    if scheme == "spawn":
        return np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(1)[0])
    if scheme == "entropy":
        return np.random.default_rng()
    raise ValueError(f"seed scheme must be one of {SEED_SCHEMES}")


def _mu(L: np.ndarray, L_uni: np.ndarray, K_arr: np.ndarray, idx: np.ndarray):
    """Per-setting and uniform pass@K (means over the questions idx), as the paper computes them."""
    LT = L[idx]
    LU = L_uni[idx]
    exp_T = np.exp(K_arr[:, None, None] * LT[None, :, :])
    mu_T = 1.0 - exp_T.mean(axis=1)               # (n_k, n_settings)
    exp_U = np.exp(K_arr[:, None] * LU[None, :])
    mu_U = 1.0 - exp_U.mean(axis=1)               # (n_k,)
    return mu_T, mu_U


def ratio_to_uniform(L: np.ndarray, k_values=None, B: int = 10000, seed: Optional[int] = 0,
                     seed_scheme: str = "plain", method: str = "fast", percentiles=(2.5, 97.5),
                     chunk: int = 500) -> dict:
    """Ratio of each single-setting pass@K to the uniform scheme's, with paired-bootstrap SE.

    L: (n, l) log(1 - P) of the included questions. Uniform uses log(1 - P) averaged over the l
    settings (K / l samples at each). Returns ratio[j][k], ratio_se[j][k] (and percentile bands),
    indexed setting-major like the paper's output.
    """
    L = np.asarray(L, dtype=np.float64)
    n, n_t = L.shape
    k_values = list(k_values) if k_values is not None else k_grid(1024)
    K_arr = np.array(k_values, dtype=np.float64)
    n_k = len(k_values)
    L_uni = L.mean(axis=1)
    mu_T_full, mu_U_full = _mu(L, L_uni, K_arr, np.arange(n))
    ratio_central = mu_T_full / mu_U_full[:, None]
    rng = make_rng(seed, seed_scheme)
    ratios_b = np.empty((B, n_k, n_t))
    if method == "paper":
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            mu_T_b, mu_U_b = _mu(L, L_uni, K_arr, idx)
            ratios_b[b] = mu_T_b / mu_U_b[:, None]
    elif method == "fast":
        E_T = np.exp(K_arr[None, :, None] * L[:, None, :]).reshape(n, n_k * n_t)   # (n, n_k * n_t)
        E_U = np.exp(K_arr[None, :] * L_uni[:, None])                               # (n, n_k)
        for start in range(0, B, chunk):
            stop = min(B, start + chunk)
            counts = np.empty((stop - start, n))
            for r in range(stop - start):
                counts[r] = np.bincount(rng.integers(0, n, size=n), minlength=n)
            mu_T_b = 1.0 - (counts @ E_T).reshape(-1, n_k, n_t) / n
            mu_U_b = 1.0 - (counts @ E_U) / n
            ratios_b[start:stop] = mu_T_b / mu_U_b[:, :, None]
    else:
        raise ValueError("method must be 'fast' or 'paper'")
    ratio_se = ratios_b.std(axis=0, ddof=1)
    out = {"k": [int(k) for k in k_values], "B": int(B), "n_included": int(n), "seed": seed,
           "seed_scheme": seed_scheme, "method": method,
           "ratio": ratio_central.T.tolist(), "ratio_se": ratio_se.T.tolist(),
           "pass_at_k": mu_T_full.T.tolist(), "pass_at_k_uniform": mu_U_full.tolist()}
    if percentiles:
        lo, hi = np.percentile(ratios_b, list(percentiles), axis=0)
        out["percentiles"] = list(percentiles)
        out["ratio_pct_lo"] = lo.T.tolist()
        out["ratio_pct_hi"] = hi.T.tolist()
    return out


def bootstrap_ratio(num_per_q: np.ndarray, den_per_q: np.ndarray, B: int = 10000, seed: Optional[int] = 0,
                    seed_scheme: str = "plain", percentiles=(2.5, 97.5)) -> dict:
    """Paired bootstrap of sum(num) / sum(den) for per-question scores of shape (n,) or (n, n_k),
    e.g. the held-out optimal allocation against the uniform scheme."""
    num = np.asarray(num_per_q, dtype=np.float64)
    den = np.asarray(den_per_q, dtype=np.float64)
    if num.ndim == 1:
        num, den = num[:, None], den[:, None]
    n = num.shape[0]
    rng = make_rng(seed, seed_scheme)
    counts = np.empty((B, n))
    for b in range(B):
        counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    r = (counts @ num) / (counts @ den)
    out = {"ratio": (num.sum(axis=0) / den.sum(axis=0)).tolist(), "se": r.std(axis=0, ddof=1).tolist(), "B": B}
    if percentiles:
        lo, hi = np.percentile(r, list(percentiles), axis=0)
        out["pct_lo"], out["pct_hi"] = lo.tolist(), hi.tolist()
    return out


def main(argv=None) -> int:
    from onetemp.core.fit_all import MIN_PEAK_PCT, load_profiles
    ap = argparse.ArgumentParser(description="Ratio to uniform with paired-bootstrap bands (Appendix F).")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--mode", default="fit", choices=("fit", "meas"))
    ap.add_argument("--min-peak", type=float, default=MIN_PEAK_PCT)
    ap.add_argument("--max-k", type=int, default=1024)
    ap.add_argument("--B", type=int, default=10000, help="bootstrap replicates (paper's figures: 10000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-scheme", default="plain", choices=SEED_SCHEMES)
    ap.add_argument("--method", default="fast", choices=("fast", "paper"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prof = load_profiles(a.profiles)
    inc = prof.included(a.min_peak, a.mode)
    res = ratio_to_uniform(prof.L(a.mode)[inc], k_grid(a.max_k), a.B, a.seed, a.seed_scheme, a.method)
    res["temps"] = prof.alloc_settings
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f)
    print(f"[bootstrap] n={res['n_included']} B={a.B} settings={len(prof.alloc_settings)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
