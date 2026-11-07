"""pass@K (Section 2.2) and the quantities built on it. The package's one definition of each estimator.

Two estimators of a question's pass@K at one setting:

* ``passk_unbiased(n, c, K)``: the unbiased estimator 1 - C(n - c, K) / C(n, K) of Chen et al. (2021) from
  c correct out of n samples, defined for K <= n. It is computed exactly in integers and rounded once to a
  double, the form the paper's min_p analyses used (Appendix I). ``unbiased_log_fail(n, c, K)`` is its
  companion, log C(n - c, K) / C(n, K), the estimator's chance that all K draws fail.
* ``passk_plugin(P, K)``: the plug-in 1 - (1 - P)^K (Eq. 2) for a single-sample success probability P. The
  paper evaluates every strategy this way on the fitted profiles P (Appendix F). The paper's code used two
  floating-point forms of it, and each analysis keeps its own so that every printed number reproduces:
  ``form="exp"``, 1 - exp(K L) with L = log(clip(1 - P, 1e-15, 1)) (Appendices F and G), and
  ``form="expm1"``, -expm1(K log1p(-P)) (Appendix H). The two agree to within a few units in the last place.

For an allocation K_1..K_l over the settings, a question's failure probability is
prod_j (1 - P_j)^(K_j) = exp(sum_j K_j log(1 - P_j)) (Eq. 7), so with L = log(1 - P) (one row per
question, one column per setting) every strategy is scored from L alone:

* one setting j at budget K:           1 - exp(K L[:, j])            passk_plugin(L=L[:, j], k=K)
* allocation q (sum q = K):            1 - exp(L @ q)                passk_plugin(L=L, k=q)
* the uniform scheme (Wu et al.):      1 - exp((K / l) sum_j L[:, j]), with K / l samples at every
  one of the l settings, fractional when l does not divide K, as in the paper's code.

The allocation solver and the table code sum these expected failures over questions directly
(n - sum exp(L q)), in the paper's order of operations (``allocation.py``, ``tables.py``, ``bootstrap.py``).

Command line (pass@K of every setting from counts, with the unbiased estimator)::

    python -m onetemp.core.passk --counts counts.jsonl --benchmark math500 --model Qwen/Qwen3-8B --k 1 8 64
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from functools import lru_cache

import numpy as np

# The K grid of the paper's figures and held-out curves (ratio to uniform, LOO); the paper uses it
# up to K_max = 1024 (Appendix D, "Maximum sample budget").
PAPER_K_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
                1536, 2048]
K_MAX = 1024
# Rows of the pass@K tables (Tables 2-6) and columns of the summary table (Appendix F).
TABLE_K = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
SUMMARY_K = [1, 16, 256, 1024]
LOG1M_FLOOR = 1e-15        # 1 - P is floored at 1e-15 before the log, as in the paper's code


def k_grid(max_k: int = K_MAX) -> list:
    """The paper's K grid up to max_k."""
    return [k for k in PAPER_K_GRID if k <= max_k]


# ───────────────────────── The unbiased estimator ─────────────────────────

@lru_cache(maxsize=None)
def _unbiased_exact(n: int, c: int, k: int) -> float:
    if k > n:
        return math.nan
    return float(1 - Fraction(math.comb(n - c, k), math.comb(n, k)))


def passk_unbiased(n, c, k: int):
    """Unbiased pass@k = 1 - C(n - c, k) / C(n, k), elementwise over n and c; NaN where k > n.

    Exact: the ratio of the two binomial coefficients is formed in integers and rounded once to a double
    (1 when n - c < k, since every k-subset then contains a correct sample). Values are cached per (n, c, k).
    Scalars in, a float out; arrays in, an array out. k is one integer >= 0.
    """
    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0")
    if np.ndim(n) == 0 and np.ndim(c) == 0:
        return _unbiased_exact(int(n), int(c), k)
    n_arr, c_arr = np.broadcast_arrays(np.asarray(n), np.asarray(c))
    out = np.empty(n_arr.shape, dtype=np.float64)
    for idx in np.ndindex(n_arr.shape):
        nn, cc = int(n_arr[idx]), int(c_arr[idx])
        if nn < 0 or not 0 <= cc <= nn:
            raise ValueError(f"need 0 <= c <= n, got n={nn} c={cc}")
        out[idx] = _unbiased_exact(nn, cc, k)
    return out


@lru_cache(maxsize=None)
def unbiased_log_fail(n: int, c: int, k: int) -> float:
    """log of C(n - c, k) / C(n, k), the unbiased estimator's chance that all k draws fail; -inf when n - c < k."""
    if n - c < k:
        return -math.inf
    return math.log(math.comb(n - c, k)) - math.log(math.comb(n, k))


# ───────────────────────── The plug-in estimator ─────────────────────────

def log1m(P, floor: float = LOG1M_FLOOR):
    """L = log(1 - P), with 1 - P floored at 1e-15 as in the paper's code."""
    return np.log(np.clip(1.0 - np.asarray(P, dtype=np.float64), floor, 1.0))


def passk_plugin(P=None, k=1, *, L=None, form: str = "exp"):
    """Plug-in pass@k = 1 - (1 - P)^k (Eq. 2), or 1 - prod_j (1 - P_j)^(q_j) for an allocation q (Eq. 7).

    P: single-sample success probabilities; or give L = log(1 - P) instead (form "exp").
    k: the budget, a number (any real > 0, broadcast against P or L), or for form "exp" an allocation q,
       a vector with one budget per column of L.
    form "exp":   1 - exp(k L), with L = log1m(P) when not given (the paper's Appendices F and G);
    form "expm1": -expm1(k log1p(-P)) (the paper's Appendix H; needs P).
    """
    if form == "expm1":
        if P is None:
            raise ValueError("form 'expm1' needs P")
        return -np.expm1(k * np.log1p(-P))
    if form != "exp":
        raise ValueError(f"form must be 'exp' or 'expm1', not {form!r}")
    if L is None:
        if P is None:
            raise ValueError("give P or L")
        L = log1m(P)
    if np.ndim(k) == 1:
        return 1.0 - np.exp(L @ np.asarray(k, dtype=np.float64))
    return 1.0 - np.exp(k * L)


def solved_single(L: np.ndarray, k) -> np.ndarray:
    """Expected number of solved questions with all k samples at one setting: one value per
    setting (sum over questions of 1 - (1 - P)^k)."""
    return np.sum(passk_plugin(L=L, k=float(k)), axis=0)


def solved_uniform(L: np.ndarray, k) -> float:
    """Expected number solved by the uniform scheme: k / l samples at each of the l settings."""
    n_q, n_s = L.shape
    return float(n_q - np.sum(np.exp((float(k) / n_s) * np.sum(L, axis=1))))


def solved_allocation(L: np.ndarray, q) -> float:
    """Expected number solved by allocation q (q_j samples at setting j)."""
    q = np.asarray(q, dtype=np.float64)
    return float(L.shape[0] - np.sum(np.exp(L @ q)))


def per_question_allocation(L: np.ndarray, q) -> np.ndarray:
    """Per-question pass probability of allocation q: 1 - exp(L_i . q)."""
    return passk_plugin(L=L, k=np.asarray(q, dtype=np.float64))


def main(argv=None) -> int:
    from onetemp.core.counts_io import AXES, load_counts
    ap = argparse.ArgumentParser(description="Unbiased pass@K of every setting, averaged over questions.")
    ap.add_argument("--counts", required=True)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--axis", default="temperature", choices=AXES)
    ap.add_argument("--fixed-temperature", type=float, default=None)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    a = ap.parse_args(argv)
    cnt = load_counts(a.counts, a.benchmark, a.model, a.axis, fixed_temperature=a.fixed_temperature)
    out = {"benchmark": cnt.benchmark, "model": cnt.model, "axis": cnt.axis, "settings": cnt.settings,
           "questions": cnt.n_questions, "estimator": "unbiased 1 - C(n-c,K)/C(n,K)", "pass_at_k": {}}
    for k in a.k:
        v = passk_unbiased(cnt.n, cnt.c, k)
        v = np.where(cnt.measured, v, np.nan)
        out["pass_at_k"][str(k)] = [None if np.isnan(x) else round(float(x), 6) for x in np.nanmean(v, axis=0)]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
