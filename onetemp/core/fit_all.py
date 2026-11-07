"""Per-question profile fit of the paper (Appendix D, "Question Profile Analyses").

For each question we observe c_j correct samples out of n_j at each setting x_j (a temperature,
or a min_p value at a fixed temperature). The fit recovers a smooth profile P(x):

1. Clopper-Pearson 1-sigma intervals per point (alpha = 1 - erf(1/sqrt 2) ~ 0.317), kept
   asymmetric: sigma_lo = rate - lo, sigma_hi = hi - rate, both clamped to >= 1e-6 percent.
2. Domain bracket [x_lo, x_hi]: from the first to the last setting with at least one correct
   sample, padded by one grid step on each side; P = 0 outside the bracket.
3. Basis: Bernstein polynomials of degree 10 on the bracket.
4. Loss: weighted least squares with side-dependent variance, residual / sigma_+ when the fit is
   above the data and / sigma_- otherwise (Appendix D, "Loss").
5. Shape constraints: non-negative, unimodal Bernstein coefficients, enforced by projection onto
   the cone of non-negative unimodal sequences (isotonic regression around each candidate peak,
   the closest projection in L2 is kept) every 20 iterations, soft-blended over the first 5% of
   iterations and hard afterwards.
6. Optimisation: projected gradient descent from zero, 50,000 iterations, step 0.99 / L with L the
   largest eigenvalue of the weighted Gram matrix (power iteration).
7. The Bernstein coefficients are converted to Legendre coefficients for evaluation.

Then, as the paper's analysis code does for every dataset:

* P(x) is evaluated at the allocation grid, divided by 100 and clipped to [0, 1]. For the
  temperature axis the allocation grid drops T = 0 (greedy decoding); for min_p every value stays.
* The peak success probability is the maximum of the fitted curve on a 1001-point grid over the
  bracket. A question is included when that peak is >= 0.1 percent, i.e. 1e-3 (Appendix D,
  "Minimum peak success probability"); questions with no correct sample have peak 0.

The gradient-descent kernel runs in C (``bernstein_kernel.c``, the paper's kernel, unchanged) or in
numpy. The numpy fallback repeats the C loops operation by operation (same order of every sum,
same comparisons), so it returns the same coefficients as the C kernel built without fused
multiply-add, ``python -m onetemp.core.fit_all --build-kernel``: bit for bit in our tests (docs/core.md),
whenever the iterates stay finite, which they do for any counts (the data are bounded and the step
is 0.99 / L). Outside that range the C kernel can return an unwritten buffer when every candidate
distance of its projection is infinite; numpy returns zeros. The paper's own build used
``-march=native``, which allows fused multiply-add; its results differ from ours in the last
digits only (``--build-flags paper`` reproduces them on x86-64).

Command line::

    python -m onetemp.core.fit_all --build-kernel                       # optional: compile the C kernel
    python -m onetemp.core.fit_all --counts counts.jsonl --benchmark livecodebench \
        --model mistralai/Mistral-7B-Instruct-v0.3 --out profiles.json --out-rows profile_rows.jsonl

``profiles.json`` holds one entry per question (bracket, Legendre and Bernstein coefficients in
percent, fit quality, peak and efficient range, P on the allocation grid); ``--out-rows`` writes
each counts.jsonl row with two more fields, ``p_fit`` and ``peak_pct`` (see ``profile_rows``).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from math import comb, lgamma
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist

from onetemp.core.counts_io import AXES, N_MODES, Counts, load_counts

# ───────────────────────── Constants (the paper's values) ─────────────────────────

CP_ALPHA: float = 0.31731050786291415      # 1-sigma two-sided

DEGREE = 10
ITERATIONS = 50000
PROJ_EVERY = 20
RAMP_FRAC = 0.05

EPS_PCT = 1e-6   # percent; clamp for sigma_lo / sigma_hi at k=0 / k=n

MIN_PEAK_PCT = 0.1          # inclusion: fitted peak >= 0.1 percent (= 1e-3)
FWHM_POINTS = 1000          # the peak and half-maximum are found on 1001 grid points
LOG1M_FLOOR = 1e-15         # log(1 - P) is taken of max(1 - P, 1e-15)

# C kernel limits (fixed stack arrays in bernstein_kernel.c)
_C_MAX_M = 32
_C_MAX_N = 64

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_SOURCE = os.path.join(_HERE, "bernstein_kernel.c")
KERNEL_ENV = "ONETEMP_BERNSTEIN_KERNEL"


def _default_kernel_path() -> str:
    ext = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
    return os.path.join(_HERE, "_bernstein_kernel" + ext)


# Build flags. "portable" (the default) disables fused multiply-add so that the numpy fallback
# reproduces the C kernel bit for bit. "paper" is the paper's own build line.
BUILD_FLAGS = {
    "portable": ["-O3", "-fno-math-errno", "-ffp-contract=off"],
    "paper": ["-O3", "-funroll-loops", "-fno-math-errno", "-march=native", "-flto"],
}


# ───────────────────────── Data types ─────────────────────────

@dataclass
class DataPoint:
    x: float          # setting (temperature or min_p)
    y: float          # pass rate in percent (0-100)
    passed: int       # raw pass count
    n: int            # sample count
    sigma_lo: float   # rate - lo_ci, in percent (clamped to >= EPS_PCT)
    sigma_hi: float   # hi_ci - rate, in percent (clamped to >= EPS_PCT)


# ───────────────────────── CP errors ─────────────────────────

def cp_interval(passed: int, n: int) -> tuple:
    """Exact Clopper-Pearson 1-sigma interval, in percent (Appendix D, Eq. for lo and hi)."""
    lo = beta_dist.ppf(CP_ALPHA / 2, passed, n - passed + 1) if passed > 0 else 0.0
    hi = beta_dist.ppf(1 - CP_ALPHA / 2, passed + 1, n - passed) if passed < n else 1.0
    return lo * 100.0, hi * 100.0


# ───────────────────────── Legendre <-> Bernstein ─────────────────────────

def _log_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def build_tlb(degree: int) -> np.ndarray:
    """Matrix taking Legendre coefficients to Bernstein coefficients on [-1, 1]."""
    m = degree + 1
    TLB = np.zeros((m, m))
    lb = np.zeros((m, m))
    for nn in range(m):
        for kk in range(nn + 1):
            lb[nn, kk] = _log_binom(nn, kk)
    for i in range(m):
        log_ci = lb[degree, i]
        for k in range(m):
            j_lo = max(0, i + k - degree)
            j_hi = min(i, k)
            v = 0.0
            for jj in range(j_lo, j_hi + 1):
                sign = 1.0 if ((k + jj) % 2 == 0) else -1.0
                v += sign * np.exp(2.0 * lb[k, jj] + lb[degree - k, i - jj] - log_ci)
            TLB[i, k] = v
    return TLB


def build_bernstein_features(pts: list, degree: int, x_lo: float, x_hi: float) -> np.ndarray:
    """B[i, j] = binom(d, j) u_i^j (1 - u_i)^(d - j), u_i the point mapped to [0, 1]."""
    n = len(pts)
    m = degree + 1
    B = np.zeros((n, m))
    span = x_hi - x_lo
    if span <= 0:
        return B
    for idx, p in enumerate(pts):
        u = max(0.0, min(1.0, (p.x - x_lo) / span))
        for j in range(m):
            B[idx, j] = comb(degree, j) * (u ** j) * ((1 - u) ** (degree - j))
    return B


# ───────────────────────── Lipschitz constant ─────────────────────────

def lipschitz_constant(B: np.ndarray, wts: np.ndarray) -> float:
    """Power-iteration largest eigenvalue of M = (2/n) B^T diag(w) B (seed 42, 300 steps).

    Pass ``w_max = max(w_lo, w_hi)`` (after boundary tightening): a conservative per-point bound on
    the curvature, since the side weight selected at any iteration is <= w_max.
    """
    n, m = B.shape
    Bw = B * wts[:, None]
    M = (2.0 / n) * (B.T @ Bw)
    rng = np.random.default_rng(42)
    v = rng.standard_normal(m)
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(300):
        u = M @ v
        s = np.linalg.norm(u)
        if s < 1e-15:
            break
        lam = s
        v = u / s
    return float(lam)


# ───────────────────────── Legendre evaluation ─────────────────────────

def poly_eval(coeffs: np.ndarray, x: float, x_lo: float, x_hi: float) -> float:
    if x_hi == x_lo:
        return 0.0
    t = 2.0 * (x - x_lo) / (x_hi - x_lo) - 1.0
    m = len(coeffs)
    if m == 0:
        return 0.0
    p0 = 1.0
    p1 = t
    s = float(coeffs[0])
    if m > 1:
        s += float(coeffs[1]) * t
    for k in range(2, m):
        p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k
        s += float(coeffs[k]) * p2
        p0, p1 = p1, p2
    return s


def eval_legendre(coeffs, x_lo, x_hi, t):
    """The analysis code's evaluator: None outside [x_lo, x_hi], else the degree-10 Legendre sum."""
    if t < x_lo or t > x_hi:
        return None
    u = 2.0 * (t - x_lo) / (x_hi - x_lo) - 1.0
    P = [1.0, u]
    for k in range(2, 11):
        P.append(((2 * k - 1) * u * P[k - 1] - (k - 1) * P[k - 2]) / k)
    return sum(c * p for c, p in zip(coeffs, P))


def eval_legendre_array(coeffs, x_lo: float, x_hi: float, ts: np.ndarray):
    """Vectorised ``eval_legendre``: the same operations in the same order, so the same values.

    The scalar evaluator adds its 11 terms with Python's built-in ``sum``, which from Python 3.12
    on is Neumaier-compensated; this function repeats that algorithm exactly (CPython's
    ``builtin_sum`` float path), so it equals the paper's evaluator as run under Python >= 3.12
    for every input. Under Python 3.10/3.11 the paper's scalar code sums naively and can differ in
    the last bit of P (relative 1e-16), which moves no printed number.

    Returns (values, inside) where ``inside`` is False where ``eval_legendre`` returns None.
    """
    ts = np.asarray(ts, dtype=np.float64)
    inside = ~((ts < x_lo) | (ts > x_hi))
    if not x_hi > x_lo:
        # an empty bracket (a dataset with one setting): the zero profile; the scalar
        # evaluator would divide by zero here
        return np.zeros_like(ts), inside
    with np.errstate(divide="ignore", invalid="ignore"):
        u = 2.0 * (ts - x_lo) / (x_hi - x_lo) - 1.0
        P = [np.ones_like(u), u]
        for k in range(2, 11):
            P.append(((2 * k - 1) * u * P[k - 1] - (k - 1) * P[k - 2]) / k)
        terms = [c * p for c, p in zip(coeffs, P)]
        f = terms[0] + 0.0            # sum() starts from int 0: 0 + x0
        comp = np.zeros_like(u)
        for x in terms[1:]:
            t = f + x
            comp = comp + np.where(np.abs(f) >= np.abs(x), (f - t) + x, (x - t) + f)
            f = t
        s = np.where((comp != 0) & np.isfinite(comp), f + comp, f)
    return np.where(inside, s, 0.0), inside


# ───────────────────────── Data preparation ─────────────────────────

def prepare_question(name: str, n_samples, passed_values: list, temps: list):
    """Convert one question into DataPoints and its [x_lo, x_hi] bracket.

    n_samples is either a scalar (one budget for every setting) or a per-setting sequence aligned
    with passed_values/temps. A setting with n = 0 was never measured and is dropped rather than
    treated as a measured zero. CP half-widths are kept asymmetric and clamped to >= EPS_PCT.
    """
    if isinstance(n_samples, (int, float)):
        n_at = [int(n_samples)] * len(temps)
    else:
        n_at = [int(v) for v in n_samples]

    non_zero = [i for i, p in enumerate(passed_values) if p > 0]
    if not non_zero:
        x_lo, x_hi = temps[0], temps[-1]
    else:
        lo_idx = min(non_zero)
        hi_idx = max(non_zero)
        if lo_idx > 0:
            lo_idx -= 1
        if hi_idx < len(temps) - 1:
            hi_idx += 1
        x_lo, x_hi = temps[lo_idx], temps[hi_idx]

    pts: list = []
    for i, t in enumerate(temps):
        if t < x_lo or t > x_hi:
            continue
        n_i = n_at[i]
        if n_i <= 0:
            continue
        passed = passed_values[i]
        rate = (passed / n_i) * 100.0
        lo_ci, hi_ci = cp_interval(passed, n_i)
        sigma_lo = max(EPS_PCT, rate - lo_ci)
        sigma_hi = max(EPS_PCT, hi_ci - rate)
        pts.append(DataPoint(x=t, y=rate, passed=passed, n=n_i, sigma_lo=sigma_lo, sigma_hi=sigma_hi))
    return pts, x_lo, x_hi


# ───────────────────────── C kernel: build and load ─────────────────────────

def build_kernel(out_path: Optional[str] = None, flags: str = "portable", cc: Optional[str] = None,
                 verbose: bool = True) -> str:
    """Compile bernstein_kernel.c into a shared library and return its path."""
    out_path = out_path or _default_kernel_path()
    cc = cc or os.environ.get("CC", "gcc")
    if flags not in BUILD_FLAGS:
        raise ValueError(f"flags must be one of {sorted(BUILD_FLAGS)}")
    cmd = [cc, *BUILD_FLAGS[flags], "-shared"]
    if platform.system() != "Windows":
        cmd.append("-fPIC")
    cmd += ["-o", out_path, KERNEL_SOURCE]
    if verbose:
        print("[build] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_path


_c_double_p = ctypes.POINTER(ctypes.c_double)
_KERNEL_CACHE: dict = {}


def load_kernel(path: Optional[str] = None):
    """Load the compiled kernel. Search: ``path``, then $ONETEMP_BERNSTEIN_KERNEL, then the default
    location next to this file. Returns (ctypes library, path) or (None, None) if none is found."""
    candidates = [p for p in (path, os.environ.get(KERNEL_ENV), _default_kernel_path()) if p]
    for p in candidates:
        if not os.path.exists(p):
            if p == path:
                raise FileNotFoundError(f"kernel library not found: {p}")
            continue
        p = os.path.abspath(p)
        if p not in _KERNEL_CACHE:
            lib = ctypes.CDLL(p)
            lib.bernstein_gd.restype = None
            lib.bernstein_gd.argtypes = [
                _c_double_p, _c_double_p, _c_double_p, _c_double_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int, ctypes.c_int,
                _c_double_p,
            ]
            _KERNEL_CACHE[p] = lib
        return _KERNEL_CACHE[p], p
    return None, None


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_c_double_p)


def run_c_kernel(lib, B: np.ndarray, w_lo: np.ndarray, w_hi: np.ndarray, y: np.ndarray,
                 lr: float, iterations: int, ramp_iters: int, proj_every: int) -> np.ndarray:
    n, m = B.shape
    if m > _C_MAX_M or n > _C_MAX_N:
        raise ValueError(f"C kernel holds at most {_C_MAX_N} points and {_C_MAX_M} coefficients")
    B_c = np.ascontiguousarray(B, dtype=np.float64)
    w_lo_c = np.ascontiguousarray(w_lo, dtype=np.float64)
    w_hi_c = np.ascontiguousarray(w_hi, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    bc_out = np.zeros(m, dtype=np.float64)
    lib.bernstein_gd(_ptr(B_c), _ptr(w_lo_c), _ptr(w_hi_c), _ptr(y_c),
                     ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(iterations),
                     ctypes.c_double(lr), ctypes.c_int(ramp_iters), ctypes.c_int(proj_every),
                     _ptr(bc_out))
    return bc_out


# ───────────────────────── numpy fallback of the C kernel ─────────────────────────

def _pava_batch(vals: np.ndarray, nondecreasing: bool) -> np.ndarray:
    """Row-wise replica of the kernel's pava_nondec / pava_noninc (pool-adjacent-violators).

    Each row is processed exactly as the C code does: repeated left-to-right passes that merge a
    violating block into the current block, until a pass merges nothing; then every position takes
    its block's mean sums/cnts. A pass that merges nothing changes nothing, so rows that finish
    early can share the remaining passes of the other rows.
    """
    R, L = vals.shape
    out = vals.copy()
    if L <= 1 or R == 0:
        return out
    sums = vals.copy()
    cnts = np.ones((R, L))
    nb = np.full(R, L, dtype=np.int64)
    rows = np.arange(R)
    while True:
        merged = False
        w = np.ones(R, dtype=np.int64)
        for bi in range(1, L):
            act = bi < nb
            if not act.any():
                break
            wm1 = w - 1
            prev_mean = sums[rows, wm1] / cnts[rows, wm1]
            cur_mean = sums[:, bi] / cnts[:, bi]
            viol = (prev_mean > cur_mean) if nondecreasing else (prev_mean < cur_mean)
            mrg = act & viol
            keep = act & ~viol
            if mrg.any():
                r = rows[mrg]
                sums[r, wm1[mrg]] = sums[r, wm1[mrg]] + sums[r, bi]
                cnts[r, wm1[mrg]] = cnts[r, wm1[mrg]] + cnts[r, bi]
                merged = True
            if keep.any():
                r = rows[keep]
                sums[r, w[keep]] = sums[r, bi]
                cnts[r, w[keep]] = cnts[r, bi]
                w[keep] += 1
        nb = w
        if not merged:
            break
    # Expand: block b covers int(cnts[b] + 0.5) positions, value sums[b] / cnts[b].
    cnt_int = (cnts + 0.5).astype(np.int64)
    blk_valid = np.arange(L)[None, :] < nb[:, None]
    cum = np.cumsum(np.where(blk_valid, cnt_int, 0), axis=1)            # (R, L) end of each block
    pos = np.arange(L)
    block_of = np.sum((cum[:, None, :] <= pos[None, :, None]) & blk_valid[:, None, :], axis=2)
    avg = sums / cnts
    out = avg[rows[:, None], np.minimum(block_of, L - 1)]
    return out


def _project_unimodal_nonneg_batch(b: np.ndarray) -> np.ndarray:
    """Row-wise replica of project_unimodal_nonneg: closest non-negative unimodal sequence (L2)."""
    Q, m = b.shape
    if m <= 1:
        return np.where(b > 0.0, b, 0.0)
    d = m - 1
    bp = np.where(b > 0.0, b, 0.0)
    best = np.zeros_like(b)
    best_dist = np.full(Q, np.inf)
    for peak in range(d + 1):
        cand = bp.copy()
        if peak > 0:
            cand[:, :peak + 1] = _pava_batch(cand[:, :peak + 1], nondecreasing=True)
        if peak < d:
            cand[:, peak:] = _pava_batch(cand[:, peak:], nondecreasing=False)
        cand = np.where(cand < 0.0, 0.0, cand)
        dist = np.zeros(Q)
        for i in range(m):
            diff = cand[:, i] - b[:, i]
            dist = dist + diff * diff
        upd = dist < best_dist
        if upd.any():
            best[upd] = cand[upd]
            best_dist[upd] = dist[upd]
    return best


def run_numpy_kernel(B_list: list, w_lo_list: list, w_hi_list: list, y_list: list, lr_list: list,
                     iterations: int, ramp_iters: int, proj_every: int) -> np.ndarray:
    """Projected gradient descent of bernstein_gd for many questions at once.

    Questions have different numbers of points; they are zero-padded at the end, and a padded
    point adds exactly zero to every sum, so each question follows its own C computation.
    """
    Q = len(B_list)
    m = B_list[0].shape[1]
    N = max(b.shape[0] for b in B_list)
    B = np.zeros((Q, N, m))
    wl = np.zeros((Q, N))
    wh = np.zeros((Q, N))
    y = np.zeros((Q, N))
    npts = np.zeros(Q)
    for q in range(Q):
        k = B_list[q].shape[0]
        B[q, :k] = B_list[q]
        wl[q, :k] = w_lo_list[q]
        wh[q, :k] = w_hi_list[q]
        y[q, :k] = y_list[q]
        npts[q] = k
    lr = np.asarray(lr_list, dtype=np.float64)[:, None]
    inv_n_times_2 = (2.0 / npts)[:, None]
    Bj = [np.ascontiguousarray(B[:, :, j]) for j in range(m)]     # (Q, N) per coefficient
    Bi = [np.ascontiguousarray(B[:, i, :]) for i in range(N)]     # (Q, m) per point
    bc = np.zeros((Q, m))
    for it in range(iterations):
        pred = np.zeros((Q, N))
        for j in range(m):
            pred = pred + Bj[j] * bc[:, j:j + 1]
        w = np.where(pred > y, wh, wl)
        residual = w * (y - pred)
        s = np.zeros((Q, m))
        for i in range(N):
            s = s + Bi[i] * residual[:, i:i + 1]
        grad = -inv_n_times_2 * s
        bc = bc - lr * grad
        if ramp_iters > 0:
            ramp = it / ramp_iters
            if ramp > 1.0:
                ramp = 1.0
        else:
            ramp = 1.0
        if (it % proj_every == 0) or (it == iterations - 1):
            bc_clamped = np.where(bc > 0.0, bc, 0.0)
            bc_uni = _project_unimodal_nonneg_batch(bc_clamped)
            bc = bc + ramp * (bc_uni - bc)
    bc_clamped = np.where(bc > 0.0, bc, 0.0)
    return _project_unimodal_nonneg_batch(bc_clamped)


# ───────────────────────── Fit pipeline ─────────────────────────

def fit_inputs(pts: list, x_lo: float, x_hi: float, degree: int = DEGREE):
    """Design matrix, side weights, targets and step size for one question (as fit_question)."""
    n_points = len(pts)
    B = build_bernstein_features(pts, degree, x_lo, x_hi)
    sigma_lo = np.array([p.sigma_lo for p in pts], dtype=np.float64)
    sigma_hi = np.array([p.sigma_hi for p in pts], dtype=np.float64)
    w_lo = 1.0 / (sigma_lo ** 2)
    w_hi = 1.0 / (sigma_hi ** 2)
    # Boundary tightening: at c = 0 (c = n) the lower (upper) side is impossible and its clamped
    # sigma would give a huge weight; the meaningful side's weight is substituted. The paper's code
    # comment says only the step size sees this, but the substituted arrays are also the ones the
    # kernel receives; we keep the code's behaviour.
    for i, p in enumerate(pts):
        if p.passed == 0:
            w_lo[i] = w_hi[i]
        elif p.passed == p.n:
            w_hi[i] = w_lo[i]
    w_max = np.maximum(w_lo, w_hi)
    s = w_max.sum()
    if s > 0:
        scale = n_points / s
        w_lo *= scale
        w_hi *= scale
        w_max *= scale
    L = lipschitz_constant(B, w_max)
    lr = 0.99 / L
    y = np.array([p.y for p in pts], dtype=np.float64)
    return B, w_lo, w_hi, y, lr


def fit_question(pts: list, x_lo: float, x_hi: float, TLB: np.ndarray, degree: int = DEGREE,
                 iterations: int = ITERATIONS, backend: str = "auto", kernel_path: Optional[str] = None):
    """Fit one question. Returns (bernstein_coeffs, legendre_coeffs)."""
    B, w_lo, w_hi, y, lr = fit_inputs(pts, x_lo, x_hi, degree)
    ramp_iters = int(iterations * RAMP_FRAC)
    lib = None
    if backend in ("auto", "c"):
        lib, _ = load_kernel(kernel_path)
        if lib is None and backend == "c":
            raise RuntimeError("backend='c' but no compiled kernel; run python -m onetemp.core.fit_all --build-kernel")
    if lib is not None and B.shape[0] <= _C_MAX_N and B.shape[1] <= _C_MAX_M:
        bc = run_c_kernel(lib, B, w_lo, w_hi, y, lr, iterations, ramp_iters, PROJ_EVERY)
    else:
        bc = run_numpy_kernel([B], [w_lo], [w_hi], [y], [lr], iterations, ramp_iters, PROJ_EVERY)[0]
    c_leg, *_ = np.linalg.lstsq(TLB, bc, rcond=None)
    return bc, c_leg


def _side_sigma(p: DataPoint, pred: float) -> float:
    """Sigma matching the loss convention: sigma_hi if fit > y else sigma_lo."""
    return p.sigma_hi if pred > p.y else p.sigma_lo


def r_squared(pts: list, c_leg: np.ndarray, x_lo: float, x_hi: float) -> float:
    """Weighted R^2 with side-dependent weights matching the loss."""
    preds = np.array([poly_eval(c_leg, p.x, x_lo, x_hi) for p in pts])
    ys = np.array([p.y for p in pts])
    sigmas = np.array([_side_sigma(p, preds[i]) for i, p in enumerate(pts)])
    ws = 1.0 / (sigmas ** 2)
    sw = ws.sum()
    if sw <= 0:
        return 1.0
    w_mean = (ws @ ys) / sw
    ss_tot = float(np.sum(ws * (ys - w_mean) ** 2))
    ss_res = float(np.sum(ws * (ys - preds) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def max_sigma_deviation(pts: list, c_leg: np.ndarray, x_lo: float, x_hi: float) -> float:
    """Worst |fit - data| / sigma_+- (side picked by sign of residual)."""
    best = 0.0
    for p in pts:
        fv = poly_eval(c_leg, p.x, x_lo, x_hi)
        sigma = _side_sigma(p, fv)
        if sigma > 0:
            dev = abs(fv - p.y) / sigma
            if dev > best:
                best = dev
    return best


# ───────────────────────── Peak, half maximum, measurements ─────────────────────────

def compute_fwhm(coeffs, x_lo: float, x_hi: float) -> Optional[dict]:
    """Peak and efficient range (P >= half the peak, Eq. 5) of a fitted profile, as the analysis
    code computes them: on 1001 points over [x_lo, x_hi], linear interpolation at the crossings.
    Returns None when the peak is <= 0. Values are in percent."""
    n_pts = FWHM_POINTS
    j = np.arange(n_pts + 1, dtype=np.float64)
    ts = x_lo + (x_hi - x_lo) * j / n_pts
    vals, inside = eval_legendre_array(coeffs, x_lo, x_hi, ts)
    key = np.where(inside, vals, -np.inf)
    peak_idx = int(np.argmax(key))
    if not inside[peak_idx]:
        return None
    peak_val = float(vals[peak_idx])
    if peak_val <= 0:
        return None
    half_val = peak_val / 2.0
    vs = [float(v) if ok else None for v, ok in zip(vals, inside)]
    t_left = None
    for jj in range(peak_idx, 0, -1):
        a, b = vs[jj - 1], vs[jj]
        if a is not None and b is not None and a <= half_val < b:
            frac = (half_val - a) / (b - a)
            t_left = float(ts[jj - 1]) + frac * (float(ts[jj]) - float(ts[jj - 1]))
            break
    if t_left is None:
        t_left = x_lo
    t_right = None
    for jj in range(peak_idx, n_pts):
        a, b = vs[jj], vs[jj + 1]
        if a is not None and b is not None and a > half_val >= b:
            frac = (half_val - a) / (b - a)
            t_right = float(ts[jj]) + frac * (float(ts[jj + 1]) - float(ts[jj]))
            break
    if t_right is None:
        t_right = x_hi
    return {
        "half_val": round(half_val, 6),
        "t_left": round(t_left, 6),
        "t_right": round(t_right, 6),
        "fwhm": round(t_right - t_left, 6),
        "peak_t": round(float(ts[peak_idx]), 6),
        "peak_val": round(peak_val, 6),
    }


def measured_rates(settings: list, c_row, n_row, greedy_zero: bool) -> list:
    """Measured pass rate per setting, in percent, rounded to 6 decimals (the analysis code's
    ``measurements``). With ``greedy_zero`` (temperature axis), T = 0 is deterministic: 0 or 100."""
    out = []
    for t, passed, n in zip(settings, c_row, n_row):
        if greedy_zero and t == 0.0:
            rate = 100.0 if passed > 0 else 0.0
        elif n:
            rate = (int(passed) / int(n)) * 100.0
        else:
            rate = 0.0
        out.append(round(rate, 6))
    return out


def compute_fwhm_meas(settings: list, rates: list, greedy_zero: bool) -> Optional[dict]:
    """Peak and half-maximum crossings of the measured rates (the analysis code's meas mode)."""
    ts, rs = list(settings), list(rates)
    if greedy_zero and ts and ts[0] == 0.0:
        ts, rs = ts[1:], rs[1:]
    if not ts or not rs or max(rs) <= 0:
        return None
    peak_idx = max(range(len(ts)), key=lambda i: rs[i])
    peak_val = rs[peak_idx]
    peak_t = ts[peak_idx]
    half_val = peak_val / 2.0
    t_left = ts[0]
    for i in range(peak_idx, 0, -1):
        if rs[i - 1] <= half_val < rs[i]:
            frac = (half_val - rs[i - 1]) / (rs[i] - rs[i - 1])
            t_left = ts[i - 1] + frac * (ts[i] - ts[i - 1])
            break
    t_right = ts[-1]
    for i in range(peak_idx, len(ts) - 1):
        if rs[i] > half_val >= rs[i + 1]:
            frac = (half_val - rs[i]) / (rs[i + 1] - rs[i])
            t_right = ts[i] + frac * (ts[i + 1] - ts[i])
            break
    return {"half_val": round(half_val, 6), "t_left": round(t_left, 6), "t_right": round(t_right, 6),
            "fwhm": round(t_right - t_left, 6), "peak_t": round(peak_t, 6), "peak_val": round(peak_val, 6)}


# ───────────────────────── Profiles of a whole dataset ─────────────────────────

@dataclass
class Profiles:
    """Fitted profiles of one dataset, plus the matrices every later step reads.

    ``P_fit[i, j]`` is the fitted single-sample success probability of question i at
    ``alloc_settings[j]`` (in [0, 1]); ``P_meas`` is the measured rate there. ``peak_fit`` and
    ``peak_meas`` are the peak success probabilities in percent (inclusion: >= 0.1).
    """

    axis: str
    settings: list            # the fit grid: every setting of the dataset
    alloc_settings: list      # the allocation grid (temperature: drops T = 0)
    ids: list
    n_mode: str
    backend: str
    x_lo: np.ndarray
    x_hi: np.ndarray
    coeffs: np.ndarray        # (Q, 11) Legendre coefficients, percent units
    bern: np.ndarray          # (Q, 11) Bernstein coefficients
    r2: np.ndarray
    max_sigma: np.ndarray
    fitted: np.ndarray        # bool: a polynomial was fitted (else the zero profile)
    peak_fit: np.ndarray      # percent
    fwhm: list                # dict or None per question (fit)
    P_fit: np.ndarray
    peak_meas: np.ndarray     # percent
    fwhm_meas: list
    P_meas: np.ndarray
    meta: dict = field(default_factory=dict)

    def P(self, mode: str = "fit") -> np.ndarray:
        if mode not in ("fit", "meas"):
            raise ValueError("mode must be 'fit' or 'meas'")
        return self.P_fit if mode == "fit" else self.P_meas

    def L(self, mode: str = "fit") -> np.ndarray:
        """log(1 - P), the matrix of Eq. (9), floored as in the paper's code."""
        return np.log(np.clip(1 - self.P(mode), LOG1M_FLOOR, 1))

    def peak(self, mode: str = "fit") -> np.ndarray:
        return self.peak_fit if mode == "fit" else self.peak_meas

    def included(self, min_peak: float = MIN_PEAK_PCT, mode: str = "fit") -> np.ndarray:
        """Inclusion mask: peak success probability >= min_peak (percent)."""
        return self.peak(mode) >= min_peak

    def to_json(self) -> dict:
        qs = []
        for i, qid in enumerate(self.ids):
            qs.append({
                "id": qid, "fitted": bool(self.fitted[i]),
                "x_lo": float(self.x_lo[i]), "x_hi": float(self.x_hi[i]),
                "legendre": [float(v) for v in self.coeffs[i]],
                "bernstein": [float(v) for v in self.bern[i]],
                "R2": float(self.r2[i]), "max_sigma": float(self.max_sigma[i]),
                "peak_fit_pct": float(self.peak_fit[i]), "fwhm": self.fwhm[i],
                "P_fit": [float(v) for v in self.P_fit[i]],
                "peak_meas_pct": float(self.peak_meas[i]), "fwhm_meas": self.fwhm_meas[i],
                "P_meas": [float(v) for v in self.P_meas[i]],
            })
        return {"format": "core.fit_all/profiles-v1", "axis": self.axis, "settings": self.settings,
                "alloc_settings": self.alloc_settings, "n_mode": self.n_mode, "backend": self.backend,
                "meta": self.meta, "questions": qs}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def from_json(cls, d: dict) -> "Profiles":
        qs = d["questions"]
        n_s = len(d["alloc_settings"])
        arr = lambda k: np.array([q[k] for q in qs], dtype=np.float64)  # noqa: E731
        mat = lambda k, w: np.array([q[k] for q in qs], dtype=np.float64).reshape(len(qs), w)  # noqa: E731
        return cls(axis=d["axis"], settings=d["settings"], alloc_settings=d["alloc_settings"],
                   ids=[q["id"] for q in qs], n_mode=d["n_mode"], backend=d["backend"],
                   x_lo=arr("x_lo"), x_hi=arr("x_hi"),
                   coeffs=mat("legendre", DEGREE + 1), bern=mat("bernstein", DEGREE + 1),
                   r2=arr("R2"), max_sigma=arr("max_sigma"),
                   fitted=np.array([q["fitted"] for q in qs], dtype=bool),
                   peak_fit=arr("peak_fit_pct"), fwhm=[q["fwhm"] for q in qs],
                   P_fit=mat("P_fit", n_s),
                   peak_meas=arr("peak_meas_pct"), fwhm_meas=[q["fwhm_meas"] for q in qs],
                   P_meas=mat("P_meas", n_s),
                   meta=d.get("meta", {}))


def load_profiles(path: str) -> Profiles:
    with open(path, "r", encoding="utf-8") as f:
        return Profiles.from_json(json.load(f))


def fit_profiles(counts: Counts, n_mode: str = "question-max", backend: str = "auto",
                 kernel_path: Optional[str] = None, iterations: int = ITERATIONS,
                 progress: bool = False) -> Profiles:
    """Fit every question of a dataset and build the matrices of the comparison (Appendix D/F).

    backend: "c" (compiled kernel required), "numpy", or "auto" (C when the library is found).
    """
    if n_mode not in N_MODES:
        raise ValueError(f"n_mode must be one of {N_MODES}")
    if backend not in ("auto", "c", "numpy"):
        raise ValueError("backend must be 'auto', 'c' or 'numpy'")
    settings = [float(s) for s in counts.settings]
    greedy_zero = counts.axis == "temperature"
    alloc_settings = [s for s in settings if s > 0.0] if greedy_zero else list(settings)
    if not alloc_settings:
        raise ValueError("no setting left in the allocation grid")
    n_fit = counts.n_for_fit(n_mode)
    if n_mode == "question-max":
        # A setting without a row counts as a measured zero with the question's n (see counts_io).
        c_fit = np.where(counts.measured, counts.c, 0)
    else:
        c_fit = counts.c
    Q = counts.n_questions
    S = len(settings)
    TLB = build_tlb(DEGREE)
    m = DEGREE + 1

    lib, lib_path = (None, None)
    if backend in ("auto", "c"):
        lib, lib_path = load_kernel(kernel_path)
        if lib is None and backend == "c":
            raise RuntimeError("backend='c' but no compiled kernel; run python -m onetemp.core.fit_all --build-kernel")
    used_backend = "c" if lib is not None else "numpy"

    x_lo = np.zeros(Q)
    x_hi = np.zeros(Q)
    bern = np.zeros((Q, m))
    coeffs = np.zeros((Q, m))
    r2 = np.ones(Q)
    max_sig = np.zeros(Q)
    fitted = np.zeros(Q, dtype=bool)
    pts_all: dict = {}
    t0 = time.perf_counter()
    jobs = []
    for i in range(Q):
        n_row = [int(v) for v in n_fit[i]]
        c_row = [int(v) for v in c_fit[i]]
        x_lo[i], x_hi[i] = settings[0], settings[-1]
        if sum(n_row) <= 0 or sum(c_row) == 0:
            continue
        pts, lo, hi = prepare_question("", n_row, c_row, settings)
        if len(pts) < 2 or hi == lo:
            continue
        x_lo[i], x_hi[i] = lo, hi
        pts_all[i] = pts
        jobs.append(i)

    ramp_iters = int(iterations * RAMP_FRAC)
    inputs = {i: fit_inputs(pts_all[i], x_lo[i], x_hi[i]) for i in jobs}
    use_c = [i for i in jobs if lib is not None and inputs[i][0].shape[0] <= _C_MAX_N]
    use_c_set = set(use_c)
    use_np = [i for i in jobs if i not in use_c_set]
    for k, i in enumerate(use_c):
        B, wl, wh, y, lr = inputs[i]
        bern[i] = run_c_kernel(lib, B, wl, wh, y, lr, iterations, ramp_iters, PROJ_EVERY)
        if progress and (k + 1) % 100 == 0:
            print(f"  [fit c] {k + 1}/{len(use_c)} {time.perf_counter() - t0:.1f}s", flush=True)
    if use_np:
        if progress:
            print(f"  [fit numpy] {len(use_np)} questions in one batch", flush=True)
        out = run_numpy_kernel([inputs[i][0] for i in use_np], [inputs[i][1] for i in use_np],
                               [inputs[i][2] for i in use_np], [inputs[i][3] for i in use_np],
                               [inputs[i][4] for i in use_np], iterations, ramp_iters, PROJ_EVERY)
        for k, i in enumerate(use_np):
            bern[i] = out[k]
    if use_np and lib is not None:
        used_backend = "c+numpy"
    for i in jobs:
        c_leg, *_ = np.linalg.lstsq(TLB, bern[i], rcond=None)
        coeffs[i] = c_leg
        r2[i] = r_squared(pts_all[i], c_leg, x_lo[i], x_hi[i])
        max_sig[i] = max_sigma_deviation(pts_all[i], c_leg, x_lo[i], x_hi[i])
        fitted[i] = True
    t_fit = time.perf_counter() - t0

    grid = np.array(alloc_settings, dtype=np.float64)
    P_fit = np.zeros((Q, len(alloc_settings)))
    peak_fit = np.zeros(Q)
    fwhm = []
    for i in range(Q):
        c_list = [float(v) for v in coeffs[i]]
        f = compute_fwhm(c_list, float(x_lo[i]), float(x_hi[i]))
        fwhm.append(f)
        peak_fit[i] = f["peak_val"] if f else 0.0
        vals, inside = eval_legendre_array(c_list, float(x_lo[i]), float(x_hi[i]), grid)
        P_fit[i] = np.clip(np.where(inside, vals, 0.0) / 100.0, 0, 1)

    P_meas = np.zeros((Q, len(alloc_settings)))
    peak_meas = np.zeros(Q)
    fwhm_meas = []
    s_arr = np.array(settings, dtype=np.float64)
    for i in range(Q):
        rates = measured_rates(settings, counts.c[i], counts.n[i] if n_mode == "per-setting" else n_fit[i],
                               greedy_zero)
        fm = compute_fwhm_meas(settings, rates, greedy_zero)
        fwhm_meas.append(fm)
        peak_meas[i] = fm["peak_val"] if fm else 0.0
        P_meas[i] = np.clip(np.interp(grid, s_arr, np.asarray(rates) / 100.0), 0, 1)

    meta = {
        "source": counts.source, "benchmark": counts.benchmark, "model": counts.model,
        "fixed": counts.fixed, "questions": Q, "fit_backend": used_backend,
        "kernel_path_basename": os.path.basename(lib_path) if lib_path else None,
        "kernel_sha256": _sha256_file(lib_path) if lib_path else None,
        "kernel_source_sha256": _sha256_file(KERNEL_SOURCE),
        "constants": {"degree": DEGREE, "iterations": iterations, "proj_every": PROJ_EVERY,
                      "ramp_frac": RAMP_FRAC, "cp_alpha": CP_ALPHA, "eps_pct": EPS_PCT,
                      "fwhm_points": FWHM_POINTS + 1, "log1m_floor": LOG1M_FLOOR},
        "fit_seconds": round(t_fit, 3),
        "numpy": np.__version__,
    }
    return Profiles(axis=counts.axis, settings=settings, alloc_settings=alloc_settings, ids=list(counts.ids),
                    n_mode=n_mode, backend=used_backend, x_lo=x_lo, x_hi=x_hi, coeffs=coeffs, bern=bern,
                    r2=r2, max_sigma=max_sig, fitted=fitted, peak_fit=peak_fit, fwhm=fwhm, P_fit=P_fit,
                    peak_meas=peak_meas, fwhm_meas=fwhm_meas, P_meas=P_meas, meta=meta)


def profile_rows(prof: Profiles, counts: Counts) -> list:
    """One row per measured (question, allocation setting): the counts.jsonl row plus

    * ``p_fit``: the fitted single-sample success probability there, clip(P(x) / 100, 0, 1), 0
      outside the question's bracket (the paper's P_grid entry);
    * ``peak_pct``: the question's fitted peak in percent, rounded to 6 decimals, 0.0 for a
      question without a fit (the paper's compute_fwhm peak_val); the inclusion rule is
      peak_pct >= 0.1.

    Settings outside the allocation grid (T = 0 on the temperature axis) get no row.
    """
    if list(counts.ids) != list(prof.ids):
        raise ValueError("counts and profiles hold different questions")
    s_index = {float(s): j for j, s in enumerate(counts.settings)}
    rows = []
    for i, qid in enumerate(prof.ids):
        for k, s in enumerate(prof.alloc_settings):
            j = s_index[float(s)]
            if not counts.measured[i, j]:
                continue
            row = {"benchmark": counts.benchmark, "model": counts.model, "id": qid}
            if counts.axis == "temperature":
                row["temperature"] = float(s)
                row["min_p"] = counts.fixed.get("min_p")
            else:
                row["temperature"] = counts.fixed.get("temperature")
                row["min_p"] = float(s)
                row["tau"] = counts.fixed.get("temperature")
            row["n"] = int(counts.n[i, j])
            row["c"] = int(counts.c[i, j])
            row["p_fit"] = float(prof.P_fit[i, k])
            row["peak_pct"] = float(prof.peak_fit[i])
            rows.append(row)
    return rows


def write_profile_rows(prof: Profiles, counts: Counts, path: str) -> int:
    """Write ``profile_rows`` as JSON lines (floats round-trip exactly). Returns the row count."""
    rows = profile_rows(prof, counts)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ───────────────────────── Command line ─────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit the paper's per-question profiles (Appendix D) to counts.jsonl.")
    ap.add_argument("--build-kernel", action="store_true", help="compile bernstein_kernel.c and exit")
    ap.add_argument("--build-flags", default="portable", choices=sorted(BUILD_FLAGS),
                    help="portable (default, no fused multiply-add) or paper (the paper's -march=native line)")
    ap.add_argument("--kernel", default=None, help="path of the compiled kernel (default: next to this file)")
    ap.add_argument("--counts", help="counts.jsonl")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--axis", default="temperature", choices=AXES)
    ap.add_argument("--fixed-temperature", type=float, default=None, help="for --axis min_p")
    ap.add_argument("--n-mode", default="question-max", choices=N_MODES)
    ap.add_argument("--backend", default="auto", choices=("auto", "c", "numpy"))
    ap.add_argument("--iterations", type=int, default=ITERATIONS, help="gradient steps (paper: 50000)")
    ap.add_argument("--out", help="output profiles.json (one entry per question)")
    ap.add_argument("--out-rows", default=None,
                    help="also write profile rows as JSONL: each counts.jsonl row plus p_fit and peak_pct")
    a = ap.parse_args(argv)
    if a.build_kernel:
        p = build_kernel(a.kernel, a.build_flags)
        print(f"[build] wrote {p}  sha256 {_sha256_file(p)}")
        return 0
    if not a.counts or not (a.out or a.out_rows):
        ap.error("--counts and --out and/or --out-rows are required (or --build-kernel)")
    cnt = load_counts(a.counts, a.benchmark, a.model, a.axis, fixed_temperature=a.fixed_temperature)
    prof = fit_profiles(cnt, n_mode=a.n_mode, backend=a.backend, kernel_path=a.kernel,
                        iterations=a.iterations, progress=True)
    if a.out:
        prof.save(a.out)
    if a.out_rows:
        n_rows = write_profile_rows(prof, cnt, a.out_rows)
        print(f"[rows] {n_rows} rows -> {a.out_rows}")
    inc = prof.included()
    print(f"[fit] {cnt.n_questions} questions, {int(prof.fitted.sum())} fitted, {int(inc.sum())} included "
          f"(peak >= {MIN_PEAK_PCT}%), backend {prof.backend}, {prof.meta['fit_seconds']} s"
          + (f" -> {a.out}" if a.out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
