"""Evaluate fitted profiles and read off their peak and efficient range, exactly as the paper's code.

The paper's profile fit (Appendix D) returns, per question, a degree-10 polynomial on a bracket
[x_lo, x_hi] and P = 0 outside it.  Its analysis code then:

* evaluates the polynomial (Legendre form, in percent) on 1,001 equally spaced points of the bracket;
* takes the **peak** tau*_q at the first grid point with the largest value (Section 3.2.1);
* takes the **efficient range** E_q = {tau : P_q(tau) >= peak / 2} (Eq. 4) as the interval [t_L, t_R]
  between the half-maximum crossings nearest the peak, each found by linear interpolation between grid
  points; if the curve does not fall to half its peak inside the bracket, the bracket end is used
  (Appendix E.1: "for a unimodal profile the efficient range is approximately an interval [t_L, t_R],
  so we report |E_q| = t_R - t_L");
* rounds every reported number to 6 decimals.

``fitted_summary`` reproduces those numbers bit for bit: the same grid arithmetic, the same comparisons and
the same summation (see ``_series_sum``).  ``measured_summary`` is the paper's fit-free variant on measured
rates c/n with linear interpolation between grid settings.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.stats import beta as _beta

from .io import FittedProfile

#: Number of intervals of the evaluation grid used for the peak and the half-maximum crossings.
N_EVAL_INTERVALS = 1000
#: Number of intervals of the grid used to draw a fitted curve (Figure 1).
N_DRAW_INTERVALS = 200
#: Exact Clopper-Pearson 1-sigma two-sided tail probability, 1 - erf(1/sqrt(2)) (Appendix D).
CP_ALPHA = 0.31731050786291415


def _series_sum(terms: list[np.ndarray]) -> np.ndarray:
    """Element-wise sum of ``terms`` in order, rounded exactly as Python's built-in ``sum``.

    The paper's code sums the 11 Legendre terms with ``sum(c * p for ...)``.  From CPython 3.12 on,
    ``sum`` of floats uses Neumaier's compensated summation; before 3.12 it adds left to right.  The two
    differ by about 1e-16 relative, which is enough to move an argmax along a flat-topped profile, so this
    function follows the running interpreter.
    """
    s = np.array(terms[0], dtype=np.float64, copy=True)
    if sys.version_info < (3, 12):
        for x in terms[1:]:
            s = s + x
        return s
    comp = np.zeros_like(s)
    for x in terms[1:]:
        t = s + x
        comp += np.where(np.abs(s) >= np.abs(x), (s - t) + x, (x - t) + s)
        s = t
    use = (comp != 0.0) & np.isfinite(comp)
    return np.where(use, s + comp, s)


def legendre_percent(coeffs, x_lo: float, x_hi: float, t) -> np.ndarray:
    """Evaluate a fitted profile (percent) at ``t``; NaN outside [x_lo, x_hi].

    Same arithmetic as the paper's ``eval_legendre``: u = 2 (t - x_lo)/(x_hi - x_lo) - 1, the three-term
    recurrence L_k = ((2k-1) u L_{k-1} - (k-1) L_{k-2}) / k up to k = 10, then sum_k c_k L_k(u).
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.full(t.shape, np.nan)
    inside = ~((t < x_lo) | (t > x_hi))
    if not inside.any():
        return out
    u = 2.0 * (t[inside] - x_lo) / (x_hi - x_lo) - 1.0
    L = [np.ones_like(u), u]
    for k in range(2, 11):
        L.append(((2 * k - 1) * u * L[k - 1] - (k - 1) * L[k - 2]) / k)
    coeffs = [float(c) for c in coeffs]
    terms = [c * Lk for c, Lk in zip(coeffs, L)]
    out[inside] = _series_sum(terms)
    return out


def profile_probability(profile: FittedProfile, t) -> np.ndarray:
    """P_q(t) as a probability in [0, 1]: the fitted percent value / 100, clipped, and 0 outside the bracket.

    This is how the paper turns a fitted profile into pass rates on the allocation grid.
    """
    v = legendre_percent(profile.legendre, profile.x_lo, profile.x_hi, t)
    v = np.where(np.isnan(v), 0.0, v)
    return np.clip(v / 100.0, 0.0, 1.0)


def _grid(x_lo: float, x_hi: float, n_intervals: int) -> np.ndarray:
    j = np.arange(n_intervals + 1, dtype=np.int64)
    return x_lo + (x_hi - x_lo) * j / n_intervals


def curve_points(profile: FittedProfile, n_intervals: int = N_DRAW_INTERVALS) -> tuple[list[float], list[float]]:
    """The drawn curve of a fitted profile: 201 points on its bracket, values in percent, rounded to 6 dp."""
    ts = _grid(profile.x_lo, profile.x_hi, n_intervals)
    vs = legendre_percent(profile.legendre, profile.x_lo, profile.x_hi, ts)
    xs, ys = [], []
    for t, v in zip(ts.tolist(), vs.tolist()):
        if not math.isnan(v):
            xs.append(round(t, 6))
            ys.append(round(v, 6))
    return xs, ys


def fitted_summary(profile: FittedProfile, n_intervals: int = N_EVAL_INTERVALS) -> dict | None:
    """Peak and efficient range of a fitted profile; ``None`` if its peak is not positive.

    Returns (all in percent or setting units, rounded to 6 decimals like the paper):
        peak_t     tau*_q, the first evaluation point with the largest value
        peak_val   max P_q, in percent
        half_val   peak_val / 2
        t_left     left end of E_q  (bracket start if the curve never falls below half on the left)
        t_right    right end of E_q (bracket end if the curve never falls below half on the right)
        fwhm       t_right - t_left before rounding, then rounded
    """
    x_lo, x_hi = float(profile.x_lo), float(profile.x_hi)
    ts = _grid(x_lo, x_hi, n_intervals)
    vs = legendre_percent(profile.legendre, x_lo, x_hi, ts)
    finite = ~np.isnan(vs)
    if not finite.any():
        return None
    masked = np.where(finite, vs, -np.inf)
    peak_idx = int(np.argmax(masked))  # first maximum, as Python's max() over the indices
    peak_val = float(vs[peak_idx])
    if peak_val <= 0:
        return None
    half_val = peak_val / 2.0

    # Left crossing: the nearest j <= peak_idx (searching downwards) with vs[j-1] <= half < vs[j].
    t_left = x_lo
    if peak_idx >= 1:
        a = vs[:peak_idx]          # vs[j-1] for j = 1..peak_idx
        b = vs[1:peak_idx + 1]     # vs[j]
        hit = np.nonzero((a <= half_val) & (half_val < b))[0]
        if hit.size:
            j = int(hit[-1]) + 1
            a_, b_ = float(vs[j - 1]), float(vs[j])
            frac = (half_val - a_) / (b_ - a_)
            t_left = float(ts[j - 1]) + frac * (float(ts[j]) - float(ts[j - 1]))

    # Right crossing: the nearest j >= peak_idx (searching upwards) with vs[j] > half >= vs[j+1].
    t_right = x_hi
    if peak_idx < n_intervals:
        a = vs[peak_idx:n_intervals]
        b = vs[peak_idx + 1:n_intervals + 1]
        hit = np.nonzero((a > half_val) & (half_val >= b))[0]
        if hit.size:
            j = peak_idx + int(hit[0])
            a_, b_ = float(vs[j]), float(vs[j + 1])
            frac = (half_val - a_) / (b_ - a_)
            t_right = float(ts[j]) + frac * (float(ts[j + 1]) - float(ts[j]))

    return {
        "half_val": round(half_val, 6),
        "t_left": round(t_left, 6),
        "t_right": round(t_right, 6),
        "fwhm": round(t_right - t_left, 6),
        "peak_t": round(float(ts[peak_idx]), 6),
        "peak_val": round(peak_val, 6),
    }


def measured_rates_percent(settings, n, c, n_mode: str = "per-setting"):
    """Measured pass rates in percent, as the paper's ``measurements`` helper.

    ``n_mode="per-setting"`` divides by each setting's own n; ``n_mode="question-max"`` divides by one
    question-level n (the largest n of the question) at every setting, as the code that drew the paper's
    LiveCodeBench profile figures did.
    """
    n = np.asarray(n, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if n_mode == "question-max":
        n_use = np.full_like(n, int(n.max()) if n.size else 0)
    elif n_mode == "per-setting":
        n_use = n
    else:
        raise ValueError(f"n_mode must be 'question-max' or 'per-setting', not {n_mode!r}")
    rates = []
    for ci, ni in zip(c.tolist(), n_use.tolist()):
        rates.append(round((ci / ni) * 100.0, 6) if ni else 0.0)
    return list(np.asarray(settings, dtype=float).tolist()), rates


def measured_summary(settings, rates_percent) -> dict | None:
    """The paper's fit-free peak and efficient range on measured rates (its ``compute_fwhm_meas``).

    Peak = first setting with the largest measured rate; t_L, t_R = half-maximum crossings by linear
    interpolation between neighbouring settings, else the ends of the grid.  Settings <= 0 are dropped.
    """
    ts = [float(t) for t in settings]
    rates = [float(r) for r in rates_percent]
    if ts and ts[0] == 0.0:
        ts, rates = ts[1:], rates[1:]
    if not ts or max(rates) <= 0:
        return None
    peak_idx = max(range(len(ts)), key=lambda i: rates[i])
    peak_val = rates[peak_idx]
    half_val = peak_val / 2.0
    t_left = ts[0]
    for i in range(peak_idx, 0, -1):
        if rates[i - 1] <= half_val < rates[i]:
            frac = (half_val - rates[i - 1]) / (rates[i] - rates[i - 1])
            t_left = ts[i - 1] + frac * (ts[i] - ts[i - 1])
            break
    t_right = ts[-1]
    for i in range(peak_idx, len(ts) - 1):
        if rates[i] > half_val >= rates[i + 1]:
            frac = (half_val - rates[i]) / (rates[i + 1] - rates[i])
            t_right = ts[i] + frac * (ts[i + 1] - ts[i])
            break
    return {
        "half_val": round(half_val, 6),
        "t_left": round(t_left, 6),
        "t_right": round(t_right, 6),
        "fwhm": round(t_right - t_left, 6),
        "peak_t": round(ts[peak_idx], 6),
        "peak_val": round(peak_val, 6),
    }


def cp_interval_percent(passed: int, n: int) -> tuple[float, float]:
    """Exact Clopper-Pearson 1-sigma interval of a binomial proportion, in percent (Appendix D)."""
    lo = _beta.ppf(CP_ALPHA / 2, passed, n - passed + 1) if passed > 0 else 0.0
    hi = _beta.ppf(1 - CP_ALPHA / 2, passed + 1, n - passed) if passed < n else 1.0
    return float(lo) * 100.0, float(hi) * 100.0
