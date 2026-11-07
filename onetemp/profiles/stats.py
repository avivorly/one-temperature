"""Statistics behind the three hardness views (Figure 2; Figures 5-13; Appendix E.1).

* ``sliding_mean_symmetric``: the "sliding window of size N" of Appendix E.1.  At position i of the
  hardness-sorted list the window holds the N/2 questions on each side and question i itself; near the ends
  it shrinks symmetrically (radius min(i, n-1-i, N/2)) and never becomes one-sided.
* ``drawn_interior``: which part of a sliding-mean line is drawn.  Appendix E.1: "Each line leaves out the
  15 questions at each end of its H-sorted list, where its window is not full ... On LiveCodeBench and
  MATH-500, each line then starts at the first remaining question whose next question is at least 5%
  harder ... The Hi-ToM lines use every full window."  The second step is ``advance_low_h``.
* ``ols_log10``: the OLS line of |E_q| on log10 H_q, with R^2 (Figures 8-10).
* ``coverage_bins``: the empirical R_Q(H) of Eq. 5 (Figures 11-13): log-uniform hardness bins between the
  smallest and largest included H, and per bin the largest share of its questions whose efficient ranges
  share one setting, found by a sweep line over the range endpoints.
"""
from __future__ import annotations

import math

import numpy as np

#: Sliding-window size N of Appendix E.1.
WINDOW = 30
#: Peak-setting cut of the red line in Figures 2 and 5-7 ("tau*_q > 0.1").
NEAR_ZERO_CUT = 0.1
#: Relative hardness step that ends the low-hardness advance of ``drawn_interior``.
ADVANCE_RATIO = 1.05
#: Number of log-uniform hardness bins used by every coverage figure of the paper.
N_BINS = 17


def sliding_mean_symmetric(ys, win: int = WINDOW) -> np.ndarray:
    """Symmetric sliding mean of ``ys`` (already sorted by hardness), radius min(i, n-1-i, win//2)."""
    arr = np.asarray(ys, dtype=float)
    n = len(arr)
    out = np.empty(n, dtype=float)
    half = int(win) // 2
    for i in range(n):
        rad = min(i, n - 1 - i, half)
        out[i] = float(arr[i - rad: i + rad + 1].mean())
    return out


def drawn_interior(xs, ys, win: int = WINDOW, advance_low_h: bool = True):
    """Return (xs, ys, start, stop): the drawn part of a sliding-mean line.

    1. Drop ``win // 2`` points at each end, where the symmetric window is not full.  A series of at most
       ``win + 1`` points is returned whole (the paper's fallback).
    2. If ``advance_low_h``: while the next hardness is less than 5% above the current one, move the start
       right.  Used on LiveCodeBench and MATH-500, where many questions share the clamp hardness 0.0869;
       not used on Hi-ToM (paper, Appendix E.1).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    half = int(win) // 2
    m = len(ys)
    if m <= 2 * half + 1:
        return xs, ys, 0, m
    start, stop = half, m - half
    if advance_low_h:
        while start + 1 < stop and xs[start + 1] / xs[start] < ADVANCE_RATIO:
            start += 1
    return xs[start:stop], ys[start:stop], start, stop


def ols_log10(H, y) -> dict:
    """OLS of y on log10 H: slope, intercept and R^2 (np.polyfit, as the paper's figure code)."""
    log_h = np.log10(np.asarray(H, dtype=float))
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or np.ptp(log_h) == 0:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "n": int(len(y))}
    slope, intercept = np.polyfit(log_h, y, 1)
    y_hat = slope * log_h + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n": int(len(y))}


def max_overlap(left, right) -> int:
    """Largest number of closed intervals [left_i, right_i] that share one point.

    Sweep line over the endpoints: +1 at a left end, -1 at a right end, and at equal coordinates every +1
    is counted before any -1, so intervals that only touch still count as overlapping.
    """
    ev_t = np.concatenate([np.asarray(left, dtype=float), np.asarray(right, dtype=float)])
    ev_s = np.concatenate([np.ones(len(left), dtype=int), -np.ones(len(right), dtype=int)])
    if ev_t.size == 0:
        return 0
    # The paper interleaves (left_i, right_i) per question before sorting; lexsort on (t, -s) gives the
    # same counts for any input order.
    order = np.lexsort((-ev_s, ev_t))
    return int(np.cumsum(ev_s[order]).max())


def log_uniform_edges(h_min: float, h_max: float, n_bins: int = N_BINS) -> list[tuple[float, float]]:
    """Bin edges [L_b, R_b) equally spaced in ln H; the last bin is closed (its R is raised by 1e-9)."""
    lo, hi = float(np.log(h_min)), float(np.log(h_max))
    d = (hi - lo) / n_bins
    out = []
    for b in range(n_bins):
        L = float(np.exp(lo + b * d))
        R = float(np.exp(lo + (b + 1) * d))
        if b == n_bins - 1:
            R = R * (1 + 1e-9)
        out.append((L, R))
    return out


def coverage_bins(H_sorted, t_left, t_right, n_bins: int = N_BINS) -> list[dict]:
    """Empirical R_Q(H) (Eq. 5) per log-uniform hardness bin; empty bins are skipped.

    ``H_sorted`` must be ascending (questions sorted by hardness); ``t_left``/``t_right`` are the efficient
    range ends in the same order.  Each returned bin has its edges, center ((L+R)/2, the bar position of
    the paper's figure), width, M (questions in the bin), the largest overlap and height = overlap / M.
    """
    Hs = np.asarray(H_sorted, dtype=float)
    tl = np.asarray(t_left, dtype=float)
    tr = np.asarray(t_right, dtype=float)
    n = len(Hs)
    if n < 2:
        return []
    n_bins = max(2, int(n_bins))
    out = []
    b_start = 0
    for b, (L, R) in enumerate(log_uniform_edges(Hs[0], Hs[-1], n_bins)):
        while b_start < n and Hs[b_start] < L:
            b_start += 1
        b_end = b_start
        while b_end < n and Hs[b_end] < R:
            b_end += 1
        M = b_end - b_start
        if M == 0:
            continue
        # The paper builds the events question by question: (t_L, +1), (t_R, -1).
        ev_t, ev_s = [], []
        for p in range(b_start, b_end):
            ev_t.append(float(tl[p])); ev_s.append(+1)
            ev_t.append(float(tr[p])); ev_s.append(-1)
        ev_t = np.asarray(ev_t, dtype=float)
        ev_s = np.asarray(ev_s, dtype=int)
        order = np.lexsort((-ev_s, ev_t))
        mx = int(np.cumsum(ev_s[order]).max())
        out.append({"bin": b, "H_left": L, "H_right": R, "center": (L + R) / 2.0, "width": R - L,
                    "M": int(M), "max_count": mx, "height": mx / M,
                    "first_index": int(b_start), "end_index": int(b_end)})
    return out


def auto_bins(n_questions: int) -> int:
    """The default bin count stated in Appendix E.1, floor(n / 20) (the paper's figures fix 17)."""
    return max(2, n_questions // 20)
