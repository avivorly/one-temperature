"""Appendix I, "Question profiles and hardness": the Section 3 analyses on a min_p grid, from measured rates.

On these grids (10 min_p values at tau = 1.5, 5 at tau = 2.0) a degree-10 profile fit is underdetermined, so
the paper works with measured pass rates P[q, v] = c / n and grid positions:

* peak    = the first grid value with the largest measured rate (the smallest min_p among ties);
* H_q     = Eq. 3 at that peak rate (same clamps), with the 0.1% inclusion floor of Appendix D;
* E_q     = the contiguous grid values around the peak whose rate is at least half the peak;
            |E_q| counts grid values;
* coverage of a hardness bin = the largest share of its questions whose E_q holds one grid value
  (17 log-uniform bins, as Figure 11);
* sliding means of the peak are means of grid position (grid values are spaced equally on the axis).

Because the peak and H_q come from the same noisy maximum, a trend is described only if it passes two checks
fixed in advance (Appendix I):

1. **split-half guard**: in 400 random splits of every cell's samples into halves (hypergeometric thinning
   of c to n/2), with the peak/range from half A and H_q (and the bins) from half B, the trend statistic
   keeps the full-data sign in at least 95% of splits;
2. **common-shape null**: every question gets the same profile shape g(v) (the pooled rate of each grid
   value), scaled to its own mean rate r_q and capped at 1, p_q(v) = min(1, r_q g(v) / mean g), with
   binomial draws at the same n in 400 replicates; the full-data statistic must lie outside the null's
   central 95% on its own side.

The trend statistics are OLS slopes on log10 H_q: of the peak grid position (all questions, and the
questions whose peak is above the smallest value), of |E_q|, and of the coverage bars (unweighted, over the
non-empty bins at their geometric centres).
"""
from __future__ import annotations

import math
import re

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator, PercentFormatter  # noqa: E402

from . import stats  # noqa: E402
from .curves import cp_interval_percent  # noqa: E402
from .hardness import MIN_PEAK_PERCENT, hardness_from_percent  # noqa: E402
from .plots import PAPER_RC, apply_h_log_axis, coverage_ylim_floor  # noqa: E402

N_GUARD = 400
N_NULL = 400
N_FPR = 100
S_FPR = 40
STATS = ("peak_slope_all", "peak_slope_subset", "range_slope", "relc_slope")
DEFAULT_COLORS = ("#1f77b4", "#d62728", "#2ca02c")
BG_COLOR = "#9A9A9A"
BG_ALPHA = 0.35
N_GREY = 48


def natural_key(s: str):
    """Sort key that orders digit runs numerically (s2_o1 < s10_o0)."""
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


# ----------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------

def peaks(c: np.ndarray, n: np.ndarray, min_peak_percent: float = MIN_PEAK_PERCENT) -> dict:
    """Per question: peak index j (first argmax), peak rate pk, included flag, E_q = [L, R] in grid positions."""
    P = c / n
    Q, nv = P.shape
    j = P.argmax(axis=1)
    pk = P[np.arange(Q), j]
    inc = (pk > 0) & (pk * 100.0 >= min_peak_percent)
    ok = P >= (pk / 2.0)[:, None]
    L, R = j.copy(), j.copy()
    for side, arr in ((-1, L), (1, R)):
        alive = np.ones(Q, dtype=bool)
        for d in range(1, nv):
            idx = j + side * d
            can = alive & (idx >= 0) & (idx < nv)
            hit = np.zeros(Q, dtype=bool)
            w = np.where(can)[0]
            hit[w] = ok[w, idx[w]]
            arr[hit] = idx[hit]
            alive = hit
    return {"P": P, "j": j, "pk": pk, "inc": inc, "L": L, "R": R, "size": R - L + 1}


def hardness_array(pk) -> np.ndarray:
    return np.array([hardness_from_percent(100.0 * p) for p in pk])


def ols(x, y) -> tuple[float, float, float]:
    """(slope, intercept, R^2) of y on x; NaN with fewer than 3 points or constant x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return math.nan, math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    yh = slope * x + intercept
    ss_res, ss_tot = float(np.sum((y - yh) ** 2)), float(np.sum((y - y.mean()) ** 2))
    return float(slope), float(intercept), (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan


def grid_coverage_bins(H, L, R, nv: int, n_bins: int = stats.N_BINS) -> list[dict]:
    """Coverage per log-uniform hardness bin when E_q is a set of grid positions [L_q, R_q]."""
    order = np.argsort(H, kind="stable")
    Hs, Ls, Rs = np.asarray(H)[order], np.asarray(L)[order], np.asarray(R)[order]
    n = len(Hs)
    out = []
    if n == 0:
        return out
    b_start = 0
    for b, (Lb, Rb) in enumerate(stats.log_uniform_edges(Hs[0], Hs[-1], n_bins)):
        while b_start < n and Hs[b_start] < Lb:
            b_start += 1
        b_end = b_start
        while b_end < n and Hs[b_end] < Rb:
            b_end += 1
        M = b_end - b_start
        if M == 0:
            continue
        cover = [int(np.sum((Ls[b_start:b_end] <= v) & (Rs[b_start:b_end] >= v))) for v in range(nv)]
        mx = max(cover)
        out.append({"bin": b, "H_left": Lb, "H_right": Rb, "center": (Lb + Rb) / 2.0, "width": Rb - Lb, "M": M,
                    "max_count": mx, "height": mx / M, "argmax_positions": [v for v in range(nv) if cover[v] == mx],
                    "count_per_position": cover})
    return out


def trend(cA, nA, cB, nB, nv: int) -> dict:
    """The four trend statistics, with peaks and ranges from (cA, nA) and hardness from (cB, nB)."""
    a, b = peaks(cA, nA), peaks(cB, nB)
    inc = a["inc"] & b["inc"]
    H = hardness_array(b["pk"][inc])
    x = np.log10(H)
    j, size = a["j"][inc], a["size"][inc]
    sub = j > 0
    bins = grid_coverage_bins(H, a["L"][inc], a["R"][inc], nv)
    bx = np.log10([math.sqrt(e["H_left"] * e["H_right"]) for e in bins])
    return {"n": int(inc.sum()), "n_subset": int(sub.sum()),
            "peak_slope_all": ols(x, j)[0], "peak_slope_subset": ols(x[sub], j[sub])[0],
            "range_slope": ols(x, size)[0], "relc_slope": ols(bx, [e["height"] for e in bins])[0]}


def split_half(rng: np.random.Generator, c: np.ndarray, n: np.ndarray):
    """Split each cell's n samples into halves of n // 2 and n - n // 2 by hypergeometric thinning of c."""
    h = n // 2
    cA = rng.hypergeometric(c, n - c, h)
    return cA, h, c - cA, n - h


def _summ(vals, full) -> dict:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    s = np.sign(full)
    return {"n_finite": int(v.size), "median": float(np.median(v)), "q025": float(np.quantile(v, 0.025)),
            "q975": float(np.quantile(v, 0.975)), "frac_same_sign_as_full": float(np.mean(np.sign(v) == s))}


def noise_checks(c: np.ndarray, n: np.ndarray, nv: int, seeds, n_guard: int = N_GUARD, n_null: int = N_NULL,
                 n_fpr: int = N_FPR, s_fpr: int = S_FPR) -> dict:
    """Split-half guard, common-shape null and (diagnostic) the guard's false-positive rate on null data.

    ``seeds``: three ``numpy.random.SeedSequence`` (guard, null, false-positive rate).  ``n_fpr = 0`` skips
    the diagnostic.  A statistic is ``trend_claimable`` if both checks pass.
    """
    full = trend(c, n, c, n, nv)
    g_rng = np.random.default_rng(seeds[0])
    guard = {k: [] for k in STATS}
    guard_n = []
    for _ in range(n_guard):
        t = trend(*split_half(g_rng, c, n), nv)
        guard_n.append(t["n"])
        for k in STATS:
            guard[k].append(t[k])
    g = c.sum(axis=0) / n.sum(axis=0)
    r = c.sum(axis=1) / n.sum(axis=1)
    p0 = np.clip(r[:, None] * (g / g.mean())[None, :], 0.0, 1.0)
    n_rng = np.random.default_rng(seeds[1])
    null = {k: [] for k in STATS}
    for _ in range(n_null):
        cs = n_rng.binomial(n, p0)
        t = trend(cs, n, cs, n, nv)
        for k in STATS:
            null[k].append(t[k])
    fpr = {k: 0 for k in STATS}
    if n_fpr:
        f_rng = np.random.default_rng(seeds[2])
        for _ in range(n_fpr):
            cs = f_rng.binomial(n, p0)
            tf = trend(cs, n, cs, n, nv)
            sg = {k: [] for k in STATS}
            for _ in range(s_fpr):
                t = trend(*split_half(f_rng, cs, n), nv)
                for k in STATS:
                    sg[k].append(t[k])
            for k in STATS:
                v = np.asarray(sg[k])
                v = v[np.isfinite(v)]
                if np.isfinite(tf[k]) and v.size and np.mean(np.sign(v) == np.sign(tf[k])) >= 0.95:
                    fpr[k] += 1
    res = {}
    for k in STATS:
        gs, ns = _summ(guard[k], full[k]), _summ(null[k], full[k])
        guard_pass = gs["frac_same_sign_as_full"] >= 0.95
        null_pass = (full[k] > ns["q975"]) if full[k] > 0 else (full[k] < ns["q025"])
        nv_arr = np.asarray(null[k], float)
        at_least = int(np.sum(nv_arr >= full[k])) if full[k] > 0 else int(np.sum(nv_arr <= full[k]))
        res[k] = {"full_data": full[k], "guard": {**gs, "per_seed": [float(x) for x in guard[k]]},
                  "null": {**ns, "per_replicate": [float(x) for x in null[k]],
                           "replicates_at_least_as_extreme": at_least},
                  "guard_false_positive_rate_on_null": (fpr[k] / n_fpr) if n_fpr else None,
                  "guard_pass": bool(guard_pass), "null_pass": bool(null_pass),
                  "trend_claimable": bool(guard_pass and null_pass)}
    res["_n"] = {"full": full["n"], "full_subset": full["n_subset"],
                 "guard_included_min": int(min(guard_n)) if guard_n else None,
                 "guard_included_max": int(max(guard_n)) if guard_n else None}
    res["_null_shape"] = {"g": [float(x) for x in g],
                          "rule": "p_q(v) = clip(r_q * g(v) / mean_v g, 0, 1), r_q = sum_v c / sum_v n, "
                                  "g(v) = sum_q c / sum_q n"}
    return res


def paper_seeds(root_seed: int, study_index: int):
    """The three seed sequences of study ``study_index``: SeedSequence(root).spawn(...)[i].spawn(3)."""
    kid = np.random.SeedSequence(root_seed, spawn_key=(int(study_index),))
    return kid.spawn(3)


# ----------------------------------------------------------------------------------------------------
# Figures (Appendix I, Figures 38-41)
# ----------------------------------------------------------------------------------------------------

def grid_labels(grid) -> list[str]:
    return [f"{v:g}" for v in grid]


def h_sorted(ids, st) -> tuple[list[int], dict]:
    """Included question indices sorted by hardness; ties keep natural id order."""
    inc = np.where(st["inc"])[0]
    base = sorted(inc.tolist(), key=lambda i: natural_key(ids[i]))
    Hd = {i: hardness_from_percent(100.0 * st["pk"][i]) for i in base}
    return sorted(base, key=lambda i: Hd[i]), Hd


def _legend_hits(fig, ax, xs, ys, marker_s):
    fig.canvas.draw()
    bb = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    rad = np.sqrt(marker_s) / 2.0 * fig.dpi / 72.0
    inside = (pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)
    touch = ((pts[:, 0] >= bb.x0 - rad) & (pts[:, 0] <= bb.x1 + rad) & (pts[:, 1] >= bb.y0 - rad)
             & (pts[:, 1] <= bb.y1 + rad))
    return int(inside.sum()), int(touch.sum())


def _headroom(fig, ax, top_row, xs, ys, marker_s=8):
    """Raise the y top from top_row + 1.5 in steps of 0.5 until the legend covers no point (cap: +6)."""
    bottom = ax.get_ylim()[0]
    h = 1.5
    while h <= 6.0:
        ax.set_ylim(bottom, top_row + h)
        fig.tight_layout()
        n_in, n_touch = _legend_hits(fig, ax, xs, ys, marker_s)
        if n_in == 0 and n_touch == 0:
            return h
        h += 0.5
    raise RuntimeError("the legend still covers points with 6 grid steps of headroom")


def _save_jpg(fig, path, **kw) -> None:
    fig.savefig(str(path), format="jpg", **kw)


def fig_grid_peak(ids, grid, st, out_path=None, win: int = stats.WINDOW):
    """Per-question peak min_p (grid position) vs hardness with the two sliding means (Figure 39).

    Saved (if ``out_path``) inside the paper's style context: JPEG, dpi 300, tight box.
    """
    nv = len(grid)
    order, Hd = h_sorted(ids, st)
    H = np.array([Hd[i] for i in order])
    pt = np.array([int(st["j"][i]) for i in order], dtype=float)
    rec = {"n": len(order)}
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.scatter(H, pt, s=8, color="#888", alpha=0.45, linewidths=0,
                   label=f"Per-question peak min_p (n={len(order)})")
        xb, yb, _, _ = stats.drawn_interior(H, stats.sliding_mean_symmetric(pt, win), win, advance_low_h=False)
        ax.plot(xb, yb, color="#2266cc", lw=2.0, label=f"Sliding avg, all (window={win})")
        mask = pt > 0
        n_sub = int(mask.sum())
        rec["blue"] = {"H": xb, "mean_grid_position": yb}
        if n_sub >= 2:
            xr, yr, _, _ = stats.drawn_interior(H[mask], stats.sliding_mean_symmetric(pt[mask], win), win,
                                                advance_low_h=False)
            ax.plot(xr, yr, color="red", lw=2.0, label=f"Sliding avg, peak min_p > {grid[0]:g} (n={n_sub})")
            rec["red"] = {"n": n_sub, "H": xr, "mean_grid_position": yr}
        ax.set_xlabel("Hardness")
        ax.set_ylabel("Peak min_p")
        ax.set_title("Per-question peak min_p vs hardness")
        apply_h_log_axis(ax, axis="x")
        ax.yaxis.set_major_locator(FixedLocator(list(range(nv))))
        ax.yaxis.set_major_formatter(FixedFormatter(grid_labels(grid)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.legend(loc="upper right", fontsize=11, framealpha=0.92)
        ax.grid(True, which="both", linestyle=":", color="#bbbbbb", alpha=0.5)
        fig.tight_layout()
        rec["headroom_grid_steps"] = _headroom(fig, ax, nv - 1, H, pt)
        if out_path is not None:
            _save_jpg(fig, out_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
    rec["points"] = {"id": [ids[i] for i in order], "H": H, "peak_grid_position": pt.astype(int)}
    return fig, rec


def fig_grid_range(ids, grid, st, out_path=None):
    """Per-question |E_q| (grid values) vs hardness with the OLS fit on log10 H (Figure 40)."""
    nv = len(grid)
    order, Hd = h_sorted(ids, st)
    H = np.array([Hd[i] for i in order])
    size = np.array([int(st["size"][i]) for i in order], dtype=float)
    log_h = np.log10(H)
    slope, intercept, r2 = ols(log_h, size)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.scatter(H, size, s=8, color="#888", alpha=0.45, linewidths=0,
                   label=f"Per-question $|E_q|$ (n={len(order)})")
        x_grid = np.linspace(log_h.min(), log_h.max(), 200)
        fit_label = (f"OLS:  $|E_q| = {slope:.3f}\\cdot\\log_{{10}} H "
                     f"+ {intercept:.3f}$\n$R^2 = {r2:.3f}$")
        ax.plot(10 ** x_grid, slope * x_grid + intercept, color="red", lw=2.2, label=fit_label)
        ax.set_xlabel("Hardness")
        ax.set_ylabel("Efficient range size (grid values)")
        ax.set_title("Efficient range size vs hardness")
        apply_h_log_axis(ax, axis="x")
        ax.yaxis.set_major_locator(FixedLocator(list(range(1, nv + 1))))
        ax.legend(loc="upper right", fontsize=11, framealpha=0.92)
        ax.grid(True, which="both", linestyle=":", color="#bbbbbb", alpha=0.5)
        fig.tight_layout()
        h = _headroom(fig, ax, nv, H, size)
        if out_path is not None:
            _save_jpg(fig, out_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
    rec = {"n": len(order), "ols": {"slope_per_decade": slope, "intercept": intercept, "R2": r2},
           "headroom_grid_steps": h,
           "points": {"id": [ids[i] for i in order], "H": H, "range_size": size.astype(int)}}
    return fig, rec


def fig_grid_coverage(ids, grid, st, out_path=None, n_bins: int = stats.N_BINS):
    """Maximal coverage per log-uniform hardness bin over grid values (Figure 41)."""
    nv = len(grid)
    order, Hd = h_sorted(ids, st)
    H = np.array([Hd[i] for i in order])
    bins = grid_coverage_bins(H, st["L"][order], st["R"][order], nv, n_bins)
    y_vals = [e["height"] for e in bins]
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        if bins:
            ax.bar([e["center"] for e in bins], y_vals, width=[e["width"] for e in bins], color="#2ca02c",
                   edgecolor="#1a6d1a", linewidth=0.6, alpha=0.78, align="center")
        ax.set_xlabel("Hardness")
        ax.set_ylabel("maximal coverage")
        ax.set_title("maximal coverage vs hardness")
        apply_h_log_axis(ax, axis="x")
        ax.set_ylim(coverage_ylim_floor(y_vals), 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(True, which="both", axis="y", linestyle=":", color="#bbbbbb", alpha=0.5)
        fig.tight_layout()
        if out_path is not None:
            _save_jpg(fig, out_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
    rec = {"n": len(order), "n_bins": n_bins,
           "bins": [{**e, "argmax_min_p": [grid[v] for v in e["argmax_positions"]]} for e in bins]}
    return fig, rec


def choose_highlighted(ids, c, st, nv):
    """Figure 38's rule: split grid positions into thirds (numpy.array_split); in each third, among questions
    peaking there, the one with the most correct samples at its peak; ties by the smallest id."""
    thirds = np.array_split(np.arange(nv), 3)
    chosen = []
    for pos in thirds:
        posset = set(pos.tolist())
        cand = [i for i in range(len(ids)) if st["j"][i] in posset and st["pk"][i] > 0]
        if not cand:
            chosen.append(None)
            continue
        top = max(int(c[i, st["j"][i]]) for i in cand)
        tied = sorted(ids[i] for i in cand if int(c[i, st["j"][i]]) == top)
        chosen.append(ids.index(tied[0]))
    return chosen, thirds


def fig_grid_profiles(ids, c, n, grid, st, out_path=None, n_grey: int = N_GREY, seed: int = 0,
                      figsize=(5.3, 3.1), dpi: float = 176.4):
    """Figure 38: measured min_p profiles, 3 highlighted (with exact CP 1-sigma bars) + ``n_grey`` grey.

    Drawn with matplotlib's default font sizes, 5.3 x 3.1 in, saved as JPEG (quality 95) at ``dpi``.
    """
    nv = len(grid)
    chosen, thirds = choose_highlighted(ids, c, st, nv)
    fg = [i for i in chosen if i is not None]
    others = [i for i in sorted(range(len(ids)), key=lambda i: natural_key(ids[i])) if i not in fg]
    rng = np.random.default_rng(seed)
    grey = [others[k] for k in sorted(rng.choice(len(others), size=n_grey, replace=False).tolist())]
    x = np.arange(nv)
    drawn_max = 0.0
    rec = {"highlighted": [], "grey": []}
    fig, ax = plt.subplots(figsize=figsize)
    glines, hlines, herr = [], [], []
    for k, i in enumerate(grey):
        y = 100.0 * c[i] / n[i]
        drawn_max = max(drawn_max, float(y.max()))
        gl, = ax.plot(x, y, color=BG_COLOR, alpha=BG_ALPHA, linewidth=0.8, zorder=1, solid_capstyle="round",
                      label=f"Other sampled questions (n={len(grey)})" if k == 0 else None)
        glines.append(gl)
        rec["grey"].append(ids[i])
    for t, (i, color) in enumerate(zip(chosen, DEFAULT_COLORS)):
        if i is None:
            rec["highlighted"].append({"third": t, "id": None})
            continue
        y = 100.0 * c[i] / n[i]
        ci = [cp_interval_percent(int(c[i, v]), int(n[i, v])) for v in range(nv)]
        lo = np.array([a for a, _ in ci])
        hi = np.array([b for _, b in ci])
        drawn_max = max(drawn_max, float(hi.max()))
        hl, = ax.plot(x, y, color=color, linewidth=2, zorder=5, label=ids[i])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, zorder=5.5)
        ax.scatter(x, y, color=color, s=22, edgecolor="white", linewidth=0.8, zorder=6)
        hlines.append(hl)
        herr.append((lo, hi))
        rec["highlighted"].append({"third": t, "id": ids[i], "peak_index": int(st["j"][i]),
                                   "peak_min_p": grid[int(st["j"][i])], "c_at_peak": int(c[i, st["j"][i]]),
                                   "peak_rate_percent": float(100.0 * st["pk"][i])})
    top = min(105.0, 5.0 * math.ceil(1.1 * drawn_max / 5.0))
    ax.set_xlabel("min_p")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("An example of question profiles")
    ax.set_xticks(x)
    ax.set_xticklabels(grid_labels(grid))
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.25)
    ax.set_xlim(left=-0.3, right=nv - 1 + 0.3)

    def n_in(bb, xs, ys):
        p = ax.transData.transform(np.column_stack([xs, ys]))
        return int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) & (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())

    def seg(bb, xs, ys):
        t = np.linspace(0, 1, 21)
        X = np.concatenate([xs[k] + t * (xs[k + 1] - xs[k]) for k in range(len(xs) - 1)])
        Y = np.concatenate([ys[k] + t * (ys[k + 1] - ys[k]) for k in range(len(ys) - 1)])
        return n_in(bb, X, Y)

    # Raise the top by 5 points until the legend covers no highlighted point, bar end or line sample and no
    # grey line sample (cap 105).
    while True:
        ax.set_ylim(bottom=0, top=top)
        fig.tight_layout()
        fig.canvas.draw()
        bb = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
        hl_cov = (sum(n_in(bb, x, h.get_ydata()) for h in hlines)
                  + sum(n_in(bb, np.r_[x, x], np.r_[lo, hi]) for lo, hi in herr)
                  + sum(seg(bb, x, h.get_ydata()) for h in hlines))
        grey_cov = sum(seg(bb, x, g.get_ydata()) for g in glines)
        if (hl_cov == 0 and grey_cov == 0) or top >= 105.0:
            break
        top = min(105.0, top + 5.0)
    rec.update({"y_top_percent": top, "thirds": [p.tolist() for p in thirds], "grey_alpha": BG_ALPHA})
    if out_path is not None:
        _save_jpg(fig, out_path, dpi=dpi, pil_kwargs={"quality": 95})
        plt.close(fig)
    return fig, rec


def study_facts(ids, grid, c, n, st) -> dict:
    """Counts the Appendix I text quotes: peaks at the smallest and largest grid value, questions with no
    correct sample at the largest value, exclusions by the 0.1% floor, H range, bins wholly above H = 100."""
    nv = len(grid)
    inc = st["inc"]
    H = hardness_array(st["pk"][inc])
    edges_bins = grid_coverage_bins(H, st["L"][inc], st["R"][inc], nv)
    return {
        "questions": len(ids),
        "included": int(inc.sum()),
        "no_correct_sample": int(np.sum(st["pk"] == 0)),
        "below_floor_with_some_correct": int(np.sum((st["pk"] > 0) & (st["pk"] * 100 < MIN_PEAK_PERCENT))),
        "peak_at_smallest_value": int(np.sum(st["j"] == 0)),
        "peak_at_largest_value": int(np.sum(st["j"] == nv - 1)),
        "peak_at_smallest_value_included": int(np.sum((st["j"] == 0) & inc)),
        "peak_at_largest_value_included": int(np.sum((st["j"] == nv - 1) & inc)),
        "no_correct_at_largest_value": int(np.sum(c[:, nv - 1] == 0)),
        "H_min": float(H.min()) if H.size else None,
        "H_max": float(H.max()) if H.size else None,
        "questions_in_bins_wholly_above_H100": int(sum(e["M"] for e in edges_bins if e["H_left"] >= 100.0)),
        "lowest_bar_bins_ge10": (min(e["height"] for e in edges_bins if e["M"] >= 10)
                                 if any(e["M"] >= 10 for e in edges_bins) else None),
        "n_bins_nonempty": len(edges_bins),
    }
