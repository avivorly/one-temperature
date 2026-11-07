"""Per-question profile tables and per-dataset summaries (Section 3.2; Appendix E).

Pipeline for one (benchmark, model) temperature sweep:

1. fitted profile per question (Appendix D; ``fitting.py``)
2. ``question_table``: peak tau*_q, peak pass rate, efficient range [t_L, t_R], |E_q| = t_R - t_L and
   hardness H_q (Eq. 3) per question (``curves.fitted_summary``, ``hardness``)
3. ``included_sorted``: the Appendix D filter (fitted peak >= 0.1%, solvable), then a stable sort by H
4. ``summarize``: the numbers behind the three hardness views
   * peak setting: blue = sliding mean over all included questions, red = over those with tau* > 0.1
     (Figures 2 left, 5-7), with the drawn part chosen by ``stats.drawn_interior``
   * efficient-range size: OLS of |E_q| on log10 H (Figures 2 middle, 8-10)
   * maximal coverage R_Q(H) in 17 log-uniform hardness bins (Figures 2 right, 11-13)

``summarize`` works for any setting axis; for min_p studies on measured rates see ``grid_profiles.py``.
"""
from __future__ import annotations

import math

import numpy as np

from . import stats
from .curves import fitted_summary, measured_rates_percent, measured_summary
from .hardness import H_CLAMP, MIN_PEAK_PERCENT, hardness_from_percent, is_included
from .io import FittedProfile, Study


def question_table(study: Study, profiles: dict[str, FittedProfile]) -> list[dict]:
    """One row per question of ``study`` (study order) from its fitted profile.

    Row fields: id, peak_t, peak_val (percent), half_val, t_left, t_right, fwhm, range (= t_right - t_left
    from the rounded ends, as the paper's figure), H (Eq. 3 at the peak) and included (Appendix D).
    Questions whose fitted peak is not positive get peak fields of None and included = False.
    """
    rows = []
    for q in study.questions:
        prof = profiles.get(q.id)
        if prof is None:
            raise KeyError(f"no fitted profile for question {q.id!r} of {study.label}")
        f = fitted_summary(prof)
        rows.append(_row(q.id, f))
    return rows


def question_table_measured(study: Study, n_mode: str = "per-setting") -> list[dict]:
    """The fit-free variant: peak and efficient range from measured rates (``curves.measured_summary``)."""
    rows = []
    for q in study.questions:
        ts, rates = measured_rates_percent(q.settings, q.n, q.c, n_mode=n_mode)
        rows.append(_row(q.id, measured_summary(ts, rates)))
    return rows


def _row(qid: str, f: dict | None) -> dict:
    if not f or f.get("peak_val", 0) <= 0:
        return {"id": qid, "peak_t": None, "peak_val": None, "half_val": None, "t_left": None,
                "t_right": None, "fwhm": None, "range": None, "H": None, "included": False}
    return {"id": qid, **f, "range": f["t_right"] - f["t_left"],
            "H": hardness_from_percent(f["peak_val"]), "included": is_included(f["peak_val"])}


def included_sorted(rows: list[dict], min_peak_percent: float = MIN_PEAK_PERCENT) -> list[dict]:
    """Included questions (positive peak >= ``min_peak_percent``), stably sorted by hardness."""
    keep = [r for r in rows if r["peak_val"] is not None and is_included(r["peak_val"], min_peak_percent)]
    return sorted(keep, key=lambda r: r["H"])


def _line(H, y, win, advance):
    xs, ys, start, stop = stats.drawn_interior(H, stats.sliding_mean_symmetric(y, win), win, advance)
    return {"H": xs, "mean": ys, "start_index": int(start), "stop_index": int(stop), "n_points": int(len(ys)),
            "first": float(ys[0]) if len(ys) else math.nan, "last": float(ys[-1]) if len(ys) else math.nan}


def summarize(sorted_rows: list[dict], advance_low_h: bool, win: int = stats.WINDOW,
              near_zero_cut: float = stats.NEAR_ZERO_CUT, n_bins: int | str = stats.N_BINS) -> dict:
    """All numbers of the three hardness views for one dataset (input: ``included_sorted`` output).

    ``advance_low_h``: apply the low-hardness advance of the drawn lines (the paper: LiveCodeBench and
    MATH-500 yes, Hi-ToM no; ``paper.ADVANCE_LOW_H``).  ``n_bins``: 17 (the paper's figures) or "auto"
    for floor(n / 20), the default stated in Appendix E.1.
    """
    n = len(sorted_rows)
    out: dict = {"n": n, "window": int(win), "near_zero_cut": near_zero_cut, "advance_low_h": bool(advance_low_h)}
    if n < 2:
        out["error"] = "fewer than 2 included questions"
        return out
    H = np.array([r["H"] for r in sorted_rows], dtype=float)
    pt = np.array([r["peak_t"] for r in sorted_rows], dtype=float)
    rng = np.array([r["range"] for r in sorted_rows], dtype=float)
    tl = np.array([r["t_left"] for r in sorted_rows], dtype=float)
    tr = np.array([r["t_right"] for r in sorted_rows], dtype=float)

    out["H_min"], out["H_max"] = float(H[0]), float(H[-1])
    out["n_at_clamp"] = int(np.sum(np.isclose(H, H_CLAMP, rtol=0, atol=1e-12)))
    mask = pt > float(near_zero_cut)
    out["n_red"] = int(mask.sum())
    out["blue"] = _line(H, pt, win, advance_low_h)
    out["red"] = _line(H[mask], pt[mask], win, advance_low_h) if mask.sum() >= 2 else None
    out["ols"] = stats.ols_log10(H, rng)
    nb = stats.auto_bins(n) if n_bins == "auto" else int(n_bins)
    bins = stats.coverage_bins(H, tl, tr, nb)
    out["coverage"] = {"n_bins": nb, "bins": bins, "n_bars": len(bins),
                       "min": min(b["height"] for b in bins) if bins else math.nan,
                       "first": bins[0]["height"] if bins else math.nan,
                       "last": bins[-1]["height"] if bins else math.nan}
    return out


def table_columns() -> list[str]:
    return ["id", "included", "H", "peak_t", "peak_val", "t_left", "t_right", "range", "half_val", "fwhm"]


def write_table_csv(path, rows: list[dict]) -> None:
    """Write a per-question table as CSV (peak_val in percent)."""
    import csv
    cols = table_columns()
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(k) is None else (repr(r[k]) if isinstance(r[k], float) else r[k]) for k in cols])


def compact_summary(s: dict) -> dict:
    """The headline numbers of ``summarize`` (no per-point arrays), for printing and the comparison tables."""
    if "error" in s:
        return dict(s)
    cov = s["coverage"]
    return {
        "n": s["n"], "n_red": s["n_red"], "n_at_clamp": s["n_at_clamp"],
        "H_min": s["H_min"], "H_max": s["H_max"],
        "blue_first": s["blue"]["first"], "blue_last": s["blue"]["last"], "blue_points": s["blue"]["n_points"],
        "red_first": s["red"]["first"] if s["red"] else None,
        "red_last": s["red"]["last"] if s["red"] else None,
        "red_points": s["red"]["n_points"] if s["red"] else 0,
        "ols_slope": s["ols"]["slope"], "ols_intercept": s["ols"]["intercept"], "ols_r2": s["ols"]["r2"],
        "coverage_bars": cov["n_bars"], "coverage_min": cov["min"], "coverage_first": cov["first"],
        "coverage_last": cov["last"],
        "coverage_heights": [b["height"] for b in cov["bins"]],
        "coverage_M": [b["M"] for b in cov["bins"]],
    }
