"""Figure 1: example temperature profiles (Section 3.1, "Unimodality and Typical Profile Shapes").

Three highlighted questions are drawn in colour with their measured pass rates as dots, over a background
of other questions in grey.  The two highlighted curves with the lower peaks are annotated with their peak
heights P_E^1 and P_E^2 (Section 3.1) and the height P_C at which they cross.

The construction follows the paper's figure script:

* every curve is the fitted profile (Appendix D) evaluated on 201 points of its bracket, then upsampled
  with monotone PCHIP and extended smoothly to tau = 0, where greedy decoding pins it at 0% or 100%
  (``smooth_to_zero``);
* background: ``n_background`` questions (60) drawn with ``numpy.random.default_rng(seed)`` (seed 0),
  12 from each peak-temperature band [0, 0.4), [0.4, 1.0), [1.0, 1.8), [1.8, 3.0), [3.0, 10);
* curves whose smoothed trace has an interior dip deeper than 5 points (two peaks) are dropped, and 3 of
  the 9 highest-peaked remaining curves are dropped at random with the same generator;
* the crossing P_C is taken between the two lower peaks (else the highest crossing).

Figure 1 was drawn from an earlier release of the Qwen3-8B LiveCodeBench data, which is not shipped; a redraw
from the shipped counts differs in the grey background (the random sample of other questions), while the
three highlighted profiles coincide.  This module reports the counts it draws.
"""
from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

DEFAULT_COLORS = ("#1f77b4", "#d62728", "#2ca02c")
BG_COLOR = "#9A9A9A"
PC_COLOR = "#5e35b1"
PEAK_T_EDGES = [0.0, 0.4, 1.0, 1.8, 3.0, 10.0]
MARKER_TICK_H = 2.5
X_MAX = 4.0


def smooth_to_zero(xs, ys, n: int = 600):
    """Monotone PCHIP upsampling, with an exponential approach to 0% or 100% at tau = 0.

    Greedy decoding is deterministic, so a profile starts at 0% or 100% (Section 3.1).  When the curve
    starts after tau = 0, the extension y(t) = target + (y0 - target) exp((t - t0) / s) matches the data's
    slope at the first point t0, and a cubic smoothstep makes y(0) exactly the target.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 4:
        return xs, np.clip(ys, 0.0, 100.0)
    if xs[0] > 1e-3:
        first_t = float(xs[0])
        first_y = float(ys[0])
        target = 100.0 if first_y >= 50.0 else 0.0
        delta = first_y - target
        data_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        ok = abs(delta) > 0.05 and abs(data_slope) > 1e-3 and delta * data_slope > 0
        if ok:
            tau = delta / data_slope
            n_pre = max(50, int(round(first_t * 300)))
            t_pre = np.linspace(0.0, first_t, n_pre + 1)[:-1]
            exp_part = target + delta * np.exp((t_pre - first_t) / tau)
            a = exp_part[0] - target
            u = t_pre / first_t
            ss = u * u * (3.0 - 2.0 * u)
            y_pre = exp_part - a * (1.0 - ss)
            xs = np.concatenate((t_pre, xs))
            ys = np.concatenate((y_pre, ys))
        else:
            xs = np.concatenate(([0.0], xs))
            ys = np.concatenate(([target], ys))
    cs = PchipInterpolator(xs, ys, extrapolate=True)
    xd = np.linspace(xs[0], xs[-1], n)
    return xd, np.clip(cs(xd), 0.0, 100.0)


def peak_of(q: dict) -> tuple[float, float]:
    """(peak value in percent, peak temperature): the fitted summary, else the curve's own maximum."""
    f = q.get("fwhm") or {}
    if f.get("peak_val") is not None:
        return f["peak_val"], f["peak_t"]
    cy, cx = q.get("curve_y") or [0], q.get("curve_x") or [0]
    i = max(range(len(cy)), key=lambda j: cy[j])
    return cy[i], cx[i]


def crossing(q1: dict, q2: dict):
    """Where two fitted curves cross: between their peaks if possible, else the highest crossing."""
    x1, y1 = np.array(q1["curve_x"]), np.array(q1["curve_y"])
    x2, y2 = np.array(q2["curve_x"]), np.array(q2["curve_y"])
    d = y1 - np.interp(x1, x2, y2)
    found = []
    for i in range(len(d) - 1):
        if d[i] == 0:
            found.append((float(x1[i]), float(y1[i])))
        elif d[i] * d[i + 1] < 0:
            t = float(x1[i] - d[i] * (x1[i + 1] - x1[i]) / (d[i + 1] - d[i]))
            found.append((t, float(np.interp(t, x1, y1))))
    if not found:
        return None
    p1t, p2t = peak_of(q1)[1], peak_of(q2)[1]
    lo, hi = min(p1t, p2t), max(p1t, p2t)
    inside = [c for c in found if lo <= c[0] <= hi]
    return max(inside or found, key=lambda c: c[1])


def _has_dip(by, depth_pp: float = 5.0) -> bool:
    """True if some value lies more than ``depth_pp`` points below the running maximum on both sides."""
    by = np.asarray(by)
    if by.size < 5:
        return False
    lmax = np.maximum.accumulate(by)
    rmax = np.maximum.accumulate(by[::-1])[::-1]
    return bool(np.minimum(lmax - by, rmax - by).max() > depth_pp)


def choose_background(data: list[dict], highlight: list[str], n_background: int = 60, seed: int = 0,
                      drop_top: int = 3, top_k: int = 9):
    """Pick the grey background curves exactly as the paper's figure: returns (indices into pool, pool, log)."""
    fg = set(highlight)
    pool = [q for q in data if q["name"] not in fg and q.get("curve_x") and q.get("curve_y")]
    rng = np.random.default_rng(seed)
    per_bin = max(1, n_background // (len(PEAK_T_EDGES) - 1))
    pool_pt = np.array([peak_of(q)[1] for q in pool])
    chosen = []
    for k in range(len(PEAK_T_EDGES) - 1):
        in_bin = np.where((pool_pt >= PEAK_T_EDGES[k]) & (pool_pt < PEAK_T_EDGES[k + 1]))[0]
        if len(in_bin) == 0:
            continue
        take = min(per_bin, len(in_bin))
        chosen.extend(rng.choice(in_bin, size=take, replace=False).tolist())
    n_sampled = len(chosen)
    smoothed, kept = {}, []
    for i in chosen:
        bx, by = smooth_to_zero(pool[int(i)]["curve_x"], pool[int(i)]["curve_y"])
        if _has_dip(by):
            continue
        smoothed[i] = (bx, by)
        kept.append(i)
    n_after_dip = len(kept)
    high = sorted(kept, key=lambda i: -peak_of(pool[int(i)])[0])[:top_k]
    if high:
        drop = set(rng.choice(high, size=min(drop_top, len(high)), replace=False).tolist())
        kept = [i for i in kept if i not in drop]
    log = {"pool": len(pool), "sampled": n_sampled, "after_dip_filter": n_after_dip, "drawn": len(kept),
           "background_ids": [pool[int(i)]["name"] for i in kept]}
    return kept, pool, smoothed, log


def build_records(study, profiles, n_mode: str) -> list[dict]:
    """Per-question records for ``plot_profile_examples`` from a study and its fitted profiles.

    The curve is the fitted profile on 201 points of its bracket; the measured points are on the whole
    dataset grid (a setting a question lacks shows as 0), with the same n rule as the fit.
    """
    from .curves import curve_points, fitted_summary, measured_rates_percent
    grid = study.grid
    data = []
    for q in study.questions:
        p = profiles[q.id]
        cx, cy = curve_points(p)
        full = {float(s): (int(ni), int(ci)) for s, ni, ci in zip(q.settings, q.n, q.c)}
        n_arr = [full.get(float(s), (0, 0))[0] for s in grid]
        c_arr = [full.get(float(s), (0, 0))[1] for s in grid]
        mt, mr = measured_rates_percent(grid, n_arr, c_arr, n_mode=n_mode)
        data.append({"name": q.id, "curve_x": cx, "curve_y": cy, "fwhm": fitted_summary(p),
                     "meas_t": mt, "meas_rate": mr})
    return data


def resolve_names(wanted: list[str], names: list[str]) -> list[str]:
    """Map the paper's question names (e.g. "GCD of Subset (v6)") to the ids of a counts file.

    Exact match first; otherwise the unique id that contains the name.  Raises ``KeyError`` listing the
    candidates when a name is missing or ambiguous (pass the ids explicitly then).
    """
    out = []
    names_set = set(names)
    for w in wanted:
        if w in names_set:
            out.append(w)
            continue
        cand = [n for n in names if w in n]
        if len(cand) != 1:
            raise KeyError(f"question {w!r}: {len(cand)} ids contain it {cand[:5]}; pass the ids explicitly")
        out.append(cand[0])
    return out


def plot_profile_examples(data: list[dict], highlight: list[str], out_path: str, bg_alpha: float = 0.35,
                          colors=DEFAULT_COLORS, marker_color: str = "match", marker_shape: str = "line",
                          n_background: int = 60, seed: int = 0, dpi: int = 110, x_max: float = X_MAX,
                          title: str = "An example of question profiles", setting_label: str = "Temperature",
                          labels: list[str] | None = None):
    """Draw Figure 1 from per-question records and save it (PNG at dpi 110 in the paper).

    ``data``: one dict per question, in the dataset's order, with keys
        name, curve_x, curve_y   (``curves.curve_points`` of the fitted profile, percent)
        fwhm                     (``curves.fitted_summary``; None if the fitted peak is not positive)
        meas_t, meas_rate        (measured settings and pass rates in percent)
    ``highlight``: the three question ids drawn in colour; ``labels``: their legend texts (default: ids).
    Returns a dict with what was drawn (P_E^1, P_E^2, P_C, background counts and ids).
    """
    by_name = {}
    for q in data:
        by_name.setdefault(q["name"], q)
    highlight = resolve_names(list(highlight), [q["name"] for q in data])
    qs = [by_name[h] for h in highlight]
    ranked = sorted(range(len(qs)), key=lambda i: peak_of(qs[i])[0])
    lo1, lo2 = ranked[0], ranked[1]
    q_lo1, q_lo2 = qs[lo1], qs[lo2]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    kept, pool, smoothed, log = choose_background(data, highlight, n_background, seed)
    bg_label = f"Other sampled questions (n={len(kept)})"
    for j, i in enumerate(kept):
        bx, by = smoothed[i]
        ax.plot(bx, by, color=BG_COLOR, alpha=bg_alpha, linewidth=0.8, zorder=1, solid_capstyle="round",
                label=bg_label if j == 0 else None)
    legend_labels = list(labels) if labels else [q["name"] for q in qs]
    for q, color, lab in zip(qs, colors, legend_labels):
        x, y = smooth_to_zero(q["curve_x"], q["curve_y"])
        ax.plot(x, y, color=color, linewidth=2, zorder=5, label=lab)
        if q.get("meas_t") and q.get("meas_rate"):
            ax.scatter(q["meas_t"], q["meas_rate"], color=color, s=22, edgecolor="white", linewidth=0.8, zorder=6)

    cr = crossing(q_lo1, q_lo2)
    if cr is not None:
        ax.axhline(cr[1], color="gray", linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.55, zorder=2)
    pe1_v, pe1_t = peak_of(q_lo1)
    pe2_v, pe2_t = peak_of(q_lo2)
    pe1_c, pe2_c = colors[lo1], colors[lo2]
    if pe1_t > pe2_t:
        pe1_v, pe1_t, pe2_v, pe2_t = pe2_v, pe2_t, pe1_v, pe1_t
        pe1_c, pe2_c = pe2_c, pe1_c
    bar_specs = [
        (pe1_t, pe1_v, pe1_c, r"$P_E^{1}$"),
        (cr[0] if cr else None, cr[1] if cr else None, PC_COLOR, r"$P_C$"),
        (pe2_t, pe2_v, pe2_c, r"$P_E^{2}$"),
    ]
    text_bbox = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85)
    cap_w = 0.07
    for t, yv, c, lbl in bar_specs:
        if t is None:
            continue
        ax.vlines(t, 0, yv, color=c, linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.85, zorder=5)
        ec = c if marker_color == "match" else "black"
        ax.hlines(yv, t - cap_w, t + cap_w, color=ec, linewidth=2.0, alpha=1.0, zorder=9)
        if marker_shape == "inv-T":
            ax.vlines(t, yv, yv + MARKER_TICK_H, color=ec, linewidth=2.0, alpha=1.0, zorder=9)
            anchor_y = yv + MARKER_TICK_H
        elif marker_shape == "T":
            ax.vlines(t, yv - MARKER_TICK_H, yv, color=ec, linewidth=2.0, alpha=1.0, zorder=9)
            anchor_y = yv
        else:
            anchor_y = yv
        ax.annotate(lbl, xy=(t, anchor_y), xytext=(0, 6), textcoords="offset points", fontsize=11, color=c,
                    zorder=11, bbox=text_bbox, ha="center", va="bottom")
    ax.set_xlabel(setting_label)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0, top=105.0)
    ax.set_xlim(left=0, right=x_max)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return {"out": str(out_path), "P_E1": {"t": pe1_t, "value_percent": pe1_v},
            "P_E2": {"t": pe2_t, "value_percent": pe2_v},
            "P_C": None if cr is None else {"t": cr[0], "value_percent": cr[1]},
            "n_highlighted": len(qs), "n_background": len(kept),
            "n_total_drawn": len(qs) + len(kept), "bg_alpha": bg_alpha, "background": log}
