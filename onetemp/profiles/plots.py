"""Redraw the hardness views: Figure 2 (main text) and Figures 5-13 (Appendix E).

Each ``fig_*`` function reproduces the paper's drawing code call for call (sizes, colours, labels, tick
positions, legend placement), so that with the same matplotlib version the saved images match the paper's.
They take the hardness-sorted included rows of ``analysis.included_sorted``.

* ``fig_peak``        per-question peak setting vs hardness + sliding means   (Figures 2 left, 5-7)
* ``fig_range``       efficient-range size vs hardness + OLS on log10 H        (Figures 2 middle, 8-10)
* ``fig_coverage``    maximal coverage per log-uniform hardness bin, Eq. 5     (Figures 2 right, 11-13)
* ``figure2_row``     the three panels side by side, as Figure 2

Appendix E figures are saved like the paper: JPEG, dpi 300, ``bbox_inches="tight"``.
"""
from __future__ import annotations

import io as _io
from contextlib import contextmanager

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, PercentFormatter  # noqa: E402

from . import stats  # noqa: E402

#: Font sizes of every paper figure drawn here.
PAPER_RC = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
}

#: Hardness-axis ticks of every hardness figure.
H_TICK_VALS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
H_TICK_TEXT = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100", "200", "500", "1000", "10000"]


@contextmanager
def paper_style():
    """Matplotlib settings of the paper's figures (Agg backend, the font sizes above)."""
    with plt.rc_context(PAPER_RC):
        yield


def apply_h_log_axis(ax, axis: str = "x") -> None:
    """Log-scaled hardness axis with the paper's fixed ticks."""
    if axis == "y":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(H_TICK_VALS))
        ax.yaxis.set_major_formatter(FixedFormatter(H_TICK_TEXT))
    else:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(H_TICK_VALS))
        ax.xaxis.set_major_formatter(FixedFormatter(H_TICK_TEXT))


def fig_peak(rows, advance_low_h: bool, win: int = stats.WINDOW, near_zero_cut: float = stats.NEAR_ZERO_CUT,
             setting_symbol: str = r"\tau", setting_name: str = "temperature"):
    """Per-question peak setting vs hardness with two symmetric sliding means (window ``win``).

    Grey: one dot per included question.  Blue: sliding mean over all of them.  Red: sliding mean over the
    questions whose peak exceeds ``near_zero_cut`` (removes the near-zero cluster).  Only the part chosen by
    ``stats.drawn_interior`` is drawn.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    n = len(rows)
    if n >= 2:
        H = np.array([r["H"] for r in rows], dtype=float)
        pt = np.array([r["peak_t"] for r in rows], dtype=float)
        ax.scatter(H, pt, s=8, color="#888", alpha=0.45, linewidths=0,
                   label=f"Per-question peak ${setting_symbol}$ (n={n})")
        xb, yb, _, _ = stats.drawn_interior(H, stats.sliding_mean_symmetric(pt, win), win, advance_low_h)
        ax.plot(xb, yb, color="#2266cc", lw=2.0, label=f"Sliding avg, all (window={int(win)})")
        mask = pt > float(near_zero_cut)
        n_sub = int(mask.sum())
        if n_sub >= 2:
            xr, yr, _, _ = stats.drawn_interior(H[mask], stats.sliding_mean_symmetric(pt[mask], win), win,
                                                advance_low_h)
            ax.plot(xr, yr, color="red", lw=2.0,
                    label=(f"Sliding avg, peak ${setting_symbol} > {near_zero_cut:.2f}$ (n={n_sub})"))
    ax.set_xlabel("Hardness")
    ax.set_ylabel(f"Peak {setting_name}")
    ax.set_title(f"Per-question peak {setting_name} vs hardness")
    apply_h_log_axis(ax, axis="x")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.92)
    ax.grid(True, which="both", linestyle=":", color="#bbbbbb", alpha=0.5)
    fig.tight_layout()
    return fig


def fig_range(rows):
    """Per-question efficient-range size t_R - t_L vs hardness, with the OLS line on log10 H (legend)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    n = len(rows)
    if n >= 2:
        H = np.array([r["H"] for r in rows], dtype=float)
        rng = np.array([r["t_right"] - r["t_left"] for r in rows], dtype=float)
        log_h = np.log10(H)
        ax.scatter(H, rng, s=8, color="#888", alpha=0.45, linewidths=0,
                   label=f"Per-question $t_R - t_L$ (n={n})")
        slope, intercept = np.polyfit(log_h, rng, 1)
        y_hat = slope * log_h + intercept
        ss_res = float(np.sum((rng - y_hat) ** 2))
        ss_tot = float(np.sum((rng - rng.mean()) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        x_grid = np.linspace(log_h.min(), log_h.max(), 200)
        y_grid = slope * x_grid + intercept
        fit_label = (f"OLS:  $t_R - t_L = {slope:.3f}\\cdot\\log_{{10}} H "
                     f"+ {intercept:.3f}$\n$R^2 = {r2:.3f}$")
        ax.plot(10 ** x_grid, y_grid, color="red", lw=2.2, label=fit_label)
    ax.set_xlabel("Hardness")
    ax.set_ylabel("Efficient range size")
    ax.set_title("Efficient range size vs hardness")
    apply_h_log_axis(ax, axis="x")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.92)
    ax.grid(True, which="both", linestyle=":", color="#bbbbbb", alpha=0.5)
    fig.tight_layout()
    return fig


def coverage_ylim_floor(heights) -> float:
    """The paper's y floor: the smallest bar rounded down to 5 points, minus 5 points (at least 0)."""
    if not heights:
        return 0.0
    min_pct = min(heights) * 100.0
    return max(0.0, (np.floor(min_pct / 5.0) * 5.0 - 5.0) / 100.0)


def fig_coverage(rows, n_bins: int = stats.N_BINS):
    """Maximal coverage (Eq. 5) per log-uniform hardness bin, one green bar per non-empty bin."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    n = len(rows)
    y_vals = []
    if n >= 2:
        H = np.array([r["H"] for r in rows], dtype=float)
        tl = np.array([r["t_left"] for r in rows], dtype=float)
        tr = np.array([r["t_right"] for r in rows], dtype=float)
        bins = stats.coverage_bins(H, tl, tr, n_bins)
        centers = [b["center"] for b in bins]
        widths = [b["width"] for b in bins]
        y_vals = [b["height"] for b in bins]
        if centers:
            ax.bar(centers, y_vals, width=widths, color="#2ca02c", edgecolor="#1a6d1a",
                   linewidth=0.6, alpha=0.78, align="center")
    ax.set_xlabel("Hardness")
    ax.set_ylabel("maximal coverage")
    ax.set_title("maximal coverage vs hardness")
    apply_h_log_axis(ax, axis="x")
    ax.set_ylim(coverage_ylim_floor(y_vals), 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(True, which="both", axis="y", linestyle=":", color="#bbbbbb", alpha=0.5)
    fig.tight_layout()
    return fig


def save_appendix_jpg(fig, path) -> None:
    """Save as the paper's Appendix E images: JPEG, dpi 300, tight bounding box."""
    fig.savefig(str(path), bbox_inches="tight", dpi=300, format="jpg")


def draw_appendix_set(rows, advance_low_h: bool, out_prefix: str, n_bins: int = stats.N_BINS,
                      titles: bool = True) -> dict:
    """Draw and save ``<out_prefix>_AveTemp.jpg``, ``_RangeSize.jpg`` and ``_relC.jpg`` (Figures 5-13 style).

    ``titles=False`` removes the in-image titles (the paper keeps them).  Returns {kind: path}.
    """
    written = {}
    with paper_style():
        for kind, fn in (("AveTemp", lambda: fig_peak(rows, advance_low_h)),
                         ("RangeSize", lambda: fig_range(rows)),
                         ("relC", lambda: fig_coverage(rows, n_bins))):
            fig = fn()
            if not titles:
                fig.axes[0].set_title("")
            path = f"{out_prefix}_{kind}.jpg"
            save_appendix_jpg(fig, path)
            plt.close(fig)
            written[kind] = path
    return written


def _panel_rgb(fig, dpi: int) -> np.ndarray:
    """Render ``fig`` like the paper's Figure 2 panels: fixed axes rectangle, full canvas, no tight box."""
    fig.subplots_adjust(left=0.135, right=0.97, top=0.91, bottom=0.135)
    buf = _io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png")
    buf.seek(0)
    img = plt.imread(buf, format="png")  # float32 in [0, 1], RGBA
    rgb = np.round(img[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def figure2_row(rows, advance_low_h: bool, out_path, dpi: int = 350, n_bins: int = stats.N_BINS) -> str:
    """Figure 2: peak, range-size and coverage panels side by side in one JPEG (quality 92).

    Each panel is drawn by the functions above, given the same axes rectangle, rendered at ``dpi`` and the
    three images are placed next to each other, as the paper's figure.
    """
    with paper_style():
        panels = []
        for fn in (lambda: fig_peak(rows, advance_low_h), lambda: fig_range(rows),
                   lambda: fig_coverage(rows, n_bins)):
            fig = fn()
            panels.append(_panel_rgb(fig, dpi))
            plt.close(fig)
    h = max(p.shape[0] for p in panels)
    if any(p.shape[0] != h for p in panels):
        raise RuntimeError("panel heights differ; all three panels are drawn at the same size")
    row = np.concatenate(panels, axis=1)
    plt.imsave(str(out_path), row, format="jpeg", pil_kwargs={"quality": 92, "optimize": True, "dpi": (0, 0)})
    return str(out_path)
