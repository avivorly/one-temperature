"""Draw a KL x ratio heatmap figure (Appendix G Figures 17-28, Appendix I Figures 42-44) from heatmap.run's numbers.

The look is the paper's: a 2 x 2 grid of K panels, each column normalised to 100, the cell fill a log-scaled viridis
from <= 0.01% to 100%, cell labels from the numbers file, and one colour bar.

cells="data"  cells drawn in data units, KL in nats on x and the ratio on y (Figures 17-20, LiveCodeBench);
cells="equal" every band drawn one unit wide and high, so all cells have the same size (Figures 21-28 and 42-44).

    python -m onetemp.robustness.plot_heatmap heatmap_numbers.json out.jpg [--cells equal]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

PCT_FLOOR, PCT_CEIL = 0.01, 100.0
LOG_FLOOR, LOG_CEIL = math.log10(PCT_FLOOR), math.log10(PCT_CEIL)
VIRIDIS_STOPS = [(0.00, (68 / 255, 1 / 255, 84 / 255)), (0.25, (59 / 255, 82 / 255, 139 / 255)),
                 (0.50, (33 / 255, 145 / 255, 140 / 255)), (0.75, (94 / 255, 201 / 255, 98 / 255)),
                 (1.00, (253 / 255, 231 / 255, 37 / 255))]
VIRIDIS = LinearSegmentedColormap.from_list("viridis_js", VIRIDIS_STOPS, N=512)
TITLE = "Score sensitivity to the question-weighting distribution"


def color_from_percent(pct: float):
    p = min(max(pct, PCT_FLOOR), PCT_CEIL)
    t = min(max((math.log10(p) - LOG_FLOOR) / (LOG_CEIL - LOG_FLOOR), 0.0), 1.0)
    return VIRIDIS(t)[:3]


def text_color_for(face) -> str:
    r, g, b = face[:3]
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"


def render_panel(ax, pk: dict, bands: dict, K: int, cells: str, show_x: bool, show_y: bool):
    col_edges, row_edges = bands["col_edges"], bands["row_edges"]
    counts = pk["counts"]
    n_cols, n_rows = len(col_edges) - 1, len(row_edges) - 1
    ax.set_facecolor("#fafafa")
    for c in range(n_cols):
        tot = sum(counts[c])
        x0, x1 = (col_edges[c], col_edges[c + 1]) if cells == "data" else (c, c + 1)
        for r in range(n_rows):
            y0, y1 = (row_edges[r], row_edges[r + 1]) if cells == "data" else (r, r + 1)
            raw = 100.0 * counts[c][r] / tot if tot else 0.0
            face = color_from_percent(raw)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=face, edgecolor="white", linewidth=0.7,
                                   zorder=2))
            lab = pk["labels"][c][r]
            if lab:
                ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), lab, ha="center", va="center", fontsize=7,
                        color=text_color_for(face), zorder=4)
    if cells == "data":
        ax.set_xlim(col_edges[0], col_edges[-1])
        ax.set_ylim(row_edges[0] + 1e-9, row_edges[-1] - 1e-9)
        ax.set_xticks([0.5 * (col_edges[c] + col_edges[c + 1]) for c in range(n_cols)])
        ax.set_yticks([0.5 * (row_edges[r] + row_edges[r + 1]) for r in range(n_rows)])
    else:
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks([c + 0.5 for c in range(n_cols)])
        ax.set_yticks([r + 0.5 for r in range(n_rows)])
    ax.set_xticklabels(bands["col_labels"] if show_x else [], fontsize=9)
    ax.set_yticklabels(bands["row_labels"] if show_y else [], fontsize=9)
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888")
    ax.grid(False)
    ax.set_title(rf"$k = {K}$", fontsize=11, loc="left", pad=6, color="#333")


def draw_figure(res: dict, out_path, cells: str = "equal", title: str = TITLE) -> Path:
    """The 2 x 2 figure (one to four K) at the paper's size, 13.5 x 10 in at 200 dpi."""
    Ks = res["K"]
    if len(Ks) > 4:
        raise ValueError(f"a figure holds at most four K panels, got {len(Ks)}; run one figure per four K")
    fig = plt.figure(figsize=(13.5, 10.0), dpi=200)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06], hspace=0.30, wspace=0.10, left=0.07, right=0.92,
                          top=0.91, bottom=0.07)
    for idx, K in enumerate(Ks):
        row, col = divmod(idx, 2)
        bottom = idx + 2 >= len(Ks)                  # the lowest panel of its column carries the KL labels
        ax = fig.add_subplot(gs[row, col])
        render_panel(ax, res["per_k"][str(K)], res["bands"], K, cells, show_x=bottom, show_y=(col == 0))
        if bottom:
            ax.set_xlabel(r"KL divergence  [$\mathrm{nats}$]", fontsize=10)
        if col == 0:
            ax.set_ylabel("ratio to optimal score", fontsize=10)
    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(cmap=VIRIDIS, norm=LogNorm(vmin=PCT_FLOOR, vmax=PCT_CEIL))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("% of column  (log scale)", fontsize=9, labelpad=6)
    cbar.set_ticks([0.01, 0.1, 1, 10, 100])
    cbar.ax.set_yticklabels(["≤0.01%", "0.1%", "1%", "10%", "100%"], fontsize=9)
    cbar.ax.minorticks_off()
    cbar.outline.set_visible(False)
    fig.suptitle(title, fontsize=14, weight="bold", y=0.96, x=0.07, ha="left")
    out_path = Path(out_path)
    fig.savefig(out_path, format="jpg", dpi=200, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("numbers", help="heatmap_numbers.json from robustness.heatmap")
    ap.add_argument("out", help="output .jpg")
    ap.add_argument("--cells", choices=["data", "equal"], default="equal")
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    res = json.loads(Path(a.numbers).read_text())
    print(draw_figure(res, a.out, cells=a.cells))


if __name__ == "__main__":
    main()
