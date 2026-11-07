"""Draw beta* against D (Appendix H Figures 29-31 and 33, Appendix I Figure 45) and the two-panel Figure 4.

draw_study: one 1 x 2 figure per study, a 10% loss on the left and a 1% loss on the right, one curve per K. Points
with beta* > 10^6 sit on the top row as open markers; the last point of each curve is the first D whose window holds
every question.
draw_figure4: (a) the 10% curves of one study (Mistral 7B Instruct on LiveCodeBench in the paper), (b) the 24 cells of
Figure 4(b) from fig4b's merged numbers, K by KL band, each cell the share of W's losing under 10%.

    python -m onetemp.robustness.plot_prot study beta_star.json out.jpg [--cut]
    python -m onetemp.robustness.plot_prot figure4 beta_star.json fig4b_cells.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

COLOR_BY_K = {4: "#3b528b", 32: "#21918c", 128: "#5ec962", 512: "#e3a01b", 256: "#e3a01b"}
MARKER_BY_K = {4: "o", 32: "s", 128: "^", 512: "D", 256: "D"}
FALLBACK = ["#440154", "#3b528b", "#21918c", "#5ec962", "#e3a01b", "#d1495b", "#7f7f7f"]
Y_INF = 1e7
FS = 8.0
DEC_LABELS = ["1", "10"] + [f"$10^{{{e}}}$" for e in range(2, 7)]
INF_LABEL = "> $10^{6}$"


def _style(K, i, Ks=()):
    if K == 256 and 512 in Ks:                     # both on one figure: K = 256 gets its own look
        return "#d1495b", "v"
    return COLOR_BY_K.get(K, FALLBACK[i % len(FALLBACK)]), MARKER_BY_K.get(K, "ox^sDv<>"[i % 8])


def _beta_drawn(res) -> np.ndarray:
    """beta* with the points where C is empty set to NaN: no setting loses that much, so there is nothing to draw."""
    b = np.asarray(res["beta_star"], dtype=float).copy()
    ce = res.get("c_empty")
    if ce is None:                                  # older files: read the reason from the points
        ce = np.zeros(b.shape, dtype=bool)
        thr_i = {round(float(t), 9): a for a, t in enumerate(res["thr"])}
        k_i = {int(K): k for k, K in enumerate(res["K"])}
        for pt in res.get("points", []):
            if pt.get("why") == "C is empty":
                ce[thr_i[round(float(pt["thr"]), 9)], k_i[int(pt["K"])], pt["D_index"]] = True
    b[np.asarray(ce, dtype=bool)] = np.nan
    return b


def _cut_x(res) -> float:
    """The last D before any curve of the figure exceeds 10^6 (Appendix I's cut panels); C-empty curves ignored."""
    b = _beta_drawn(res)
    D = np.asarray(res["D_grid"])
    bad = np.isinf(b).any(axis=(0, 1))
    first = int(np.argmax(bad)) if bad.any() else len(D)
    return float(D[max(first - 1, 0)])


def draw_study(res: dict, out_path, title: str | None = None, cut: bool = False, x_max: float | None = None):
    b = _beta_drawn(res)
    D, Ks, thrs = np.asarray(res["D_grid"], dtype=float), res["K"], res["thr"]
    reached = ~np.all(np.isnan(b), axis=(0, 1))
    if x_max is None:
        x_max = _cut_x(res) if cut else max(5.0, float(np.ceil(D[reached].max())) if reached.any() else 5.0)
    plt.rcParams.update({"font.size": FS, "axes.titlesize": FS + 0.5, "axes.labelsize": FS,
                         "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5})
    fig, axes = plt.subplots(1, len(thrs), figsize=(3.5 * len(thrs), 3.6), dpi=200, sharex=True, sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor("white")
    ticks, tick_labels = [10.0 ** e for e in range(7)] + [Y_INF], DEC_LABELS + [INF_LABEL]
    for a, (ax, thr) in enumerate(zip(axes[0], thrs)):
        for k, K in enumerate(Ks):
            col, mk = _style(K, k, Ks)
            bk_all = b[a, k]
            keep = ~np.isnan(bk_all) & (D <= x_max + 1e-9)
            dk, bk = D[keep], bk_all[keep]
            if not len(dk):
                continue
            fin = np.isfinite(bk)
            ax.plot(dk, np.where(fin, bk, Y_INF), color=col, lw=1.1, zorder=2, label=f"K = {K}")
            ax.plot(dk[fin], bk[fin], ls="none", marker=mk, ms=3.2, color=col, zorder=3, clip_on=False)
            if (~fin).any():
                ax.plot(dk[~fin], np.full((~fin).sum(), Y_INF), ls="none", marker=mk, ms=3.6, mfc="white", mec=col,
                        mew=0.9, zorder=4, clip_on=False)
        ax.set_title(f"{round((1 - thr) * 100, 6):g}% loss", pad=7.0)
        ax.set_yscale("log")
        ax.grid(True, which="major", color="#dddddd", lw=0.5, zorder=0)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.8, Y_INF)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticklabels(tick_labels)
    fig.supxlabel("D: protected half-width around H = K (decades)", fontsize=FS, y=0.02)
    fig.supylabel("β*: max weight change outside\nneeded for the loss", fontsize=FS, x=0.01, y=0.465)
    fig.suptitle(title or f"{res['model']}, {res['benchmark']} (n = {res['n_inc']})", fontsize=FS + 1.0, y=0.985)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.925),
               handlelength=1.8, columnspacing=1.2)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.78, bottom=0.15, wspace=0.06)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)
    return out_path


def draw_figure4(res: dict, cells: dict, out_path, thr: float = 0.9, x_max: float = 2.5):
    """Figure 4: (a) beta* against D at a 10% loss, (b) the share of W's losing under 10% per K and KL band."""
    b = _beta_drawn(res)
    D = np.asarray(res["D_grid"], dtype=float)
    a = [round(t, 9) for t in res["thr"]].index(round(thr, 9))
    Ks = res["K"]
    fig = plt.figure(figsize=(7.2, 2.6), dpi=200)
    ax = fig.add_axes([0.08, 0.2, 0.36, 0.68])
    for k, K in enumerate(Ks):
        col, mk = _style(K, k, Ks)
        keep = ~np.isnan(b[a, k]) & (D <= x_max + 1e-9)
        dk, bk = D[keep], b[a, k][keep]
        fin = np.isfinite(bk)
        ax.plot(dk[fin], bk[fin], color=col, lw=1.3, marker=mk, ms=2.6, label=f"K = {K}")
        if (~fin).any():
            ax.plot(dk[~fin], np.full((~fin).sum(), Y_INF), ls="none", marker=mk, ms=3.2, mfc="white", mec=col,
                    mew=0.9, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(0.8, Y_INF)
    ax.set_xlim(0, x_max)
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(7)] + [Y_INF]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels(DEC_LABELS + [INF_LABEL], fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(True, color="#e5e5e5", lw=0.5)
    ax.set_xlabel("protected half-width $D$ (decades)", fontsize=7)
    ax.set_ylabel(r"reweighting factor $\beta^*$", fontsize=7)
    ax.set_title(r"(a) Worst-case reweighting: $\beta^*$ to lose $10\%$", fontsize=7.5, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    bx = fig.add_axes([0.56, 0.2, 0.42, 0.68])
    order = sorted(cells["K"], reverse=True)
    vals = [[float(c["share_pct"]) for c in cells["cells"][str(K)] if c.get("share_pct") is not None] for K in order]
    smin = min(min(r) for r in vals if r)
    for r, K in enumerate(order):
        col, _ = _style(K, r)
        for c, cell in enumerate(cells["cells"][str(K)]):
            share = cell.get("share_pct")
            t = 0.0 if share is None else min(1.0, 0.05 + (float(share) - smin) / 42.0)
            face = tuple((1 - t) + t * np.array(to_rgb(col)))
            bx.add_patch(Rectangle((c, len(order) - 1 - r), 1, 1, facecolor=face, edgecolor="white", lw=1.0))
            lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
            bx.text(c + 0.5, len(order) - 1 - r + 0.5, cell["label"], ha="center", va="center", fontsize=6.5,
                    color="#222222" if lum > 0.55 else "white")
        bx.text(-0.15, len(order) - 1 - r + 0.5, f"K = {K}", ha="right", va="center", fontsize=6.5, color=col)
    bx.set_xlim(0, len(cells["bands"]))
    bx.set_ylim(0, len(order))
    bx.set_xticks(np.arange(len(cells["bands"])) + 0.5)
    bx.set_xticklabels(cells["bands"], fontsize=6.5)
    bx.set_yticks([])
    for s_ in bx.spines.values():
        s_.set_visible(False)
    bx.set_xlabel(r"KL$(W\,\|\,\mathrm{uniform})$ band (nats)", fontsize=7)
    bx.set_title(r"(b) Random reweighting: % losing under $10\%$", fontsize=7.5, loc="left")
    out_path = Path(out_path)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study")
    s.add_argument("beta_star_json")
    s.add_argument("out")
    s.add_argument("--cut", action="store_true", help="show D only up to the last value before any curve passes 1e6")
    s.add_argument("--title", default=None)
    f = sub.add_parser("figure4")
    f.add_argument("beta_star_json")
    f.add_argument("fig4b_cells_json")
    f.add_argument("out")
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    res = json.loads(Path(a.beta_star_json).read_text())
    if a.cmd == "study":
        print(draw_study(res, a.out, title=a.title, cut=a.cut))
    else:
        print(draw_figure4(res, json.loads(Path(a.fig4b_cells_json).read_text()), a.out))


if __name__ == "__main__":
    main()
