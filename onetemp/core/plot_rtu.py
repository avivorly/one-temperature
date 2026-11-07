"""Ratio-to-uniform figure (Figure 3; Figures 14-16 of Appendix F) in the paper's style.

Every y value is a pass@K divided by the uniform scheme's pass@K at the same K (dashed line at 1).
Coloured curves: single-setting strategies, drawn only for the settings that are the dominant
(largest) entry of the in-sample optimal integer allocation at some K, each with its paired
bootstrap +/- 1 SE band. Red curve: the held-out optimal allocation (held-out total over the
in-sample uniform total), without a band. Below 1 the y axis is compressed by a factor 0.35
("squeezed below 1"), as in the paper. Figure 3 draws the 10-fold red curve; the appendix grids
draw leave-one-out.

Command line::

    python -m onetemp.core.plot_rtu --rtu rtu.json --heldout heldout_kf10.json \
        --title "Mistral 7B Instruct, LiveCodeBench" --out rtu_mistral.png
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

DISTINCT_PALETTE = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#bcbd22", "#7f7f7f", "#000080", "#008080",
                    "#4b0082", "#006400"]
RED = "#d62728"
SQUEEZE_FACTOR = 0.35
RC = {"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 11,
      "ytick.labelsize": 11, "legend.fontsize": 11}
Y_TICKS = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]


def squeeze(v):
    """y >= 1 unchanged; y < 1 mapped to 1 - (1 - y) * 0.35."""
    a = np.asarray(v, dtype=float)
    return np.where(a >= 1.0, a, 1.0 - (1.0 - a) * SQUEEZE_FACTOR)


def ever_dominant(heldout: dict, n_settings: int) -> set:
    """Settings that are the largest entry of the in-sample integer optimum at some K."""
    out = set()
    for e in heldout.get("per_k", []):
        q = e.get("q_int_full") or []
        if len(q) == n_settings:
            out.add(int(np.argmax(np.asarray(q))))
    return out


def red_curve(heldout: dict):
    ks, ys = [], []
    for e in heldout.get("per_k", []):
        if e.get("case3_total", 0) > 0:
            ks.append(float(e["k"]))
            ys.append(e["mean_solved"] / e["case3_total"])
    return np.asarray(ks), np.asarray(ys)


def render(rtu: dict, heldout: dict, title: str = "", settings=None, label_fmt: str = "T={:.2f}",
           red_label: str = "Optimal (held-out)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    settings = settings if settings is not None else rtu.get("temps")
    k_values = [int(k) for k in rtu["k"]]
    k_arr = np.asarray(k_values, dtype=float)
    ratio = np.asarray(rtu["ratio"])
    se = np.asarray(rtu["ratio_se"])
    dom = sorted(ever_dominant(heldout, len(settings)), key=lambda i: float(settings[i]))
    colour = {ti: DISTINCT_PALETTE[i % len(DISTINCT_PALETTE)] for i, ti in enumerate(dom)}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(10.5, 5.8), constrained_layout=True)
        ax = fig.add_subplot(111)
        for ti in dom:
            y = ratio[ti]
            ax.plot(k_arr, squeeze(y), color=colour[ti], linewidth=1.6, label=label_fmt.format(float(settings[ti])),
                    zorder=2)
            ax.fill_between(k_arr, squeeze(y - se[ti]), squeeze(y + se[ti]), color=colour[ti], alpha=0.15,
                            linewidth=0, zorder=1)
        kx, ky = red_curve(heldout)
        if len(kx):
            ax.plot(kx, squeeze(ky), color=RED, linewidth=2.4, label=red_label, zorder=4)
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, zorder=0)
        ax.yaxis.set_major_locator(FixedLocator([float(squeeze(v)) for v in Y_TICKS]))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{v:.2f}" for v in Y_TICKS]))
        ax.set_xscale("log")
        ax.set_xlim(min(k_values), max(k_values))
        major = [k for k in k_values if k in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}]
        ax.xaxis.set_major_locator(FixedLocator(major))
        ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in major]))
        ax.xaxis.set_minor_locator(FixedLocator(k_values))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=2)
        ax.set_ylabel("pass@K / pass@K(uniform)  (squeezed below 1)")
        ax.set_xlabel("K (samples per question)")
        if title:
            ax.set_title(title)
        ax.grid(True, which="major", axis="both", linestyle=":", color="#bbbbbb", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.92, ncol=1)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Draw the ratio-to-uniform figure (Figure 3 style).")
    ap.add_argument("--rtu", required=True, help="output of onetemp.core.bootstrap")
    ap.add_argument("--heldout", required=True, help="output of onetemp.core.heldout (10-fold for Figure 3, LOO for the appendix)")
    ap.add_argument("--title", default="")
    ap.add_argument("--axis-label", default="T", help="curve label prefix, e.g. T or min_p")
    ap.add_argument("--red-label", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", required=True, help=".png, .jpg or .pdf")
    a = ap.parse_args(argv)
    with open(a.rtu, encoding="utf-8") as f:
        rtu = json.load(f)
    with open(a.heldout, encoding="utf-8") as f:
        ho = json.load(f)
    red = a.red_label or ("Optimal (LOO)" if ho.get("cv_mode") == "loo" else f"Optimal ({ho.get('n_folds')}-fold)")
    fig = render(rtu, ho, a.title, label_fmt=a.axis_label + "={:.2f}" if a.axis_label == "T" else a.axis_label + "={:g}",
                 red_label=red)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"[plot] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
