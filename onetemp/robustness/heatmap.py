"""Random Dirichlet reweighting (Appendix G, Figure 4(b), and Appendix I's min_p version): KL x ratio heatmaps.

For each K, Reference A is the best single setting tau_A under uniform weights, with score_A = S_U(tau_A).
Reweighted B draws W from the Dirichlet smooth-base sampler and elects tau_B = argmax_tau S_W(tau); the score of
tau_B is then taken back under uniform weights, so ratio = S_U(tau_B) / score_A is in (0, 1]. Each W is placed by
KL(W || uniform) in a KL band (columns) and by its ratio in a ratio band (rows). The same W's are used at every K,
so the panels differ only through K.

Cells (Appendix G, "Reading the plot"): counts per (KL band, ratio band); each column normalised to 100; labels
rounded by largest remainders so that a column sums to 100.00, "100" only when every W of the column lands in that
row, "" below 0.005. The top row, ratio >= 0.9, holds the W's whose tau_B loses under 10% of score_A: its
per-band share is the "% losing under 10%" of Figure 4(b) and of the appendix tables. top_row_table.md prints each
share rounded on its own, so it can differ by 0.01 from the figure's label of the same cell (largest remainders).

KL bands: [0,1), [1,2), [2,3), [3,4), [4,5), [5, ln n) with ln n the entropy ceiling; when ln n <= 5 the columns are
1 nat wide up to the last integer below ln n and the last one ends at ln n (Qwen3-0.6B on LiveCodeBench: 0-1, ...,
4-4.44). W's with KL >= ln n (near one-hot) or KL < 0 are not counted.

Ratio bands ("rows"):
  "script"  the paper's LiveCodeBench rule: from floor(10 r_min) / 10 to ceil(10 r_max) / 10 with lines at 0.5, ..., 0.9
            only, r over the ratios of every W of all K panels, the near one-hot W's that are not counted included
            (the script's own rule); --row-bottom fixes the bottom edge, as the paper kept the bottom its original
            sampler found (Figures 17-20);
  "tenths0" ten bands 0.0-0.1, ..., 0.9-1.0 (Figures 21-28, MATH-500 and Hi-ToM);
  "tenths1" nine bands 0.1-0.2, ..., 0.9-1.0 (Figures 42-44, min_p).

Presets reproduce the paper's figures (see docs/robustness.md):
  paper-g-lcb    draws "replay", election "exact", rates "fit", rows "script", cells drawn in nats x ratio units
  paper-g        draws "replay", election "exact", rates "fit", rows "tenths0", equal cells
  paper-i        draws "linear", election "argmax", rates "measured", rows "tenths1", equal cells, questions
                 sorted by id (--order id)

    python -m onetemp.robustness.heatmap --input counts.jsonl --profiles profiles.json --benchmark livecodebench \
        --model mistralai/Mistral-7B-Instruct-v0.3 --preset paper-g-lcb --row-bottom 0.4 --out out/lcb_mistral --plot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import contexts, dirichlet, election
from .data import attach_core_profiles, environment, guard_outputs, load_study

K_PAPER = (4, 32, 128, 512)
N_W = 10000
PRESETS = {
    "paper-g-lcb": {"draw": "replay", "election": "exact", "rates": "fit", "rows": "script", "cells": "data"},
    "paper-g": {"draw": "replay", "election": "exact", "rates": "fit", "rows": "tenths0", "cells": "equal"},
    "paper-i": {"draw": "linear", "election": "argmax", "rates": "measured", "rows": "tenths1", "cells": "equal",
                "order": "id"},
}


def sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def kl_col_edges(n_inc: int) -> np.ndarray:
    """KL band edges: [0, 1, 2, 3, 4, 5, ln n] when ln n > 5, else 1-nat bands up to the last integer below ln n."""
    ceil_ = math.log(n_inc)
    if ceil_ > 5.0:
        return np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ceil_])
    last = math.ceil(ceil_) - 1
    return np.array([float(i) for i in range(last + 1)] + [ceil_])


def kl_col_labels(col_edges) -> list:
    out = []
    for c in range(len(col_edges) - 1):
        lo, hi = col_edges[c], col_edges[c + 1]
        out.append(f"{lo:.0f}–{hi:.2f}" if c == len(col_edges) - 2 else f"{lo:.0f}–{hi:.0f}")
    return out


def ratio_row_edges(rule: str, ratios=None, bottom: float | None = None) -> np.ndarray:
    """Ratio band edges for one figure (all K panels share them)."""
    if rule == "tenths0":
        return np.array([0.0 - 1e-9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9])
    if rule == "tenths1":
        return np.array([0.1 - 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9])
    if rule != "script":
        raise ValueError(f"rows must be script, tenths0 or tenths1, not {rule!r}")
    r = np.asarray(ratios, dtype=np.float64)
    y_bot = math.floor(float(r.min()) * 10.0) / 10.0 if bottom is None else float(bottom)
    y_top = math.ceil(float(r.max()) * 10.0) / 10.0
    h_lines = [v for v in (0.5, 0.6, 0.7, 0.8, 0.9) if y_bot < v < y_top]
    return np.array([y_bot - 1e-9, *h_lines, y_top + 1e-9])


def ratio_row_labels(row_edges) -> list:
    return [f"{row_edges[r] + 1e-9:.1f}–{row_edges[r + 1] - 1e-9:.1f}" for r in range(len(row_edges) - 1)]


def round_pcts_to_sum_100(pcts):
    """The paper's label rounding: two decimals by largest remainders, so a column sums to exactly 100.00."""
    n = len(pcts)
    for i, p in enumerate(pcts):
        if p >= 100.0 - 1e-12:
            result = [0.0] * n
            result[i] = 100.0
            return result
    scaled = [p * 100.0 for p in pcts]
    floored = [min(int(s), 9999) for s in scaled]
    fractional = [s - f for s, f in zip(scaled, floored)]
    residual = int(round(sum(scaled) - sum(floored)))
    idx_by_frac = sorted(range(n), key=lambda i: -fractional[i])
    distributed = 0
    pos = 0
    while distributed < residual and pos < 4 * n:
        i = idx_by_frac[pos % n]
        if floored[i] < 9999:
            floored[i] += 1
            distributed += 1
        pos += 1
    return [f / 100.0 for f in floored]


def cell_label(pct: float, is_true_100: bool = False) -> str:
    """'100' only when every W of the column is in the cell; '' below 0.005; else two decimals."""
    if is_true_100:
        return "100"
    if pct < 0.005:
        return ""
    return f"{pct:.2f}"


def bin_cells(kl, ratio, col_edges, row_edges) -> dict:
    """Counts [column][row], column totals, displayed percentages and labels (column 0 = lowest KL band, row 0 =
    lowest ratio band), plus the top-row share of every column."""
    col_edges = np.asarray(col_edges, dtype=np.float64)
    n_cols, n_rows = len(col_edges) - 1, len(row_edges) - 1
    counts = np.zeros((n_cols, n_rows), dtype=np.int64)
    ci = np.searchsorted(col_edges, kl, side="right") - 1
    ri = np.searchsorted(row_edges, ratio, side="right") - 1
    valid = (ci >= 0) & (ci < n_cols) & (ri >= 0) & (ri < n_rows)
    np.add.at(counts, (ci[valid], ri[valid]), 1)
    tot = counts.sum(axis=1)
    disp, labels, top = [], [], []
    for c in range(n_cols):
        t = int(tot[c])
        d = [0.0] * n_rows if t == 0 else round_pcts_to_sum_100([100.0 * counts[c, r] / t for r in range(n_rows)])
        disp.append([float(v) for v in d])
        labels.append([cell_label(d[r], is_true_100=(t > 0 and counts[c, r] == t)) for r in range(n_rows)])
        top.append(100.0 * counts[c, n_rows - 1] / t if t else float("nan"))
    kl_valid = (ci >= 0) & (ci < n_cols)
    return {"counts": counts.tolist(), "col_totals": tot.tolist(), "displayed_pcts": disp, "labels": labels,
            "top_row_pcts": top, "n_counted": int(valid.sum()), "n_not_counted": int((~valid).sum()),
            "n_kl_at_or_above_ceiling": int((kl >= col_edges[-1]).sum()), "n_kl_below_0": int((kl < 0).sum()),
            "n_kl_in_band_ratio_outside_rows": int((kl_valid & ~valid).sum())}


def pooled_share(counts, col_edges, kl_lo: float, kl_hi: float | None = None) -> float:
    """Top-row share pooled over the KL bands that lie in [kl_lo, kl_hi) (all bands from kl_lo up when kl_hi is None)."""
    counts = np.asarray(counts)
    num = den = 0
    for c in range(counts.shape[0]):
        lo, hi = col_edges[c], col_edges[c + 1]
        if lo >= kl_lo - 1e-12 and (kl_hi is None or hi <= kl_hi + 1e-12):
            num += int(counts[c, -1])
            den += int(counts[c].sum())
    return 100.0 * num / den if den else float("nan")


def run(study, Ks=K_PAPER, n_w: int = N_W, seed: int = dirichlet.SEED, draw: str = "replay", rule: str = "exact",
        rates: str = "fit", rows: str = "script", row_bottom: float | None = None, drop_all_zero: bool = True,
        check_replay: bool = True) -> dict:
    """One figure: the same n_w W's at every K. Returns the numbers of every panel and the per-W arrays."""
    t0 = time.perf_counter()
    ctx = contexts.kl_context(study, Ks, rates=rates)
    dr = dirichlet.sample(ctx["u_sorted"], n_w, draw, seed=seed, check_replay=check_replay)
    kl = dr["kl"]
    all_zero = dr["W"].sum(axis=1) == 0.0
    keep = ~all_zero if drop_all_zero else np.ones(len(kl), dtype=bool)
    per_k, ratios_all, el_all = {}, [], {}
    for K in Ks:
        ck = ctx["per_k"][int(K)]
        el, n_tied = election.elect(rule, dr, ck)
        ratio = ck["ratio_per_t"][el]
        el_all[int(K)] = el
        ratios_all.append(ratio[keep])
        per_k[int(K)] = {"tau_A": ck["tau_A"], "score_A": ck["score_A"], "ratio_per_t": ck["ratio_per_t"].tolist(),
                         "score_per_t": ck["score_per_t"].tolist(), "ratio": ratio, "n_tied": n_tied,
                         "n_ties_over_1": int((n_tied > 1).sum()),
                         "elected_counts": np.bincount(el, minlength=len(ctx["settings"])).tolist()}
    col_edges = kl_col_edges(ctx["n_inc"])
    row_edges = ratio_row_edges(rows, np.concatenate(ratios_all), row_bottom)
    out_k = {}
    for K in Ks:
        d = per_k[int(K)]
        cells = bin_cells(kl[keep], d["ratio"][keep], col_edges, row_edges)
        out_k[str(int(K))] = {k: v for k, v in d.items() if k not in ("ratio", "n_tied")} | cells | {
            "ratio_min": float(d["ratio"][keep].min()), "ratio_max": float(d["ratio"][keep].max()),
            "kl_over_3_pct": pooled_share(cells["counts"], col_edges, 3.0),
            "kl_1_2_pct": pooled_share(cells["counts"], col_edges, 1.0, 2.0)}
    res = {"study": study.label, "benchmark": study.benchmark, "model": study.model, "axis": study.axis,
           "tau": study.tau, "settings": ctx["settings"].tolist(), "n_questions": ctx["n_questions"],
           "n_inc": ctx["n_inc"], "kl_ceiling": math.log(ctx["n_inc"]), "K": [int(k) for k in Ks],
           "method": {"n_w": n_w, "seed": seed, "draw": draw, "election": rule, "rates": rates, "rows": rows,
                      "row_bottom": row_bottom, "drop_all_zero": drop_all_zero},
           "bands": {"col_edges": col_edges.tolist(), "col_labels": kl_col_labels(col_edges),
                     "row_edges": row_edges.tolist(), "row_labels": ratio_row_labels(row_edges)},
           "sampler": {"n_all_zero": dr["n_all_zero"], "n_all_zero_left_out": int((all_zero & ~keep).sum()),
                       "n_gamma_underflow": dr["n_gamma_underflow"],
                       "n_kappa_above_ceiling": int((dr["params"]["kappa"] > math.log(ctx["n_inc"])).sum()),
                       "W_sha256": sha_arr(dr["W"]), "kl_sha256": sha_arr(kl),
                       "calibration": {"c_grid": dr["c_grid"].tolist(), "kl_med": dr["kl_med"].tolist()}},
           "per_k": out_k, "seconds": round(time.perf_counter() - t0, 2)}
    res["summary"] = summary_table(res)
    return res, {"kl": kl, "keep": keep, "elected": el_all, "W": dr["W"]}


def summary_table(res: dict) -> dict:
    """The "% losing under 10%" table: top-row share per K and KL band, KL > 3 and KL 1-2 pooled, and their mean."""
    labels = res["bands"]["col_labels"]
    rows = []
    for K in res["K"]:
        pk = res["per_k"][str(K)]
        rows.append({"K": K, "tau_A": pk["tau_A"], "score_A": pk["score_A"],
                     "top_row_pct": dict(zip(labels, pk["top_row_pcts"])), "kl_over_3_pct": pk["kl_over_3_pct"],
                     "kl_1_2_pct": pk["kl_1_2_pct"]})
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    return {"rows": rows, "kl_over_3_mean_over_K": mean("kl_over_3_pct"), "kl_1_2_mean_over_K": mean("kl_1_2_pct")}

def share_text(pk: dict, c: int) -> str:
    """A top-row share as the figures print it: "100" only when every W of the band is in the top row, a share that
    would round to 100.00 otherwise prints 99.99, "" for an empty band."""
    tot = int(sum(pk["counts"][c]))
    if tot == 0:
        return ""
    top = int(pk["counts"][c][-1])
    if top == tot:
        return "100"
    s = f"{100.0 * top / tot:.2f}"
    return "99.99" if s == "100.00" else s


def table_markdown(res: dict) -> str:
    """The summary as a markdown table (one row per K), shares printed as the figures print them."""
    labels = res["bands"]["col_labels"]
    head = "| K | tau_A | score_A | " + " | ".join(f"KL {l}" for l in labels) + " | KL > 3 | KL 1-2 |"
    sep = "|" + "---|" * (len(labels) + 5)
    lines = [head, sep]
    for r in res["summary"]["rows"]:
        pk = res["per_k"][str(r["K"])]
        cells = " | ".join(share_text(pk, c) for c in range(len(labels)))
        lines.append(f"| {r['K']} | {r['tau_A']:g} | {r['score_A']:.2f} | {cells} | {r['kl_over_3_pct']:.2f} | "
                     f"{r['kl_1_2_pct']:.2f} |")
    lines.append(f"| mean over K | | | {' | '.join('' for _ in labels)} | "
                 f"{res['summary']['kl_over_3_mean_over_K']:.2f} | {res['summary']['kl_1_2_mean_over_K']:.2f} |")
    return "\n".join(lines) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="counts.jsonl (or a profiles.jsonl with p_fit/peak_pct)")
    ap.add_argument("--profiles", default=None, help="profiles.json of the profile fit (python -m onetemp.core.fit_all)")
    ap.add_argument("--benchmark")
    ap.add_argument("--model")
    ap.add_argument("--axis", choices=["temperature", "min_p"])
    ap.add_argument("--tau", type=float, help="the fixed temperature of a min_p study")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="the paper's settings for a figure family")
    ap.add_argument("--K", default="4,32,128,512", help="comma-separated K values (default: the paper's)")
    ap.add_argument("--n-w", type=int, default=N_W, help="number of random W's (default 10000)")
    ap.add_argument("--seed", type=int, default=dirichlet.SEED)
    ap.add_argument("--draw", choices=dirichlet.MODES)
    ap.add_argument("--election", choices=["argmax", "exact"])
    ap.add_argument("--rates", choices=["fit", "measured"])
    ap.add_argument("--rows", choices=["script", "tenths0", "tenths1"])
    ap.add_argument("--row-bottom", type=float, default=None, help="bottom ratio edge for --rows script")
    ap.add_argument("--keep-all-zero", action="store_true", help="count all-zero W's (only the linear draw makes them)")
    ap.add_argument("--no-replay-check", action="store_true", help="skip the bit-for-bit check of the replayed draws")
    ap.add_argument("--out", required=True, help="output directory; existing output files are never overwritten")
    ap.add_argument("--plot", action="store_true", help="also draw the heatmap JPG")
    ap.add_argument("--cells", choices=["data", "equal"], help="heatmap cell geometry")
    ap.add_argument("--order", choices=["file", "id"],
                    help="question order: as in the file (default), or sorted by id (preset paper-i, as the paper's "
                         "min_p figures read them; the order decides which question each random weight goes to)")
    a = ap.parse_args(argv)
    cfg = dict(PRESETS.get(a.preset, {"draw": "replay", "election": "exact", "rates": "fit", "rows": "tenths0",
                                      "cells": "equal"}))
    cfg.setdefault("order", "file")
    for k in ("draw", "election", "rates", "rows", "cells", "order"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    out = Path(a.out)
    jpath, tpath, ipath = out / "heatmap_numbers.json", out / "top_row_table.md", out / "heatmap.jpg"
    guard_outputs([jpath, tpath] + ([ipath] if a.plot else []))
    Ks = sorted({int(k) for k in a.K.split(",")})
    if not Ks or Ks[0] < 1:
        sys.exit("--K must hold positive integers")
    study = load_study(a.input, a.benchmark, a.model, a.axis, a.tau,
                       require_fit=(cfg["rates"] == "fit" and not a.profiles))
    if a.profiles:
        study = attach_core_profiles(study, a.profiles)
    if cfg["order"] == "id":
        # The paper's min_p heatmap script (Figures 42-44) read the questions sorted by id (string order). The order
        # matters: the W's are assigned to the questions in hardness order, and questions of equal hardness (common
        # with measured rates) keep their input order.
        study = study.take(np.array(sorted(range(len(study.ids)), key=lambda i: study.ids[i]), dtype=np.int64))
    res, arrays = run(study, Ks, a.n_w, a.seed, cfg["draw"], cfg["election"], cfg["rates"], cfg["rows"],
                      a.row_bottom, drop_all_zero=not a.keep_all_zero, check_replay=not a.no_replay_check)
    res["method"]["question_order"] = cfg["order"]
    res["environment"] = environment()
    res["dropped_settings"] = study.dropped_settings
    out.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(_jsonable(res), indent=1))
    tpath.write_text(table_markdown(res))
    print(table_markdown(res))
    if a.plot:
        from .plot_heatmap import draw_figure
        draw_figure(res, ipath, cells=cfg["cells"])
    print(f"wrote {jpath} ({res['seconds']} s)")


if __name__ == "__main__":
    main()
