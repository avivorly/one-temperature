"""pass@K tables in the style of Appendix F (Tables 2-6) and the summary table of Appendix F.2.

pass@K table, one row per K (the paper's rows 1, 2, 3, 4, 6, ..., 512):

* one column per setting: pass@K with all K samples at that setting, on the fitted profiles
  (the paper's code rounds the expected solved count to 6 decimals before dividing by n);
* "Best tau", two numbers: in-sample (the best setting chosen and scored on all included
  questions) above, held-out (leave-one-out: chosen on the other questions) below;
* "Uniform": the uniform scheme;
* "Optimal (in / out)": the optimal allocation in-sample above, held-out below.

Bold, as in the paper: on the in-sample row the larger of Best tau and Optimal (Optimal on ties
within 1e-6); on the held-out row likewise. Values print with 3 decimals.

Summary table, per K in {1, 16, 256, 1024}: "Optimal (LOO) / uniform" (held-out optimum over the
in-sample uniform total) and "best single tau (in-sample)" with its ratio to uniform in
parentheses (the setting with the highest ratio on all included questions).

Command line::

    python -m onetemp.core.tables --profiles profiles.json --heldout heldout_loo.json --rtu rtu.json \
        --name "Mistral 7B Instruct" --out-prefix tables/mistral_lcb
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import Optional

import numpy as np

from onetemp.core.passk import SUMMARY_K, TABLE_K


def fmt3(rate: float) -> str:
    """Three-decimal fixed format; negative values clamp to 0 (the paper's ``fmt``)."""
    if rate < 0:
        rate = 0.0
    return f"{rate:.3f}"


def fmt_setting(t: float) -> str:
    """A setting as the paper prints it: 0.3, 1.0, 1.125."""
    return repr(float(t))


def per_setting_rates(L: np.ndarray, k_values) -> np.ndarray:
    """pass@K per setting, as the table reads it: round(n - sum_i exp(K L_ij), 6) / n."""
    n = L.shape[0]
    out = np.empty((len(k_values), L.shape[1]))
    for r, k in enumerate(k_values):
        scores = n - np.sum(np.exp(float(k) * L), axis=0)
        out[r] = np.round(scores, 6) / n
    return out


def uniform_rates(L: np.ndarray, k_values) -> np.ndarray:
    """pass@K of the uniform scheme, as the table reads it (the paper's case3, rounded to 6)."""
    n, l = L.shape
    row_sum = np.sum(L, axis=1)
    return np.array([np.round(n - np.sum(np.exp((float(k) / l) * row_sum)), 6) / n for k in k_values])


def passk_table_rows(L: np.ndarray, settings, heldout: dict, k_values=None) -> list:
    """One dict per K with every number of the pass@K table (unformatted)."""
    k_values = list(k_values) if k_values is not None else TABLE_K
    n = L.shape[0]
    rates = per_setting_rates(L, k_values)
    uni = uniform_rates(L, k_values)
    by_k = {int(e["k"]): e for e in heldout["per_k"]}
    rows = []
    for r, k in enumerate(k_values):
        e = by_k.get(int(k))
        row = {"K": int(k), "per_setting": {fmt_setting(t): float(v) for t, v in zip(settings, rates[r])},
               "uniform": float(uni[r])}
        if e is not None:
            row.update({
                "best_in": e["case2_total"] / n,
                "best_in_setting": float(settings[int(e["case2_best_T_idx"])]),
                "best_out": (e["case2_oos_total"] / n) if e.get("case2_oos_total") is not None else None,
                "optimal_in": e["in_sample_solved"] / n,
                "optimal_out": e["mean_rate"],
                "optimal_in_allocation": e["q_int_full"],
            })
        rows.append(row)
    return rows


def _aggregate_cells(row: dict) -> tuple:
    """LaTeX cells Best tau / Uniform / Optimal with the paper's bold rule."""
    eps = 1e-6
    best_in, opt_in = row["best_in"], row["optimal_in"]
    best_out, opt_out = row.get("best_out"), row["optimal_out"]
    if opt_in >= best_in - eps:
        opt_top, best_top = f"\\textbf{{{fmt3(opt_in)}}}", fmt3(best_in)
    else:
        opt_top, best_top = fmt3(opt_in), f"\\textbf{{{fmt3(best_in)}}}"
    if best_out is None:
        best_cell = best_top
        opt_bot = fmt3(opt_out)
    else:
        if opt_out >= best_out - eps:
            opt_bot, best_bot = f"\\textbf{{{fmt3(opt_out)}}}", fmt3(best_out)
        else:
            opt_bot, best_bot = fmt3(opt_out), f"\\textbf{{{fmt3(best_out)}}}"
        best_cell = f"{{\\makecell{{{best_top} \\\\ \\footnotesize {best_bot}}}}}"
    opt_cell = f"{{\\makecell{{{opt_top} \\\\ \\footnotesize {opt_bot}}}}}"
    return best_cell, fmt3(row["uniform"]), opt_cell


def passk_table_latex(rows: list, settings, caption: str, label: str, split: Optional[int] = None) -> str:
    """The paper's layout: low settings in a first sub-table, the others with the aggregate
    columns (Best tau, Uniform, Optimal in / out) in a second one."""
    keys = [fmt_setting(t) for t in settings]
    split = split if split is not None else math.ceil(len(keys) / 2)
    parts = [keys[:split], keys[split:]]
    lines = ["\\begin{table}[p]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\renewcommand{\\arraystretch}{1.1}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", ""]
    for pi, cols in enumerate(parts):
        last = pi == len(parts) - 1
        spec = "r" + "S[table-format=1.3]" * len(cols)
        if last:
            spec += "@{\\hskip 0.8em}cS[table-format=1.3]c"
        head = ["\\textbf{$K$}"] + [f"\\multicolumn{{1}}{{c}}{{\\textbf{{{c}}}}}" for c in cols]
        if last:
            head += ["\\multicolumn{1}{c}{\\textbf{Best $\\tau$}}", "\\multicolumn{1}{c}{\\textbf{Uniform}}",
                     "\\multicolumn{1}{c}{\\textbf{\\makecell{Optimal\\\\(in / out)}}}"]
        lines += ["\\rowcolors{3}{}{gray!8}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
                  " & ".join(head) + " \\\\", "\\midrule"]
        for ri, row in enumerate(rows):
            cells = [f"$\\mathbf{{{row['K']}}}$"] + [fmt3(row["per_setting"][c]) for c in cols]
            if last and "optimal_in" in row:
                cells += list(_aggregate_cells(row))
            lines.append("  " + " & ".join(cells) + " \\\\")
            if ri % 4 == 3 and ri != len(rows) - 1:
                lines.append("  \\addlinespace[2pt]")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        if not last:
            lines += ["\\par\\vspace{8pt}", ""]
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def passk_table_csv(rows: list, settings, path: str) -> None:
    keys = [fmt_setting(t) for t in settings]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K"] + [f"tau={k}" for k in keys] + ["best_in", "best_in_setting", "best_out", "uniform",
                                                          "optimal_in", "optimal_out"])
        for row in rows:
            w.writerow([row["K"]] + [f"{row['per_setting'][k]:.6f}" for k in keys] +
                       [_f6(row.get("best_in")), row.get("best_in_setting"), _f6(row.get("best_out")),
                        f"{row['uniform']:.6f}", _f6(row.get("optimal_in")), _f6(row.get("optimal_out"))])


def _f6(v) -> str:
    return "" if v is None else f"{v:.6f}"


def passk_table_markdown(rows: list, settings) -> str:
    keys = [fmt_setting(t) for t in settings]
    head = ["K"] + keys + ["Best tau (in / out)", "Uniform", "Optimal (in / out)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        best = f"{fmt3(row['best_in'])} / {fmt3(row['best_out'])}" if row.get("best_out") is not None else ""
        opt = f"{fmt3(row['optimal_in'])} / {fmt3(row['optimal_out'])}" if "optimal_in" in row else ""
        out.append("| " + " | ".join([str(row["K"])] + [fmt3(row["per_setting"][k]) for k in keys] +
                                     [best, fmt3(row["uniform"]), opt]) + " |")
    return "\n".join(out) + "\n"


def summary_rows(settings, heldout: dict, rtu: dict, k_values=None) -> list:
    """Summary-table quantities per K: Optimal (LOO) / uniform and the best single setting
    (in-sample) with its ratio to uniform."""
    k_values = list(k_values) if k_values is not None else SUMMARY_K
    by_k = {int(e["k"]): e for e in heldout["per_k"]}
    rk = [int(k) for k in rtu["k"]]
    ratio = np.asarray(rtu["ratio"])          # (n_settings, n_k)
    out = []
    for k in k_values:
        row = {"K": int(k)}
        e = by_k.get(int(k))
        if e is not None:
            row["optimal_loo_over_uniform"] = e["mean_solved"] / e["case3_total"]
        if int(k) in rk:
            col = ratio[:, rk.index(int(k))]
            j = int(np.argmax(col))
            row["best_single_setting"] = float(settings[j])
            row["best_single_ratio"] = float(col[j])
        out.append(row)
    return out


def summary_latex(name: str, rows: list) -> str:
    """Two LaTeX rows in the paper's summary-table format."""
    a = " & ".join(f"${r['optimal_loo_over_uniform']:.3f}$" if "optimal_loo_over_uniform" in r else "" for r in rows)
    b = " & ".join(f"${fmt_setting(r['best_single_setting'])}$ (${r['best_single_ratio']:.3f}$)"
                   if "best_single_setting" in r else "" for r in rows)
    return (f"   & {name} & Optimal (LOO) / uniform & {a} \\\\\n"
            f"  & & best single $\\tau$ (in-sample) & {b} \\\\\n")


def main(argv=None) -> int:
    from onetemp.core.fit_all import MIN_PEAK_PCT, load_profiles
    ap = argparse.ArgumentParser(description="pass@K tables and summary rows in the style of Appendix F.")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--heldout", required=True, help="leave-one-out result of onetemp.core.heldout (max K >= 512)")
    ap.add_argument("--rtu", default=None, help="ratio-to-uniform result of onetemp.core.bootstrap, for the summary rows")
    ap.add_argument("--mode", default="fit", choices=("fit", "meas"))
    ap.add_argument("--min-peak", type=float, default=MIN_PEAK_PCT)
    ap.add_argument("--name", default="Model")
    ap.add_argument("--label", default="tab:passk")
    ap.add_argument("--split", type=int, default=None, help="settings in the first sub-table (default: half)")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args(argv)
    prof = load_profiles(a.profiles)
    inc = prof.included(a.min_peak, a.mode)
    L = prof.L(a.mode)[inc]
    with open(a.heldout, encoding="utf-8") as f:
        ho = json.load(f)
    rows = passk_table_rows(L, prof.alloc_settings, ho, [k for k in TABLE_K if k <= ho["max_k"]])
    s = prof.alloc_settings
    cap = (f"Pass@$K$ for {a.name}. Low settings $\\tau \\in [{fmt_setting(s[0])},\\,"
           f"{fmt_setting(s[(a.split or math.ceil(len(s) / 2)) - 1])}]$ above, the others together with the "
           f"row-aggregate columns below.")
    with open(a.out_prefix + "_passk.tex", "w", encoding="utf-8") as f:
        f.write(passk_table_latex(rows, s, cap, a.label, a.split))
    passk_table_csv(rows, s, a.out_prefix + "_passk.csv")
    with open(a.out_prefix + "_passk.md", "w", encoding="utf-8") as f:
        f.write(passk_table_markdown(rows, s))
    with open(a.out_prefix + "_passk.json", "w", encoding="utf-8") as f:
        json.dump({"settings": s, "n_included": int(inc.sum()), "rows": rows}, f, indent=1)
    msg = f"[tables] {a.out_prefix}_passk.{{tex,csv,md,json}}"
    if a.rtu:
        with open(a.rtu, encoding="utf-8") as f:
            rtu = json.load(f)
        srows = summary_rows(s, ho, rtu)
        with open(a.out_prefix + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(srows, f, indent=1)
        with open(a.out_prefix + "_summary.tex", "w", encoding="utf-8") as f:
            f.write(summary_latex(a.name, srows))
        msg += f" and {a.out_prefix}_summary.{{tex,json}}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
