#!/usr/bin/env python3
"""The first-stage screen: which questions get the second stage (Appendix C "Datasets"; Appendix I "Setup").

    python -m onetemp.sampling.screen --rule hitom --counts counts_screen.jsonl --model Qwen/Qwen3-0.6B --out kept.json
    python -m onetemp.sampling.screen --rule minp  --counts counts_minp_screen.jsonl --out kept_minp.json
    python -m onetemp.sampling.screen --rule lcb-mistral --counts counts_lcb_main.jsonl --out lcb_deep.json

Input: counts.jsonl written by a grader from the FIRST-STAGE samples only (one line per question and setting:
{"benchmark", "model", "id", "temperature", "min_p", "n", "c"}). For every question

    peak = max over the stage's settings of c / n          (the best single setting's measured pass rate)

and the rule keeps the question if and only if  low < peak  (when the rule has a lower bound)  and  peak < high
(or peak <= high). The paper's rules (paper_runs.SCREEN_RULES):
    hitom        keep iff peak < 0.30, no lower bound          (500 samples per temperature; Mistral 1,000)
    minp         keep iff peak < 0.30 over the 10 min_p values  (Qwen3-0.6B at tau = 1.5, 100 samples per value)
    lcb-mistral  keep iff 0 < peak < 0.30                       (LiveCodeBench, 1,500 samples per temperature)
    lcb          keep iff 0 < peak <= 0.30                      (LiveCodeBench, GPT-OSS 20B and Qwen3-8B)
The rule compares integers, so no rounding can move a question across 0.30: c / n < 0.30 is 10 c < 3 n.
A question that lacks a setting of the grid, or has a different n somewhere, is reported as undecided and is
neither kept nor dropped, unless --partial-ok.

Output: {"ids": [kept ids, in input order], "rule": ..., "n_kept", "n_dropped", "undecided": [...],
"peaks": {id: [peak, argmax setting]}} - give it to sample_vllm.py --ids-file for the second stage.
Selecting on low observed rates biases the selected questions' first-stage pass rates down; the paper pools
both stages for its tables and figures and says so (Appendix I discusses the effect for Mistral 7B Instruct).
"""
import argparse
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
from onetemp.counts import iter_counts

from . import paper_runs as pr  # noqa: E402


def read_counts(path, benchmark=None, model=None, axis=None, tau=None):
    """The rows of one study in a counts.jsonl file (read and validated by onetemp.counts)."""
    rows = []
    for r in iter_counts(path):
        if benchmark and r["benchmark"] != benchmark:
            continue
        if model and r["model"] != model:
            continue
        mp = r["min_p"]
        if axis == "temperature" and mp is not None:
            continue
        if axis == "min_p":
            if mp is None:
                continue
            if tau is not None and abs(float(r["temperature"]) - tau) > 1e-12:
                continue
        rows.append(r)
    return rows


def setting_of(r, axis):
    return float(r["min_p"]) if axis == "min_p" else float(r["temperature"])


def keep(peak, rule):
    """peak is a Fraction c/n; the comparisons are exact."""
    hi = Fraction(str(rule["high"]))
    ok = peak <= hi if rule["high_inclusive"] else peak < hi
    if rule["low_exclusive"] is not None:
        ok = ok and peak > Fraction(str(rule["low_exclusive"]))
    return ok


def rule_text(rule, axis):
    lo = f"{rule['low_exclusive']} < " if rule["low_exclusive"] is not None else ""
    op = "<=" if rule["high_inclusive"] else "<"
    return f"keep iff {lo}max over {axis} of c/n {op} {rule['high']}"


def screen(rows, rule, axis, settings=None, partial_ok=False):
    by_q, order = {}, []
    for r in rows:
        q = r["id"]
        if q not in by_q:
            by_q[q] = {}
            order.append(q)
        s = setting_of(r, axis)
        if s in by_q[q]:                         # several lines of one cell (e.g. per GPU): add them up
            n0, c0 = by_q[q][s]
            by_q[q][s] = (n0 + int(r["n"]), c0 + int(r["c"]))
        else:
            by_q[q][s] = (int(r["n"]), int(r["c"]))
    grid = sorted(settings) if settings else sorted({s for d in by_q.values() for s in d})
    kept, dropped, undecided, peaks = [], [], [], {}
    for q in order:
        cells = by_q[q]
        ns = {cells[s][0] for s in grid if s in cells}
        if not partial_ok and (any(s not in cells for s in grid) or len(ns) != 1):
            undecided.append(q)
            continue
        best, arg = None, None
        for s in grid:                           # the lowest setting reaching the peak is its argmax
            if s not in cells or cells[s][0] == 0:
                continue
            f = Fraction(cells[s][1], cells[s][0])
            if best is None or f > best:
                best, arg = f, s
        if best is None:
            undecided.append(q)
            continue
        peaks[q] = [float(best), arg]
        (kept if keep(best, rule) else dropped).append(q)
    return {"ids": kept, "n_kept": len(kept), "n_dropped": len(dropped), "undecided": undecided,
            "grid": grid, "peaks": peaks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True, help="counts.jsonl of the first stage")
    ap.add_argument("--rule", required=True, choices=sorted(pr.SCREEN_RULES), help="the paper's keep rule")
    ap.add_argument("--benchmark", help="only rows of this benchmark")
    ap.add_argument("--model", help="only rows of this model (HF id)")
    ap.add_argument("--axis", choices=["temperature", "min_p"], help="default: min_p for --rule minp")
    ap.add_argument("--tau", type=float, help="with --axis min_p: only rows at this temperature")
    ap.add_argument("--settings", help="the stage's grid (comma-separated); default: every setting in the file")
    ap.add_argument("--partial-ok", action="store_true", help="decide questions with missing settings too")
    ap.add_argument("--out", required=True, help="kept-ids JSON to write")
    a = ap.parse_args()
    axis = a.axis or ("min_p" if a.rule == "minp" else "temperature")
    rule = pr.SCREEN_RULES[a.rule]
    rows = read_counts(a.counts, a.benchmark, a.model, axis, a.tau)
    if not rows:
        raise SystemExit("no counts rows match the filters")
    models = sorted({r.get("model") for r in rows})
    if len(models) > 1:
        raise SystemExit(f"rows of several models {models}: pass --model")
    settings = [float(x) for x in a.settings.split(",")] if a.settings else None
    res = screen(rows, rule, axis, settings, a.partial_ok)
    res.update(rule=rule_text(rule, axis), rule_name=a.rule, axis=axis, model=models[0], counts_file=a.counts)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    print(f"{res['rule']}: kept {res['n_kept']}, dropped {res['n_dropped']}, undecided {len(res['undecided'])} "
          f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
