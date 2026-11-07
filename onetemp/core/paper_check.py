"""Check the paper's numbers from counts files: Figure 3, the pass@K tables (Tables 2-6) and the
summary table of Appendix F.

For every (benchmark, model) of the paper found in the given counts files, this runs the paper's
analysis (profile fit with the paper's settings, leave-one-out and, for Figure 3, 10-fold) and
compares, number by number, with ``paper_numbers.json``:

* Figure 3 (Mistral 7B Instruct, LiveCodeBench, n = 359): the optimal allocation's held-out ratio
  to uniform at each K, 10-fold (Figure 3) and leave-one-out (Figure 14b), at 4 decimals;
* pass@K tables (LiveCodeBench, Tables 2-6): every printed cell at 3 decimals, and the bold marks;
* summary table (12 rows): Optimal (LOO) / uniform and the best single setting with its ratio;
* the min_p table of Appendix I (Qwen3-0.6B, Hi-ToM, tau = 1.5, measured rates): the ratio to
  uniform of 7 min_p values at 4 K, the LOO optimum, and the uniform scheme's pass@K.

The included-question count n of every dataset is checked against the paper's figure captions.
It prints one line per comparison (paper | ours | difference | where) and writes JSON with
--out. The exit code is 0 when every number matches.

    python -m onetemp.core.paper_check --counts FILE ... --out results/paper_check.json

FILE ...: one counts file per study; ``python -m onetemp.reproduce tables`` passes the paper's 13 (the 12
temperature sweeps, LiveCodeBench in the question_n parse, and the min_p sweep at tau = 1.5 with 500 samples).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from onetemp.core.bootstrap import ratio_to_uniform
from onetemp.core.counts_io import list_groups, load_counts
from onetemp.core.fit_all import fit_profiles
from onetemp.core.heldout import compare, reference_index
from onetemp.core.passk import TABLE_K, k_grid
from onetemp.core import tables as tb

PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_numbers.json")


def _find(paths, benchmark, model, axis="temperature", fixed_temperature=None):
    """The one given file that holds this study; None when none does. Two files with the same study (for example
    both LiveCodeBench parses, or a min_p file and its screen) are refused rather than one picked silently."""
    hits = []
    for p in paths:
        for g in list_groups(p):
            if g["benchmark"] != benchmark or g["model"] != model or g["axis"] != axis:
                continue
            if axis == "min_p" and fixed_temperature is not None \
                    and abs(float(g["fixed"]["temperature"]) - fixed_temperature) > 1e-9:
                continue
            if p not in hits:
                hits.append(p)
    if len(hits) > 1:
        raise SystemExit(f"{benchmark} / {model} ({axis}) is in {len(hits)} of the given files: {hits}; pass one "
                         f"file per study (python -m onetemp.reproduce tables picks the paper's)")
    return hits[0] if hits else None


def check_minp(path, spec, integer="paper", backend="auto", kernel=None, log=print):
    """The Appendix I table: ratio to uniform on measured rates over the min_p axis."""
    out = {"benchmark": spec["benchmark"], "model": spec["model"], "axis": "min_p", "checks": []}
    add = out["checks"].append
    t0 = time.perf_counter()
    cnt = load_counts(path, spec["benchmark"], spec["model"], axis="min_p",
                      fixed_temperature=spec["fixed_temperature"])
    prof = fit_profiles(cnt, n_mode="question-max", backend=backend, kernel_path=kernel)
    inc = prof.included(mode=spec["mode"])
    L = prof.L(spec["mode"])[inc]
    ks = k_grid(spec["max_k"])
    add({"what": "included questions n", "where": spec["where"], "paper": spec["n"], "ours": int(inc.sum()),
         "ok": spec["n"] == int(inc.sum())})
    loo = compare(L, ks, None, integer, reference_index(prof.alloc_settings, "min_p"))
    rtu = ratio_to_uniform(L, ks, B=2, seed=0)
    ratio = np.asarray(rtu["ratio"])
    for value, row in spec["ratio_to_uniform"].items():
        j = prof.alloc_settings.index(float(value))
        for k, want in zip(spec["K"], row):
            got = f"{ratio[j][ks.index(k)]:.3f}"
            add({"what": f"min_p={value}, K={k}", "where": spec["where"], "paper": want, "ours": got, "ok": got == want})
    by_k = {e["k"]: e for e in loo["per_k"]}
    for k, want_o, want_u in zip(spec["K"], spec["optimal_loo_over_uniform"], spec["uniform_pass_at_k_percent"]):
        got_o = f"{by_k[k]['ratio_optimal_heldout']:.3f}"
        add({"what": f"Optimal (LOO) / uniform, K={k}", "where": spec["where"], "paper": want_o, "ours": got_o,
             "ok": got_o == want_o})
        got_u = f"{100 * rtu['pass_at_k_uniform'][ks.index(k)]:.2f}"
        add({"what": f"uniform pass@K (%), K={k}", "where": spec["where"], "paper": want_u, "ours": got_u,
             "ok": got_u == want_u})
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["all_ok"] = all(c["ok"] for c in out["checks"])
    for c in out["checks"]:
        if not c["ok"] or c["what"].startswith(("included", "Optimal")):
            log(f"  {'OK ' if c['ok'] else 'BAD'} {c['what']}: paper {c['paper']} | ours {c['ours']} | {c['where']}")
    return out


def check_dataset(path, benchmark, model, paper, integer="paper", backend="auto", kernel=None, log=print):
    out = {"benchmark": benchmark, "model": model, "checks": []}
    add = out["checks"].append
    t0 = time.perf_counter()
    cnt = load_counts(path, benchmark, model)
    prof = fit_profiles(cnt, n_mode="question-max", backend=backend, kernel_path=kernel)
    inc = prof.included()
    L = prof.L()[inc]
    n_inc = int(inc.sum())
    ref = reference_index(prof.alloc_settings)
    ks = k_grid(1024)
    loo = compare(L, ks, None, integer, ref)
    want_n = next((s["n"] for s in paper["summary_table"] if s["benchmark"] == benchmark and s["model"] == model), None)
    add({"what": "included questions n", "where": "figure caption", "paper": want_n, "ours": n_inc,
         "ok": want_n == n_inc})
    f3 = paper["figure3"]
    if benchmark == f3["benchmark"] and model == f3["model"]:
        kf = compare(L, ks, 10, integer, ref)
        for name, res, where in (("kfold10", kf, "Figure 3 (10-fold)"), ("loo", loo, "Figure 14(b) (leave-one-out)")):
            for k, want, e in zip(f3["K"], f3[name], res["per_k"]):
                got = e["mean_solved"] / e["case3_total"]
                add({"what": f"optimal held-out / uniform, K={k}", "where": where, "paper": want, "ours": f"{got:.4f}",
                     "diff": float(got - float(want)), "ok": f"{got:.4f}" == want})
    tab = paper["passk_tables"].get(model) if benchmark == "livecodebench" else None
    if tab:
        rows = tb.passk_table_rows(L, prof.alloc_settings, loo, TABLE_K)
        bad = 0
        total = 0
        for want, got in zip(tab["rows"], rows):
            k = want["K"]
            for s, v in want["per_setting"].items():
                total += 1
                mine = tb.fmt3(got["per_setting"][tb.fmt_setting(float(s))])
                if mine != v:
                    bad += 1
                    add({"what": f"pass@{k} at tau={s}", "where": tab["where"], "paper": v, "ours": mine, "ok": False})
            best_cell, uni_cell, opt_cell = tb._aggregate_cells(got)
            agg_mine = {"best_in": tb.fmt3(got["best_in"]), "best_out": tb.fmt3(got["best_out"]),
                        "uniform": uni_cell, "optimal_in": tb.fmt3(got["optimal_in"]),
                        "optimal_out": tb.fmt3(got["optimal_out"])}
            bold_mine = {"best_in_bold": "\\textbf{" + agg_mine["best_in"] + "}" in best_cell.split("\\\\")[0],
                         "best_out_bold": "\\textbf{" + agg_mine["best_out"] + "}" in best_cell.split("\\\\")[-1],
                         "optimal_in_bold": "\\textbf{" + agg_mine["optimal_in"] + "}" in opt_cell.split("\\\\")[0],
                         "optimal_out_bold": "\\textbf{" + agg_mine["optimal_out"] + "}" in opt_cell.split("\\\\")[-1]}
            for key in ("best_in", "best_out", "uniform", "optimal_in", "optimal_out"):
                if key in want:
                    total += 1
                    if agg_mine[key] != want[key]:
                        bad += 1
                        add({"what": f"K={k} {key}", "where": tab["where"], "paper": want[key], "ours": agg_mine[key],
                             "ok": False})
            for key, v in bold_mine.items():
                if key in want and want[key] != v:
                    bad += 1
                    add({"what": f"K={k} bold {key}", "where": tab["where"], "paper": want[key], "ours": v, "ok": False})
        add({"what": f"pass@K table cells ({total})", "where": tab["where"], "paper": total, "ours": total - bad,
             "ok": bad == 0})
    srow = next((s for s in paper["summary_table"] if s["benchmark"] == benchmark and s["model"] == model), None)
    if srow:
        rtu = ratio_to_uniform(L, ks, B=2, seed=0)   # only the full-data ratios are used here
        mine = tb.summary_rows(prof.alloc_settings, loo, rtu)
        for k, want_o, (want_t, want_r), r in zip(srow["K"], srow["optimal_loo_over_uniform"],
                                                  srow["best_single_in_sample"], mine):
            got_o = f"{r['optimal_loo_over_uniform']:.3f}"
            add({"what": f"Optimal (LOO) / uniform, K={k}", "where": "summary table", "paper": want_o, "ours": got_o,
                 "diff": float(r["optimal_loo_over_uniform"] - float(want_o)), "ok": got_o == want_o})
            got_t, got_r = tb.fmt_setting(r["best_single_setting"]), f"{r['best_single_ratio']:.3f}"
            add({"what": f"best single tau (ratio), K={k}", "where": "summary table", "paper": f"{want_t} ({want_r})",
                 "ours": f"{got_t} ({got_r})", "ok": float(got_t) == float(want_t) and got_r == want_r})
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["all_ok"] = all(c["ok"] for c in out["checks"])
    for c in out["checks"]:
        if not c["ok"] or "table cells" in c["what"] or c["what"].endswith("K=2") or "n" == c["what"][-1]:
            diff = f" | diff {c['diff']:+.2e}" if "diff" in c else ""
            log(f"  {'OK ' if c['ok'] else 'BAD'} {c['what']}: paper {c['paper']} | ours {c['ours']}{diff} | {c['where']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the paper's Figure 3, pass@K tables, summary table and the "
                                             "Appendix I min_p table.")
    ap.add_argument("--counts", nargs="+", required=True, help="counts.jsonl file(s)")
    ap.add_argument("--integer", default="paper", choices=("paper", "exact"))
    ap.add_argument("--backend", default="auto", choices=("auto", "c", "numpy"))
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to benchmark:model entries (the min_p table: hitom:Qwen/Qwen3-0.6B:min_p)")
    ap.add_argument("--out", default=None, help="write all comparisons as JSON")
    a = ap.parse_args(argv)
    with open(PAPER, encoding="utf-8") as f:
        paper = json.load(f)
    targets = [(s["benchmark"], s["model"]) for s in paper["summary_table"]]
    if a.only:
        targets = [t for t in targets if f"{t[0]}:{t[1]}" in a.only]
    results = []
    for bench, model in targets:
        path = _find(a.counts, bench, model)
        if path is None:
            print(f"[skip] {bench} / {model}: not in the given counts files")
            results.append({"benchmark": bench, "model": model, "skipped": True})
            continue
        print(f"[check] {bench} / {model}", flush=True)
        r = check_dataset(path, bench, model, paper, a.integer, a.backend, a.kernel)
        n_ok = sum(c["ok"] for c in r["checks"])
        print(f"  -> {n_ok}/{len(r['checks'])} checks match ({r['seconds']} s)", flush=True)
        results.append(r)
    spec = paper.get("minp_table")
    if spec and (not a.only or f"{spec['benchmark']}:{spec['model']}:min_p" in a.only):
        path = _find(a.counts, spec["benchmark"], spec["model"], "min_p", spec["fixed_temperature"])
        if path is None:
            print(f"[skip] {spec['where']}: no min_p group at tau = {spec['fixed_temperature']} in the counts files")
            results.append({"benchmark": spec["benchmark"], "model": spec["model"], "axis": "min_p", "skipped": True})
        else:
            print(f"[check] {spec['benchmark']} / {spec['model']} / min_p at tau = {spec['fixed_temperature']}", flush=True)
            r = check_minp(path, spec, a.integer, a.backend, a.kernel)
            n_ok = sum(c["ok"] for c in r["checks"])
            print(f"  -> {n_ok}/{len(r['checks'])} checks match ({r['seconds']} s)", flush=True)
            results.append(r)
    done = [r for r in results if not r.get("skipped")]
    all_ok = bool(done) and all(r["all_ok"] for r in done)
    print(f"[reproduce] {len(done)} datasets checked, {'all match' if all_ok else 'differences found'}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"results": results, "all_ok": all_ok}, f, indent=1)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
