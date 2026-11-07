"""End-to-end comparison of one (benchmark, model, setting axis): counts -> profiles -> comparison.

Steps (each also available as its own command):

1. ``fit_all``: fit every question's profile (Appendix D) and keep the questions whose fitted peak
   is >= 0.1 percent;
2. ``heldout``: at every K of the paper's grid, the optimal allocation in-sample and held-out
   (leave-one-out and/or k-fold with the paper's fold rule), best single setting in-sample and
   held-out, uniform scheme (Appendix F);
3. ``bootstrap``: ratio to uniform of every single setting with paired-bootstrap SE bands;
4. ``tables``: the pass@K table (Tables 2-6 style) and the summary-table rows;
5. ``plot_rtu`` (with --plot): the ratio-to-uniform figure (Figure 3 style).

Outputs in --out-dir: profiles.json, heldout_<folds>.json, rtu.json, passk_table.{tex,csv,md,json},
summary.{json,tex}, rtu_<folds>.png, and run.json (inputs, options, versions, timings).

Example (Figure 3 and Table 3, Mistral 7B Instruct on LiveCodeBench)::

    python -m onetemp.core.pipeline \
        --counts data/counts/livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl \
        --folds loo 10 --plot --name "Mistral 7B Instruct" --out-dir results/fig3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import scipy

from onetemp.core.bootstrap import SEED_SCHEMES, ratio_to_uniform
from onetemp.core.counts_io import AXES, N_MODES, load_counts
from onetemp.core.fit_all import ITERATIONS, MIN_PEAK_PCT, fit_profiles
from onetemp.core.heldout import compare, reference_index
from onetemp.core.passk import TABLE_K, k_grid
from onetemp.core import tables as tb


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(counts_path: str, out_dir: str, benchmark=None, model=None, axis: str = "temperature",
        fixed_temperature=None, n_mode: str = "question-max", backend: str = "auto", kernel=None,
        mode: str = "fit", min_peak: float = MIN_PEAK_PCT, max_k: int = 1024, folds=("loo",),
        integer: str = "paper", B: int = 10000, seed: int = 0, seed_scheme: str = "plain",
        name: str = "", plot: bool = False, iterations: int = ITERATIONS, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.perf_counter()
    info = {"inputs": {"counts": os.path.basename(counts_path), "counts_sha256": _sha256(counts_path)},
            "options": {"benchmark": benchmark, "model": model, "axis": axis, "fixed_temperature": fixed_temperature,
                        "n_mode": n_mode, "backend": backend, "mode": mode, "min_peak_pct": min_peak,
                        "max_k": max_k, "folds": list(folds), "integer": integer, "B": B, "seed": seed,
                        "seed_scheme": seed_scheme, "iterations": iterations},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "timings_s": {}}
    cnt = load_counts(counts_path, benchmark, model, axis, fixed_temperature=fixed_temperature)
    info["counts_summary"] = cnt.summary()
    t0 = time.perf_counter()
    prof = fit_profiles(cnt, n_mode=n_mode, backend=backend, kernel_path=kernel, iterations=iterations)
    prof.save(os.path.join(out_dir, "profiles.json"))
    info["timings_s"]["fit"] = round(time.perf_counter() - t0, 2)
    inc = prof.included(min_peak, mode)
    L = prof.L(mode)[inc]
    n_inc = int(inc.sum())
    info["n_questions"] = cnt.n_questions
    info["n_included"] = n_inc
    info["fit_backend"] = prof.backend
    info["allocation_settings"] = prof.alloc_settings
    log(f"[fit] {cnt.n_questions} questions, {n_inc} included (peak >= {min_peak}%), {len(prof.alloc_settings)} "
        f"settings, backend {prof.backend}, {info['timings_s']['fit']} s")

    ks = k_grid(max_k)
    ref = reference_index(prof.alloc_settings, axis)
    results = {}
    for fspec in folds:
        k_folds = None if str(fspec) == "loo" else int(fspec)
        t0 = time.perf_counter()
        res = compare(L, ks, k_folds, integer, ref)
        res.update({"settings": prof.alloc_settings, "mode": mode, "min_peak": min_peak})
        tag = "loo" if res["cv_mode"] == "loo" else f"kf{k_folds}"
        with open(os.path.join(out_dir, f"heldout_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f)
        results[tag] = res
        info["timings_s"][f"heldout_{tag}"] = round(time.perf_counter() - t0, 2)
        log(f"[heldout] {tag}: {res['seconds']} s, folds sha256 {res['folds_sha256'][:12]}")

    t0 = time.perf_counter()
    rtu = ratio_to_uniform(L, ks, B, seed, seed_scheme)
    rtu["temps"] = prof.alloc_settings
    with open(os.path.join(out_dir, "rtu.json"), "w", encoding="utf-8") as f:
        json.dump(rtu, f)
    info["timings_s"]["bootstrap"] = round(time.perf_counter() - t0, 2)

    ho_tab = results.get("loo") or next(iter(results.values()))
    rows = tb.passk_table_rows(L, prof.alloc_settings, ho_tab, [k for k in TABLE_K if k <= max_k])
    s = prof.alloc_settings
    cap = f"Pass@$K$ for {name or (model or 'the model')}."
    with open(os.path.join(out_dir, "passk_table.tex"), "w", encoding="utf-8") as f:
        f.write(tb.passk_table_latex(rows, s, cap, "tab:passk"))
    tb.passk_table_csv(rows, s, os.path.join(out_dir, "passk_table.csv"))
    with open(os.path.join(out_dir, "passk_table.md"), "w", encoding="utf-8") as f:
        f.write(tb.passk_table_markdown(rows, s))
    with open(os.path.join(out_dir, "passk_table.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": s, "n_included": n_inc, "heldout": ho_tab["cv_mode"], "rows": rows}, f, indent=1)
    srows = tb.summary_rows(s, ho_tab, rtu)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"heldout": ho_tab["cv_mode"], "rows": srows}, f, indent=1)
    with open(os.path.join(out_dir, "summary.tex"), "w", encoding="utf-8") as f:
        f.write(tb.summary_latex(name or (model or "Model"), srows))

    if plot:
        from onetemp.core.plot_rtu import render
        for tag, res in results.items():
            red = "Optimal (LOO)" if tag == "loo" else f"Optimal ({res['n_folds']}-fold)"
            fig = render(rtu, res, f"{name} -- pass@K relative to uniform allocation" if name else "",
                         settings=s, label_fmt=("T={:.2f}" if axis == "temperature" else "min_p={:g}"), red_label=red)
            fig.savefig(os.path.join(out_dir, f"rtu_{tag}.png"), dpi=200, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
    info["timings_s"]["total"] = round(time.perf_counter() - t_all, 2)
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    for tag, res in results.items():
        e2 = next((e for e in res["per_k"] if e["k"] == 2), None)
        if e2:
            log(f"[result] {tag}: optimal held-out / uniform at K=2 = {e2['ratio_optimal_heldout']:.4f}")
    log(f"[done] {info['timings_s']['total']} s -> {out_dir}")
    return {"info": info, "heldout": results, "rtu": rtu, "table": rows, "summary": srows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="counts.jsonl -> profiles -> held-out comparison -> tables (and plot).")
    ap.add_argument("--counts", required=True)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--axis", default="temperature", choices=AXES)
    ap.add_argument("--fixed-temperature", type=float, default=None, help="for --axis min_p (Appendix I)")
    ap.add_argument("--n-mode", default="question-max", choices=N_MODES)
    ap.add_argument("--backend", default="auto", choices=("auto", "c", "numpy"))
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--mode", default="fit", choices=("fit", "meas"),
                    help="fit: fitted profiles (the paper); meas: measured rates c/n")
    ap.add_argument("--min-peak", type=float, default=MIN_PEAK_PCT)
    ap.add_argument("--max-k", type=int, default=1024)
    ap.add_argument("--folds", nargs="+", default=["loo"], help="'loo' and/or fold counts, e.g. --folds loo 10")
    ap.add_argument("--integer", default="paper", choices=("paper", "exact"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-scheme", default="plain", choices=SEED_SCHEMES)
    ap.add_argument("--name", default="", help="model label for tables and the plot title")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    run(a.counts, a.out_dir, a.benchmark, a.model, a.axis, a.fixed_temperature, a.n_mode, a.backend, a.kernel,
        a.mode, a.min_peak, a.max_k, a.folds, a.integer, a.B, a.seed, a.seed_scheme, a.name, a.plot, a.iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
