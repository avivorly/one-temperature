"""Reproduce the paper's figures, tables and printed numbers from the shipped counts, and compare them.

Every target reads the per-question counts in data/counts/, fits the question profiles with this package's own
fit where the analysis needs them (``onetemp.core.fit_all``), runs the analysis, and compares the result with the
paper: with the numbers the paper prints, and with the paper's own runs shipped in data/reference/ (the cell
labels of every heatmap, every beta* value, the 2,400 chunk tallies of Figure 4(b), and the sha256 of the paper's
image files).

    python -m onetemp.reproduce list              # the targets, what each covers, and its run time
    python -m onetemp.reproduce fig3              # Figure 3 (the quickstart)
    python -m onetemp.reproduce fig4b             # Figure 4(b): seeded check of 50 of the 2,400 chunks
    python -m onetemp.reproduce fig17             # one item: runs what it needs
    python -m onetemp.reproduce all --jobs 3      # everything (about 45 minutes on one core, 26 with --jobs 3)

Outputs go to --out (default ``results/``): one folder per group, with every intermediate file and log; the
comparison is printed, one line per result, and the last line counts the results and the mismatches. A result that
already exists in --out is reused, so a stopped run resumes; delete the folder to recompute from scratch. The run
times are with the C kernel of the profile fit (``python -m onetemp.core.fit_all --build-kernel``); without it
each fit takes about 14 times longer. The exit status is 0 when every comparison matches (known, documented differences
included), 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from onetemp.paths import counts_dir, reference_dir

PKG_ROOT = Path(__file__).resolve().parent.parent      # the folder that holds the onetemp package

MODELS = [("gpt-oss-20b", "openai/gpt-oss-20b", "GPT OSS 20B"),
          ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct"),
          ("Qwen3-8B", "Qwen/Qwen3-8B", "Qwen3-8B"),
          ("Qwen3-0.6B", "Qwen/Qwen3-0.6B", "Qwen3-0.6B")]
BENCHMARKS = [("livecodebench", "LiveCodeBench"), ("math500", "MATH-500"), ("hitom", "Hi-ToM")]
MINP_FILES = ["hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl", "hitom/Qwen3-0.6B__min_p_tau2.0__nu400.jsonl",
              "hitom/Mistral-7B-Instruct-v0.3__min_p_tau2.0__nu1500.jsonl"]
# Bottom edge of the ratio rows of Figures 17-20 (the paper kept the edge its original sampler found).
ROW_BOTTOM = {"gpt-oss-20b": 0.7, "Mistral-7B-Instruct-v0.3": 0.4, "Qwen3-8B": 0.7, "Qwen3-0.6B": 0.2}
MISTRAL_LCB_Q = "livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl"
MISTRAL_LCB_CELL = "livecodebench/Mistral-7B-Instruct-v0.3__temperature__cell_n.jsonl"
SIM_SHAPES = ("peaked", "reversed", "flat")
SIM_PRINTED = {"n_templates": 109, "n_above_1e4": {"peaked": 966, "reversed": 985, "flat": 980}}


def temperature_file(bench: str, mfile: str, parse: str = "question_n") -> str:
    """The counts file of one temperature sweep. LiveCodeBench has two parses: question_n (one sample count per
    question, as the paper's Figures 1-3, Figure 4(b) and Appendices E-G read it) and cell_n (each temperature's own
    count, Appendix H and Figure 4(a))."""
    if bench == "livecodebench":
        return f"livecodebench/{mfile}__temperature__{parse}.jsonl"
    return f"{bench}/{mfile}__temperature.jsonl"


def paper_files() -> list[str]:
    """The 12 temperature sweeps (LiveCodeBench in the question_n parse) and the min_p study of the Appendix I
    table, the inputs of the core checks."""
    return [temperature_file(b, m) for m, _, _ in MODELS for b, _ in BENCHMARKS] + [MINP_FILES[0]]


# ----------------------------------------------------------------------------------------------------------------
# Running the package's own commands
# ----------------------------------------------------------------------------------------------------------------

def _python() -> list[str]:
    f = [sys.executable]
    if sys.flags.isolated:
        f.append("-I")
    if sys.flags.dont_write_bytecode:
        f.append("-B")
    return f


def run_module(module: str, args: list[str], log: Path) -> int:
    """Run ``python -m module args``, its output to ``log``. Returns the exit code."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("$ python -m " + module + " " + " ".join(args) + "\n")
        fh.flush()
        p = subprocess.run(_python() + ["-m", module] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def run_all(jobs: list, n_jobs: int, what: str) -> list:
    """Run (label, function) jobs, n_jobs at a time; report each as it finishes."""
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
        futs = [(label, ex.submit(fn)) for label, fn in jobs]
        for label, f in futs:
            r = f.result()
            results.append(r)
            print(f"  [{what}] {label}: {r if isinstance(r, str) else 'done'} ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
    return results


class Ctx:
    def __init__(self, out: Path, jobs: int, data: Path):
        self.out, self.jobs, self.data = out, jobs, data
        self.ok = True
        self.lines: list[str] = []

    def counts(self, rel: str) -> Path:
        return self.data / rel

    def note(self, line: str, ok: bool = True):
        self.ok &= bool(ok)
        self.lines.append(("" if ok else "MISMATCH ") + line)
        print(("  " if ok else "  MISMATCH ") + line, flush=True)

    def fit(self, rel: str, n_mode: str) -> Path:
        """profiles.json of the package's fit of one counts file (computed once, then reused)."""
        p = self.out / "fits" / (rel.replace("/", "__").replace(".jsonl", "") + f"__{n_mode}.json")
        if p.exists():
            return p
        tmp = p.with_suffix(".tmp.json")
        rc = run_module("onetemp.core.fit_all", ["--counts", str(self.counts(rel)), "--n-mode", n_mode,
                                                 "--out", str(tmp)], p.with_suffix(".log"))
        if rc != 0:
            raise RuntimeError(f"profile fit of {rel} failed, see {p.with_suffix('.log')}")
        os.replace(tmp, p)
        return p

    def fits(self, items: list) -> None:
        todo = [(rel, m) for rel, m in dict.fromkeys(items)
                if not (self.out / "fits" / (rel.replace("/", "__").replace(".jsonl", "") + f"__{m}.json")).exists()]
        if todo:
            print(f"[fits] fitting {len(todo)} studies (Appendix D profile fit)", flush=True)
            run_all([(f"{rel} ({m})", (lambda rel=rel, m=m: self.fit(rel, m) and "fitted")) for rel, m in todo],
                    self.jobs, "fit")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ref(name: str) -> dict:
    return _load(reference_dir() / name)


def _same_array(a, b) -> tuple[int, int]:
    """(equal entries, all entries) of two arrays: NaN equals NaN, infinity equals infinity, all else exactly."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 0, int(max(a.size, b.size))
    eq = (a == b) | (np.isnan(a) & np.isnan(b))
    return int(eq.sum()), int(eq.size)


def _study_name(rel: str) -> str:
    bench, rest = rel.split("/", 1)
    mfile = rest.split("__")[0]
    b = dict(BENCHMARKS)[bench]
    m = next(lab for f, _, lab in MODELS if f == mfile)
    tail = ""
    if "min_p" in rest:
        tail = ", min_p at tau = " + rest.split("tau")[1].split("__")[0]
    return f"{b}, {m}{tail}"


# ----------------------------------------------------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------------------------------------------------

def group_tables(c: Ctx, only: list | None = None) -> None:
    """Figure 3, the pass@K tables (Tables 2-6), the summary table of Appendix F (whose values are the red and the
    best-temperature curves of Figures 14-16 at K = 1, 16, 256, 1024) and the Appendix I table (Figure 34)."""
    out = c.out / "tables"
    res = out / ("paper_check.json" if not only else "paper_check_" + "_".join(o.split(":")[0] for o in only) + ".json")
    files = paper_files()
    if not res.exists():
        args = ["--counts"] + [str(c.counts(f)) for f in files] + ["--out", str(res)]
        if only:
            args += ["--only"] + only
        print(f"[tables] {'all 13 studies' if not only else ', '.join(only)} (profile fit, leave-one-out over 20 K, "
              f"10-fold for Figure 3); about {'15 minutes' if not only else 'one minute'}", flush=True)
        run_module("onetemp.core.paper_check", args, res.with_suffix(".log"))
    if not res.exists():
        c.note(f"the core check wrote no result, see {res.with_suffix('.log')}", False)
        return
    r = _load(res)
    n_all = n_ok = 0
    for d in r["results"]:
        if d.get("skipped"):
            continue
        k = sum(x["ok"] for x in d["checks"])
        n_all += len(d["checks"])
        n_ok += k
        name = f"{d['benchmark']}, {d['model']}" + (", min_p" if d.get("axis") == "min_p" else "")
        c.note(f"{name}: {k}/{len(d['checks'])} checks match", k == len(d["checks"]))
    c.note(f"tables total: {n_ok}/{n_all} checks match the printed numbers", n_ok == n_all)


def figure3(c: Ctx) -> None:
    """Figure 3: the optimal allocation's held-out pass@K over the uniform scheme's (Mistral 7B Instruct,
    LiveCodeBench, n = 359), 10-fold (Figure 3) and leave-one-out (Figure 14(b)), at every K."""
    group_tables(c, ["livecodebench:mistralai/Mistral-7B-Instruct-v0.3"])
    res = c.out / "tables" / "paper_check_livecodebench.json"
    if not res.exists():
        return
    d = next(x for x in _load(res)["results"] if not x.get("skipped"))
    rows = {}
    for x in d["checks"]:
        if x["what"].startswith("optimal held-out / uniform, K="):
            k = int(x["what"].split("K=")[1])
            rows.setdefault(k, {})["kf" if "10-fold" in x["where"] else "loo"] = (x["ours"], x["paper"], x["ok"])
    print("\n  Figure 3: optimal allocation, held-out pass@K / uniform pass@K (ours | paper)")
    print("     K    10-fold            leave-one-out")
    for k in sorted(rows):
        kf, lo = rows[k]["kf"], rows[k]["loo"]
        print(f"  {k:5d}    {kf[0]} | {kf[1]}    {lo[0]} | {lo[1]}")
    n_kf = sum(v["kf"][2] for v in rows.values())
    n_lo = sum(v["loo"][2] for v in rows.values())
    c.note(f"Figure 3: 10-fold {n_kf}/{len(rows)} K and leave-one-out {n_lo}/{len(rows)} K equal at 4 decimals; "
           f"K = 2: 10-fold {rows[2]['kf'][0]}, leave-one-out {rows[2]['loo'][0]}",
           n_kf == n_lo == len(rows) and len(rows) > 0)


def group_fig4b(c: Ctx, n_sample: int) -> None:
    """Figure 4(b): recompute a seeded sample of the 2,400 chunks and compare with the paper's tallies."""
    prof = c.fit(MISTRAL_LCB_Q, "question-max")
    out = c.out / "fig4b"
    res = out / f"check_{n_sample}.json"
    if not res.exists():
        print(f"[fig4b] recomputing {n_sample} of the 2,400 chunks (about {4 * n_sample // 60 + 1} minutes)", flush=True)
        run_module("onetemp.robustness.fig4b", ["check", "--input", str(c.counts(MISTRAL_LCB_Q)), "--profiles",
                                                str(prof), "--n-sample", str(n_sample), "--out", str(res)],
                   out / f"check_{n_sample}.log")
    if not res.exists():
        c.note(f"the Figure 4(b) check wrote no result, see {out / f'check_{n_sample}.log'}", False)
        return
    r = _load(res)
    cmp = r["compare"]
    c.note(f"Figure 4(b): {cmp['n_chunks_equal']} of {len(r['chunks_checked'])} recomputed chunks equal the paper's "
           f"run (W and KL hashes, band counts, every tally)", r["all_equal"])
    from onetemp.robustness.fig4b import headline, panel_markdown
    print("\n" + panel_markdown(r["panel"]) + "\n\n" + headline(r["panel"]) + "\n")
    p = r["panel"]
    # The 24 cells Figure 4(b) prints, per K, in the order of its KL bands (0-1, 1-2, 2-3, 3-4, 4-5, 5-5.88).
    figure = {"512": ["99.99", "99.85", "98.75", "97.12", "93.38", "87.97"],
              "128": ["100", "99.79", "98.39", "96.34", "91.31", "84.03"],
              "32": ["100", "99.91", "98.63", "96.59", "92.01", "85.16"],
              "4": ["99.92", "98.04", "93.47", "87.19", "78.76", "69.40"]}
    n_cells = sum(len(v) for v in figure.values())
    got = {k: [x["label"] for x in p["cells"].get(k, [])] for k in figure}
    n_equal = sum(a == b for k, v in figure.items() for a, b in zip(got[k], v))
    c.note(f"Figure 4(b): {n_equal} of {n_cells} printed cells equal (K = 4, KL 5-5.88: "
           f"{p['cells']['4'][-1]['label']})", n_equal == n_cells)
    # The paper prints the KL > 3 share only as "on average roughly 90%" and the KL 1-2 share as "99%" (both
    # compared by the text group); the per-K KL > 3 shares are those of the paper's run, from its tallies.
    run = {"4": "78.80", "32": "91.48", "128": "90.80", "512": "93.00"}
    per_k = " / ".join(f"{p['kl_over_3_pct'][k]:.2f}" for k in run)
    ok = per_k == " / ".join(run.values()) and f"{p['kl_over_3_mean_over_K']:.2f}" == "88.52"
    c.note(f"Figure 4(b): KL > 3 per K = 4, 32, 128, 512: {per_k}, mean {p['kl_over_3_mean_over_K']:.2f}%, "
           + ("as in the paper's run" if ok else "the paper's run has 78.80 / 91.48 / 90.80 / 93.00, mean 88.52%"), ok)


def _heat_jobs(c: Ctx, studies: list) -> list:
    jobs = []
    for rel, st in studies:
        out = c.out / "heatmaps" / rel.replace("/", "__").replace(".jsonl", "")
        if (out / "heatmap_numbers.json").exists():
            continue
        if st["axis"] == "min_p":
            args = ["--input", str(c.counts(rel)), "--axis", "min_p", "--tau", str(st["tau"]), "--preset", "paper-i",
                    "--K", ",".join(map(str, st["K"])), "--out", str(out), "--plot"]
        else:
            prof = c.fit(rel, "question-max")
            mfile = rel.split("/")[1].split("__")[0]
            args = ["--input", str(c.counts(rel)), "--profiles", str(prof), "--out", str(out), "--plot"]
            if rel.startswith("livecodebench/"):
                args += ["--preset", "paper-g-lcb", "--row-bottom", str(ROW_BOTTOM[mfile])]
            else:
                args += ["--preset", "paper-g"]
        jobs.append((rel, (lambda args=args, out=out: (run_module("onetemp.robustness.heatmap", args,
                                                                  out.parent / (out.name + ".log")), "drawn")[1])))
    return jobs


def group_heatmaps(c: Ctx, only: list | None = None) -> None:
    """Appendix G (Figures 17-28) and the min_p heatmaps of Appendix I (Figures 42-44): every cell label and count,
    tau_A and score_A of every K, against the paper's runs (data/reference/heatmaps.json)."""
    ref = _ref("heatmaps.json")["studies"]
    studies = [(rel, st) for rel, st in ref.items() if not only or rel in only]
    c.fits([(rel, "question-max") for rel, st in studies if st["axis"] == "temperature"])
    jobs = _heat_jobs(c, studies)
    if jobs:
        print(f"[heatmaps] {len(jobs)} heatmaps of 10,000 random W's each (15 s to 4 minutes each)", flush=True)
        run_all(jobs, c.jobs, "heatmap")
    tot_l = tot_ok = 0
    moved = []
    for rel, st in sorted(studies, key=lambda x: x[1]["figure"]):
        p = c.out / "heatmaps" / rel.replace("/", "__").replace(".jsonl", "") / "heatmap_numbers.json"
        if not p.exists():
            c.note(f"Figure {st['figure']} ({_study_name(rel)}): no result, see its log", False)
            continue
        mine = _load(p)
        n_lab = n_eq = n_cnt = n_cnt_eq = 0
        scal = []
        for K, r in st["per_K"].items():
            m = mine["per_k"][K]
            for col in range(len(r["labels"])):
                for row in range(len(r["labels"][col])):
                    n_lab += 1
                    same = m["labels"][col][row] == r["labels"][col][row]
                    n_eq += same
                    if not same:
                        moved.append(f"Figure {st['figure']}, K = {K}, column {col}, row {row}: paper "
                                     f"{r['labels'][col][row]!r}, ours {m['labels'][col][row]!r}")
                    n_cnt += 1
                    n_cnt_eq += m["counts"][col][row] == r["counts"][col][row]
            scal.append(abs(float(m["tau_A"]) - float(r["tau_A"])) < 1e-9
                        and abs(float(m["score_A"]) - float(r["score_A"])) <= 1e-9 * max(1.0, abs(float(r["score_A"]))))
        tot_l += n_lab
        tot_ok += n_eq
        c.note(f"Figure {st['figure']} ({_study_name(rel)}): {n_eq}/{n_lab} cell labels and {n_cnt_eq}/{n_cnt} cell "
               f"counts equal; tau_A and score_A equal at {sum(scal)}/{len(scal)} K",
               n_eq == n_lab and n_cnt_eq == n_cnt and all(scal))
    c.note(f"heatmaps total: {tot_ok}/{tot_l} cell labels equal the paper's", tot_ok == tot_l)
    for line in moved[:40]:
        print("    label moved: " + line)
    (c.out / "heatmaps").mkdir(parents=True, exist_ok=True)
    (c.out / "heatmaps" / "compare.json").write_text(json.dumps({"labels": tot_l, "equal": tot_ok,
                                                                 "moved": moved}, indent=1))


def group_beta(c: Ctx, only: list | None = None) -> None:
    """Appendix H (Figures 4(a) and 29-31) and Appendix I (Figure 45): beta* at every (loss, K, D), after the LP sweep
    and after the exact recheck, against the paper's runs (data/reference/beta_star.json)."""
    ref = _ref("beta_star.json")["studies"]
    studies = [(rel, st) for rel, st in ref.items() if not only or rel in only]
    c.fits([(rel, "per-setting") for rel, st in studies if st["axis"] == "temperature"])
    jobs = []
    for rel, st in studies:
        out = c.out / "beta" / rel.replace("/", "__").replace(".jsonl", "")
        if (out / "beta_star.json").exists():
            continue
        args = ["--input", str(c.counts(rel)), "--K", ",".join(map(str, st["K"])), "--out", str(out), "--plot"]
        if st["axis"] == "min_p":
            args += ["--axis", "min_p", "--tau", str(st["tau"]), "--rates", "measured"]
        else:
            args += ["--profiles", str(c.fit(rel, "per-setting"))]
        jobs.append((rel, (lambda args=args, out=out: (run_module("onetemp.robustness.protected", args,
                                                                  out.parent / (out.name + ".log")), "solved")[1])))
    if jobs:
        print(f"[beta] {len(jobs)} beta* sweeps with exact certification (10 s to 3 minutes each)", flush=True)
        run_all(jobs, c.jobs, "beta*")
    tot = tot_ok = 0
    for rel, st in sorted(studies, key=lambda x: (x[1]["figure"], x[0])):
        p = c.out / "beta" / rel.replace("/", "__").replace(".jsonl", "") / "beta_star.json"
        if not p.exists():
            c.note(f"Figure {st['figure']} ({_study_name(rel)}): no result, see its log", False)
            continue
        mine = _load(p)
        e_s, n_s = _same_array(mine["beta_star_sweep"], st["beta_star_sweep"])
        e_f, n_f = _same_array(mine["beta_star"], st["beta_star"])
        tA = [float(mine["tau_A"][str(k)]) for k in st["K"]]
        sA = [float(mine["score_A"][str(k)]) for k in st["K"]]
        scal = all(abs(a - b) < 1e-9 for a, b in zip(tA, st["tau_A"])) and \
            all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(sA, st["score_A"]))
        tot += n_f
        tot_ok += e_f
        fig = f"Figure {st['figure']}" + (" and 4(a)" if rel == MISTRAL_LCB_CELL else "")
        cert = mine.get("certify", {}).get("counts", {})
        c.note(f"{fig} ({_study_name(rel)}, n = {mine['n_inc']}): beta* {e_f}/{n_f} equal after the exact recheck, "
               f"{e_s}/{n_s} after the sweep; tau_A and score_A {'equal' if scal else 'DIFFERENT'}"
               + (f"; certificates: {cert.get('lower_unproven', '?')} lower bounds and "
                  f"{cert.get('inf_unproven', '?')} '> 10^6' points unproven" if cert else ""),
               e_f == n_f and e_s == n_s and scal and mine["n_inc"] == st["n_included"])
    c.note(f"beta* total: {tot_ok}/{tot} entries equal the paper's", tot_ok == tot)


def group_simulated(c: Ctx) -> None:
    """Appendix H, simulated hardness distributions (Figure 33): beta* of the three pools, and the numbers the text
    gives for them (109 templates; 966, 985 and 980 of the 1000 questions above H = 10^4)."""
    ref = _ref("simulated_pools.json")["pools"]
    prof = c.fit(MISTRAL_LCB_CELL, "per-setting")
    jobs = []
    for shape in SIM_SHAPES:
        out = c.out / "simulated" / shape
        if not (out / "beta_star.json").exists():
            args = ["--input", str(c.counts(MISTRAL_LCB_CELL)), "--profiles", str(prof), "--shape", shape,
                    "--out", str(out), "--plot"]
            jobs.append((shape, (lambda args=args, out=out: (run_module("onetemp.robustness.simulate", args,
                                                                        out.parent / (out.name + ".log")), "solved")[1])))
    if jobs:
        print(f"[simulated] {len(jobs)} simulated pools of 1000 questions", flush=True)
        run_all(jobs, c.jobs, "pool")
    for shape in SIM_SHAPES:
        p = c.out / "simulated" / shape / "beta_star.json"
        if not p.exists():
            c.note(f"Figure 33, {shape} pool: no result", False)
            continue
        mine = _load(p)
        e, n = _same_array(mine["beta_star_sweep"], ref[shape]["beta_star_sweep"])
        c.note(f"Figure 33, {shape} pool: beta* {e}/{n} equal; {mine['n_templates']} templates (printed "
               f"{SIM_PRINTED['n_templates']}); {mine['n_above_1e4']} questions above H = 10^4 (printed "
               f"{SIM_PRINTED['n_above_1e4'][shape]})",
               e == n and mine["n_templates"] == SIM_PRINTED["n_templates"]
               and mine["n_above_1e4"] == SIM_PRINTED["n_above_1e4"][shape])


def group_profiles(c: Ctx) -> None:
    """Section 3 and Appendices E and I: Figures 1, 2, 5-13 and 38-41, every printed profile number, and the image
    bytes against the paper's files."""
    out = c.out / "profiles"
    if not (out / "repro.json").exists():
        print("[profiles] fitting the 12 sweeps and drawing Figures 1, 2, 5-13 and 38-41 (about 2 minutes)", flush=True)
        run_module("onetemp.profiles", ["repro", "--counts-dir", str(c.data), "--out-dir", str(out)],
                   c.out / "profiles.log")
    if not (out / "repro.json").exists():
        c.note("the profile reproduction wrote no result, see profiles.log", False)
        return
    rows = _load(out / "repro.json")["comparisons"]
    img = [r for r in rows if r["what"].endswith(" bytes")]
    num = [r for r in rows if not r["what"].endswith(" bytes")]
    known = [r for r in num if not r["ok"] and r.get("known")]
    bad = [r for r in num if not r["ok"] and not r.get("known")]
    img_bad = [r for r in img if not r["ok"]]
    import matplotlib
    import PIL
    # the paper's images were drawn with matplotlib 3.10.6 and Pillow 11.3.0; other versions can differ in rendering,
    # and then the numbers decide
    pinned = matplotlib.__version__ == "3.10.6" and PIL.__version__ == "11.3.0"
    c.note(f"profiles: {sum(r['ok'] for r in img)}/{len(img)} images byte-identical to the paper's; "
           f"{sum(r['ok'] for r in num)}/{len(num)} printed numbers and checks equal; {len(known)} known difference"
           f"{'s' if len(known) != 1 else ''}", not bad and (not img_bad or not pinned))
    if img_bad and not pinned:
        print(f"    note: {len(img_bad)} images differ in bytes; the paper's were drawn with matplotlib 3.10.6 and "
              f"Pillow 11.3.0 (here {matplotlib.__version__} and {PIL.__version__}), so this is not counted as a "
              f"mismatch")
    for r in known:
        print(f"    known difference: {r['what']}: paper {r['paper']}, ours {r['ours']} ({r['known']})")
    for r in bad + (img_bad if pinned else []):
        print(f"    MISMATCH: {r['what']}: paper {r['paper']}, ours {r['ours']} ({r['where']})")


def group_text(c: Ctx) -> None:
    """Numbers of the main text that no figure or table prints: the sampling setup of Section 2.3, the efficient
    hardness range of Section 4.3.2 (Eq. 16), and the Figure 4(b) shares quoted in Section 4.3.2."""
    from collections import Counter

    from scipy.optimize import brentq

    from onetemp.counts import read_counts, table
    from onetemp.robustness.fig4b import N_CHUNKS, load_tallies, merge_tallies
    sizes, lo, hi, drawn = set(), [], [], Counter()
    for rel in [temperature_file(b, m) for m, _, _ in MODELS for b, _ in BENCHMARKS]:
        t = table(read_counts(c.counts(rel)))
        sampled = [s for s in t.settings if s > 0]          # temperature 0 (greedy decoding) is not a sampling setting
        sizes.add(len(sampled))
        lo.append(min(sampled))
        hi.append(max(sampled))
        for i in range(len(t.ids)):
            for j, s in enumerate(t.settings):
                if s > 0:
                    drawn[t.cell_n(i, j)] += 1
    typical = drawn.most_common(1)[0][0]
    c.note(f"Section 2.3: {min(sizes)} to {max(sizes)} temperatures per sweep (grids of {sorted(sizes)}), between "
           f"{min(lo):g} and {max(hi):g}; the most common number of samples per (question, temperature) is {typical:,} "
           f"(printed: typically 1,500 samples at each of 16 to 38 temperatures between 0.075 and 4.2)",
           (min(sizes), max(sizes), min(lo), max(hi), typical) == (16, 38, 0.075, 4.2, 1500))
    f = lambda a: a * math.exp(1.0 - a) - math.exp(-1.0)            # Eq. 16 at the level 1/e
    a_lo, a_hi = brentq(f, 1e-9, 1.0, xtol=1e-15), brentq(f, 1.0, 50.0, xtol=1e-15)
    c.note(f"Section 4.3.2: alpha e^(1 - alpha) >= 1/e for {a_lo:.4f} <= alpha <= {a_hi:.4f}, a factor of "
           f"{a_hi / a_lo:.1f} (printed: 0.16 <~ alpha <~ 3.15, roughly a factor of 20)",
           f"{a_lo:.2f}" == "0.16" and f"{a_hi:.2f}" == "3.15" and round(a_hi / a_lo) == 20)
    p = merge_tallies(list(load_tallies(reference_dir() / "fig4b_chunk_tallies.jsonl").values()), N_CHUNKS)
    c.note(f"Section 4.3.2: KL > 3, mean over K of the share losing under 10%: {p['kl_over_3_mean_over_K']:.2f}% "
           f"(printed: on average roughly 90%); KL 1-2: {p['kl_1_2_mean_over_K']:.2f}% (printed: rises to 99%); "
           f"from the paper's 2,400 Figure 4(b) tallies, which 'python -m onetemp.reproduce fig4b' checks",
           round(p["kl_over_3_mean_over_K"] / 10) == 9 and round(p["kl_1_2_mean_over_K"]) == 99)


def group_data(c: Ctx) -> None:
    """The data files themselves (manifest hashes and schema), and the numbers that need no profile fit: Appendix C,
    the LiveCodeBench parses of Appendix H, and the measured-rate numbers of Appendix I (Figures 35-37)."""
    out = c.out / "data"
    out.mkdir(parents=True, exist_ok=True)
    rc1 = run_module("onetemp.data.check", [], out / "check.log")
    tail = (out / "check.log").read_text(encoding="utf-8").strip().splitlines()[-1:]
    c.note(f"data check: {tail[0] if tail else 'no output'}", rc1 == 0)
    rc2 = run_module("onetemp.data.fitfree", ["--json", str(out / "fitfree.json")], out / "fitfree.log")
    tail = (out / "fitfree.log").read_text(encoding="utf-8").strip().splitlines()[-1:]
    c.note(f"fit-free numbers: {tail[0] if tail else 'no output'}", rc2 == 0)


# ----------------------------------------------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------------------------------------------

def _hm(fig: int) -> list:
    return [rel for rel, st in _ref("heatmaps.json")["studies"].items() if st["figure"] == fig]


def _bs(fig: int) -> list:
    return [rel for rel, st in _ref("beta_star.json")["studies"].items() if st["figure"] == fig]


NOT_INCLUDED = {
    "tab1": "Table 1 lists the shape constraints of the profile fit; it has no computed entries. "
            "onetemp/core/fit_all.py applies them.",
    "fig32": "the drawing of the three simulated hardness shapes; the numbers the text gives for them are compared by "
             "'python -m onetemp.reproduce simulated'.",
    "fig35": "the drawing; the numbers the text gives with it (the gains of min_p 0.0316 at tau = 2.0 over 0.003 at "
             "tau = 1.5) and the Appendix I table of the best values at tau = 2.0 and 1.5 are compared by "
             "'python -m onetemp.reproduce data'.",
    "fig36": "the drawing; its table in Appendix I (the best value, 0.0056, against the best plain temperature at "
             "each K) is compared by 'python -m onetemp.reproduce data'.",
}

GROUPS = {
    "fig3": ("Figure 3 (the quickstart): 10-fold and leave-one-out held-out ratios at 20 K", "1 minute"),
    "fig4b": ("Figure 4(b): 50 of the 2,400 chunks recomputed, then the panel from the 2,400 shipped tallies",
              "4 minutes"),
    "tables": ("Figure 3, Tables 2-6, the Appendix F summary table (Figures 14-16 at K = 1, 16, 256, 1024) and the "
               "Appendix I table (Figure 34): 189 checks", "15 minutes"),
    "profiles": ("Figures 1, 2, 5-13, 38-41: images (sha256) and printed profile numbers", "2 minutes"),
    "heatmaps": ("Figures 17-28 and 42-44: every cell label and count", "12 minutes"),
    "beta": ("Figures 4(a), 29-31 and 45: every beta*, with exact certification", "10 minutes"),
    "simulated": ("Figure 33 and its printed counts", "1 minute"),
    "text": ("the main-text numbers of Sections 2.3 and 4.3.2", "10 seconds"),
    "data": ("data files (hashes, schema) and the fit-free numbers of Appendices C, H and I (Figures 35-37)",
             "10 seconds"),
}


def run_target(t: str, c: Ctx, n_sample: int) -> None:
    t = t.lower()
    n = int(t[3:]) if t.startswith("fig") and t[3:].isdigit() else None
    if t in NOT_INCLUDED:
        print(f"{t}: not included: {NOT_INCLUDED[t]}")
    elif t == "fig3":
        figure3(c)
    elif t == "fig4b":
        group_fig4b(c, n_sample)
    elif t == "fig4a":
        group_beta(c, [MISTRAL_LCB_CELL])
    elif t == "fig4":
        group_beta(c, [MISTRAL_LCB_CELL])
        group_fig4b(c, n_sample)
    elif t in ("tables", "fig14", "fig15", "fig16", "fig34") or t in ("tab2", "tab3", "tab4", "tab5", "tab6"):
        group_tables(c)
    elif t == "profiles" or n in (1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 38, 39, 40, 41):
        group_profiles(c)
    elif t == "heatmaps":
        group_heatmaps(c)
    elif n is not None and (17 <= n <= 28 or 42 <= n <= 44):
        group_heatmaps(c, _hm(n))
    elif t == "beta":
        group_beta(c)
    elif n is not None and (29 <= n <= 31 or n == 45):
        group_beta(c, _bs(n))
    elif t in ("simulated", "fig33"):
        group_simulated(c)
    elif t == "text":
        group_text(c)
    elif t in ("data", "fig37", "appc"):
        group_data(c)
    elif t == "all":
        for g in ("data", "text", "profiles", "fig4b", "heatmaps", "beta", "simulated", "tables"):
            print(f"\n===== {g} =====", flush=True)
            run_target(g, c, n_sample)
    else:
        raise SystemExit(f"unknown target {t!r}; see 'python -m onetemp.reproduce list'")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="list, all, a group (" + ", ".join(GROUPS) + ") or an item "
                    "(fig1 ... fig45, fig4a, fig4b, tab1 ... tab6, appc)")
    ap.add_argument("--out", default="results", help="output folder (default: results)")
    ap.add_argument("--jobs", type=int, default=1, help="independent runs at a time (default 1)")
    ap.add_argument("--n-sample", type=int, default=50, help="Figure 4(b): chunks to recompute (default 50)")
    ap.add_argument("--data", default=None, help="the counts folder (default: data/counts of this repository)")
    a = ap.parse_args(argv)
    if a.targets == ["list"]:
        print("groups:")
        for g, (what, tm) in GROUPS.items():
            print(f"  {g:10s} {what} [{tm}]")
        print("  all        every group above [about 45 minutes on one core; 26 minutes with --jobs 3]")
        print("items: fig1 ... fig45, fig4a, fig4b, tab1 ... tab6, appc run the group they belong to; not included:")
        for t, why in NOT_INCLUDED.items():
            print(f"  {t:10s} {why}")
        return 0
    c = Ctx(Path(a.out), a.jobs, Path(a.data) if a.data else counts_dir())
    c.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for t in a.targets:
        run_target(t, c, a.n_sample)
    print(f"\n[reproduce] {len(c.lines)} results, {sum(1 for x in c.lines if x.startswith('MISMATCH'))} mismatches "
          f"({time.perf_counter() - t0:.0f} s); outputs in {c.out}")
    return 0 if c.ok else 1


if __name__ == "__main__":
    sys.exit(main())
