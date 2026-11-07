"""Recompute, from the released counts alone, the paper's numbers that need no profile fit.

    python -m onetemp.data.fitfree            # prints one line per number: paper | ours | difference | where
    python -m onetemp.data.fitfree --json OUT.json

Covered (each line names its place in the paper):

* Appendix C: the included questions, samples per (question, temperature) and temperature grids.
* Appendix H: how many profiles the corrected parse of LiveCodeBench changes, and how many questions it adds.
* Appendix I (min_p on Hi-ToM), every number computed from measured pass rates:
  - the tau = 1.5 ratio-to-uniform table (single values and the uniform row; the LOO row needs the
    allocation optimizer and is left to the analysis code);
  - pass@K with the unbiased estimator 1 - C(n - c, K) / C(n, K), and paired standard errors over
    questions, in the tau = 2.0 against tau = 1.5 table, the comparison with the best plain temperature,
    and the Mistral 7B Instruct table;
  - the ratio-to-uniform statements at tau = 2.0, the peak counts of the profile figure, the inclusion
    floor of Mistral 7B Instruct, the hardness bins of the coverage figure, and the OLS slopes of the
    peak position on log10 hardness.
* For scale, measured pass@1 against the fitted pass@1 printed in the LiveCodeBench pass@K tables
  (Tables 2 to 6). These differ by design: the tables use the fitted profiles.

It uses numpy and this package's estimators (onetemp.core.passk). Pass rates are instance-weighted: every sample counts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from onetemp.core.passk import passk_plugin, passk_unbiased
from onetemp.counts import CountsTable, read_counts, table
from onetemp.paths import data_dir

DATA = data_dir()          # the repository's data/ folder (onetemp.paths)
C = DATA / "counts"
LINES: list[dict] = []
_INC = None


def included_ids(rel: str) -> list[str]:
    """The included questions of a counts file (the paper's fitted-peak or measured floor), from included_ids.json."""
    global _INC
    if _INC is None:
        _INC = json.loads((DATA / "included_ids.json").read_text(encoding="utf-8"))["files"]
    return _INC["counts/" + rel]["ids"]


def T(rel: str, included: bool = False, ids: list[str] | None = None) -> CountsTable:
    """Load one counts file as a table: every question, its included questions, or the given ids (file order)."""
    rows = read_counts(C / rel)
    keep = set(included_ids(rel)) if included else (set(ids) if ids is not None else None)
    if keep is not None:
        rows = [r for r in rows if r["id"] in keep]
    return table(rows)


def add(where: str, what: str, paper, ours, tol=None, unit="", dec=None) -> None:
    """Record one comparison at the paper's printed precision: half a unit of its last printed decimal.

    The paper prints percentages and percentage points with 2 decimals and ratios with 3, so those are
    the defaults for unit '%', 'pp' and 'ratio'; ``dec`` overrides them, ``tol`` replaces the rule."""
    if isinstance(paper, (int, float)) and isinstance(ours, (int, float)):
        diff = ours - paper
        if tol is None:
            if dec is None:
                dec = {"%": 2, "pp": 2, "ratio": 3}.get(unit)
            if dec is None:
                s = repr(paper)
                dec = len(s.split(".")[1]) if "." in s else 0
            tol = 0.5 * 10 ** (-dec) + 1e-12
        ok = abs(diff) <= tol
    else:
        diff = None
        ok = paper == ours
    LINES.append({"where": where, "what": what, "paper": paper, "ours": ours, "diff": diff, "ok": ok,
                  "unit": unit})


# --------------------------------------------------------------------------------------------
# Estimators (Section 2.2 and Appendix F)
# --------------------------------------------------------------------------------------------

def unbiased(n: int, c: int, K: int) -> float:
    """Unbiased pass@K from c correct of n samples (onetemp.core.passk.passk_unbiased); requires K <= n."""
    if K > n:
        raise ValueError("K > n")
    return passk_unbiased(n, c, K)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def paired_se(a, b) -> float:
    """Standard error of the mean of the per-question differences a_q - b_q (sample sd, ddof = 1)."""
    d = [x - y for x, y in zip(a, b)]
    m = mean(d)
    return math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d))


def col(t: CountsTable, s: float) -> int:
    return t.settings.index(s)


def unb_col(t: CountsTable, j: int, K: int, rows=None) -> list[float]:
    rows = range(len(t.ids)) if rows is None else rows
    return [unbiased(t.cell_n(i, j), t.c[i][j], K) for i in rows]


def _rates(t: CountsTable) -> np.ndarray:
    return np.array([[t.rate(i, j) for j in range(len(t.settings))] for i in range(len(t.ids))], dtype=np.float64)


def plug_single(t: CountsTable, j: int, K: int) -> float:
    """Measured-rate pass@K of one setting: mean over questions of 1 - (1 - p)^K (the plug-in of onetemp.core.passk)."""
    return float(np.mean(passk_plugin(_rates(t)[:, j], K)))


def plug_uniform(t: CountsTable, K: int) -> float:
    """Measured-rate pass@K of the uniform scheme, K/l samples at each of the l settings."""
    l = len(t.settings)
    return float(np.mean(passk_plugin(_rates(t), np.full(l, K / l))))


def best_setting(t: CountsTable, K: int, rows=None):
    """In-sample best setting at K by unbiased pass@K (first in grid order on a tie)."""
    vals = [mean(unb_col(t, j, K, rows)) for j in range(len(t.settings))]
    j = max(range(len(vals)), key=lambda k: (vals[k], -k))
    return j, vals[j]


def peak_index(t: CountsTable, i: int) -> int:
    """Grid position of the largest measured pass rate, the smallest setting among ties."""
    r = [t.rate(i, j) for j in range(len(t.settings))]
    m = max(r)
    return r.index(m)


def hardness_at_best(t: CountsTable, i: int) -> float:
    """H_q = -1 / ln(1 - P) at the question's best measured pass rate (Equation 3), with P clamped to
    [1e-9, 0.99999] as the paper's plotting code does (only questions with P > 0 are passed in)."""
    p = max(t.rate(i, j) for j in range(len(t.settings)))
    p = min(max(p, 1e-9), 0.99999)
    return -1.0 / math.log1p(-p)


def ols_slope(x, y) -> float:
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx


# --------------------------------------------------------------------------------------------
# Appendix C and H
# --------------------------------------------------------------------------------------------

def appendix_c() -> None:
    inc = {
        ("livecodebench", "gpt-oss-20b"): (766, 16, "~1.5k"), ("livecodebench", "Mistral-7B-Instruct-v0.3"): (359, 16, "~1.5k"),
        ("livecodebench", "Qwen3-8B"): (647, 38, "~1.5k"), ("livecodebench", "Qwen3-0.6B"): (85, 20, None),
        ("math500", "gpt-oss-20b"): (500, 16, 1500), ("math500", "Mistral-7B-Instruct-v0.3"): (455, 16, 1500),
        ("math500", "Qwen3-8B"): (496, 20, 750), ("math500", "Qwen3-0.6B"): (492, 20, 1500),
        ("hitom", "gpt-oss-20b"): (678, 31, 1500), ("hitom", "Mistral-7B-Instruct-v0.3"): (615, 31, 1500),
        ("hitom", "Qwen3-8B"): (466, 31, 1500), ("hitom", "Qwen3-0.6B"): (743, 31, 1500),
    }
    totals = {"livecodebench": {"gpt-oss-20b": 975, "Mistral-7B-Instruct-v0.3": 975, "Qwen3-8B": 975,
                                "Qwen3-0.6B": 110},
              "math500": dict.fromkeys(("gpt-oss-20b", "Mistral-7B-Instruct-v0.3", "Qwen3-8B", "Qwen3-0.6B"), 500)}
    for (b, m), (n_inc, n_t, nu) in inc.items():
        rel = (f"{b}/{m}__temperature__question_n.jsonl" if b == "livecodebench" else f"{b}/{m}__temperature.jsonl")
        t = T(rel)
        if b in totals:
            add(f"App. C, Datasets, {m}, {b}", "questions in the study (975 LiveCodeBench V1-V6, 110 for "
                "Qwen3-0.6B; 500 MATH-500)", totals[b][m], len(t.ids), tol=0)
        n_here = len(included_ids(rel))
        fit_note = ("the paper's fitted-peak filter, recomputed with the paper's own code on this export "
                    "(docs/data.md, section 11); the list is in included_ids.json")
        if (b, m) == ("livecodebench", "Qwen3-8B"):
            # The Appendix C table prints 646, the count of the earlier data release; every figure of the
            # paper, and the text of Appendix G, use the later release, in which 647 pass the filter.
            add(f"App. C table, {m}, {b}", "included questions (later release, as the figures print)",
                647, n_here, tol=0)
            LINES[-1]["note"] = fit_note
            LINES.append({"where": f"App. C table, {m}, {b}", "what": "included questions, as the table prints",
                          "paper": 646, "ours": n_here, "diff": n_here - 646, "ok": None, "unit": "",
                          "note": "646 is the earlier release (100,568,250 samples), which is not in this export; "
                                  "Appendix G states both counts"})
            drawn = sum(t.cell_n(i, j) for i in range(len(t.ids)) for j in range(len(t.settings)))
            add(f"App. G, Experiment, {m}, {b}", "samples of the later release", 119228250, drawn, tol=0)
        else:
            add(f"App. C table, {m}, {b}", "included questions", n_inc, n_here, tol=0)
            LINES[-1]["note"] = fit_note
        add(f"App. C table, {m}, {b}", "temperatures in the grid", n_t, len(t.settings), tol=0)
        if isinstance(nu, int):
            ns = {t.cell_n(i, j) for i in range(len(t.ids)) for j in range(len(t.settings))}
            add(f"App. C table, {m}, {b}", "samples per (q, tau)", nu, max(ns) if len(ns) == 1 else -1, tol=0)
        elif b == "livecodebench":
            ns = sorted(t.cell_n(i, j) for i in range(len(t.ids)) for j in range(len(t.settings)))
            med = ns[len(ns) // 2]
            LINES.append({"where": f"App. C table, {m}, {b}", "what": "samples per (q, tau): median / max",
                          "paper": nu or "approximately doubled on some hard questions",
                          "ours": f"{med} / {ns[-1]}", "diff": None, "ok": True, "unit": ""})
    g = T("livecodebench/Qwen3-8B__temperature__question_n.jsonl").settings
    add("App. C table, Qwen3-8B, livecodebench", "grid: 0.1 .. 3.5 step 0.1, then 3.7, 4.0, 4.2", True,
        g == [round(0.1 * k, 6) for k in range(1, 36)] + [3.7, 4.0, 4.2])
    g = T("livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl").settings
    add("App. C table, GPT OSS 20B / Mistral, livecodebench and MATH-500", "grid: 0.1 .. 1.2, 1.4, 1.5, 1.7, 2.0",
        True, g == [round(0.1 * k, 6) for k in range(1, 13)] + [1.4, 1.5, 1.7, 2.0])
    h = T("hitom/Qwen3-0.6B__temperature.jsonl").settings
    add("App. C table, Hi-ToM", "grid: 0.075 .. 1.95 step 0.075, then 2.0, 2.2, 2.5, 3.0, 4.0",
        True, h == [round(0.075 * k, 6) for k in range(1, 27)] + [2.0, 2.2, 2.5, 3.0, 4.0])


def appendix_h() -> None:
    printed_changed = {"Mistral-7B-Instruct-v0.3": 4, "gpt-oss-20b": 18, "Qwen3-8B": 197, "Qwen3-0.6B": 3}
    printed_n = {"Mistral-7B-Instruct-v0.3": 359, "gpt-oss-20b": 766, "Qwen3-8B": 677, "Qwen3-0.6B": 86}
    for m in printed_changed:
        q = T(f"livecodebench/{m}__temperature__question_n.jsonl", included=True)
        c = T(f"livecodebench/{m}__temperature__cell_n.jsonl", included=True)
        add(f"App. H, Data, {m}", "included questions, corrected parse", printed_n[m], len(c.ids), tol=0)
        LINES[-1]["note"] = "the paper's fitted-peak filter on the cell_n counts (docs/data.md, section 5)"
        # A question's profile changes with the parse iff a temperature whose own n differs from the
        # question-level n lies inside its fit bracket: fit_all.prepare_question fits only the temperatures
        # from one step below the first temperature with a correct sample to one step above the last.
        changed = 0
        qset = set(q.ids)
        for i, qid in enumerate(c.ids):
            if qid not in qset:
                continue
            hits = [j for j in range(len(c.settings)) if c.c[i][j] > 0]
            if not hits:
                continue
            lo, hi = max(0, min(hits) - 1), min(len(c.settings) - 1, max(hits) + 1)
            q_n = max(c.n[i])
            changed += any(c.n[i][j] != q_n for j in range(lo, hi + 1))
        add(f"App. H, Data, {m}", "profiles changed by the corrected parse (included under both parses, with a "
                                  "temperature of uneven n inside the fit bracket)", printed_changed[m], changed, tol=0)
    for m, added in (("Qwen3-8B", 30), ("Qwen3-0.6B", 1)):
        q = T(f"livecodebench/{m}__temperature__question_n.jsonl", included=True)
        c = T(f"livecodebench/{m}__temperature__cell_n.jsonl", included=True)
        add(f"App. H, Data, {m}", "included questions added by the corrected parse", added,
            len(set(c.ids) - set(q.ids)), tol=0)
        add(f"App. H, Data, {m}", "included questions lost by the corrected parse", 0,
            len(set(q.ids) - set(c.ids)), tol=0)


# --------------------------------------------------------------------------------------------
# Appendix I
# --------------------------------------------------------------------------------------------

def appendix_i() -> None:
    t15 = T("hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl")
    t15d = T("hitom/Qwen3-0.6B__min_p_tau1.5__stage2_nu400.jsonl")
    t20 = T("hitom/Qwen3-0.6B__min_p_tau2.0__nu400.jsonl")
    mis = T("hitom/Mistral-7B-Instruct-v0.3__min_p_tau2.0__nu1500.jsonl")
    # the paper's floor on measured rates: keep a question if its best pass rate is at least 0.1 %
    floor = [q for i, q in enumerate(mis.ids) if 100 * max(mis.rate(i, j) for j in range(5)) >= 0.1]
    misi = T("hitom/Mistral-7B-Instruct-v0.3__min_p_tau2.0__nu1500.jsonl", ids=floor)
    plain_m = T("hitom/Mistral-7B-Instruct-v0.3__temperature.jsonl")
    # the comparison of Appendix I runs on the sorted ids kept by both screens
    top = T("hitom/Qwen3-0.6B__temperature__topup1000.jsonl")
    Qids = sorted(set(top.ids) & set(t20.ids))
    pos = {q: i for i, q in enumerate(top.ids)}
    plain_q = CountsTable(top.benchmark, top.model, top.axis, Qids, top.settings,
                          [top.n[pos[q]] for q in Qids], [top.c[pos[q]] for q in Qids])
    w = "App. I"
    add(f"{w}, Setup table", "kept questions, Qwen3-0.6B tau 1.5 / 2.0 / Mistral tau 2.0",
        "648/648/617", f"{len(t15.ids)}/{len(t20.ids)}/{len(mis.ids)}")
    add(f"{w}, Setup table", "nu, Qwen3-0.6B tau 1.5 / 2.0 / Mistral tau 2.0", "500/400/1500",
        f"{t15.n[0][0]}/{t20.n[0][0]}/{mis.n[0][0]}")
    add(f"{w}, Setup", "tau 1.5 grid", "0,0.001,0.003,0.01,0.03,0.1,0.2,0.3,0.5,0.7",
        ",".join(f"{x:g}" for x in t15.settings))
    add(f"{w}, Setup", "tau 2.0 grid", "0.0056,0.0128,0.0316,0.0721,0.178", ",".join(f"{x:g}" for x in t20.settings))

    # tau = 1.5 ratio-to-uniform table (measured rates, nu = 500)
    Ks = (1, 16, 64, 384)
    printed = {0.0: (0.904, 0.979, 1.024, 1.008), 0.001: (1.284, 1.169, 1.081, 1.014),
               0.003: (1.318, 1.174, 1.076, 1.013), 0.01: (1.329, 1.166, 1.065, 1.006),
               0.03: (1.291, 1.132, 1.037, 0.983), 0.1: (1.151, 1.015, 0.940, 0.914),
               0.2: (0.987, 0.873, 0.808, 0.780)}
    unif = {K: plug_uniform(t15, K) for K in Ks}
    for m, vals in printed.items():
        for K, pv in zip(Ks, vals):
            add(f"{w}, tau 1.5 table", f"ratio to uniform, min_p {m:g}, K = {K}", pv,
                plug_single(t15, col(t15, m), K) / unif[K], unit="ratio")
    for K, pv in zip(Ks, (7.32, 58.38, 85.32, 97.94)):
        add(f"{w}, tau 1.5 table", f"uniform pass@{K}", pv, 100 * unif[K], unit="%")

    # absolute gains at tau 1.5, nu = 500, unbiased estimator
    j0 = col(t15, 0.0)
    b1 = col(t15, 0.01)
    a0 = unb_col(t15, j0, 1)
    add(f"{w}, tau 1.5 text", "pass@1 at min_p 0", 6.62, 100 * mean(a0), unit="%")
    jb, vb = best_setting(t15, 1)
    add(f"{w}, tau 1.5 text", "best value at K = 1 (min_p)", 0.01, t15.settings[jb], tol=0)
    add(f"{w}, tau 1.5 text", "pass@1 at the best value", 9.73, 100 * vb, unit="%")
    for K, (g_best, se_best) in ((1, (3.11, 0.14)), (16, (11.34, 0.51)), (64, (4.63, 0.45))):
        jb, _ = best_setting(t15, K)
        a = unb_col(t15, jb, K)
        b = unb_col(t15, j0, K)
        add(f"{w}, tau 1.5 text", f"gain of the best value ({t15.settings[jb]:g}) over min_p 0, pass@{K}",
            g_best, 100 * (mean(a) - mean(b)), unit="pp")
        add(f"{w}, tau 1.5 text", f"paired SE of that gain, pass@{K}", se_best, 100 * paired_se(a, b), unit="pp")
    for K, g in ((1, 3.03), (16, 11.34), (64, 4.21)):
        j3 = col(t15, 0.003)
        add(f"{w}, tau 1.5 text", f"gain of min_p 0.003 over min_p 0, pass@{K}", g,
            100 * (mean(unb_col(t15, j3, K)) - mean(unb_col(t15, j0, K))), unit="pp")

    # tau = 2.0 against tau = 1.5, both on the 400 second-stage samples
    rows = ((1, 0.0316, 10.42, 0.01, 9.71, 0.71, 0.11), (4, 0.0316, 33.72, 0.01, 31.51, 2.22, 0.28),
            (16, 0.0316, 72.54, 0.003, 68.97, 3.57, 0.38), (64, 0.0128, 95.11, 0.001, 92.92, 2.20, 0.37),
            (256, 0.0128, 99.73, 0.001, 99.25, 0.48, 0.23), (400, 0.0128, 100.00, 0.001, 99.69, 0.31, 0.22))
    for K, m2, p2, m1, p1, gap, se in rows:
        j2, v2 = best_setting(t20, K)
        j1, v1 = best_setting(t15d, K)
        add(f"{w}, tau 2.0 vs 1.5 table", f"best tau 2.0 value at K = {K}", m2, t20.settings[j2], tol=0)
        add(f"{w}, tau 2.0 vs 1.5 table", f"its pass@{K}", p2, 100 * v2, unit="%")
        add(f"{w}, tau 2.0 vs 1.5 table", f"best tau 1.5 value at K = {K}", m1, t15d.settings[j1], tol=0)
        add(f"{w}, tau 2.0 vs 1.5 table", f"its pass@{K}", p1, 100 * v1, unit="%")
        a, b = unb_col(t20, j2, K), unb_col(t15d, j1, K)
        add(f"{w}, tau 2.0 vs 1.5 table", f"gap at K = {K}", gap, 100 * (mean(a) - mean(b)), unit="pp")
        add(f"{w}, tau 2.0 vs 1.5 table", f"paired SE at K = {K}", se, 100 * paired_se(a, b), unit="pp")
    # 0.0316 at tau 2.0 against 0.003 at tau 1.5
    jA, jB = col(t20, 0.0316), col(t15d, 0.003)
    for K, g, se in ((1, 0.77, 0.10), (16, 3.57, 0.38), (256, 0.31, 0.34)):
        a, b = unb_col(t20, jA, K), unb_col(t15d, jB, K)
        add(f"{w}, text after Fig. 35", f"0.0316 (tau 2.0) minus 0.003 (tau 1.5), pass@{K}", g,
            100 * (mean(a) - mean(b)), unit="pp")
        add(f"{w}, text after Fig. 35", f"paired SE, pass@{K}", se, 100 * paired_se(a, b), unit="pp")

    # comparison with the best plain temperature, on the 642 questions both screens kept
    Q = plain_q.ids
    i20 = {q: i for i, q in enumerate(t20.ids)}
    i15 = {q: i for i, q in enumerate(t15d.ids)}
    r20 = [i20[q] for q in Q]
    r15 = [i15[q] for q in Q]
    add(f"{w}, comparison table", "questions both stages kept", 642, len(Q), tol=0)
    comp = ((1, 1.275, 1.52, 0.10, 0.79, 0.07), (4, 1.275, 4.49, 0.28, 2.22, 0.19),
            (16, 1.275, 7.28, 0.43, 3.68, 0.29), (64, 1.35, 4.49, 0.41, 2.27, 0.34),
            (256, 1.35, 1.02, 0.22, 0.54, 0.22), (400, 1.35, 0.56, 0.19, 0.25, 0.16))
    for K, tp, g2, s2, g1, s1 in comp:
        jp, vp = best_setting(plain_q, K)
        add(f"{w}, comparison table", f"best plain tau at K = {K}", tp, plain_q.settings[jp], tol=0)
        pref = unb_col(plain_q, jp, K)
        j2, _ = best_setting(t20, K, r20)
        j1, _ = best_setting(t15d, K, r15)
        a2 = unb_col(t20, j2, K, r20)
        a1 = unb_col(t15d, j1, K, r15)
        add(f"{w}, comparison table", f"gap at tau 2.0, K = {K}", g2, 100 * (mean(a2) - mean(pref)), unit="pp")
        add(f"{w}, comparison table", f"its paired SE, K = {K}", s2, 100 * paired_se(a2, pref), unit="pp")
        add(f"{w}, comparison table", f"gap at tau 1.5, K = {K}", g1, 100 * (mean(a1) - mean(pref)), unit="pp")
        add(f"{w}, comparison table", f"its paired SE, K = {K}", s1, 100 * paired_se(a1, pref), unit="pp")
    j2, v2 = best_setting(t20, 1, r20)
    j1, v1 = best_setting(t15d, 1, r15)
    jp, vp = best_setting(plain_q, 1)
    add(f"{w}, comparison text", "pass@1, best value at tau 2.0", 10.37, 100 * v2, unit="%")
    add(f"{w}, comparison text", "pass@1, best value at tau 1.5", 9.65, 100 * v1, unit="%")
    add(f"{w}, comparison text", "pass@1, best plain temperature", 8.86, 100 * vp, unit="%")
    add(f"{w}, comparison text", "pass@1, plain tau 2.0 in the sweep", 1.83,
        100 * mean(unb_col(plain_q, col(plain_q, 2.0), 1)), unit="%")
    a = unb_col(t15d, col(t15d, 0.0), 1, r15)
    b = unb_col(plain_q, col(plain_q, 1.5), 1)
    add(f"{w}, comparison text", "min_p 0 at tau 1.5 (this study) minus plain tau 1.5 (sweep), pass@1",
        -0.04, 100 * (mean(a) - mean(b)), unit="pp")
    add(f"{w}, comparison text", "its paired SE", 0.06, 100 * paired_se(a, b), unit="pp")

    # Mistral 7B Instruct at tau = 2.0 against the best plain temperature, all 617 questions, nu = 1500
    im = {q: i for i, q in enumerate(plain_m.ids)}
    rm = [im[q] for q in mis.ids]
    add(f"{w}, Mistral table", "questions", 617, len(mis.ids), tol=0)
    mrows = ((1, 7.33, 1.575, 6.17, 1.16, 0.06), (4, 23.72, 1.575, 20.20, 3.52, 0.17),
             (16, 53.19, 1.575, 46.52, 6.68, 0.31), (64, 78.56, 1.725, 72.48, 6.08, 0.43),
             (256, 91.60, 1.8, 89.81, 1.79, 0.49), (1024, 97.12, 2.0, 98.00, -0.88, 0.58),
             (1500, 97.89, 2.0, 99.19, -1.30, 0.60))
    for K, pm, tp, pp, gap, se in mrows:
        jm, vm = best_setting(mis, K)
        jp, vp = best_setting(plain_m, K, rm)
        add(f"{w}, Mistral table", f"best min_p value at K = {K}", 0.0056, mis.settings[jm], tol=0)
        add(f"{w}, Mistral table", f"its pass@{K}", pm, 100 * vm, unit="%")
        add(f"{w}, Mistral table", f"best plain tau at K = {K}", tp, plain_m.settings[jp], tol=0)
        add(f"{w}, Mistral table", f"its pass@{K}", pp, 100 * vp, unit="%")
        a, b = unb_col(mis, jm, K), unb_col(plain_m, jp, K, rm)
        add(f"{w}, Mistral table", f"gap at K = {K}", gap, 100 * (mean(a) - mean(b)), unit="pp")
        add(f"{w}, Mistral table", f"paired SE at K = {K}", se, 100 * paired_se(a, b), unit="pp")
    for K, rel in ((1, 19), (16, 14)):
        jm, vm = best_setting(mis, K)
        jp, vp = best_setting(plain_m, K, rm)
        add(f"{w}, Mistral text", f"relative gain at K = {K}", rel, 100 * (vm / vp - 1), unit="%", dec=0)
    j56, jp2 = col(mis, 0.0056), col(plain_m, 2.0)
    for K, g in ((1, 3.41), (16, 15.57), (64, 11.42)):
        a, b = unb_col(mis, j56, K), unb_col(plain_m, jp2, K, rm)
        add(f"{w}, Mistral text", f"0.0056 minus plain tau 2.0, pass@{K}", g, 100 * (mean(a) - mean(b)), unit="pp")
        if K == 1:
            add(f"{w}, Mistral text", "paired SE, pass@1", 0.14, 100 * paired_se(a, b), unit="pp")
            add(f"{w}, Mistral text", "plain tau 2.0, pass@1", 3.92, 100 * mean(b), unit="%")
    add(f"{w}, Mistral text", "min_p 0.0128, pass@1", 7.32, 100 * mean(unb_col(mis, col(mis, 0.0128), 1)), unit="%")

    # ratio to uniform at tau = 2.0 (measured rates)
    Kq = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    best_by_K = {}
    for K in Kq:
        u = plug_uniform(t20, K)
        r = [plug_single(t20, j, K) / u for j in range(len(t20.settings))]
        best_by_K[K] = (t20.settings[max(range(len(r)), key=lambda k: r[k])], r)
    add(f"{w}, rtu at tau 2.0, Qwen3-0.6B", "best single value at every K <= 32", 0.0316,
        0.0316 if all(best_by_K[K][0] == 0.0316 for K in Kq if K <= 32) else -1, tol=0)
    add(f"{w}, rtu at tau 2.0, Qwen3-0.6B", "best single value from K = 48 on", 0.0128,
        0.0128 if all(best_by_K[K][0] == 0.0128 for K in Kq if K >= 48) else -1, tol=0)
    add(f"{w}, rtu at tau 2.0, Qwen3-0.6B", "ratio of 0.0316 at K = 1", 1.107,
        best_by_K[1][1][col(t20, 0.0316)], unit="ratio")
    add(f"{w}, rtu at tau 2.0, Qwen3-0.6B", "largest ratio of 0.178", 0.974,
        max(best_by_K[K][1][col(t20, 0.178)] for K in Kq), unit="ratio")
    Km = Kq + [512, 768, 1024]
    rat = {}
    for K in Km:
        u = plug_uniform(misi, K)
        rat[K] = [plug_single(misi, j, K) / u for j in range(len(misi.settings))]
    j56 = col(misi, 0.0056)
    add(f"{w}, rtu at tau 2.0, Mistral", "included questions (fitted floor 0.1 %)", 598, len(misi.ids), tol=0)
    add(f"{w}, rtu at tau 2.0, Mistral", "0.0056 best single value at every K", True,
        all(max(range(5), key=lambda k: rat[K][k]) == j56 for K in Km))
    for K, pv in ((1, 1.057), (32, 1.109), (1024, 1.025)):
        add(f"{w}, rtu at tau 2.0, Mistral", f"ratio of 0.0056 at K = {K}", pv, rat[K][j56], unit="ratio")
    j178 = col(misi, 0.178)
    for K, pv in ((1, 0.884), (1024, 0.558)):
        add(f"{w}, rtu at tau 2.0, Mistral", f"ratio of 0.178 at K = {K}", pv, rat[K][j178], unit="ratio")

    # inclusion floor, profile peaks, hardness bins and slopes (measured rates)
    best = [max(mis.rate(i, j) for j in range(5)) for i in range(len(mis.ids))]
    add(f"{w}, Fig. 37 caption", "Mistral questions below the 0.1 % floor", 19, sum(1 for b in best if b < 0.001), tol=0)
    add(f"{w}, Fig. 37 caption", "of them with no correct sample", 8, sum(1 for b in best if b == 0), tol=0)
    pk15 = [peak_index(t15, i) for i in range(len(t15.ids))]
    add(f"{w}, profiles text", "tau 1.5: questions peaking at min_p 0", 33, pk15.count(0), tol=0)
    add(f"{w}, profiles text", "tau 1.5: questions peaking at min_p 0.7", 24, pk15.count(9), tol=0)
    add(f"{w}, profiles text", "tau 1.5: questions with no correct sample at 0.7", 576,
        sum(1 for i in range(len(t15.ids)) if t15.c[i][9] == 0), tol=0)
    add(f"{w}, profiles text", "tau 2.0: Qwen3-0.6B questions peaking at 0.178", 133,
        sum(1 for i in range(len(t20.ids)) if peak_index(t20, i) == 4), tol=0)
    add(f"{w}, profiles text", "tau 2.0: Mistral questions peaking at 0.178", 119,
        sum(1 for i in range(len(misi.ids)) if peak_index(misi, i) == 4), tol=0)
    for name, t, nb_hi, slope_all, slope_sub in (("Qwen3-0.6B, tau 1.5", t15, 9, -3.10, -2.86),
                                                 ("Qwen3-0.6B, tau 2.0", t20, 2, -1.98, -1.35),
                                                 ("Mistral, tau 2.0", misi, 68, -1.51, -1.48)):
        H = [hardness_at_best(t, i) for i in range(len(t.ids))]
        lo, hi = math.log10(min(H)), math.log10(max(H))
        edges = [10 ** (lo + (hi - lo) * k / 17) for k in range(18)]
        above = [k for k in range(17) if edges[k] >= 100.0]
        cnt = 0
        for h in H:
            k = min(16, int((math.log10(h) - lo) / (hi - lo) * 17))
            cnt += k in above
        add(f"{w}, Fig. 41 caption", f"{name}: questions in the bins wholly above H = 100", nb_hi, cnt, tol=0)
        x = [math.log10(h) for h in H]
        pk = [peak_index(t, i) for i in range(len(t.ids))]
        add(f"{w}, Peak text", f"{name}: OLS slope of the peak position on log10 H", slope_all, ols_slope(x, pk), dec=2)
        sub = [k for k in range(len(pk)) if pk[k] > 0]
        add(f"{w}, Peak text", f"{name}: the same without questions peaking at the smallest value", slope_sub,
            ols_slope([x[k] for k in sub], [pk[k] for k in sub]), dec=2)


# --------------------------------------------------------------------------------------------
# LiveCodeBench pass@1: measured against the fitted values of Tables 2-6
# --------------------------------------------------------------------------------------------

def lcb_tables() -> None:
    printed = {  # K = 1 row of each table, first temperatures (fitted profiles)
        "Mistral-7B-Instruct-v0.3": ("Table 3", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                                     (0.178, 0.184, 0.186, 0.185, 0.181, 0.177, 0.171, 0.164)),
        "Qwen3-8B": ("Table 4", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                     (0.585, 0.585, 0.583, 0.582, 0.580, 0.578, 0.576, 0.575, 0.574, 0.573)),
    }
    for m, (tab, temps, vals) in printed.items():
        t = T(f"livecodebench/{m}__temperature__question_n.jsonl", included=True)
        for tau, pv in zip(temps, vals):
            j = col(t, tau)
            meas = mean(t.rate(i, j) for i in range(len(t.ids)))
            LINES.append({"where": f"{tab} (fitted), {m}", "what": f"pass@1 at tau {tau:g}: measured c/n vs fitted",
                          "paper": pv, "ours": meas, "diff": meas - pv, "ok": None, "unit": "",
                          "note": "the table uses the fitted profiles; the measured mean is shown for scale"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=None, help="also write every comparison to this JSON file")
    a = ap.parse_args(argv)
    appendix_c()
    appendix_h()
    appendix_i()
    lcb_tables()
    # ok: True = equal at the paper's printed precision; False = not; None = shown for scale or a
    # known, explained difference (not a reproduction claim)
    good = [x for x in LINES if x["ok"] is True]
    bad = [x for x in LINES if x["ok"] is False]
    info = [x for x in LINES if x["ok"] is None]
    for x in LINES:
        d = "" if x["diff"] is None else f"{x['diff']:+.4g}"
        o = x["ours"] if not isinstance(x["ours"], float) else f"{x['ours']:.6g}"
        tag = {True: "ok  ", False: "OFF ", None: "info"}[x["ok"]]
        print(f"{tag} | {x['where']} | {x['what']} | paper {x['paper']} | ours {o} | {d}"
              + (f" | {x['note']}" if x.get("note") else ""))
    print(f"\n{len(good) + len(bad)} reproduced numbers: {len(good)} equal at the paper's printed precision, "
          f"{len(bad)} not; {len(info)} more lines shown for scale or as a known difference")
    if a.json:
        Path(a.json).write_text(json.dumps(LINES, indent=1) + "\n", encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
