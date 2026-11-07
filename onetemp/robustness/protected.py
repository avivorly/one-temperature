"""Reweighting outside a protected hardness window: beta* against D (Appendix H, Figure 4(a), Appendix I's min_p version).

Test (Appendix H, "Test"). For a question q, s_q(tau) is its pass@K at setting tau and S_W(tau) = sum_q W_q s_q(tau).
tau_A maximises S_U (uniform weights) and score_A = S_U(tau_A). The settings that lose at least 10% form
C = {tau : S_U(tau) <= 0.9 score_A} (0.99 for a 1% loss). The protected window holds the questions with
|log10 H_q - log10 K| <= D, which keep W_q = 1; every other question has W_q in [1/beta, beta], with no
normalisation of sum W. A setting tau_c in C wins under W if S_W(tau_c) - S_W(tau') >= 1e-9 for every tau' not in C.
beta* is the smallest beta on the grid 1.005^i (capped at 10^6) at which some tau_c in C wins under some such W;
"> 10^6" when no beta up to 10^6 suffices (also when C is empty, recorded as its own reason). Since a larger D
protects more questions, beta* is nondecreasing in D. Each curve (loss, K) stops at the first D whose window holds
every question.

How (the paper's code): tau_c is attainable at beta iff the max-margin LP
    max t  s.t.  sum_{q out} W_q (s_q(tau') - s_q(tau_c)) + t <= -sum_{q prot} (s_q(tau') - s_q(tau_c))  for tau' not in C,
                 1/beta <= W_q <= beta
has t* >= 1e-9 (HiGHS dual simplex, then IPM, then the default HiGHS, with tight tolerances). The candidates of C are
tried closest-to-the-loss first; the first attainable one is bisected over the whole grid, later ones only below the
best index so far. Every D is solved cold. The witness is the LP's W at beta* x 1.005, clipped onto its box.
The exact rational checks of every point are in certify.py and run after the sweep (--certify, on by default).

Windows (--window):
  sym    protected iff |x_q - log10 K| <= D (the paper's test);
  hard   only the questions more than d decades harder than K are reweighted (x_q > log10 K + d);
  easy   only the questions more than d decades easier than K are reweighted (x_q < log10 K - d);
here x_q = log10 H_q at the question's best setting (--hardness best), or at tau_A (--hardness tauA, the variant of
"One side at a time" that measures easiness where tau_A is used).

    python -m onetemp.robustness.protected --input profiles.jsonl --benchmark livecodebench \
        --model mistralai/Mistral-7B-Instruct-v0.3 --out out/prot_lcb_mistral --plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from . import contexts
from .data import add_input_args, environment, guard_outputs, study_from_args

THRS = (0.9, 0.99)
K_PAPER = (4, 32, 128, 512)
D_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
          5.0, 5.5, 6.0)
D_GRID_SIM = D_GRID + (6.5, 7.0, 7.5)
EPS_ABS = 1e-9
TIGHT = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
         "ipm_optimality_tolerance": 1e-12}
ATTEMPTS = (("highs-ds", 20.0), ("highs-ipm", 60.0), ("highs", 60.0))
BETA_MAX = 1e6
STEP = 1.005
I_TOP = int(np.floor(np.log(BETA_MAX) / np.log(STEP))) + 1          # 2771
BETA_GRID = np.array([STEP ** i for i in range(I_TOP)] + [BETA_MAX])  # BETA_GRID[I_TOP] = 1e6


def window_masks(x: np.ndarray, K: float, D: float, window: str = "sym"):
    """(protected, outside) boolean masks of one window."""
    c = math.log10(float(K))
    if window == "sym":
        prot = np.abs(x - c) <= D
    elif window == "hard":
        prot = ~(x > c + D)
    elif window == "easy":
        prot = ~(x < c - D)
    else:
        raise ValueError(f"window must be sym, hard or easy, not {window!r}")
    return prot, ~prot


def margin_lp(s_k, c, non, prot, out, beta, log=None):
    """t* = max over admissible W of min over tau' not in C of S_W(tau_c) - S_W(tau'), and the outside W.

    Returns (t or None, W_out or None); None when no HiGHS method solves the LP.
    """
    Dm = (s_k[:, non] - s_k[:, [c]]).T                     # (|non|, n)
    const = Dm[:, prot].sum(axis=1)
    A = np.hstack([Dm[:, out], np.ones((len(non), 1))])
    cvec = np.zeros(int(out.sum()) + 1)
    cvec[-1] = -1.0
    bounds = [(1.0 / beta, beta)] * int(out.sum()) + [(None, None)]
    for meth, limit in ATTEMPTS:
        t0 = time.perf_counter()
        res = linprog(cvec, A_ub=A, b_ub=-const, bounds=bounds, method=meth, options=dict(TIGHT, time_limit=limit))
        if log is not None:
            log.append([int(c), meth, int(res.status), round(time.perf_counter() - t0, 4)])
        if res.status == 0:
            return float(res.x[-1]), np.asarray(res.x[:-1], dtype=np.float64)
    return None, None


def candidates(s_k: np.ndarray, thr: float):
    """C (in the paper's order: closest to the loss first, then by index) and the settings outside C."""
    S_U = s_k.sum(axis=0)
    cand = [int(t) for t in np.flatnonzero(S_U <= thr * S_U.max())]
    non = np.setdiff1d(np.arange(s_k.shape[1]), cand)
    order = sorted(cand, key=lambda t: (-S_U[t], t))
    return cand, non, order, S_U


def sweep_curve(x, s_k, tau, K, thr, D_grid, window="sym", stop_when_all_protected=True):
    """beta* at every D of one curve (one loss threshold, one K). Returns a list of point dicts."""
    T = s_k.shape[1]
    n = len(x)
    cand, non, order, S_U = candidates(s_k, thr)
    score_A = float(S_U.max())
    pts = []
    for j, D in enumerate(D_grid):
        tp = time.perf_counter()
        prot, out = window_masks(x, K, D, window)
        log: list = []
        pt = {"thr": thr, "K": K, "D": float(D), "D_index": j, "window": window, "n_protected": int(prot.sum()),
              "n_outside": int(out.sum()), "n_candidates": len(cand), "beta_star": None, "beta_star_index": None,
              "tau_c": None}
        if out.sum() == 0:
            pt.update({"result": "> 1e6", "why": "no question outside", "n_lp": 0})
            pts.append(pt)
            if stop_when_all_protected:
                break
            continue
        if not cand:
            pt.update({"result": "> 1e6", "why": "C is empty", "n_lp": 0})
            pts.append(pt)
            continue
        cache: dict = {}

        def att(c, i):
            if (c, i) not in cache:
                t, _ = margin_lp(s_k, c, non, prot, out, float(BETA_GRID[i]), log)
                cache[(c, i)] = t
            t = cache[(c, i)]
            return None if t is None else bool(t >= EPS_ABS)

        best_i, best_c, unres = None, None, 0
        for c in order:
            hi = I_TOP if best_i is None else best_i - 1
            if hi < 0:
                continue
            v = att(c, hi)
            unres += v is None
            if not v:
                continue
            lo = -1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                v = att(c, mid)
                unres += v is None
                if v:
                    hi = mid
                else:
                    lo = mid
            best_i, best_c = hi, c
        if best_i is None:
            pt.update({"result": "> 1e6", "why": f"no W with beta <= 1e6 ({int(out.sum())} outside)"})
        else:
            b = float(BETA_GRID[best_i])
            bw = b * 1.005
            t_w, Wo = margin_lp(s_k, best_c, non, prot, out, bw, log)
            W = np.ones(n)
            if Wo is not None:
                W[out] = np.clip(Wo, 1.0 / bw, bw)
            S_W = W @ s_k
            pt.update({"result": "finite", "beta_star": b, "beta_star_index": int(best_i), "tau_c": float(tau[best_c]),
                       "tau_c_index": int(best_c), "ratio_tau_c": float(S_U[best_c] / score_A),
                       "witness": {"beta_w": bw, "lp_t": t_w, "margin_after_clip": float(S_W[best_c] - S_W[non].max())},
                       "_W": W})
        pt.update({"n_lp": len(log), "unresolved_lp": unres, "seconds": round(time.perf_counter() - tp, 3)})
        pts.append(pt)
    return pts


def sweep(inp: dict, Ks=K_PAPER, thrs=THRS, D_grid=D_GRID, window="sym", x=None):
    """Every curve of one study. ``inp`` is contexts.prot_inputs(...); ``x`` overrides its hardness."""
    x = inp["x"] if x is None else x
    tau = inp["settings"]
    curves = {}
    for thr in thrs:
        for K in Ks:
            curves[(float(thr), int(K))] = sweep_curve(x, inp["s"][int(K)], tau, int(K), float(thr), D_grid, window)
    return curves


def beta_array(curves, Ks, thrs, D_grid, key="beta_star"):
    """(thr, K, D) array: the value of ``key`` (inf for "> 1e6", nan where the curve stopped)."""
    arr = np.full((len(thrs), len(Ks), len(D_grid)), np.nan)
    for a, thr in enumerate(thrs):
        for k, K in enumerate(Ks):
            for pt in curves[(float(thr), int(K))]:
                v = pt.get(key)
                arr[a, k, pt["D_index"]] = np.inf if v is None else v
    return arr


def save_witnesses(path, curves, Ks, thrs, D_grid, n):
    """The sweep's witnesses (W at beta* x 1.005, every question) as an npz array (thr, K, D, n); nan where none."""
    W = np.full((len(thrs), len(Ks), len(D_grid), n), np.nan)
    for a, thr in enumerate(thrs):
        for k, K in enumerate(Ks):
            for pt in curves[(float(thr), int(K))]:
                if "_W" in pt:
                    W[a, k, pt["D_index"]] = pt["_W"]
    np.savez_compressed(path, W=W, thr=np.asarray(thrs), K=np.asarray(Ks), D=np.asarray(D_grid))


def monotone_violations(arr, D_grid):
    viol = []
    for a in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            seq = [(D_grid[j], v) for j, v in enumerate(arr[a, k]) if not np.isnan(v)]
            viol += [(a, k, seq[i][0]) for i in range(len(seq) - 1) if seq[i + 1][1] < seq[i][1]]
    return viol


def hardness_at(inp: dict, which: str, K: int):
    """x_q at the best setting (the paper) or at tau_A of this K (Appendix H's tau_A variant)."""
    if which == "best":
        return inp["x"]
    ia = inp["t_a_idx"][int(K)]
    P = np.minimum(inp["P"][:, ia], contexts.P_CAP)
    with np.errstate(divide="ignore"):
        return np.log10(-1.0 / np.log1p(-P))


def parse_grids(K: str, thr: str, D: str | None):
    """Sorted, unique, validated K, loss thresholds and D grid from their comma-separated forms."""
    Ks = sorted({int(k) for k in K.split(",")})
    thrs = sorted({float(t) for t in thr.split(",")})
    D_grid = list(D_GRID) if D is None else sorted({float(d) for d in D.split(",")})
    if not Ks or Ks[0] < 1:
        raise SystemExit("--K must hold positive integers")
    if any(not (0.0 < t < 1.0) for t in thrs):
        raise SystemExit("--thr must lie strictly between 0 and 1 (0.9 is a 10% loss)")
    if any(d < 0 for d in D_grid):
        raise SystemExit("--D must be >= 0")
    return Ks, thrs, D_grid


def c_empty_array(curves, Ks, thrs, D_grid):
    """(thr, K, D) booleans: the points where C is empty (no setting loses that much; no curve to draw)."""
    arr = np.zeros((len(thrs), len(Ks), len(D_grid)), dtype=bool)
    for a, thr in enumerate(thrs):
        for k, K in enumerate(Ks):
            for pt in curves[(float(thr), int(K))]:
                arr[a, k, pt["D_index"]] = pt.get("why") == "C is empty"
    return arr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--K", default="4,32,128,512")
    ap.add_argument("--thr", default="0.9,0.99", help="loss thresholds: 0.9 = a 10%% loss, 0.99 = 1%%")
    ap.add_argument("--D", default=None, help="comma-separated D grid (default: the paper's 23 values, 0 to 6)")
    ap.add_argument("--window", choices=["sym", "hard", "easy"], default="sym")
    ap.add_argument("--hardness", choices=["best", "tauA"], default="best")
    ap.add_argument("--no-certify", action="store_true", help="skip the exact rational certification")
    ap.add_argument("--out", required=True, help="output directory; existing output files are never overwritten")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    Ks, thrs, D_grid = parse_grids(a.K, a.thr, a.D)
    out = Path(a.out)
    jp, wp, ip = out / "beta_star.json", out / "witness.npz", out / "beta_star_vs_D.jpg"
    guard_outputs([jp, wp] + ([ip] if a.plot else []))
    study = study_from_args(a)
    t0 = time.perf_counter()
    inp = contexts.prot_inputs(study, Ks, rates=a.rates)
    xs = {K: hardness_at(inp, a.hardness, K) for K in Ks}
    curves = {}
    for K in Ks:
        for thr in thrs:
            curves[(thr, K)] = sweep_curve(xs[K], inp["s"][K], inp["settings"], K, thr, D_grid, a.window)
    bstar = beta_array(curves, Ks, thrs, D_grid)
    res = {"study": study.label, "benchmark": study.benchmark, "model": study.model, "axis": study.axis,
           "tau": study.tau, "rates": a.rates, "n_questions": inp["n_questions"], "n_inc": inp["n_inc"],
           "settings": inp["settings"].tolist(), "dropped_settings": study.dropped_settings, "K": Ks, "thr": thrs,
           "D_grid": D_grid, "window": a.window, "hardness": a.hardness,
           "tau_A": {str(k): inp["tau_A"][k] for k in Ks}, "score_A": {str(k): inp["score_A"][k] for k in Ks},
           "x_min": float(inp["x"].min()), "x_max": float(inp["x"].max()), "x": inp["x"].tolist(),
           "beta_star_sweep": bstar.tolist(),
           "c_empty": c_empty_array(curves, Ks, thrs, D_grid).tolist(),
           "points": [pt for key in curves for pt in curves[key]], "invariant_violations_sweep":
               len(monotone_violations(bstar, D_grid)), "seconds_sweep": round(time.perf_counter() - t0, 1),
           "environment": environment()}
    if not a.no_certify:
        from . import certify
        cert = certify.certify_study(inp, curves, Ks, thrs, D_grid, a.window, xs)
        res["certify"] = cert["summary"]
        res["beta_star"] = cert["beta_star_final"].tolist()
        res["reachability"] = cert["reachability"]
        res["invariant_violations_final"] = len(monotone_violations(cert["beta_star_final"], D_grid))
    else:
        res["beta_star"] = bstar.tolist()
    out.mkdir(parents=True, exist_ok=True)
    save_witnesses(wp, curves, Ks, thrs, D_grid, inp["n_inc"])
    res["points"] = [{k: v for k, v in pt.items() if not k.startswith("_")} for pt in res["points"]]
    jp.write_text(json.dumps(res, indent=1, default=float))
    print(curve_table(res))
    if a.plot:
        from .plot_prot import draw_study
        draw_study(res, ip, title=a.title)
    print(f"wrote {jp} ({round(time.perf_counter() - t0, 1)} s)")


def curve_table(res: dict) -> str:
    """beta* per (loss, K) and D, as text ("> 1e6" for infinite, "no C" where C is empty, blank after a stop)."""
    lines = []
    arr = np.asarray(res["beta_star"], dtype=float)
    ce = np.asarray(res.get("c_empty", np.zeros(arr.shape, dtype=bool)), dtype=bool)
    for a, thr in enumerate(res["thr"]):
        lines.append(f"loss {round((1 - thr) * 100, 6):g}%: D = " + " ".join(f"{d:g}" for d in res["D_grid"]))
        for k, K in enumerate(res["K"]):
            vals = []
            for j, v in enumerate(arr[a, k]):
                vals.append("" if np.isnan(v) else ("no C" if ce[a, k, j] else (">1e6" if np.isinf(v) else f"{v:.4g}")))
            lines.append(f"  K={K}: " + " | ".join(vals))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
