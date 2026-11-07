"""Simulated hardness distributions (Appendix H, "Simulated hardness distributions", Figures 32-33).

The measured questions' hardness stays below 3 x 10^3, so to test large budgets the protected-window test is
repeated on three simulated datasets of 1000 questions each, with hardness up to L = 500,000:
  peaked    f(H) = (H/25000)^2.9928468843 ((500000 - H)/475000)^56.8640908012   (maximum 1 at H = 25,000)
  reversed  h(H) = 1 - (H/25000)^1.3727444798 ((500000 - H)/475000)^26.0821451156 (minimum 0 at H = 25,000)
  flat      y = 1/e
(each equals 1/e at H = 50,000), read as unnormalised densities of questions on a linear hardness axis. Question i
sits at the (i - 0.5)/1000 quantile, from a cumulative trapezoid on 4,000,001 points of [0, L] inverted by linear
interpolation.
Profiles: the measured questions with 2 <= log10 H < 3 (Mistral 7B Instruct on LiveCodeBench in the paper: 109),
sorted by hardness then id, are the templates. In order of hardness, a simulated question with log10 H_i >= 2 takes
template k mod n_templates; one below 2 takes the measured question nearest in log10 H. Each template T is scaled in
hardness by 1 - P_i(tau) = (1 - P_T(tau))^(H_T/H_i), so that H_i(tau) = (H_i/H_T) H_T(tau) at every setting.
The test then runs unchanged with K in {4, 32, 512, 10^4, 10^5, 10^6} and D up to 7.5.

    python -m onetemp.robustness.simulate --input profiles.jsonl --benchmark livecodebench \
        --model mistralai/Mistral-7B-Instruct-v0.3 --shape peaked --out out/sim_peaked [--certify] [--plot]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from onetemp.core.passk import passk_plugin

from . import contexts, protected
from .data import attach_core_profiles, environment, guard_outputs, load_study

L_MAX, M_PEAK, R_E = 500000.0, 25000.0, 50000.0
N_Q = 1000
N_FINE = 4_000_001
K_SIM = (4, 32, 512, 10_000, 100_000, 1_000_000)


def peaked(H):
    return (H / 25000.0) ** 2.9928468843 * ((500000.0 - H) / 475000.0) ** 56.8640908012


def reversed_(H):
    return 1.0 - (H / 25000.0) ** 1.3727444798 * ((500000.0 - H) / 475000.0) ** 26.0821451156


def flat(H):
    return np.full_like(np.asarray(H, dtype=np.float64), np.exp(-1.0))


SHAPES = {"peaked": peaked, "reversed": reversed_, "flat": flat}


def quantiles(fn, n_fine=N_FINE, n_q=N_Q):
    g = np.linspace(0.0, L_MAX, n_fine)
    y = fn(g)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(g))])
    c /= c[-1]
    u = (np.arange(1, n_q + 1) - 0.5) / n_q
    return np.interp(u, c, g)


def build(P_m, x_m, qid_m, shape: str, n_q: int = N_Q, n_fine: int = N_FINE):
    """(x, P) of one simulated pool from measured profiles P_m (n, T) with hardness x_m and tie-break ids qid_m."""
    H_m = -1.0 / np.log1p(-P_m.max(axis=1))
    tidx = np.flatnonzero((x_m >= 2.0) & (x_m < 3.0))
    tidx = tidx[np.lexsort((qid_m[tidx], x_m[tidx]))]
    if not len(tidx):
        raise ValueError("no measured question with 2 <= log10 H < 3 to use as a template")
    Hq = quantiles(SHAPES[shape], n_fine, n_q)
    x = np.log10(Hq)
    tmpl = np.empty(n_q, dtype=np.int64)
    hi = np.flatnonzero(x >= 2.0)
    tmpl[hi] = tidx[np.arange(len(hi)) % len(tidx)]
    for i in np.flatnonzero(x < 2.0):
        d = np.abs(x_m - x[i])
        cands = np.flatnonzero(d == d.min())
        tmpl[i] = cands[np.argmin(qid_m[cands])]
    expo = H_m[tmpl] / Hq
    P = passk_plugin(P_m[tmpl], expo[:, None], form="expm1")      # 1 - (1 - P)^expo, Appendix H form
    return {"x": x, "H": Hq, "P": P, "template": tmpl, "n_templates": int(len(tidx))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="counts or profiles.jsonl of the template study")
    ap.add_argument("--profiles", default=None, help="profiles.json of the profile fit (python -m onetemp.core.fit_all)")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--shape", choices=sorted(SHAPES), required=True)
    ap.add_argument("--K", default=",".join(map(str, K_SIM)))
    ap.add_argument("--D", default=None, help="default: the paper's grid up to 7.5")
    ap.add_argument("--certify", action="store_true", help="also run the exact certification (slower)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out)
    guard_outputs([out / "beta_star.json"] + ([out / "beta_star_vs_D.jpg"] if a.plot else []))
    study = load_study(a.input, a.benchmark, a.model, "temperature", require_fit=not a.profiles)
    if a.profiles:
        study = attach_core_profiles(study, a.profiles)
    base = contexts.prot_inputs(study, [1], rates="fit")
    P_m = passk_plugin(study.p_fit, 1)[[study.ids.index(q) for q in base["ids"]]]
    try:
        qid = np.asarray([int(q) for q in base["ids"]], dtype=np.int64)
    except ValueError:
        qid = np.arange(len(base["ids"]), dtype=np.int64)
    sim = build(P_m, base["x"], qid, a.shape)
    Ks = [int(k) for k in a.K.split(",")]
    D_grid = list(protected.D_GRID_SIM) if a.D is None else [float(d) for d in a.D.split(",")]
    inp = {"x": sim["x"], "P": sim["P"], "settings": base["settings"], "ids": [f"sim{i}" for i in range(N_Q)],
           "n_inc": N_Q, "n_questions": N_Q, "s": {}, "tau_A": {}, "score_A": {}, "t_a_idx": {}}
    for K in Ks:
        s = contexts.passk_from_P(sim["P"], K)
        S_U = s.sum(axis=0)
        ia = int(np.argmax(S_U))
        inp["s"][K], inp["t_a_idx"][K], inp["tau_A"][K], inp["score_A"][K] = s, ia, float(base["settings"][ia]), float(S_U[ia])
    curves = {(thr, K): protected.sweep_curve(sim["x"], inp["s"][K], inp["settings"], K, thr, D_grid)
              for thr in protected.THRS for K in Ks}
    bstar = protected.beta_array(curves, Ks, protected.THRS, D_grid)
    res = {"study": f"simulated {a.shape} pool from {study.label}", "benchmark": f"simulated-{a.shape}",
           "model": study.model, "n_inc": N_Q, "n_templates": sim["n_templates"], "K": Ks, "thr": list(protected.THRS),
           "D_grid": D_grid, "window": "sym", "settings": inp["settings"].tolist(),
           "H_min": float(sim["H"].min()), "H_median": float(np.median(sim["H"])), "H_max": float(sim["H"].max()),
           "n_above_1e4": int((sim["H"] > 1e4).sum()),
           "per_decade": {f"[{e},{e + 1})": int(((sim["x"] >= e) & (sim["x"] < e + 1)).sum()) for e in range(6)},
           "beta_star_sweep": bstar.tolist(),
           "points": [{k: v for k, v in pt.items() if not k.startswith("_")} for key in curves for pt in curves[key]]}
    if a.certify:
        from . import certify
        cert = certify.certify_study(inp, curves, Ks, list(protected.THRS), D_grid, "sym", reach=False)
        res["certify"], res["beta_star"] = cert["summary"], cert["beta_star_final"].tolist()
    else:
        res["beta_star"] = bstar.tolist()
    res["c_empty"] = protected.c_empty_array(curves, Ks, list(protected.THRS), D_grid).tolist()
    res["environment"] = environment()
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "beta_star.json"
    jp.write_text(json.dumps(res, indent=1, default=float))
    print(protected.curve_table(res))
    if a.plot:
        from .plot_prot import draw_study
        draw_study(res, out / "beta_star_vs_D.jpg", title=f"simulated {a.shape} pool (n = {N_Q})")
    print(f"wrote {jp} ({round(time.perf_counter() - t0, 1)} s)")


if __name__ == "__main__":
    main()
