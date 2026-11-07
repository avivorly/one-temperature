"""Aggregate statistics of Appendix H from several beta_star.json files (one per study, from robustness.protected).

  counts    finite points, "> 10^6" points with questions outside, and their kinds (no finite beta / finite beta above
            10^6), over all studies ("Exact certification" and "Points with beta* > 10^6");
  slopes    per curve (study, loss, K) with at least five finite points, the least-squares slope of log10 beta* on D
            over its finite points; the median per dataset and loss, and the range at K = 4 and a 10% loss
            ("beta* against D"); a quadratic term with a nominal p-value below 0.01 that bends the fit by more than
            0.25 decades from its chord, and whether the curve steepens;
  role_of_K for runs with D = 0 over many K: per (study, loss), the Spearman correlation of beta*(D = 0) with K, whether
            it never falls as K grows, and, pooled over studies and K per loss, its Spearman correlation with the share
            of questions easier than K and with log10 K ("The role of K"); "> 10^6" ranks above every finite value.

counts and slopes use only the symmetric-window runs (window "sym"); one-sided runs are skipped and listed. Keep each
kind of run in its own folder so that a glob picks the intended files.

    python -m onetemp.robustness.prot_summary counts out/H/sym/*/beta_star.json
    python -m onetemp.robustness.prot_summary slopes out/H/sym/*/beta_star.json
    python -m onetemp.robustness.prot_summary role_of_K out/H/roleK/*/beta_star.json     (hardness read from each run)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _load(paths):
    out = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        r["_path"] = str(p)
        out.append(r)
    return out


def _sym(results):
    keep = [r for r in results if r.get("window", "sym") == "sym"]
    skipped = [r["_path"] for r in results if r.get("window", "sym") != "sym"]
    return keep, skipped


def counts(results) -> dict:
    results, skipped = _sym(results)
    tot = {"studies": len(results), "skipped_one_sided": skipped, "finite": 0, "beyond_with_outside": 0,
           "no_finite_beta": 0, "finite_beta_above_1e6": 0, "C_empty": 0, "undecided": 0, "inconsistent": 0,
           "attain_at_beta_star": 0, "attain_at_next_step": 0, "attain_none": 0, "lower_unproven": 0,
           "inf_unproven": 0, "not_certified": []}
    for r in results:
        if "certify" not in r:
            tot["not_certified"].append(r["_path"])
            continue
        c = r["certify"]["counts"]
        k = r["certify"]["kinds_of_beyond_cap_points"]
        b = np.asarray(r["beta_star"], dtype=float)
        tot["finite"] += int(np.isfinite(b).sum())               # final beta*, after the exact recheck
        tot["beyond_with_outside"] += c.get("inf_outside", 0)
        tot["no_finite_beta"] += k.get("no finite beta", 0)
        tot["finite_beta_above_1e6"] += k.get("finite beta above 1e6", 0)
        tot["C_empty"] += k.get("C is empty", 0)
        tot["undecided"] += k.get("undecided", 0)
        tot["inconsistent"] += k.get("inconsistent", 0)
        for key in ("attain_at_beta_star", "attain_at_next_step", "attain_none", "lower_unproven", "inf_unproven"):
            tot[key] += c.get(key, 0)
    return tot


def _quad_bend(D, y):
    """(p-value of the quadratic term, max |quadratic fit - its chord| in decades, curvature sign)."""
    X = np.column_stack([np.ones_like(D), D, D * D])
    coef, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    n, k = len(D), 3
    if n <= k or rank < k:
        return None, 0.0, 0.0
    resid = y - X @ coef
    s2 = float(resid @ resid) / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[2, 2]))
    p = float(2 * stats.t.sf(abs(coef[2] / se), n - k)) if se > 0 else 0.0
    grid = np.linspace(D.min(), D.max(), 201)
    fit = coef[0] + coef[1] * grid + coef[2] * grid ** 2
    chord = fit[0] + (fit[-1] - fit[0]) * (grid - grid[0]) / (grid[-1] - grid[0])
    return p, float(np.abs(fit - chord).max()), float(np.sign(coef[2]))


def slopes(results, min_finite: int = 5) -> dict:
    results, skipped = _sym(results)
    rows = []
    for r in results:
        b = np.asarray(r["beta_star"], dtype=float)
        D = np.asarray(r["D_grid"], dtype=float)
        for a, thr in enumerate(r["thr"]):
            for k, K in enumerate(r["K"]):
                fin = np.isfinite(b[a, k])
                if fin.sum() < min_finite:
                    continue
                y = np.log10(b[a, k][fin])
                slope = float(np.polyfit(D[fin], y, 1)[0])
                p, bend, sign = _quad_bend(D[fin], y)
                rows.append({"benchmark": r["benchmark"], "model": r["model"], "thr": thr, "K": K,
                             "n_finite": int(fin.sum()), "slope": slope, "quad_p": p, "bend_decades": bend,
                             "steepens": sign > 0})
    by = {}
    for row in rows:
        by.setdefault(f"{row['benchmark']}|{row['thr']}", []).append(row["slope"])
    k4 = [row["slope"] for row in rows if row["K"] == 4 and row["thr"] == 0.9]
    bent = [row for row in rows if row["quad_p"] is not None and row["quad_p"] < 0.01 and row["bend_decades"] > 0.25]
    return {"n_curves": len(rows), "skipped_one_sided": skipped,
            "median_slope_by_dataset_and_loss": {k: float(np.median(v)) for k, v in by.items()},
            "slope_range_K4_10pct": [float(min(k4)), float(max(k4))] if k4 else None,
            "n_bent": len(bent), "n_bent_steepening": int(sum(row["steepens"] for row in bent)), "curves": rows}


def role_of_K(results, x_of: dict) -> dict:
    """beta*(D = 0) against K; x_of maps (benchmark, model) to the questions' log10 hardness (for the easy share)."""
    per, pooled, undefined = [], {}, 0
    for r in results:
        b = np.asarray(r["beta_star"], dtype=float)
        D = np.asarray(r["D_grid"], dtype=float)
        if not (D == 0.0).any():
            raise SystemExit(f"{r['_path']}: role_of_K needs a run with D = 0 on its grid")
        j0 = int(np.flatnonzero(D == 0.0)[0])
        Ks = np.asarray(r["K"], dtype=float)
        x = np.asarray(r["x"]) if "x" in r else x_of.get((r["benchmark"], r["model"]))
        for a, thr in enumerate(r["thr"]):
            v = b[a, :, j0]
            ok = ~np.isnan(v)
            if ok.sum() < 3:
                continue
            vv = np.where(np.isinf(v[ok]), 1e300, v[ok])
            rho = float(stats.spearmanr(Ks[ok], vv).correlation) if np.unique(vv).size > 1 else float("nan")
            undefined += int(np.isnan(rho))
            per.append({"benchmark": r["benchmark"], "model": r["model"], "thr": thr, "spearman_K": rho,
                        "never_falls": bool(np.all(np.diff(vv) >= 0))})
            if x is not None:
                easy = np.array([(x < np.log10(K)).mean() for K in Ks[ok]])
                pooled.setdefault(thr, {"beta": [], "easy": [], "logK": []})
                pooled[thr]["beta"] += vv.tolist()
                pooled[thr]["easy"] += easy.tolist()
                pooled[thr]["logK"] += np.log10(Ks[ok]).tolist()
    pool = {str(t): {"spearman_easy_share": float(stats.spearmanr(d["easy"], d["beta"]).correlation),
                     "spearman_log10K": float(stats.spearmanr(d["logK"], d["beta"]).correlation),
                     "n": len(d["beta"])} for t, d in pooled.items()}
    rhos = [p["spearman_K"] for p in per]
    return {"n_curves": len(per), "n_undefined_constant_curves": undefined,
            "median_spearman_K": float(np.nanmedian(rhos)) if per and undefined < len(per) else None,
            "n_never_falls": int(sum(p["never_falls"] for p in per)), "pooled": pool, "curves": per}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["counts", "slopes", "role_of_K"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--x-from", default=None,
                    help="role_of_K: a folder of profiles.jsonl files, to read each study's hardness")
    ap.add_argument("--rates", choices=["fit", "measured"], default="fit")
    ap.add_argument("--out", default=None, help="write the full result as JSON (never overwrites)")
    a = ap.parse_args(argv)
    if a.out and Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    res = _load(a.files)
    if a.what == "counts":
        doc = counts(res)
    elif a.what == "slopes":
        doc = slopes(res)
    else:
        from . import contexts
        from .data import load_study
        x_of = {}
        if a.x_from:
            for p in sorted(Path(a.x_from).glob("*.jsonl")):
                st = load_study(p, require_fit=(a.rates == "fit"), quiet=True)
                x_of[(st.benchmark, st.model)] = contexts.prot_inputs(st, [1], rates=a.rates)["x"]
        doc = role_of_K(res, x_of)
    print(json.dumps({k: v for k, v in doc.items() if k != "curves"}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
