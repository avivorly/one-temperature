"""Strategy comparison of Section 4.2 and Appendix F: best single setting, uniform baseline, ratio
to uniform, and the held-out (out-of-sample) evaluation of the optimal allocation.

At each budget K, on the included questions (rows of L = log(1 - P)):

* single setting j: sum_i 1 - exp(K L[i, j]);
* best single setting (in-sample): the setting with the largest such total on all questions;
* uniform scheme: K / l samples at each of the l settings, sum_i 1 - exp(L_i . (K/l) 1);
* optimal allocation (in-sample): the integer optimum of Eq. 9 on all questions;
* optimal allocation (held-out): the questions are split into folds; for each fold the allocation
  is re-optimised on the other folds (relaxed by SLSQP, then the integer step) and scored on the
  fold's questions; the scores are summed over all folds. The paper's fold rule: leave-one-out
  is one question per fold in question order; 10-fold permutes the questions with
  ``numpy.random.default_rng(0).permutation(n)`` and cuts the permutation into 10 parts with
  ``numpy.array_split`` (9 folds of 36 and one of 35 for n = 359);
* best single setting (held-out): the best setting is re-selected on the other folds and scored on
  the fold, the same footing as the held-out optimum.

The ratio to uniform of a strategy is its total divided by the uniform total at the same K; the
held-out optimum is divided by the in-sample uniform total, as in the paper's figures. The
paired delta-method standard error of each ratio N/D is sd(num_i - (N/D) den_i) sqrt(n) / D.

This module is a port of the paper's ``_compute_case4_loo``: the same SLSQP settings, warm starts,
folds and output fields, with the integer step selectable ("paper" reproduces the paper's numbers,
"exact" uses the certified optimum, seeded with the paper's result).

Command line::

    python -m onetemp.core.heldout --profiles profiles.json --folds 10 --out heldout_kf10.json
    python -m onetemp.core.heldout --profiles profiles.json --folds loo --out heldout_loo.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Callable, Optional

import numpy as np

from onetemp.core.allocation import INTEGER_METHODS, integer_exact, integer_paper, solve_relaxed
from onetemp.core.passk import k_grid, passk_plugin


def make_folds(n: int, k_folds: Optional[int] = None, seed: int = 0):
    """The paper's folds. k_folds None or >= n: leave-one-out (one question per fold, in order).
    Otherwise default_rng(seed).permutation(n) cut by array_split into k_folds parts (seed 0 in
    the paper). Returns (list of index arrays, label)."""
    if k_folds is None or k_folds >= n:
        return [np.array([i], dtype=np.int64) for i in range(n)], "loo"
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    perm = np.random.default_rng(seed).permutation(n)
    return [np.asarray(f, dtype=np.int64) for f in np.array_split(perm, k_folds)], f"k={k_folds}"


def folds_sha256(folds) -> str:
    """Fingerprint of a fold list (sha256 of its JSON), to check that two runs used the same split."""
    return hashlib.sha256(json.dumps([list(map(int, f)) for f in folds]).encode()).hexdigest()


def paired_stderr_ratio(num_per_q: np.ndarray, den_per_q: np.ndarray) -> float:
    """Delta-method standard error of sum(num) / sum(den) with paired per-question terms."""
    n = len(num_per_q)
    N = float(num_per_q.sum())
    D = float(den_per_q.sum())
    if D <= 0 or n < 2:
        return 0.0
    r = N / D
    c = num_per_q - r * den_per_q
    return float(np.std(c, ddof=1) * np.sqrt(n) / D)


def _integer_step(method: str, q_relaxed: np.ndarray, k: int, L: np.ndarray, tol: float):
    q_p, s_p = integer_paper(q_relaxed, k, L, L.shape[0], radius=2)
    if method == "paper":
        if int(np.sum(q_p)) != int(k) or np.any(q_p < 0):
            raise RuntimeError(f"the paper's integer step returned {q_p.tolist()}, not an allocation of K={k}")
        return q_p, s_p, None
    q_e, s_e, cert = integer_exact(L, k, x_relaxed=q_relaxed, incumbent=q_p, tol=tol)
    info = {"proven": cert["proven"], "nodes": cert["nodes"], "improved": cert["improved_on_incumbent"]}
    return q_e, s_e, info


def compare(L: np.ndarray, k_values=None, k_folds: Optional[int] = None, integer: str = "paper",
            ref_index: Optional[int] = None, fold_seed: int = 0, tol: float = 1e-9,
            keep_per_question: bool = False, progress: Optional[Callable] = None) -> dict:
    """The held-out comparison at every K (the paper's ``_compute_case4_loo``).

    L: (n_included, n_settings) log(1 - P). ref_index: column of the reference setting for
    ``case1`` (the paper: the temperature closest to 1.0); None skips it.
    Returns a dict with one entry per K holding, in expected solved questions (divide by n for
    pass@K): in_sample_solved (optimal, in-sample), mean_solved (optimal, held-out), case2_total
    (best single, in-sample), case2_oos_total (best single, held-out), case3_total (uniform),
    case1_total (reference setting), the delta-method SEs, and the in-sample integer allocation.
    """
    if integer not in ("paper", "exact"):
        raise ValueError("integer must be 'paper' or 'exact'")
    L = np.asarray(L, dtype=np.float64)
    n_inc, n_temps = L.shape
    if n_inc < 2:
        raise ValueError("need at least 2 questions")
    k_values = list(k_values) if k_values is not None else k_grid(512)
    folds, cv_mode = make_folds(n_inc, k_folds, fold_seed)
    all_idx = np.arange(n_inc)
    per_k = []
    checks = {"exact_proven_all": True, "exact_improved_count": 0, "exact_nodes_max": 0}
    t0 = time.perf_counter()
    for ki, k in enumerate(k_values):
        k_f = float(k)
        q_full = solve_relaxed(L, k_f, np.full(n_temps, k_f / n_temps))
        q_int_full, in_sample_solved, info_full = _integer_step(integer, q_full, int(k_f), L, tol)
        in_sample_solved = float(in_sample_solved)

        pred1 = passk_plugin(L=L[:, ref_index], k=k_f) if ref_index is not None else None
        q_uni = np.full(n_temps, k_f / n_temps)
        pred3 = passk_plugin(L=L, k=q_uni)
        col_scores = np.sum(passk_plugin(L=L, k=k_f), axis=0)
        t2_idx = int(np.argmax(col_scores))
        pred2 = passk_plugin(L=L[:, t2_idx], k=k_f)

        pred4 = np.empty(n_inc)
        pred2_oos = np.empty(n_inc)
        fold_allocs = []
        fold_infos = []
        for fold in folds:
            train = np.setdiff1d(all_idx, fold, assume_unique=True)
            q_i = solve_relaxed(L[train], k_f, q_full)
            q_i_int, _, info = _integer_step(integer, q_i, int(k_f), L[train], tol)
            pred4[fold] = passk_plugin(L=L[fold], k=q_i_int.astype(np.float64))
            train_col_scores = np.sum(passk_plugin(L=L[train], k=k_f), axis=0)
            t2_idx_train = int(np.argmax(train_col_scores))
            pred2_oos[fold] = passk_plugin(L=L[fold, t2_idx_train], k=k_f)
            fold_allocs.append(q_i_int.astype(int).tolist())
            if info is not None:
                fold_infos.append(info)
        for info in ([info_full] if info_full else []) + fold_infos:
            checks["exact_proven_all"] &= bool(info["proven"])
            checks["exact_improved_count"] += int(info["improved"])
            checks["exact_nodes_max"] = max(checks["exact_nodes_max"], int(info["nodes"]))

        total4 = float(pred4.sum())
        entry = {
            "k": int(k),
            "mean_rate": total4 / n_inc,
            "mean_solved": total4,
            "in_sample_solved": in_sample_solved,
            "gap": in_sample_solved - total4,
            "stderr_solved_abs": float(pred4.std(ddof=1) * np.sqrt(n_inc)),
            "stderr_vs_case1": paired_stderr_ratio(pred4, pred1) if pred1 is not None else None,
            "stderr_vs_case2": paired_stderr_ratio(pred4, pred2),
            "stderr_vs_case3": paired_stderr_ratio(pred4, pred3),
            "case1_total": float(pred1.sum()) if pred1 is not None else None,
            "case2_total": float(pred2.sum()),
            "case2_oos_total": float(pred2_oos.sum()),
            "case3_total": float(pred3.sum()),
            "case2_best_T_idx": t2_idx,
            "q_int_full": q_int_full.astype(int).tolist(),
            # derived ratios (the plotted quantities)
            "ratio_optimal_heldout": total4 / float(pred3.sum()),
            "ratio_optimal_insample": in_sample_solved / float(pred3.sum()),
            "ratio_best_single_insample": float(pred2.sum()) / float(pred3.sum()),
            "ratio_best_single_heldout": float(pred2_oos.sum()) / float(pred3.sum()),
        }
        if keep_per_question:
            entry["per_question"] = {"optimal_heldout": pred4.tolist(), "best_single_heldout": pred2_oos.tolist(),
                                     "uniform": pred3.tolist(), "best_single_insample": pred2.tolist()}
            entry["fold_allocations"] = fold_allocs
        per_k.append(entry)
        if progress:
            progress(ki + 1, len(k_values), int(k), time.perf_counter() - t0)
    return {"n_included": n_inc, "n_temps": n_temps, "k_values": [int(k) for k in k_values],
            "max_k": int(max(k_values)), "cv_mode": cv_mode, "n_folds": len(folds),
            "fold_rule": ("one question per fold, question order" if cv_mode == "loo" else
                          f"numpy.random.default_rng({fold_seed}).permutation({n_inc}) split by "
                          f"numpy.array_split into {k_folds}"),
            "folds_sha256": folds_sha256(folds), "integer": integer, "tol": tol if integer == "exact" else None,
            "exact_checks": checks if integer == "exact" else None,
            "seconds": round(time.perf_counter() - t0, 2), "per_k": per_k}


def reference_index(settings, axis: str = "temperature") -> Optional[int]:
    """The paper's reference setting for case1: the temperature closest to 1.0 (min_p: 0)."""
    s = np.asarray(settings, dtype=np.float64)
    target = 1.0 if axis == "temperature" else 0.0
    return int(np.argmin(np.abs(s - target))) if len(s) else None


def main(argv=None) -> int:
    from onetemp.core.fit_all import MIN_PEAK_PCT, load_profiles
    ap = argparse.ArgumentParser(description="Held-out comparison of the optimal allocation (Appendix F).")
    ap.add_argument("--profiles", required=True, help="profiles.json written by onetemp.core.fit_all")
    ap.add_argument("--mode", default="fit", choices=("fit", "meas"))
    ap.add_argument("--min-peak", type=float, default=MIN_PEAK_PCT)
    ap.add_argument("--folds", default="loo", help="'loo' or a number of folds, e.g. 10")
    ap.add_argument("--fold-seed", type=int, default=0, help="seed of the k-fold permutation (paper: 0)")
    ap.add_argument("--max-k", type=int, default=1024)
    ap.add_argument("--k", type=int, nargs="+", default=None, help="explicit K values")
    ap.add_argument("--integer", default="paper", choices=("paper", "exact"))
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--per-question", action="store_true", help="also store per-question scores and fold allocations")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.integer not in INTEGER_METHODS:
        ap.error("bad --integer")
    prof = load_profiles(a.profiles)
    inc = prof.included(a.min_peak, a.mode)
    L = prof.L(a.mode)[inc]
    k_folds = None if a.folds == "loo" else int(a.folds)
    ks = a.k or k_grid(a.max_k)

    def _p(i, nk, k, dt):
        print(f"  [{i}/{nk}] K={k} {dt:.1f}s", flush=True)

    res = compare(L, ks, k_folds, a.integer, reference_index(prof.alloc_settings, prof.axis), a.fold_seed,
                  a.tol, a.per_question, _p)
    res["settings"] = prof.alloc_settings
    res["mode"] = a.mode
    res["min_peak"] = a.min_peak
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f)
    for e in res["per_k"]:
        print(f"K={e['k']:5d}  optimal held-out / uniform {e['ratio_optimal_heldout']:.4f}  "
              f"in-sample {e['ratio_optimal_insample']:.4f}  best single {e['ratio_best_single_insample']:.4f}")
    print(f"[heldout] {res['cv_mode']} n={res['n_included']} integer={a.integer} {res['seconds']}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
