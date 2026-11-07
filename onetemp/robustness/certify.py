"""Exact certification of beta* (Appendix H, "Exact certification"), in rational arithmetic on the stored s_q(tau).

Every double is a rational number, so every comparison below is decided exactly with fractions.Fraction.

1. Correction (the paper's exact recheck of the LP sweep). For a finite beta*: the sweep's witness (its W at
   beta* x 1.005, clipped onto the box) must give S_W(tau_c) - S_W(tau') >= 1e-9 exactly for every tau' not in C;
   if not, a certificate is searched at beta* (LP proposals for every tau_c in C), and if none exists beta* moves up
   to the smallest certified grid index. For a "> 10^6" point with questions outside and C non-empty, a certificate
   is searched at 10^6; if one exists, "> 10^6" was wrong and the smallest certified index is found by bisection.
   The corrected array is beta_star_final.
2. Certificates of every point with a question outside the window, both sides.
   - Attainable: an explicit W (1 inside the window, every other weight in [1/beta*, beta*] as a double) under which
     tau_c wins by >= 1e-9 exactly ("attain_at_beta_star"). When only the sweep's own witness certifies, in the box
     of the next grid step beta* x 1.005, beta* is certified to within one grid step ("attain_at_next_step", the
     paper's fallback; it did not occur on the paper's data); "attain_none" otherwise.
   - Not attainable one grid step below (and, for "> 10^6", at 10^6), for EACH tau_c in C: weights lambda >= 0 on
     the settings tau' not in C, summing to 1, with g_q = sum_tau' lambda_tau' (s_q(tau_c) - s_q(tau')) and
         UB = sum_{q protected} g_q + sum_{q outside} max(beta g_q, g_q / beta) < 1e-9          (Eq. 51)
     By weak duality no W allowed at beta lets tau_c win. lambda proposals: the HiGHS dual values of the sweep's
     LP (dual simplex, then IPM) and the minimiser of UB from its own LP; entries <= 1e-13 dropped.
3. Reachability of the "> 10^6" points (Appendix H, "Points with beta* > 10^6"): a point is
   - "no finite beta" (true-inf) when every tau_c in C has a lambda with g_q <= 0 for every outside question and
     sum_{protected} g_q < 1e-9: then the bound above holds at every beta;
   - "finite beta above 10^6" (beyond-cap) when some tau_c has an outside W > 0 (any size) under which it wins by
     >= 1e-9 exactly;
   - "C is empty", or "no question outside" (the stop), by definition; "undecided" otherwise (reported), and
     "inconsistent" if a W proof needed beta <= 10^6 (a contradiction; never trusted).
   A curve with a true-inf point at D = 0 has no reachable loss at any D.
"""
from __future__ import annotations

import math
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

from .protected import BETA_GRID, I_TOP, TIGHT, candidates, window_masks

EPS = Fraction(1, 10 ** 9)
DROP = 1e-13
W_BETAS = (1e7, 1e8, 1e9, 1e10, 1e11, 1e12)
BB_MAX_EXP = 400


def frac_matrix(s_k: np.ndarray):
    cache: dict = {}
    out = []
    for row in s_k.tolist():
        r = []
        for v in row:
            f = cache.get(v)
            if f is None:
                f = cache[v] = Fraction(v)
            r.append(f)
        out.append(r)
    return out


def box_lo(b: float) -> float:
    """The first double >= 1/b exactly."""
    lo = 1.0 / b
    while Fraction(lo) < 1 / Fraction(b):
        lo = math.nextafter(lo, math.inf)
    return lo


def clip_out(v, b):
    return np.minimum(np.maximum(np.asarray(v, dtype=np.float64), box_lo(b)), b)


def exact_margin(sF, c, non, W) -> Fraction:
    """min over tau' not in C of S_W(tau_c) - S_W(tau'), exactly, for a double vector W over every question."""
    WF = [Fraction(float(v)) for v in W]
    n = len(sF)
    return min(sum(WF[q] * (sF[q][c] - sF[q][tp]) for q in range(n) if sF[q][c] != sF[q][tp]) for tp in non)


def exact_margin_box(sF, c, non, out, w_out, beta) -> Fraction:
    """As exact_margin, the outside weights clipped onto the exact rational box [1/beta, beta]."""
    bF = Fraction(float(beta))
    loF, hiF = 1 / bF, bF
    W = [Fraction(1)] * len(sF)
    for q, v in zip(np.flatnonzero(out), w_out):
        W[q] = min(max(Fraction(float(v)), loF), hiF)
    return min(sum(W[q] * (sF[q][c] - sF[q][tp]) for q in range(len(sF))) for tp in non)


def lp_ws(s_k, c, non, prot, out, b):
    """Outside-W proposals at b: the sweep's LP (dual simplex) and a [0, 1] reparameterisation (IPM)."""
    ws = []
    Dm = (s_k[:, non] - s_k[:, [c]]).T
    const = Dm[:, prot].sum(axis=1)
    A = np.hstack([Dm[:, out], np.ones((len(non), 1))])
    cvec = np.zeros(int(out.sum()) + 1)
    cvec[-1] = -1.0
    res = linprog(cvec, A_ub=A, b_ub=-const, bounds=[(1.0 / b, b)] * int(out.sum()) + [(None, None)],
                  method="highs-ds", options=dict(TIGHT, time_limit=60.0))
    if res.status == 0:
        ws.append(res.x[:-1])
    lo, hi = 1.0 / b, b
    idx = np.flatnonzero(out)
    rows, rhs = [], []
    for tp in non:
        d = s_k[:, c] - s_k[:, tp]
        rows.append(np.concatenate([-(hi - lo) * d[idx], [1.0]]))
        rhs.append(lo * d[idx].sum() + d[prot].sum())
    res = linprog(cvec, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(0.0, 1.0)] * len(idx) + [(None, None)],
                  method="highs-ipm", options=dict(TIGHT, time_limit=60.0))
    if res.status == 0:
        ws.append(lo + (hi - lo) * res.x[:-1])
    return ws


def certified_box(s_k, sF, cand, non, prot, out, beta) -> bool:
    """Some tau_c in C has an exact certificate at beta (W clipped onto the exact rational box)."""
    for c in cand:
        for wv in lp_ws(s_k, c, non, prot, out, beta):
            W = np.ones(len(sF))
            W[out] = np.clip(wv, 1.0 / beta, beta)
            if float((W @ (s_k[:, [c]] - s_k[:, non])).min()) < -1e-3:
                continue
            if exact_margin_box(sF, c, non, out, wv, beta) >= EPS:
                return True
    return False


def smallest_certified(s_k, sF, cand, non, prot, out, lo, hi):
    """The smallest grid index in (lo, hi] that certifies, given that hi does (bisection)."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if certified_box(s_k, sF, cand, non, prot, out, float(BETA_GRID[mid])):
            hi = mid
        else:
            lo = mid
    return hi


def sweep_lp_duals(s_k, c, non, prot, out, b, meth):
    Dm = (s_k[:, non] - s_k[:, [c]]).T
    const = Dm[:, prot].sum(axis=1)
    A = np.hstack([Dm[:, out], np.ones((len(non), 1))])
    cvec = np.zeros(int(out.sum()) + 1)
    cvec[-1] = -1.0
    res = linprog(cvec, A_ub=A, b_ub=-const, bounds=[(1.0 / b, b)] * int(out.sum()) + [(None, None)],
                  method=meth, options=dict(TIGHT, time_limit=60.0))
    if res.status != 0:
        return None
    return -np.asarray(res.ineqlin.marginals, dtype=np.float64)


def ub_lp_lambda(s_k, c, non, prot, out, b):
    """The lambda minimising UB(lambda) at b, from its own LP."""
    L, idx = len(non), np.flatnonzero(out)
    dmat = s_k[:, [c]] - s_k[:, non]
    cvec = np.concatenate([dmat[prot].sum(axis=0), np.ones(len(idx))])
    rows = []
    for r, q in enumerate(idx):
        for f in (b, 1.0 / b):
            row = np.zeros(L + len(idx))
            row[:L] = f * dmat[q]
            row[L + r] = -1.0
            rows.append(row)
    A_eq = np.concatenate([np.ones(L), np.zeros(len(idx))])[None, :]
    res = linprog(cvec, A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0.0, None)] * L + [(None, None)] * len(idx), method="highs-ds",
                  options=dict(TIGHT, time_limit=60.0))
    if res.status != 0:
        return None
    return np.asarray(res.x[:L], dtype=np.float64)


def exact_ub(sF, c, non, prot, lam_raw, b):
    """UB(lambda) of Eq. (51), exactly; lambda = raw / sum(raw) in rationals."""
    if not (lam_raw >= 0).all() or lam_raw.sum() <= 0:
        return None
    lF = [Fraction(float(v)) for v in lam_raw]
    tot = sum(lF)
    nz = [(int(non[r]), lF[r] / tot) for r in range(len(non)) if lF[r] != 0]
    bF = Fraction(float(b))
    ub = Fraction(0)
    for q in range(len(sF)):
        row = sF[q]
        g = row[c] - sum(l * row[tp] for tp, l in nz)
        ub += g if prot[q] else (bF * g if g > 0 else g / bF)
    return ub


def dual_for(s_k, sF, c, non, prot, out, b):
    """(lambda_raw, exact UB, proposal name) of the best proposal; UB < 1e-9 proves tau_c cannot win at b."""
    best = (None, None, None)
    for name in ("ds", "ipm", "ub-lp"):
        lam = (sweep_lp_duals(s_k, c, non, prot, out, b, "highs-" + name) if name != "ub-lp"
               else ub_lp_lambda(s_k, c, non, prot, out, b))
        if lam is None:
            continue
        lam = np.where(lam > DROP, lam, 0.0)
        ub = exact_ub(sF, c, non, prot, lam, b)
        if ub is not None and (best[1] is None or ub < best[1]):
            best = (lam, ub, name)
        if ub is not None and ub < EPS:
            break
    return best


# ------------------------------------------------------------------------------------------- reachability (walls)
def lambda_proof_every_beta(s_k, sF, c, non, prot, out) -> bool:
    """A lambda with g_q <= 0 on every outside question and sum_protected g_q < 1e-9, exactly (no finite beta)."""
    L = len(non)
    dmat = s_k[:, [c]] - s_k[:, non]                     # (n, L): s_q(c) - s_q(t')
    props = [np.eye(L)[r] for r in range(L)]
    if out.any():
        scale = np.maximum(np.abs(dmat[out]).max(axis=1, keepdims=True), 1e-300)
        for tight in (0.0, 1e-12, 1e-9, 1e-6):
            res = linprog(dmat[prot].sum(axis=0), A_ub=dmat[out] / scale, b_ub=np.full(int(out.sum()), -tight),
                          A_eq=np.ones((1, L)), b_eq=[1.0], bounds=[(0.0, None)] * L, method="highs-ds",
                          options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
                                   "time_limit": 60.0})
            if res.status == 0:
                props.append(np.where(res.x > DROP, res.x, 0.0))
    for lam in props:
        if not (lam >= 0).all() or lam.sum() <= 0:
            continue
        lF = [Fraction(float(v)) for v in lam]
        tot = sum(lF)
        nz = [(int(non[r]), lF[r] / tot) for r in range(L) if lF[r] != 0]
        ok, s_prot = True, Fraction(0)
        for q in range(len(sF)):
            row = sF[q]
            g = row[c] - sum(l * row[tp] for tp, l in nz)
            if out[q]:
                if g > 0:
                    ok = False
                    break
            else:
                s_prot += g
        if ok and s_prot < EPS:
            return True
    return False


def w_proof_some_beta(s_k, sF, c, non, prot, out):
    """An outside W > 0 of any size under which tau_c wins by >= 1e-9 exactly; returns the beta it needs, or None."""
    d = s_k[:, [c]] - s_k[:, non]                         # (n, L)
    idx = np.flatnonzero(out)
    supports = []
    fav_all = out & (d >= 0).all(axis=1) & (d > 0).any(axis=1)
    fav_sum = out & (d.sum(axis=1) > 0)
    supports += [fav_all, fav_sum] + [out & (d[:, r] > 0) for r in range(d.shape[1])]
    for S in supports:
        if not S.any():
            continue
        M = [sum(sF[q][c] - sF[q][non[r]] for q in np.flatnonzero(prot)) for r in range(len(non))]
        Pp = [sum(sF[q][c] - sF[q][non[r]] for q in np.flatnonzero(S)) for r in range(len(non))]
        Nn = [sum(sF[q][c] - sF[q][non[r]] for q in idx if not S[q]) for r in range(len(non))]
        if any(p < 0 for p in Pp):
            continue
        for k in range(0, BB_MAX_EXP + 1):
            big = Fraction(10) ** k
            if all(M[r] + big * Pp[r] + Nn[r] / big >= EPS for r in range(len(non))):
                return float(10.0 ** k)
    for b in W_BETAS:
        for wv in lp_ws(s_k, c, non, prot, out, b):
            W = np.ones(len(sF))
            W[out] = np.clip(wv, 1.0 / b, b)
            if exact_margin(sF, c, non, W) >= EPS:
                return float(max(W.max(), 1.0 / W.min()))
    return None


def classify_beyond_cap(s_k, sF, cand, non, prot, out) -> str:
    """The kind of a "> 1e6" point (see the module docstring). A W proof that needs beta <= 1e6 would contradict the
    exact proof that 1e6 is not enough, and is reported as "inconsistent" instead of being trusted."""
    if not out.any():
        return "no question outside"
    if not cand:
        return "C is empty"
    if all(lambda_proof_every_beta(s_k, sF, c, non, prot, out) for c in cand):
        return "no finite beta"
    for c in cand:
        b = w_proof_some_beta(s_k, sF, c, non, prot, out)
        if b is not None:
            return "finite beta above 1e6" if b > 1e6 else "inconsistent"
    return "undecided"


# ------------------------------------------------------------------------------------------- one study
def certify_study(inp, curves, Ks, thrs, D_grid, window="sym", xs=None, reach=True):
    """Corrections, both-sided certificates and reachability for every point of one study's sweep."""
    t0 = time.perf_counter()
    tau = inp["settings"]
    nA, nK, nD = len(thrs), len(Ks), len(D_grid)
    bfinal = np.full((nA, nK, nD), np.nan)
    cnt = {"finite_sweep": 0, "finite": 0, "witness_exact": 0, "certified_at_beta_star_box": 0, "moved_up": 0,
           "uncertified": 0, "inf_checked": 0, "inf_wrong": 0, "attain_at_beta_star": 0, "attain_at_next_step": 0,
           "attain_none": 0, "lower_proven": 0, "lower_unproven": 0, "lower_trivial_beta1": 0, "inf_outside": 0,
           "inf_proven": 0, "inf_unproven": 0, "inf_no_outside": 0, "inf_C_empty": 0}
    kinds = {"no finite beta": 0, "finite beta above 1e6": 0, "C is empty": 0, "no question outside": 0,
             "undecided": 0, "inconsistent": 0}
    reach_curves, notes = {}, []
    for k, K in enumerate(Ks):
        s_k = inp["s"][int(K)]
        sF = frac_matrix(s_k)
        x = inp["x"] if xs is None else xs[int(K)]
        n = len(x)
        for a, thr in enumerate(thrs):
            cand, non, order, S_U = candidates(s_k, thr)
            pts = curves[(float(thr), int(K))]
            for pt in pts:
                j, D = pt["D_index"], pt["D"]
                prot, out = window_masks(x, K, D, window)
                b = pt["beta_star"]
                # ---- 1. correction
                if b is not None:
                    cnt["finite_sweep"] += 1
                    i = int(pt["beta_star_index"])
                    c = int(pt["tau_c_index"])
                    W = pt.get("_W")
                    if W is not None and exact_margin_box(sF, c, non, out, W[out], b * 1.005) >= EPS:
                        cnt["witness_exact"] += 1
                        bf = b
                    elif certified_box(s_k, sF, cand, non, prot, out, float(BETA_GRID[i])):
                        cnt["certified_at_beta_star_box"] += 1
                        bf = b
                    elif certified_box(s_k, sF, cand, non, prot, out, float(BETA_GRID[I_TOP])):
                        bf = float(BETA_GRID[smallest_certified(s_k, sF, cand, non, prot, out, i, I_TOP)])
                        cnt["moved_up"] += 1
                    else:
                        bf = np.inf
                        cnt["uncertified"] += 1
                else:
                    bf = np.inf
                    if out.any() and cand:
                        cnt["inf_checked"] += 1
                        if certified_box(s_k, sF, cand, non, prot, out, float(BETA_GRID[I_TOP])):
                            hi = smallest_certified(s_k, sF, cand, non, prot, out, -1, I_TOP)
                            bf = float(BETA_GRID[hi])
                            cnt["inf_wrong"] += 1
                            notes.append({"thr": thr, "K": K, "D": D, "certified_beta": bf})
                bfinal[a, k, j] = bf
                pt["beta_star_final"] = None if np.isinf(bf) else bf
                # ---- 2. both sides
                dual_beta = None
                if np.isfinite(bf):
                    cnt["finite"] += 1                      # final beta*, after the correction
                    i = int(np.flatnonzero(BETA_GRID == bf)[0])
                    found = None
                    for c in ([int(pt["tau_c_index"])] if pt.get("tau_c_index") is not None else []) + order:
                        for wv in lp_ws(s_k, c, non, prot, out, bf):
                            Wv = np.ones(n)
                            Wv[out] = clip_out(wv, bf)
                            if float((Wv @ (s_k[:, [c]] - s_k[:, non])).min()) < -1e-3:
                                continue
                            m = exact_margin(sF, c, non, Wv)
                            if m >= EPS:
                                found = (c, float(m))
                                break
                        if found:
                            break
                    where = "beta*" if found else None
                    if found is None and pt.get("_W") is not None and pt.get("beta_star") == bf:
                        # the paper's fallback: the sweep's witness in the box of the next grid step, beta* x 1.005;
                        # it proves attainability there, so beta* is certified to within one grid step only
                        Wv = np.ones(n)
                        Wv[out] = clip_out(pt["_W"][out], bf * 1.005)
                        m = exact_margin(sF, int(pt["tau_c_index"]), non, Wv)
                        if m >= EPS:
                            found, where = (int(pt["tau_c_index"]), float(m)), "beta* x 1.005"
                    cnt[{"beta*": "attain_at_beta_star", "beta* x 1.005": "attain_at_next_step",
                         None: "attain_none"}[where]] += 1
                    pt["attain_exact"] = None if found is None else {"tau_c": float(tau[found[0]]), "margin": found[1],
                                                                     "box": where}
                    if i == 0:
                        cnt["lower_trivial_beta1"] += 1
                    else:
                        dual_beta = float(BETA_GRID[i - 1])
                else:
                    if not out.any():
                        cnt["inf_no_outside"] += 1
                    elif not cand:
                        cnt["inf_C_empty"] += 1
                    else:
                        cnt["inf_outside"] += 1
                        dual_beta = float(BETA_GRID[I_TOP])
                if dual_beta is not None:
                    ok_all, worst = True, None
                    for c in cand:
                        lam, ub, name = dual_for(s_k, sF, c, non, prot, out, dual_beta)
                        if ub is None or ub >= EPS:
                            ok_all = False
                        if ub is not None:
                            worst = float(ub) if worst is None else max(worst, float(ub))
                    side = "lower" if np.isfinite(bf) else "inf"
                    cnt[f"{side}_{'proven' if ok_all else 'unproven'}"] += 1
                    pt["dual_exact"] = {"beta": dual_beta, "proven": ok_all, "max_exact_ub": worst}
                # ---- 3. reachability of "> 1e6"
                if reach and not np.isfinite(bf):
                    kind = classify_beyond_cap(s_k, sF, cand, non, prot, out)
                    kinds[kind] += 1
                    pt["kind"] = kind
                    if j == 0:
                        reach_curves[f"{thr}|{K}"] = "unreachable" if kind == "no finite beta" else (
                            "C is empty" if kind == "C is empty" else "reachable (beyond 1e6 at D = 0)")
                elif j == 0:
                    reach_curves[f"{thr}|{K}"] = "reachable"
    summary = {"counts": cnt, "kinds_of_beyond_cap_points": kinds, "inf_corrected_to_finite": notes,
               "seconds": round(time.perf_counter() - t0, 1)}
    return {"beta_star_final": bfinal, "summary": summary, "reachability": reach_curves}
