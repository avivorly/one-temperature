"""Per-question pass@K, hardness and the uniform reference choice, in the two conventions of the paper.

Both reweighting tests compare settings by the per-question pass@K s_q(tau) and the score S_W(tau) = sum_q W_q s_q(tau)
(Section 4.3.2, Appendix G, Appendix H). They differ in small numerical conventions, which are kept exactly as the
paper's code had them so that every printed number reproduces:

Appendix G (random Dirichlet reweighting, Figures 4(b) and 17-28, and Figures 42-44 for min_p): :func:`kl_context`
  - rates "fit": L = log(clip(1 - p_fit, 1e-15, 1)) and pass@K = 1 - exp(K L) (Eq. 2 on the fitted profile);
    the peak is the fitted ``peak_pct``;
  - rates "measured" (the min_p studies, whose 10- and 5-value grids cannot support the degree-10 fit): pass@K is
    the unbiased estimator 1 - C(n - c, K) / C(n, K), computed exactly in integers and rounded once to a double, and
    the peak is the largest measured rate 100 c / n;
  - inclusion: peak >= 0.1 percent (the 10^-3 filter of Appendix D);
  - hardness H = -1 / ln(1 - peak) (Eq. 3) at the peak, clipped to [0.01, 10^4] for the sampler's hardness coordinate
    u = 2 (log10 H - log10 0.01) / (log10 10^4 - log10 0.01) - 1, and questions are put in hardness order
    (numpy's default argsort, as the paper's code);
  - tau_A = the first setting maximising the uniform score, score_A its score, and ratio(tau) = S_U(tau) / score_A.

Appendix H (protected window, Figure 4(a), Figures 29-33, and Figure 45 for min_p): :func:`prot_inputs`
  - rates "fit": P = 1 - exp(log(clip(1 - p_fit, 1e-15, 1))) (the pass matrix of Appendix G at K = 1), capped at
    1 - 10^-12; s_q = -expm1(K log1p(-P)); inclusion: fitted peak >= 0.1 percent;
  - rates "measured": P = c / n, s_q = the same exact unbiased estimator; inclusion: largest measured rate >= 10^-3;
  - x_q = log10 H_q with H_q = min over the grid of -1 / log1p(-min(P, 1 - 10^-12)), the hardness at the question's
    best setting (Appendix H, "Data").
"""
from __future__ import annotations


import numpy as np

from onetemp.core.passk import passk_plugin, passk_unbiased, unbiased_log_fail

from .data import Study

MIN_PEAK_PERCENT = 0.1        # Appendix D: questions with a peak success probability below 10^-3 are left out
H_LO, H_HI = 0.01, 10000.0    # the sampler's hardness coordinate range (Appendix G)
P_CAP = 1.0 - 1e-12           # Appendix H: fitted P capped below 1 so that every hardness is finite


def unbiased_passk(n: int, c: int, K: int) -> float:
    """The unbiased pass@K of onetemp.core.passk (exact in integers, rounded once), refusing K > n."""
    if K > n:
        raise ValueError(f"pass@{K} needs K <= n (n = {n})")
    return passk_unbiased(n, c, K)


def _unbiased_matrices(n: np.ndarray, c: np.ndarray, K: int):
    pm = np.zeros(c.shape, dtype=np.float64)
    lf = np.zeros(c.shape, dtype=np.float64)
    cache: dict = {}
    for (q, j), cv in np.ndenumerate(c):
        key = (int(n[q, j]), int(cv))
        if key not in cache:
            cache[key] = (unbiased_passk(key[0], key[1], K), unbiased_log_fail(key[0], key[1], K))
        pm[q, j], lf[q, j] = cache[key]
    return pm, lf


def kl_context(study: Study, Ks, rates: str = "fit") -> dict:
    """The per-K inputs of the random reweighting (Appendix G), exactly as the paper's script builds them.

    Returns {"n_inc", "ids" (hardness order), "u_sorted", "H_sorted", "settings", "per_k": {K: {"pass_mat",
    "log_fail", "log_pass", "score_per_t", "ratio_per_t", "t_a_idx", "tau_A", "score_A"}}}.
    """
    if rates == "fit":
        if study.p_fit is None or study.peak_pct is None:
            raise ValueError("rates='fit' needs p_fit and peak_pct in the input (a profiles.jsonl)")
        peak_vals = np.asarray(study.peak_pct, dtype=np.float64)
        L_all = np.log(np.clip(1 - study.p_fit, 1e-15, 1))
    elif rates == "measured":
        peak_vals = (100.0 * study.c / study.n).max(axis=1)
        L_all = None
    else:
        raise ValueError(f"rates must be 'fit' or 'measured', not {rates!r}")
    mask = peak_vals >= MIN_PEAK_PERCENT
    n_inc = int(mask.sum())
    if n_inc < 2:
        raise ValueError(f"only {n_inc} questions pass the 10^-3 peak filter")
    peak_rate = np.clip(peak_vals[mask] / 100.0, 1e-15, 1.0 - 1e-15)
    H = -1.0 / np.log1p(-peak_rate)
    H_clip = np.clip(H, H_LO, H_HI)
    log_H = np.log10(H_clip)
    u = 2.0 * (log_H - np.log10(H_LO)) / (np.log10(H_HI) - np.log10(H_LO)) - 1.0
    order = np.argsort(H_clip)
    ids_inc = [q for q, m in zip(study.ids, mask) if m]
    ctx = {"n_inc": n_inc, "n_questions": len(study.ids), "ids": [ids_inc[i] for i in order],
           "u_sorted": u[order], "H_sorted": H_clip[order], "settings": np.asarray(study.settings, dtype=np.float64),
           "rates": rates, "L_sorted": None if L_all is None else L_all[mask][order], "per_k": {}}
    for K in Ks:
        K = int(K)
        if rates == "fit":
            L = L_all[mask][order]
            pass_mat = passk_plugin(L=L, k=K)
            log_fail = K * L
            with np.errstate(divide="ignore"):
                log_pass = np.log(-np.expm1(log_fail))
        else:
            if K > int(study.n.min()):
                raise ValueError(f"K = {K} exceeds the smallest sample count {int(study.n.min())}")
            pm, lf = _unbiased_matrices(study.n, study.c, K)
            pass_mat = pm[mask][order]
            log_fail = lf[mask][order]
            with np.errstate(divide="ignore"):
                log_pass = np.log(pass_mat)
        score_per_t = pass_mat.sum(axis=0)
        t_a = int(np.argmax(score_per_t))
        score_a = float(score_per_t[t_a])
        ctx["per_k"][K] = {"pass_mat": pass_mat, "log_fail": log_fail, "log_pass": log_pass,
                           "score_per_t": score_per_t, "ratio_per_t": score_per_t / score_a, "t_a_idx": t_a,
                           "tau_A": float(ctx["settings"][t_a]), "score_A": score_a}
    return ctx


def prot_inputs(study: Study, Ks, rates: str = "fit") -> dict:
    """The inputs of the protected-window test (Appendix H): x = log10 H at the best setting, and s[K] = pass@K.

    Returns {"x", "P", "s": {K: (Q, T)}, "settings", "ids", "n_inc", "n_questions", "tau_A": {K}, "score_A": {K}}.
    """
    if rates == "fit":
        if study.p_fit is None or study.peak_pct is None:
            raise ValueError("rates='fit' needs p_fit and peak_pct in the input (a profiles.jsonl)")
        P_all = passk_plugin(study.p_fit, 1)                 # = 1 - exp(log(clip(1 - p_fit, 1e-15, 1)))
        keep = np.asarray(study.peak_pct) >= MIN_PEAK_PERCENT
        P = np.minimum(P_all[keep], P_CAP)
    elif rates == "measured":
        P_all = study.c / study.n
        keep = P_all.max(axis=1) >= MIN_PEAK_PERCENT / 100.0
        P = P_all[keep]
    else:
        raise ValueError(f"rates must be 'fit' or 'measured', not {rates!r}")
    with np.errstate(divide="ignore"):                  # a setting with P = 0 has H = inf; the min ignores it
        H = np.min(-1.0 / np.log1p(-np.minimum(P, P_CAP)), axis=1)
    x = np.log10(H)
    if not np.isfinite(x).all():
        raise ValueError("a kept question has no finite hardness")
    out = {"x": x, "P": P, "s": {}, "settings": np.asarray(study.settings, dtype=np.float64),
           "ids": [q for q, k in zip(study.ids, keep) if k], "n_inc": int(keep.sum()), "n_questions": len(study.ids),
           "rates": rates, "tau_A": {}, "score_A": {}, "t_a_idx": {}}
    for K in Ks:
        K = int(K)
        if rates == "fit":
            s = passk_plugin(P, K, form="expm1")
        else:
            if K > int(study.n.min()):
                raise ValueError(f"K = {K} exceeds the smallest sample count {int(study.n.min())}")
            s, _ = _unbiased_matrices(study.n[keep], study.c[keep], K)
        S_U = s.sum(axis=0)
        ia = int(np.argmax(S_U))
        out["s"][K] = s
        out["tau_A"][K], out["score_A"][K], out["t_a_idx"][K] = float(out["settings"][ia]), float(S_U[ia]), ia
    return out


def passk_from_P(P: np.ndarray, K) -> np.ndarray:
    """Appendix H's pass@K of a capped P matrix: -expm1(K log1p(-P)) (any real K > 0)."""
    return passk_plugin(P, float(K), form="expm1")
