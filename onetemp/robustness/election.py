"""Choosing tau_B under a weight vector W: the reweighted argmax of Appendix G ("Reweighted B").

tau_B(W) = argmax_tau S_W(tau), S_W(tau) = sum_q W_q pass@K_q(tau). Three ways to compute it:

  "argmax"  np.argmax(W @ pass_mat) in batches of 2048 W's, the paper's original election. pass@K is stored as a
            double, so at large K it rounds to exactly 1 for every easy question, and W's that weight those questions
            score several settings identically; np.argmax then takes the lowest setting. Used for the min_p heatmaps of Appendix I (Figures 42-44).
  "exact"   the same argmax, computed where it is accurate: S_t = 1 - F_t with F_t = sum_q W_q (1 - p_qt)^K the
            W-weighted chance that all K samples fail. log F_t = logsumexp_q(log W_q + K log(1 - p_qt)) decides when
            S_t > 1/2 (pass@K near 1 is not rounded to 1), and log S_t = logsumexp_q(log W_q + log pass@K_qt) decides
            when S_t <= 1/2 (a tiny weighted pass chance is not lost in 1 - F). The two join into one increasing key,
            log S_t if S_t <= 1/2 and 2 log(1/2) - log F_t otherwise. Settings within 1e-12 of the best key are tied,
            and the lowest one is taken. Used for the twelve heatmaps of Appendix G (Figures 17-28). Needs log W, so it
            runs on the "replay" or "log" draws.
  "split"   argmin_t log F_t alone; settings whose log F is within 1e-12 of the minimum are tied, and each tied
            setting counts with weight 1 / (number tied), the expected outcome of an even tie-break. Used for the 24
            million W's of Figure 4(b).
"""
from __future__ import annotations

import math

import numpy as np

from .dirichlet import BATCH

TOL = 1e-12
SUB = 500
RULES = ("argmax", "exact", "split")


def elect_argmax(W: np.ndarray, pass_mat: np.ndarray) -> np.ndarray:
    """Index of the best setting of each W, np.argmax of W @ pass_mat per batch of BATCH W's."""
    n = W.shape[0]
    el = np.empty(n, dtype=np.int64)
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        el[s:e] = np.argmax(W[s:e] @ pass_mat, axis=1)
    return el


def lse(logW: np.ndarray, B: np.ndarray) -> np.ndarray:
    """logsumexp_q(logW[w, q] + B[q, t]) for every W and setting, in blocks of SUB W's; -inf where all terms are -inf."""
    n, nt = logW.shape[0], B.shape[1]
    out = np.empty((n, nt))
    for s in range(0, n, SUB):
        e = min(s + SUB, n)
        A = logW[s:e, :, None] + B[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = A.max(axis=1)
            ok = np.isfinite(mx)
            mxs = np.where(ok, mx, 0.0)
            out[s:e] = np.where(ok, mxs + np.log(np.exp(A - mxs[:, None, :]).sum(axis=1)), -np.inf)
    return out


def elect_exact(logW: np.ndarray, log_fail: np.ndarray, log_pass: np.ndarray, tol: float = TOL):
    """Rule "exact": (elected index, number of tied settings) per W."""
    logF = lse(logW, log_fail)
    logS = lse(logW, log_pass)
    half = math.log(0.5)
    key = np.where(logS <= half, logS, 2.0 * half - logF)
    kmax = key.max(axis=1)
    tied = key >= (kmax - tol)[:, None]
    return np.argmax(tied, axis=1), tied.sum(axis=1)


def tied_split(logW: np.ndarray, log_fail: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Rule "split": the (n, T) mask of settings tied at the minimum of log F (within tol)."""
    logF = lse(logW, log_fail)
    fmin = logF.min(axis=1)
    return np.where(np.isfinite(fmin)[:, None], logF <= fmin[:, None] + tol, np.isneginf(logF))


def split_share(tied: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Per W, the expected share of an even tie-break landing on a 'good' setting (good: (T,) bool)."""
    return (tied & good[None, :]).sum(axis=1) / np.maximum(tied.sum(axis=1), 1)


def elect(rule: str, draws: dict, ctx_k: dict):
    """Elected setting index per W under ``rule`` ("argmax" or "exact"), and the number of tied settings."""
    if rule == "argmax":
        el = elect_argmax(draws["W"], ctx_k["pass_mat"])
        return el, np.ones(len(el), dtype=np.int64)
    if rule == "exact":
        if draws.get("logW") is None:
            raise ValueError("the exact election needs log W: draw with mode 'replay' or 'log'")
        return elect_exact(draws["logW"], ctx_k["log_fail"], ctx_k["log_pass"])
    raise ValueError(f"rule must be 'argmax' or 'exact' here ('split' gives shares, see tied_split); got {rule!r}")
