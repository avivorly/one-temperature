"""The worst-case loss R(beta, D) of the uniform choice tau_A (Appendix H, "The loss of the chosen temperature").

    R(beta, D) = max_W ( 1 - S_W(tau_A) / max_tau S_W(tau) )                                   (Eq. 52)

over the W with W_q = 1 inside the window and W_q in [1/beta, beta] outside it (no normalisation). Equivalently
R = 1 - 1/rho*, rho* = max_tau max_W S_W(tau) / S_W(tau_A).

Float layer, the sort form: for a fixed tau the ratio (a + sum_out W u) / (b + sum_out W v), u_q = s_q(tau),
v_q = s_q(tau_A), a and b the protected sums, is maximised by a bang-bang W: beta on the questions with the largest
u_q / v_q and 1/beta on the rest, so rho_tau is the best of the n_out + 1 prefixes of the outside questions sorted by
u_q / v_q (descending; u_q / v_q = inf when v_q = 0 < u_q).
Exact layer, integer arithmetic on the doubles of s (each double times a common power of two is an integer, and
beta = p/q is rational): the float maximiser W* is re-evaluated exactly, rho_hat = S_W*(tau_max) / S_W*(tau_A), and
certified optimal by
    F_tau(rho_hat) = sum_prot e_q + sum_out max(beta e_q, e_q / beta) <= 0 for every tau,  e_q = s_q(tau) - rho_hat s_q(tau_A);
if some F_tau > 0, an exact Dinkelbach step (W = beta exactly where e_q > 0) gives a strictly better W, and the check
repeats. Reported R is the exact rational, so "R <= 1%" and "R = 0" are decided exactly.

Windows: "sym" (|x_q - log10 K| <= D protected), "easy" (only the questions more than d decades easier than K are
reweighted), "hard" (only those more than d decades harder), "all" (no protected window).

    python -m onetemp.robustness.worst_loss --input profiles.jsonl --benchmark math500 --model Qwen/Qwen3-8B \
        --K 1,4,16,32,64,128,256,512,1024 --beta 3/2,2,3,10,30,100,1000 --D 0,0.25,0.5,1,1.5,2,3 --out out/R
"""
from __future__ import annotations

import argparse
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import contexts
from .data import add_input_args, environment, guard_outputs, study_from_args

K_GRID = (1, 4, 16, 32, 64, 128, 256, 512, 1024)
D_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
BETAS = ("3/2", "2", "3", "10", "30", "100", "1000")
MAX_DINKEL = 60


def outside_mask(x, K, window, D):
    c = math.log10(float(K))
    if window == "sym":
        return np.abs(x - c) > D
    if window == "easy":
        return x < c - D
    if window == "hard":
        return x > c + D
    if window == "all":
        return np.ones(len(x), dtype=bool)
    raise ValueError(window)


def rho_float(s, A, out, beta: float):
    """The sort form for every tau: (rho (T,), best prefix length (T,), order (n_out, T))."""
    h, l = beta, 1.0 / beta
    prot = ~out
    v = s[:, A]
    a = s[prot].sum(axis=0)
    b = float(v[prot].sum())
    so, vo = s[out], v[out]
    nO, T = so.shape
    if nO == 0:
        return a / b, np.zeros(T, dtype=np.int64), np.zeros((0, T), dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = so / vo[:, None]
    r = np.where(vo[:, None] > 0, r, np.where(so > 0, np.inf, 0.0))
    order = np.argsort(-r, axis=0, kind="stable")
    Us = np.take_along_axis(so, order, axis=0)
    Vs = vo[order]
    U = np.vstack([np.zeros((1, T)), np.cumsum(Us, axis=0)])
    V = np.vstack([np.zeros((1, T)), np.cumsum(Vs, axis=0)])
    num = a[None, :] + h * U + l * (U[-1][None, :] - U)
    den = b + h * V + l * (V[-1][None, :] - V)
    ratio = num / den
    jstar = np.argmax(ratio, axis=0)
    return ratio[jstar, np.arange(T)], jstar, order


def _dy_exp(v: float) -> int:
    return 0 if v == 0.0 else float(v).as_integer_ratio()[1].bit_length() - 1


class ExactBlock:
    """s scaled by 2^ks to Python integers, for exact sums."""

    def __init__(self, s: np.ndarray):
        self.s = s
        uniq = np.unique(s).tolist()
        self.ks = max((_dy_exp(v) for v in uniq), default=0)
        m = {}
        for v in uniq:
            n_, d_ = float(v).as_integer_ratio()
            m[v] = n_ << (self.ks - (d_.bit_length() - 1))
        self.S = np.array([[m[v] for v in row] for row in s.tolist()], dtype=object)
        self.n, self.T = s.shape

    def weights(self, out, up, beta: Fraction):
        p, q = beta.numerator, beta.denominator
        w = np.full(self.n, p * q, dtype=object)
        w[out & up] = p * p
        w[out & ~up] = q * q
        return w

    def certify(self, A, out, beta: Fraction, tau_c: int, up):
        p, q = beta.numerator, beta.denominator
        pq, p2, q2 = p * q, p * p, q * q
        prot = ~out
        S, SA = self.S, self.S[:, A]
        w = self.weights(out, up, beta)
        SW = w.dot(S)
        N, M = SW[tau_c], SW[A]
        if N < M:
            tau_c, up = A, np.zeros(self.n, dtype=bool)
            SW = self.weights(out, up, beta).dot(S)
            N, M = SW[A], SW[A]
        n_iter = 0
        while True:
            E = M * S - N * SA[:, None]
            pos = E > 0
            F = np.where(prot[:, None], pq * E, np.where(pos, p2 * E, q2 * E)).sum(axis=0)
            bad = [t for t in range(self.T) if F[t] > 0]
            if not bad or n_iter >= MAX_DINKEL:
                return {"N": N, "M": M, "tau_max": int(tau_c), "up": up & out, "n_iter": n_iter, "certified": not bad}
            n_iter += 1
            best = None
            for t in bad:
                up_t = out & pos[:, t].astype(bool)
                SW_t = self.weights(out, up_t, beta).dot(S)
                if best is None or SW_t[t] * best[2][A] > best[2][best[0]] * SW_t[A]:
                    best = (t, up_t, SW_t)
            tau_c, up, SW = best
            N, M = SW[tau_c], SW[A]


def worst_loss(blk: ExactBlock, x, K, A: int, window: str, D, beta: Fraction) -> dict:
    """One point: exact R as a Fraction, its float, the inflicting setting and the up-weighted questions."""
    out = outside_mask(x, K, window, D)
    rho, jstar, order = rho_float(blk.s, A, out, float(beta))
    tf = int(np.argmax(rho))
    oi = np.flatnonzero(out)
    up = np.zeros(blk.n, dtype=bool)
    if len(oi):
        up[oi[order[: int(jstar[tf]), tf]]] = True
    c = blk.certify(A, out, beta, tf, up)
    R = Fraction(c["N"] - c["M"], c["N"]) if c["N"] != 0 else Fraction(0)
    lk = math.log10(float(K))
    return {"window": window, "D": D, "beta": str(beta), "n_out": int(out.sum()), "R": R, "R_float_sort":
            1.0 - 1.0 / float(rho[tf]), "certified": c["certified"], "dinkelbach_steps": c["n_iter"],
            "tau_max_index": c["tau_max"], "n_up": int(c["up"].sum()),
            "n_up_easier": int((c["up"] & (x < lk)).sum()), "n_up_harder": int((c["up"] & (x > lk)).sum())}


def run(inp: dict, Ks, betas, D_grid, windows=("sym",), include_all=True) -> list:
    rows = []
    for K in Ks:
        s = inp["s"][int(K)]
        A = inp["t_a_idx"][int(K)]
        blk = ExactBlock(s)
        for beta in betas:
            for window in windows:
                for D in D_grid:
                    r = worst_loss(blk, inp["x"], K, A, window, D, beta)
                    rows.append({"K": int(K), "tau_A": inp["tau_A"][int(K)], **r})
            if include_all:
                r = worst_loss(blk, inp["x"], K, A, "all", None, beta)
                rows.append({"K": int(K), "tau_A": inp["tau_A"][int(K)], **r})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--K", default=",".join(map(str, K_GRID)))
    ap.add_argument("--beta", default=",".join(BETAS), help="rationals >= 1, e.g. 3/2,10,100")
    ap.add_argument("--D", default=",".join(map(str, D_GRID)))
    ap.add_argument("--window", default="sym", help="comma-separated: sym, easy, hard (the no-window R is added)")
    ap.add_argument("--out", required=True, help="output directory; existing output files are never overwritten")
    a = ap.parse_args(argv)
    out = Path(a.out)
    jp = out / "worst_loss.json"
    guard_outputs([jp])
    Ks = sorted({int(k) for k in a.K.split(",")})
    betas = sorted({Fraction(b) for b in a.beta.split(",")})
    D_grid = sorted({float(d) for d in a.D.split(",")})
    windows = tuple(dict.fromkeys(w.strip() for w in a.window.split(",")))
    if not Ks or Ks[0] < 1 or betas[0] < 1 or D_grid[0] < 0:
        raise SystemExit("--K must be positive integers, --beta at least 1, --D at least 0")
    if any(w not in ("sym", "easy", "hard") for w in windows):
        raise SystemExit("--window takes sym, easy and hard")
    study = study_from_args(a)
    t0 = time.perf_counter()
    inp = contexts.prot_inputs(study, Ks, rates=a.rates)
    rows = run(inp, Ks, betas, D_grid, windows)
    for r in rows:
        r["R_exact"] = [r["R"].numerator, r["R"].denominator]
        r["R"] = float(r["R"])
    doc = {"study": study.label, "n_inc": inp["n_inc"], "K": Ks, "beta": [str(b) for b in betas], "D": D_grid,
           "windows": list(windows), "points": rows, "all_certified": all(r["certified"] for r in rows),
           "seconds": round(time.perf_counter() - t0, 1), "environment": environment()}
    out.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(doc, indent=1))
    for r in rows:
        if r["window"] == "sym" and str(r["beta"]) == "10" and r["D"] == 1.5:
            print(f"K={r['K']}: R(beta=10, D=1.5) = {100 * r['R']:.4f}%")
    print(f"wrote {jp}; all certified: {doc['all_certified']} ({doc['seconds']} s)")


if __name__ == "__main__":
    main()
