"""Tilting the question mix by hardness (Appendix H, "Tilting the mix").

Every weight is tilted by W_q(gamma) = 10^(gamma (log10 H_q - log10 K)): a factor 10^gamma per decade of hardness,
favouring harder questions for gamma > 0. The best setting under the tilt is tau*(gamma) = argmax_tau
sum_q W_q s_q(tau), and the loss of tau_A is L(gamma) = 1 - S_gamma(tau_A) / max_tau S_gamma(tau).

Reported per (study, K), K by default every power of 2 from 1 to 1024:
  - tau*(gamma) on G11 = {-1, -0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5, 1} and on a 0.01 grid from -1 to 1;
    whether it is nondecreasing in gamma (a tilt toward harder questions never moves the best setting to a lower
    one, i.e. colder on the temperature axis);
  - the direction at gamma = +-0.25: does +0.25 move it to a lower setting (colder), does -0.25 move it to a
    higher one (hotter); on a min_p axis "lower" is a smaller min_p;
  - the tolerance of tau_A: the smallest |gamma| on a 0.01 scan up to 4 at which L reaches 1% (and 10%), on each
    side; null when it never does (the text's "> 4"). 10^gamma is "the factor per decade" of the text.
Decisions with a float relative gap below 1e-6 between the best two settings are re-decided exactly (rational
arithmetic on the doubles of W and s), and L within 1e-9 of a threshold is re-decided exactly.

    python -m onetemp.robustness.tilt --input profiles.jsonl --benchmark hitom --model Qwen/Qwen3-8B --out out/tilt
"""
from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import contexts
from .data import add_input_args, environment, guard_outputs, study_from_args

G11 = (-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
GFINE = tuple(round(-1.0 + 0.01 * i, 2) for i in range(201))
GSCAN = tuple(round(0.01 * i, 2) for i in range(1, 401))
K_POW2 = tuple(2 ** i for i in range(11))


def tilt_weights(x, K, gamma):
    return 10.0 ** (gamma * (x - math.log10(float(K))))


def _exact_scores(W, s):
    WF = [Fraction(float(w)) for w in W]
    cols = []
    for t in range(s.shape[1]):
        cols.append(sum(WF[q] * Fraction(float(s[q, t])) for q in range(s.shape[0])))
    return cols


def best_setting(W, s):
    """(argmax index set, exact?) of S_W over the settings; near-ties re-decided exactly."""
    S = W @ s
    order = np.argsort(-S, kind="stable")
    top, second = S[order[0]], S[order[1]] if len(order) > 1 else -np.inf
    if top > 0 and (top - second) / top >= 1e-6:
        return [int(order[0])], False
    ex = _exact_scores(W, s)
    mx = max(ex)
    return [t for t, v in enumerate(ex) if v == mx], True


def loss_of(W, s, A):
    S = W @ s
    return 1.0 - S[A] / S.max()


def loss_reaches(W, s, A, thr) -> bool:
    L = loss_of(W, s, A)
    if abs(L - thr) >= 1e-9:
        return L >= thr
    ex = _exact_scores(W, s)
    return 1 - ex[A] / max(ex) >= Fraction(str(thr))


def curve(x, s, A, K):
    """Everything reported for one (study, K)."""
    g11 = {g: best_setting(tilt_weights(x, K, g), s) for g in G11}
    gfine = {g: best_setting(tilt_weights(x, K, g), s) for g in GFINE}

    def monotone(d):
        gs = sorted(d)
        lo = [min(d[g][0]) for g in gs]
        hi = [max(d[g][0]) for g in gs]
        return all(lo[i] <= lo[i + 1] for i in range(len(gs) - 1)) and all(hi[i] <= hi[i + 1] for i in range(len(gs) - 1))

    tol = {}
    for thr in (0.01, 0.10):
        for side, sign in (("harder", 1.0), ("easier", -1.0)):
            hit = None
            for g in GSCAN:
                if loss_reaches(tilt_weights(x, K, sign * g), s, A, thr):
                    hit = g
                    break
            tol[f"{side}|{thr:g}"] = hit
    return {"K": int(K), "tau_A_index": int(A),
            "tau_star_G11": {f"{g:g}": v[0] for g, v in g11.items()},
            "monotone_G11": monotone(g11), "monotone_Gfine": monotone(gfine),
            "plus_0.25_moves_lower": min(g11[0.25][0]) < A, "minus_0.25_moves_higher": max(g11[-0.25][0]) > A,
            "exact_redecisions": int(sum(v[1] for v in g11.values()) + sum(v[1] for v in gfine.values())),
            "gamma_threshold": tol,
            "factor_per_decade": {k: (None if v is None else 10.0 ** v) for k, v in tol.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--K", default=",".join(map(str, K_POW2)))
    ap.add_argument("--out", required=True, help="output directory; existing output files are never overwritten")
    a = ap.parse_args(argv)
    out = Path(a.out)
    jp = out / "tilt.json"
    guard_outputs([jp])
    Ks = sorted({int(k) for k in a.K.split(",")})
    if not Ks or Ks[0] < 1:
        raise SystemExit("--K must hold positive integers")
    study = study_from_args(a)
    inp = contexts.prot_inputs(study, Ks, rates=a.rates)
    rows = [curve(inp["x"], inp["s"][K], inp["t_a_idx"][K], K) for K in Ks]
    out.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({"study": study.label, "axis": study.axis, "settings": inp["settings"].tolist(),
                              "curves": rows, "environment": environment()}, indent=1))
    for r in rows:
        print(f"K={r['K']}: monotone G11 {r['monotone_G11']} fine {r['monotone_Gfine']}; +0.25 lower "
              f"{r['plus_0.25_moves_lower']}; -0.25 higher {r['minus_0.25_moves_higher']}; 1% at "
              f"harder {r['gamma_threshold']['harder|0.01']} easier {r['gamma_threshold']['easier|0.01']}")
    print(f"wrote {jp}")


if __name__ == "__main__":
    main()
