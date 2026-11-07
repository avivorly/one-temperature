"""Per-question pass rates and pass@K per temperature.

Input: a counts file (JSONL), one line per (question, temperature), as written by the graders:
    {"id": "<question id>", "temperature": 0.7, "n": 1500, "c": 312}
where n = completions drawn and c = completions graded correct.

Two pass@K estimators are provided:
  * pass_at_k_unbiased(n, c, K) = 1 - C(n-c, K) / C(n, K)   (Chen et al., 2021). Needs K <= n.
  * pass_at_k_from_rate(P, K)    = 1 - (1 - P)^K           (the paper's formula, Sec. 2.2 and App.
    "Temperature Sampling Schemes Comparisons"), with P the single-attempt success probability of
    the question at that temperature. The paper uses its fitted temperature profile for P; this
    package uses the measured rate P = c / n (see README, "What is not included").

Usage:
    python passk.py counts.jsonl                  # pass@K per temperature, K grid of the paper
    python passk.py counts.jsonl --k 1,16,256 --estimator unbiased
"""
import argparse
import json
from collections import defaultdict

import numpy as np

# The paper's K grid (ratio-to-uniform figures and tables use K <= 1024).
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """1 - C(n-c, k) / C(n, k), in product form to avoid overflow (same code as our analysis)."""
    if k > n:
        raise ValueError(f"unbiased pass@{k} needs at least {k} samples, got n={n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def pass_at_k_from_rate(p, k):
    """Probability that at least one of k independent attempts succeeds: 1 - (1 - p)^k."""
    return 1.0 - (1.0 - np.asarray(p, dtype=float)) ** k


def load_counts(path):
    """Read a counts file into matrices.

    Returns (ids, temps, N, C), with N[i, j] and C[i, j] the sample and correct counts of question
    ids[i] at temperature temps[j]. A (question, temperature) cell that is missing is an error:
    every question must be sampled at every temperature of the grid.
    """
    cells = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = round(float(r["temperature"]), 6)
            if t in cells[r["id"]]:
                raise ValueError(f"duplicate cell: id={r['id']} T={t}")
            n, c = int(r["n"]), int(r["c"])
            if not (n > 0 and 0 <= c <= n):
                raise ValueError(f"bad counts: {r}")
            cells[r["id"]][t] = (n, c)
    ids = sorted(cells)
    temps = sorted({t for d in cells.values() for t in d})
    N = np.zeros((len(ids), len(temps)), dtype=np.int64)
    C = np.zeros_like(N)
    for i, q in enumerate(ids):
        for j, t in enumerate(temps):
            if t not in cells[q]:
                raise ValueError(f"missing cell: id={q} T={t}")
            N[i, j], C[i, j] = cells[q][t]
    return ids, np.array(temps), N, C


def pass_rates(N, C):
    """Measured per-question single-attempt pass rates P[i, j] = c / n."""
    return C / N


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("counts", help="counts JSONL: one {id, temperature, n, c} line per cell")
    ap.add_argument("--k", default=",".join(map(str, K_GRID)), help="comma-separated K values")
    ap.add_argument("--estimator", choices=["rate", "unbiased"], default="rate",
                    help="rate: 1-(1-c/n)^K (any K); unbiased: 1-C(n-c,K)/C(n,K) (K <= n)")
    args = ap.parse_args()

    ids, temps, N, C = load_counts(args.counts)
    ks = [int(k) for k in args.k.split(",")]
    P = pass_rates(N, C)
    print(f"{len(ids)} questions x {len(temps)} temperatures; mean pass@K over questions")
    print("K".rjust(6) + "".join(f"T={t:g}".rjust(10) for t in temps))
    for k in ks:
        if args.estimator == "rate":
            row = pass_at_k_from_rate(P, k).mean(axis=0)
        else:
            row = [np.mean([pass_at_k_unbiased(int(N[i, j]), int(C[i, j]), k) for i in range(len(ids))])
                   for j in range(len(temps))]
        print(f"{k:6d}" + "".join(f"{v:10.4f}" for v in row))


if __name__ == "__main__":
    main()
