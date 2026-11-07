"""Natural sub-datasets (Appendix H, "Natural sub-datasets").

The labels of each dataset define sub-datasets: the subject and level of MATH-500, the order, deception, story length
and prompting type of Hi-ToM, and the difficulty, platform and version of LiveCodeBench, together with their pairwise
intersections; those with at least 20 questions are kept. On a sub-dataset S, the setting chosen on the full dataset,
t_full = argmax_t S_pool(t), is compared with the sub-dataset's own best setting t_S = argmax_t S_S(t), by its loss

    L_S = 1 - S_S(t_full) / S_S(t_S).

Labels file: JSON lines {"benchmark": ..., "id": "<question id>", "labels": {"<factor>": <level>, ...}}, with the
same ids as the counts/profiles file; a missing or null level leaves the question out of that factor's subsets. The
labels come from the public datasets' own metadata (for example MATH-500's ``subject`` and ``level``); they carry no
answers.

    python -m onetemp.robustness.subsets --input profiles.jsonl --labels labels.jsonl --benchmark math500 \
        --model Qwen/Qwen3-8B --out out/subsets
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import contexts
from .data import add_input_args, environment, guard_outputs, study_from_args

K_GRID = (1, 4, 16, 64, 256, 1024)
N_MIN = 20


def read_labels(path, benchmark=None) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if benchmark is None or r.get("benchmark") in (None, benchmark):
                    out[str(r["id"])] = r["labels"]
    return out


def subsets_of(ids, labels: dict, n_min: int = N_MIN):
    """Every single-factor level and every 2-factor cell with at least n_min questions: [(name, mask)]."""
    factors = sorted({f for q in ids for f in labels.get(q, {})})
    vals = {f: [labels.get(q, {}).get(f) for q in ids] for f in factors}
    out = []
    for f in factors:
        for lv in sorted({v for v in vals[f] if v is not None}, key=str):
            m = np.array([v == lv for v in vals[f]])
            if m.sum() >= n_min:
                out.append((f"{f}={lv}", m))
    for f, g in itertools.combinations(factors, 2):
        for lf in sorted({v for v in vals[f] if v is not None}, key=str):
            for lg in sorted({v for v in vals[g] if v is not None}, key=str):
                m = np.array([a == lf and b == lg for a, b in zip(vals[f], vals[g])])
                if m.sum() >= n_min:
                    out.append((f"{f}={lf} & {g}={lg}", m))
    return out


def natural_losses(s: np.ndarray, subsets) -> list:
    t_full = int(np.argmax(s.sum(axis=0)))
    rows = []
    for name, m in subsets:
        S = s[m].sum(axis=0)
        t_S = int(np.argmax(S))
        rows.append({"subset": name, "n": int(m.sum()), "t_full_index": t_full, "t_S_index": t_S,
                     "loss": float(1.0 - S[t_full] / S[t_S]) if S[t_S] > 0 else 0.0})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--labels", required=True, help="labels.jsonl: one line per question with its labels")
    ap.add_argument("--K", default=",".join(map(str, K_GRID)))
    ap.add_argument("--n-min", type=int, default=N_MIN)
    ap.add_argument("--out", required=True, help="output directory; existing output files are never overwritten")
    a = ap.parse_args(argv)
    out = Path(a.out)
    jp = out / "subsets.json"
    guard_outputs([jp])
    Ks = sorted({int(k) for k in a.K.split(",")})
    study = study_from_args(a)
    inp = contexts.prot_inputs(study, Ks, rates=a.rates)
    labels = read_labels(a.labels, study.benchmark)
    missing = [q for q in inp["ids"] if q not in labels]
    subs = subsets_of(inp["ids"], labels, a.n_min)
    rows = []
    for K in Ks:
        for r in natural_losses(inp["s"][K], subs):
            rows.append({"K": K, **r})
    losses = np.array([r["loss"] for r in rows]) if rows else np.zeros(0)
    summ = {"n_subsets": len(subs), "n_cases": len(rows), "n_questions_without_labels": len(missing),
            "median_loss": float(np.median(losses)) if len(rows) else None,
            "n_loss_ge_10pct": int((losses >= 0.10).sum()), "n_loss_ge_1pct": int((losses >= 0.01).sum()),
            "share_ge_1pct_by_K": {str(K): float(np.mean([r["loss"] >= 0.01 for r in rows if r["K"] == K]))
                                   for K in Ks if any(r["K"] == K for r in rows)}}
    out.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({"study": study.label, "summary": summ, "rows": rows, "environment": environment()},
                             indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
