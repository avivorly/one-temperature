#!/usr/bin/env python3
"""Pool the counts of two sampling stages the way the paper did (Appendix C).

    # sum: Hi-ToM (500 + 1,000), Mistral on Hi-ToM (1,000 + 500), LiveCodeBench Mistral and Qwen3-8B
    # (1,500 + 10,000), min_p at tau = 1.5 (100 + 400)
    python -m onetemp.sampling.pool_counts --policy sum --inputs counts_stage1.jsonl counts_stage2.jsonl --out counts.jsonl

    # replace: LiveCodeBench GPT-OSS 20B, where a question's 10,000 second-stage samples replace its 1,500
    python -m onetemp.sampling.pool_counts --policy replace --inputs counts_stage1.jsonl counts_stage2.jsonl --out counts.jsonl

    # Hi-ToM and min_p: keep only the questions the first-stage screen kept, as the paper's data files do
    python -m onetemp.sampling.pool_counts --policy sum --inputs counts_screen.jsonl counts_topup.jsonl \
        --keep kept.json --out counts.jsonl

Rows are keyed by (benchmark, model, id, temperature, min_p). With `sum`, n and c of equal keys are added (also
for any optional integer field present in every row with that key, such as c_strict). With `replace`, a question
present in a later input takes all of its rows from the latest input that has it. Every sample counts once.
Without --keep, every question of every input is written, including the questions a screen dropped (with their
first-stage samples only). --keep takes the screen's output (or any JSON list or text file of ids) and writes only
those questions; each of them must be in the inputs. The output is sorted by (benchmark, model, id, setting).
"""
import argparse
import json
import sys

from onetemp.counts import read_counts
from onetemp.sampling.public_data import read_ids_file

KEY = ("benchmark", "model", "id", "temperature", "min_p")


def read(path):
    """All rows of a counts.jsonl file (read and validated by onetemp.counts)."""
    return read_counts(path)


def key(r):
    return tuple(r.get(k) for k in KEY)


def pool(inputs, policy):
    if policy == "replace":
        owner = {}
        for i, rows in enumerate(inputs):
            for r in rows:
                owner[(r["benchmark"], r["model"], r["id"])] = i
        inputs = [[r for r in rows if owner[(r["benchmark"], r["model"], r["id"])] == i]
                  for i, rows in enumerate(inputs)]
    out = {}
    for rows in inputs:
        for r in rows:
            k = key(r)
            if k not in out:
                out[k] = dict(r)
                continue
            o = out[k]
            for f in set(o) | set(r):
                if f in KEY:
                    continue
                a, b = o.get(f), r.get(f)
                if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
                    o[f] = a + b
                elif f in o and f not in r or f in r and f not in o:
                    o.pop(f, None)                 # a field only one stage has cannot be pooled
    for o in out.values():
        if o["c"] > o["n"]:
            raise SystemExit(f"pooled c > n for {key(o)}")
    return [out[k] for k in sorted(out, key=lambda k: (k[0], k[1], k[2], k[3], -1.0 if k[4] is None else k[4]))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="counts files, earliest stage first")
    ap.add_argument("--policy", choices=["sum", "replace"], required=True)
    ap.add_argument("--keep", default=None, help="write only these questions: the screen's kept-ids JSON, a JSON "
                                                 "list, or a text file with one id per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = pool([read(p) for p in a.inputs], a.policy)
    if a.keep:
        keep = set(read_ids_file(a.keep))
        missing = keep - {r["id"] for r in rows}
        if missing:
            raise SystemExit(f"{len(missing)} ids of {a.keep} are in none of the inputs, e.g. {sorted(missing)[:3]}")
        rows = [r for r in rows if r["id"] in keep]
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} pooled rows ({a.policy}) -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
