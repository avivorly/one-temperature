"""Convert a pass-count file in the layout the paper's analysis code reads into counts.jsonl.

This is the converter that produced every file under ``counts/``. Its input is the JSON layout
that the paper's analysis code (the profile fit and the allocation comparison) loads: a list with
one record per question,

    {"task_id": "...", "question_idx": 0, "version": "...", "n": 1500,
     "results": [{"temperature": 0.1, "n": 1500, "passed": 54}, ...]}

and, for the min_p sweeps of Appendix I, the same layout with a ``min_p`` key in every result
next to its fixed ``temperature``. Only ``task_id``/``question_idx``, ``n``, ``results[].temperature``,
``results[].min_p``, ``results[].n`` and ``results[].passed`` are read.

Two parses of the sample count exist (docs/data.md, "Two parses on LiveCodeBench"):

* ``--parse question_n``: every temperature of a question gets the question-level ``n`` as its
  denominator. This is how the paper's original analysis code read the files, and it is what
  Figures 2, 3 and 4(b) and Appendices E, F and G used on LiveCodeBench. Rows whose own count
  ``results[].n`` differs from that ``n`` keep it in the optional field ``n_cell``.
* ``--parse cell_n``: every temperature gets its own ``results[].n``. This is the corrected parse
  of Appendix H and Figure 4(a).

On MATH-500 and Hi-ToM every result has the question's ``n``, so the two parses give the same file.

Usage:

    python -m onetemp.data.convert --input FILE --benchmark livecodebench \\
        --model mistralai/Mistral-7B-Instruct-v0.3 --parse question_n \\
        [--id-from task_id|question_idx] [--id-map MAP.json] [--keep IDS.txt] --out OUT.jsonl

``--id-map`` is a JSON object from the source id (``task_id`` or ``question_idx`` as a string) to the
public question id; ``--keep`` is a text file with one public id per line, and only those questions
are written, in the order of the input file. Rows are written question by question in input order,
settings ascending, one compact JSON object per line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from onetemp.counts import BENCHMARKS, validate_row  # noqa: E402

PARSES = ("question_n", "cell_n")


def _round6(x: float) -> float:
    """Settings are rounded to 6 decimals, as the paper's parser does."""
    return round(float(x), 6)


def records_to_rows(records: list, benchmark: str, model: str, parse: str,
                    id_from: str = "task_id", id_map: dict | None = None,
                    keep: list[str] | None = None) -> list[dict]:
    """Turn passDATA records into counts rows.

    Raises ValueError on anything the analysis would silently mis-read: a duplicate question id,
    a duplicate setting inside a question, a result without n or passed, passed outside 0..n,
    a min_p sweep whose fixed temperature varies, or a --keep id that is not in the input."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}")
    if parse not in PARSES:
        raise ValueError(f"parse must be one of {PARSES}")
    if not isinstance(records, list):
        raise ValueError("the input must be a JSON list of question records")
    keep_set = set(keep) if keep is not None else None
    rows: list[dict] = []
    seen_ids: set[str] = set()
    fixed_temps: set[float] = set()
    for rec in records:
        src = str(rec[id_from])
        qid = id_map[src] if id_map is not None else src
        if qid in seen_ids:
            raise ValueError(f"duplicate question id {qid!r}")
        seen_ids.add(qid)
        if keep_set is not None and qid not in keep_set:
            continue
        q_n = int(rec["n"])
        cells = {}
        for r in rec["results"]:
            n_cell, passed = int(r["n"]), int(r["passed"])
            if "min_p" in r:
                temp, mp = _round6(r["temperature"]), _round6(r["min_p"])
                fixed_temps.add(temp)
                key = mp
            else:
                temp, mp = _round6(r["temperature"]), None
                key = temp
            if key in cells:
                raise ValueError(f"{qid}: setting {key} appears twice")
            if not 0 <= passed <= n_cell:
                raise ValueError(f"{qid}: passed {passed} outside 0..{n_cell}")
            if parse == "question_n":
                if n_cell > q_n:
                    raise ValueError(f"{qid}: a result n ({n_cell}) exceeds the question n ({q_n})")
                row = {"benchmark": benchmark, "model": model, "id": qid, "temperature": temp,
                       "min_p": mp, "n": q_n, "c": passed}
                if n_cell != q_n:
                    row["n_cell"] = n_cell
            else:
                row = {"benchmark": benchmark, "model": model, "id": qid, "temperature": temp,
                       "min_p": mp, "n": n_cell, "c": passed}
            cells[key] = row
        for key in sorted(cells):
            validate_row(cells[key], qid, strict=True)
            rows.append(cells[key])
    if len(fixed_temps) > 1:
        raise ValueError(f"a min_p sweep must have one fixed temperature, found {sorted(fixed_temps)}")
    if keep_set is not None:
        missing = keep_set - seen_ids
        if missing:
            raise ValueError(f"{len(missing)} --keep ids are not in the input, e.g. {sorted(missing)[:3]}")
    return rows


def write_rows(rows: list[dict], out_path) -> None:
    """Write rows as compact JSON lines (key order as in the schema, '\\n' line ends)."""
    with open(out_path, "x", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":"), ensure_ascii=True) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="pass-count JSON (list of question records)")
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--model", required=True, help="Hugging Face repo id of the model")
    ap.add_argument("--parse", required=True, choices=PARSES)
    ap.add_argument("--id-from", default="task_id", choices=("task_id", "question_idx"))
    ap.add_argument("--id-map", default=None, help="JSON object: source id -> public question id")
    ap.add_argument("--keep", default=None, help="text file, one public id per line: write only these")
    ap.add_argument("--out", required=True, help="output counts.jsonl (must not exist)")
    a = ap.parse_args(argv)
    records = json.loads(Path(a.input).read_text(encoding="utf-8"))
    id_map = json.loads(Path(a.id_map).read_text(encoding="utf-8")) if a.id_map else None
    keep = ([ln.strip() for ln in Path(a.keep).read_text(encoding="utf-8").splitlines() if ln.strip()]
            if a.keep else None)
    rows = records_to_rows(records, a.benchmark, a.model, a.parse, a.id_from, id_map, keep)
    write_rows(rows, a.out)
    print(f"wrote {len(rows)} rows to {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
