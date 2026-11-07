"""The ``counts.jsonl`` format, and the one reader of it in this package.

Every analysis reads, and every grader writes, the per-question counts in this format: one JSON object per
line, one line per (benchmark, model, question, sampling setting):

    {"benchmark": "livecodebench" | "math500" | "hitom",
     "model": "<Hugging Face repo id>",
     "id": "<question id>",
     "temperature": <float>,        # the sampling temperature tau
     "min_p": <float> | null,       # the min_p value, null when min_p was not set
     "n": <int>,                    # number of samples
     "c": <int>}                    # number of correct samples, 0 <= c <= n

Optional fields, each checked when present:

* ``n_cell`` (int): only in the LiveCodeBench files of the ``question_n`` parse under ``data/counts/``, on the
  rows whose own sample count differs from ``n``. There ``n`` is the denominator the paper's analysis code used
  for every temperature of the question (the question's largest per-temperature count), and ``n_cell`` is the
  count actually drawn at that temperature (``c <= n_cell < n``).
* ``tau`` (float): in a min_p row, the fixed temperature; it must equal ``temperature``.
* ``c_strict`` (int): written by the MATH-500 grader, the samples that pass its exact layer alone (``<= c``).
* ``p_fit`` (float in [0, 1]) and ``peak_pct`` (float): written by ``python -m onetemp.core.fit_all --out-rows``,
  the fitted success probability at the row's setting and the question's fitted peak in percent.

The shipped files (``data/counts/``) hold the required keys and ``n_cell`` only; ``strict=True`` (used by
``python -m onetemp.data.check``) enforces that, and that the benchmark is one of the three. Other keys are
ignored otherwise. Nothing else is stored: no prompt, no completion, no reference answer, no test case.

The "setting" of a row is its temperature for a temperature sweep, or its min_p value for a min_p sweep at a
fixed temperature (Appendix I). ``table()`` returns the counts as question-by-setting matrices in the question
order of the file, which is the order the paper's analysis code read the questions in (its 10-fold split and its
seeded draws depend on that order). The analysis packages build their own views from the rows this module
yields (``onetemp.core.counts_io``, ``onetemp.profiles.io``, ``onetemp.robustness.data``).

Usage as a module:

    from onetemp.counts import read_counts, table
    rows = read_counts("data/counts/livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl")
    t = table(rows)            # t.ids, t.settings, t.n[i][j], t.c[i][j], t.axis

Usage from the shell:

    python -m onetemp.counts summary data/counts/livecodebench/*.jsonl
    python -m onetemp.counts figure fig:TempsCom          # which file(s) and population a figure uses
    python -m onetemp.counts table FILE --csv out.csv     # wide table: id, then n and c per setting
    python -m onetemp.counts list-figures
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from onetemp.paths import data_dir

DEFAULT_MANIFEST = data_dir() / "manifest.json"

BENCHMARKS = ("livecodebench", "math500", "hitom")
REQUIRED_KEYS = ("benchmark", "model", "id", "temperature", "min_p", "n", "c")
SHIPPED_OPTIONAL_KEYS = ("n_cell",)
OPTIONAL_KEYS = ("n_cell", "tau", "c_strict", "p_fit", "peak_pct")
ALLOWED_KEYS = frozenset(REQUIRED_KEYS + SHIPPED_OPTIONAL_KEYS)     # the keys of a shipped file


# --------------------------------------------------------------------------------------------
# Reading and validating rows
# --------------------------------------------------------------------------------------------

def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def validate_row(row: dict, where: str = "", strict: bool = False) -> None:
    """Raise ValueError unless ``row`` follows the counts.jsonl schema.

    Always: the required keys present; benchmark and model non-empty strings; id a non-empty string (or an
    integer); temperature a finite number >= 0; min_p null or a finite number in [0, 1]; n an int >= 0 and c an
    int with 0 <= c <= n; and each optional field, when present, as described in the module docstring.
    ``strict=True`` (the shipped files): only the required keys and ``n_cell``, benchmark one of the three, id a
    string, and n >= 1."""
    if not isinstance(row, dict):
        raise ValueError(f"{where}: a row must be a JSON object")
    if strict:
        extra = set(row) - ALLOWED_KEYS
        if extra:
            raise ValueError(f"{where}: unknown keys {sorted(extra)}")
    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")
    if not isinstance(row["benchmark"], str) or not row["benchmark"]:
        raise ValueError(f"{where}: benchmark must be a non-empty string")
    if strict and row["benchmark"] not in BENCHMARKS:
        raise ValueError(f"{where}: benchmark {row['benchmark']!r} not in {BENCHMARKS}")
    if not isinstance(row["model"], str) or not row["model"]:
        raise ValueError(f"{where}: model must be a non-empty string")
    qid = row["id"]
    if not ((isinstance(qid, str) and qid) or (_is_int(qid) and not strict)):
        raise ValueError(f"{where}: id must be a non-empty string")
    if not _is_num(row["temperature"]) or row["temperature"] < 0:
        raise ValueError(f"{where}: temperature must be a finite number >= 0")
    mp = row["min_p"]
    if mp is not None and (not _is_num(mp) or not 0.0 <= mp <= 1.0):
        raise ValueError(f"{where}: min_p must be null or a number in [0, 1]")
    n, c = row["n"], row["c"]
    if not _is_int(n) or n < (1 if strict else 0):
        raise ValueError(f"{where}: n must be an int >= {1 if strict else 0}")
    if _is_int(c) and c > n:
        raise ValueError(f"{where}: c > n (c={c}, n={n})")
    if not _is_int(c) or c < 0:
        raise ValueError(f"{where}: c must be an int with 0 <= c <= n")
    if "n_cell" in row:
        nc = row["n_cell"]
        if not _is_int(nc) or not c <= nc < n:
            raise ValueError(f"{where}: n_cell must be an int with c <= n_cell < n")
    if row.get("tau") is not None:
        if mp is None or not _is_num(row["tau"]) or round(float(row["tau"]), 6) != round(float(row["temperature"]), 6):
            raise ValueError(f"{where}: tau={row['tau']!r} must be a min_p row's temperature ({row['temperature']})")
    if "c_strict" in row and (not _is_int(row["c_strict"]) or not 0 <= row["c_strict"] <= c):
        raise ValueError(f"{where}: c_strict must be an int with 0 <= c_strict <= c")
    if "p_fit" in row and (not _is_num(row["p_fit"]) or not 0.0 <= row["p_fit"] <= 1.0):
        raise ValueError(f"{where}: p_fit must be a number in [0, 1]")
    if "peak_pct" in row and not _is_num(row["peak_pct"]):
        raise ValueError(f"{where}: peak_pct must be a finite number")


def iter_counts(path, validate: bool = True, strict: bool = False, with_lineno: bool = False) -> Iterator:
    """Yield the rows of one counts.jsonl (or .jsonl.gz) file, validating each by default.

    This is the one place in the package that parses a counts file. ``with_lineno=True`` yields
    (line number, row) pairs instead of rows."""
    path = Path(path)
    with _open_text(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({e})") from None
            if validate:
                validate_row(row, f"{path}:{lineno}", strict=strict)
            yield (lineno, row) if with_lineno else row


def read_counts(path, validate: bool = True, strict: bool = False) -> list[dict]:
    """Return all rows of one counts file as a list of dicts."""
    return list(iter_counts(path, validate=validate, strict=strict))


def setting_of(row: dict) -> float:
    """The swept setting of a row: its min_p value if min_p is set, else its temperature."""
    return row["temperature"] if row["min_p"] is None else row["min_p"]


def axis_of(rows: Iterable[dict]) -> str:
    """'min_p' if every row has a min_p value, 'temperature' if none has; mixed files are refused."""
    kinds = {r["min_p"] is None for r in rows}
    if kinds == {True}:
        return "temperature"
    if kinds == {False}:
        return "min_p"
    raise ValueError("file mixes rows with and without min_p")


# --------------------------------------------------------------------------------------------
# Question-by-setting tables
# --------------------------------------------------------------------------------------------

@dataclass
class CountsTable:
    """Counts of one file as matrices: n[i][j] samples and c[i][j] correct for question ids[i]
    at setting settings[j]. ``fixed`` records the setting that is constant in the file (the
    temperature of a min_p sweep), ``n_cell`` the optional per-cell counts (None where absent)."""
    benchmark: str
    model: str
    axis: str
    ids: list[str]
    settings: list[float]
    n: list[list[int]]
    c: list[list[int]]
    fixed: dict = field(default_factory=dict)
    n_cell: list[list[int | None]] | None = None

    def rate(self, i: int, j: int) -> float:
        """Measured pass rate c/n of question i at setting j."""
        return self.c[i][j] / self.n[i][j]

    def cell_n(self, i: int, j: int) -> int:
        """The sample count actually drawn in cell (i, j): n_cell where present, else n."""
        if self.n_cell is not None and self.n_cell[i][j] is not None:
            return self.n_cell[i][j]
        return self.n[i][j]


def table(rows: list[dict], require_complete: bool = True) -> CountsTable:
    """Arrange rows into a CountsTable. Questions keep their order of first appearance in the
    file; settings are sorted ascending. With ``require_complete`` (default) every question must
    have every setting exactly once, as in every released file."""
    if not rows:
        raise ValueError("no rows")
    bench = {r["benchmark"] for r in rows}
    model = {r["model"] for r in rows}
    if len(bench) != 1 or len(model) != 1:
        raise ValueError("a table holds one benchmark and one model")
    axis = axis_of(rows)
    fixed: dict = {}
    if axis == "min_p":
        temps = {r["temperature"] for r in rows}
        if len(temps) != 1:
            raise ValueError("a min_p table must have one fixed temperature")
        fixed["temperature"] = temps.pop()
    ids: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            ids.append(r["id"])
    settings = sorted({setting_of(r) for r in rows})
    qi = {q: i for i, q in enumerate(ids)}
    sj = {s: j for j, s in enumerate(settings)}
    n = [[0] * len(settings) for _ in ids]
    c = [[0] * len(settings) for _ in ids]
    ncell = None
    filled = [[False] * len(settings) for _ in ids]
    for r in rows:
        i, j = qi[r["id"]], sj[setting_of(r)]
        if filled[i][j]:
            raise ValueError(f"duplicate cell: id {r['id']!r}, setting {setting_of(r)}")
        filled[i][j] = True
        n[i][j], c[i][j] = r["n"], r["c"]
        if "n_cell" in r:
            if ncell is None:
                ncell = [[None] * len(settings) for _ in ids]
            ncell[i][j] = r["n_cell"]
    if require_complete:
        holes = sum(1 for row in filled for f in row if not f)
        if holes:
            raise ValueError(f"{holes} (question, setting) cells are missing")
    return CountsTable(bench.pop(), model.pop(), axis, ids, settings, n, c, fixed, ncell)


# --------------------------------------------------------------------------------------------
# Manifest helpers
# --------------------------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path=DEFAULT_MANIFEST) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def file_entry(manifest: dict, rel_path: str) -> dict:
    for f in manifest["files"]:
        if f["path"] == rel_path:
            return f
    raise KeyError(rel_path)


def figure_entries(manifest: dict, label: str) -> list[dict]:
    """The uses (file, population size, note) recorded for one figure or table label."""
    out = [u for u in manifest["uses"] if u["label"] == label]
    if not out:
        raise KeyError(f"no figure or table labelled {label!r} in the manifest")
    return out


def read_figure(label: str, manifest_path=DEFAULT_MANIFEST) -> dict[str, list[dict]]:
    """Return {file path: rows} for every counts file a figure or table uses."""
    man = load_manifest(manifest_path)
    base = Path(manifest_path).resolve().parent
    return {u["file"]: read_counts(base / u["file"]) for u in figure_entries(man, label)}


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------

def _summary(paths: list[str]) -> int:
    for p in paths:
        rows = read_counts(p)
        t = table(rows)
        n_vals = sorted({v for row in t.n for v in row})
        tot_n = sum(t.cell_n(i, j) for i in range(len(t.ids)) for j in range(len(t.settings)))  # drawn
        tot_c = sum(map(sum, t.c))
        fixed = f" fixed={t.fixed}" if t.fixed else ""
        print(f"{p}\n  {t.benchmark} | {t.model} | axis={t.axis}{fixed}\n"
              f"  rows={len(rows)} questions={len(t.ids)} settings={len(t.settings)} "
              f"[{t.settings[0]:g} .. {t.settings[-1]:g}]\n"
              f"  n per cell: min={n_vals[0]} max={n_vals[-1]} distinct={len(n_vals)}; "
              f"samples={tot_n:,} correct={tot_c:,}"
              + ("; rows with n_cell=" + str(sum(1 for r in rows if 'n_cell' in r))
                 if t.n_cell is not None else ""))
    return 0


def _figure(label: str, manifest: str) -> int:
    man = load_manifest(manifest)
    print(f"{label}: {man['labels'].get(label, '')}")
    for u in figure_entries(man, label):
        f = file_entry(man, u["file"])
        inc = f.get("included", {}).get("n")
        pop = "all questions" if u.get("population") == "all" else "the included questions"
        printed = "" if u.get("n_printed") is None else f"; the paper prints n = {u['n_printed']}"
        print(f"  {u['file']}\n    {f['questions']} questions, {inc} included, {f['settings_count']} settings; "
              f"uses {pop}{printed}"
              + (f"\n    note: {u['note']}" if u.get("note") else "")
              + (f"\n    known difference: {u['expected_difference']}" if u.get("expected_difference") else ""))
    return 0


def _list_figures(manifest: str) -> int:
    man = load_manifest(manifest)
    seen: list[str] = []
    for u in man["uses"]:
        if u["label"] not in seen:
            seen.append(u["label"])
    for lab in seen:
        print(f"{lab}  {man['labels'].get(lab, '')}")
        for u in man["uses"]:
            if u["label"] == lab:
                n = "" if u.get("n_printed") is None else f"  (n = {u['n_printed']})"
                print(f"    {u['file']}{n}")
    return 0


def _table_csv(path: str, out: str | None) -> int:
    t = table(read_counts(path))
    fh = open(out, "w", newline="", encoding="utf-8") if out else sys.stdout
    try:
        w = csv.writer(fh)
        w.writerow(["id"] + [f"n@{s:g}" for s in t.settings] + [f"c@{s:g}" for s in t.settings])
        for i, q in enumerate(t.ids):
            w.writerow([q] + t.n[i] + t.c[i])
    finally:
        if out:
            fh.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="validate files and print their shape and totals")
    s.add_argument("files", nargs="+")
    f = sub.add_parser("figure", help="print the counts file(s) a figure or table label uses")
    f.add_argument("label")
    f.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    lf = sub.add_parser("list-figures", help="list every figure and table label in the manifest")
    lf.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    t = sub.add_parser("table", help="write one file as a wide CSV table (id, n per setting, c per setting)")
    t.add_argument("file")
    t.add_argument("--csv", default=None, help="output path (default: stdout)")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        return _summary(a.files)
    if a.cmd == "figure":
        return _figure(a.label, a.manifest)
    if a.cmd == "list-figures":
        return _list_figures(a.manifest)
    return _table_csv(a.file, a.csv)


if __name__ == "__main__":
    raise SystemExit(main())
