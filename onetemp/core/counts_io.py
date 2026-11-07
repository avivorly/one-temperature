"""Read the common ``counts.jsonl`` format into per-question count matrices.

Format (one JSON object per line, one line per benchmark, model, question and setting)::

    {"benchmark": "livecodebench" | "math500" | "hitom",
     "model": "<Hugging Face repo id>",
     "id": "<question id>",
     "temperature": float,
     "min_p": float or null,
     "n": int,          # number of samples drawn
     "c": int}          # number of correct samples

Optional fields are ignored here, except ``"tau"``: when a min_p row carries it, it must equal
``temperature`` (it names the fixed temperature of a min_p sweep, Appendix I).

The setting axis is generic. ``axis="temperature"`` selects the temperature sweeps (rows whose
``min_p`` is null, unless ``fixed_min_p`` is given). ``axis="min_p"`` selects min_p sweeps at one
fixed temperature (rows whose ``min_p`` is not null; ``fixed_temperature`` picks the temperature
when the file holds several).

Question order is the order in which ids first appear in the file. It matters: the paper's 10-fold
split permutes the included questions in this order (``heldout.make_folds``), so a file exported in
the paper's question order reproduces the paper's folds.

Sample counts per question: the analyses take an ``n_mode``.

* ``"question-max"`` (the default): every setting of a question is divided by one count, the
  largest n of that question over its settings; a setting with no row counts as a measured zero
  with that n. This is the parsing of the LiveCodeBench counts that Figures 1-3, Figure 4(b),
  Appendices E-G and Tables 2-6 use (the "earlier parsing" of Appendix H).
* ``"per-setting"``: each setting uses its own n; a setting with no row is not measured and is left
  out of the fit. This is the "corrected parsing" of Appendix H (and Figure 4(a)). On MATH-500 and
  Hi-ToM every question has one n at every setting, so the two modes give the same numbers there.

Command line::

    python -m onetemp.core.counts_io --counts counts.jsonl --list
    python -m onetemp.core.counts_io --counts counts.jsonl --benchmark livecodebench \
        --model mistralai/Mistral-7B-Instruct-v0.3 --summary
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from onetemp.counts import iter_counts

AXES = ("temperature", "min_p")
N_MODES = ("question-max", "per-setting")
REQUIRED_FIELDS = ("benchmark", "model", "id", "temperature", "min_p", "n", "c")


def _round_setting(v: float) -> float:
    """Settings are compared after rounding to 6 decimals, as the paper's loader does."""
    return round(float(v), 6)


def iter_rows(path: str) -> Iterable[dict]:
    """Yield the rows of a counts.jsonl (or .jsonl.gz) file, validated by the package's one reader
    (``onetemp.counts.iter_counts``); each row also gets ``_line``, its line number, for messages."""
    for lineno, row in iter_counts(path, with_lineno=True):
        row["_line"] = lineno
        yield row


@dataclass
class Counts:
    """Per-question count matrices for one (benchmark, model, axis, fixed value).

    ``n[i, j]`` and ``c[i, j]`` are the samples and correct samples of question ``ids[i]`` at
    setting ``settings[j]``; a setting the question has no row for has ``n = c = 0`` and
    ``measured[i, j] = False``.
    """

    benchmark: Optional[str]
    model: Optional[str]
    axis: str
    fixed: dict
    settings: list
    ids: list
    n: np.ndarray
    c: np.ndarray
    measured: np.ndarray
    source: str = ""
    notes: list = field(default_factory=list)

    @property
    def n_questions(self) -> int:
        return len(self.ids)

    def n_for_fit(self, n_mode: str = "question-max") -> np.ndarray:
        """The denominators the fit uses, per question and setting (see the module docstring)."""
        if n_mode not in N_MODES:
            raise ValueError(f"n_mode must be one of {N_MODES}, got {n_mode!r}")
        if n_mode == "per-setting":
            return self.n.copy()
        n_q = self.n.max(axis=1, keepdims=True)
        return np.broadcast_to(n_q, self.n.shape).astype(np.int64).copy()

    def summary(self) -> dict:
        n_unequal = int(np.sum([len(set(int(v) for v, m in zip(self.n[i], self.measured[i]) if m)) > 1
                                for i in range(self.n_questions)]))
        return {
            "benchmark": self.benchmark, "model": self.model, "axis": self.axis, "fixed": self.fixed,
            "questions": self.n_questions, "settings": self.settings,
            "rows": int(self.measured.sum()),
            "missing_cells": int((~self.measured).sum()),
            "total_samples": int(self.n.sum()), "total_correct": int(self.c.sum()),
            "questions_with_unequal_n": n_unequal,
            "questions_with_no_correct_sample": int(np.sum(self.c.sum(axis=1) == 0)),
        }


def list_groups(path: str) -> list:
    """All (benchmark, model, axis, fixed value) groups present in a counts.jsonl file."""
    groups: dict = {}
    for row in iter_rows(path):
        if row["min_p"] is None:
            key = (row["benchmark"], row["model"], "temperature", "min_p", None)
            val = _round_setting(row["temperature"])
        else:
            key = (row["benchmark"], row["model"], "min_p", "temperature", _round_setting(row["temperature"]))
            val = _round_setting(row["min_p"])
        g = groups.setdefault(key, {"ids": set(), "settings": set(), "rows": 0})
        g["ids"].add(str(row["id"]))
        g["settings"].add(val)
        g["rows"] += 1
    out = []
    for (bench, model, axis, fixed_name, fixed_val), g in groups.items():
        out.append({"benchmark": bench, "model": model, "axis": axis, "fixed": {fixed_name: fixed_val},
                    "questions": len(g["ids"]), "settings": sorted(g["settings"]), "rows": g["rows"]})
    return out


def load_counts(path: str, benchmark: Optional[str] = None, model: Optional[str] = None,
                axis: str = "temperature", fixed_temperature: Optional[float] = None,
                fixed_min_p: Optional[float] = None) -> Counts:
    """Select one group from a counts.jsonl file and return its count matrices.

    benchmark, model: required when the file holds more than one; None accepts the only one.
    axis: "temperature" or "min_p" (the generic setting axis).
    fixed_temperature: for axis="min_p", the fixed temperature (tau) of the sweep.
    fixed_min_p: for axis="temperature", select rows with this min_p instead of min_p = null.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
    rows = []
    benches, models, fixed_vals = set(), set(), set()
    for row in iter_rows(path):
        if benchmark is not None and row["benchmark"] != benchmark:
            continue
        if model is not None and row["model"] != model:
            continue
        if axis == "temperature":
            if fixed_min_p is None:
                if row["min_p"] is not None:
                    continue
            elif row["min_p"] is None or _round_setting(row["min_p"]) != _round_setting(fixed_min_p):
                continue
        else:
            if row["min_p"] is None:
                continue
            if fixed_temperature is not None and _round_setting(row["temperature"]) != _round_setting(fixed_temperature):
                continue
            fixed_vals.add(_round_setting(row["temperature"]))
        benches.add(row["benchmark"])
        models.add(row["model"])
        rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path} for benchmark={benchmark!r} model={model!r} axis={axis!r} "
                         f"fixed_temperature={fixed_temperature!r} fixed_min_p={fixed_min_p!r}")
    if len(benches) > 1:
        raise ValueError(f"{path} holds several benchmarks {sorted(benches)}; pass benchmark=")
    if len(models) > 1:
        raise ValueError(f"{path} holds several models {sorted(models)}; pass model=")
    if axis == "min_p" and len(fixed_vals) > 1:
        raise ValueError(f"{path} holds min_p sweeps at several temperatures {sorted(fixed_vals)}; "
                         f"pass fixed_temperature=")

    ids: list = []
    index: dict = {}
    settings_set = set()
    for row in rows:
        qid = str(row["id"])
        if qid not in index:
            index[qid] = len(ids)
            ids.append(qid)
        settings_set.add(_round_setting(row[axis]))
    settings = sorted(settings_set)
    s_index = {s: j for j, s in enumerate(settings)}
    n = np.zeros((len(ids), len(settings)), dtype=np.int64)
    c = np.zeros_like(n)
    measured = np.zeros(n.shape, dtype=bool)
    for row in rows:
        i = index[str(row["id"])]
        j = s_index[_round_setting(row[axis])]
        if measured[i, j]:
            raise ValueError(f"{path}:{row['_line']}: duplicate row for question {row['id']!r} at "
                             f"{axis}={settings[j]}")
        measured[i, j] = True
        n[i, j] = row["n"]
        c[i, j] = row["c"]
    if axis == "temperature":
        fixed = {"min_p": None if fixed_min_p is None else _round_setting(fixed_min_p)}
    else:
        fixed = {"temperature": next(iter(fixed_vals))}
    return Counts(benchmark=next(iter(benches)), model=next(iter(models)), axis=axis, fixed=fixed,
                  settings=settings, ids=ids, n=n, c=c, measured=measured, source=str(path))


def counts_from_arrays(settings, n, c, ids=None, axis: str = "temperature", benchmark=None, model=None,
                       measured=None) -> Counts:
    """Build a Counts object from arrays (used by the tests and by callers with in-memory data)."""
    n = np.asarray(n, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if n.shape != c.shape or n.ndim != 2 or n.shape[1] != len(settings):
        raise ValueError("n and c must be (questions, settings) arrays matching settings")
    if np.any(c < 0) or np.any(c > n):
        raise ValueError("need 0 <= c <= n")
    settings = [_round_setting(s) for s in settings]
    if settings != sorted(settings) or len(set(settings)) != len(settings):
        raise ValueError("settings must be strictly increasing")
    ids = [str(i) for i in (ids if ids is not None else range(n.shape[0]))]
    measured = np.ones(n.shape, dtype=bool) if measured is None else np.asarray(measured, dtype=bool)
    fixed = {"min_p": None} if axis == "temperature" else {"temperature": None}
    return Counts(benchmark=benchmark, model=model, axis=axis, fixed=fixed, settings=settings, ids=ids,
                  n=n, c=c, measured=measured, source="<arrays>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a counts.jsonl file (the common data format).")
    ap.add_argument("--counts", required=True, help="path to counts.jsonl")
    ap.add_argument("--list", action="store_true", help="list the (benchmark, model, axis) groups")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--axis", default="temperature", choices=AXES)
    ap.add_argument("--fixed-temperature", type=float, default=None,
                    help="for --axis min_p: the fixed temperature of the sweep")
    ap.add_argument("--fixed-min-p", type=float, default=None,
                    help="for --axis temperature: select rows with this min_p instead of null")
    ap.add_argument("--summary", action="store_true", help="print a summary of the selected group")
    a = ap.parse_args(argv)
    if a.list:
        for g in list_groups(a.counts):
            print(json.dumps(g))
        return 0
    cnt = load_counts(a.counts, a.benchmark, a.model, a.axis, a.fixed_temperature, a.fixed_min_p)
    print(json.dumps(cnt.summary(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
