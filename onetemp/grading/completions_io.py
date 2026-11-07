#!/usr/bin/env python3
"""Shared input and output of the three graders (MATH-500, Hi-ToM, LiveCodeBench).

The graders turn sampled completions into per-question success counts. Section 2 of the paper estimates the
single-sample success probability P_q(tau) of question q at setting tau from these counts, and Appendix C describes
how each benchmark is graded. This module holds the parts the three graders share:

INPUT: completions files written by the sampling step. One JSON object per line, one line per (question, setting)
cell. Files end in .jsonl or .jsonl.gz and may sit anywhere under the paths given on the command line.

    {"benchmark": "math500" | "hitom" | "livecodebench",      optional; if present it must match the grader
     "model": "<HF repo id>",                                  or provenance.model, else the grader's --model
     "id": "<question id>",
     "temperature": float,
     "min_p": float or null,                                   optional; null or absent = min_p not used
     "completions": ["<model text>", ...],                     one entry per sample, in sampling order
     "finish_reasons": ["stop" | "length", ...]}               optional; same length as completions

    Question ids: MATH-500 unique_id ("test/algebra/101.json"); Hi-ToM "s{sample_id}_o{question_order}_
    {prompting_type}" ("s12_o3_CoTP"); LiveCodeBench the dataset's question_id ("1873_A", "abc301_a", "2848").

    A cell may be split over several lines (for example one line per GPU). The counts of all lines with the same
    (benchmark, model, id, temperature, min_p) are summed.

OUTPUT: counts.jsonl, the one format every analysis reads. One JSON object per line, one per
(benchmark, model, question, setting):

    {"benchmark": ..., "model": ..., "id": ..., "temperature": float, "min_p": float or null, "n": int, "c": int}

    n is the number of samples and c the number graded correct. Every sample counts once (instance-weighted), even
    when the model repeats itself: the success probability of Section 2 is a rate over samples, not over distinct
    texts. Optional fields a grader may add are documented in that grader (MATH-500 adds "c_strict").

The sampling parameter is generic: a setting is (temperature, min_p). In the temperature sweeps min_p is null; in the
min_p runs (Appendix I) the temperature is the fixed tau and min_p varies.
"""
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.request
from collections import Counter, OrderedDict

from onetemp.counts import BENCHMARKS, REQUIRED_KEYS as COUNTS_KEYS, validate_row
from onetemp.paths import default_cache_dir as _shared_cache_dir



# ---------------------------------------------------------------------------------------
# Reading completions files
# ---------------------------------------------------------------------------------------
def list_input_files(paths):
    """Every .jsonl / .jsonl.gz file under the directories in `paths`, plus every other path given explicitly (any
    name; a pipe such as /dev/stdin works too), sorted, each listed once."""
    found = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs.sort()
                for name in sorted(files):
                    if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
                        found.append(os.path.join(root, name))
        elif os.path.exists(p):
            found.append(p)
        else:
            raise SystemExit(f"input path does not exist: {p}")
    real = [os.path.realpath(f) for f in found]
    dup = [f for f, k in Counter(real).items() if k > 1]
    if dup:
        raise SystemExit(f"the same file is given twice (its counts would be summed twice): {dup[0]}")
    return sorted(found)


def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _setting(value, name, where):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"{where}: {name} must be a finite number or null, got {value!r}")
    return float(value)


def iter_cells(paths, benchmark, model=None):
    """Yield one dict per input line: benchmark, model, id, temperature, min_p, completions, finish_reasons, source.

    Only the keys listed in the module docstring are read. Nothing else in a line (a prompt, an answer, provenance)
    can reach a grader, so the graded string is always the model's own text.
    """
    assert benchmark in BENCHMARKS, benchmark
    for path in list_input_files(paths):
        with _open_text(path) as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                where = f"{path}:{lineno}"
                try:
                    rec = json.loads(line)
                except ValueError as ex:
                    raise SystemExit(f"{where}: not JSON ({ex})")
                if not isinstance(rec, dict):
                    raise SystemExit(f"{where}: a line must be a JSON object")
                b = rec.get("benchmark", benchmark)
                if b != benchmark:
                    raise SystemExit(f"{where}: benchmark {b!r}, but this grader is for {benchmark!r}")
                prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
                m = rec.get("model") or prov.get("model") or model
                if not isinstance(m, str) or not m:
                    raise SystemExit(f"{where}: no model id (add \"model\" to the line or pass --model)")
                qid = rec.get("id")
                if not isinstance(qid, str) or not qid:
                    raise SystemExit(f"{where}: \"id\" must be a non-empty string")
                t = _setting(rec.get("temperature"), "temperature", where)
                if t is None:
                    raise SystemExit(f"{where}: \"temperature\" is required")
                mp = _setting(rec.get("min_p"), "min_p", where)
                comps = rec.get("completions")
                if not isinstance(comps, list) or not all(isinstance(c, str) for c in comps):
                    raise SystemExit(f"{where}: \"completions\" must be a list of strings")
                fins = rec.get("finish_reasons")
                if fins is not None and (not isinstance(fins, list) or len(fins) != len(comps)):
                    raise SystemExit(f"{where}: \"finish_reasons\" must be a list as long as \"completions\"")
                if "n" in rec and rec["n"] != len(comps):          # the sampling step writes n; a short file is refused
                    raise SystemExit(f"{where}: \"n\" is {rec['n']} but the line holds {len(comps)} completions")
                extra = {k: rec[k] for k in ("lcb_version", "lcb_row") if k in rec}
                yield {"benchmark": benchmark, "model": m, "id": qid, "temperature": t, "min_p": mp,
                       "completions": comps, "finish_reasons": fins, "source": where, **extra}


def cell_key(cell):
    return (cell["benchmark"], cell["model"], cell["id"], cell["temperature"], cell["min_p"])


def texts_sha256(texts):
    """sha256 of a list of completions, to notice a shard that is included twice."""
    h = hashlib.sha256()
    for t in texts:
        b = t.encode("utf-8", "surrogatepass")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# Aggregating and writing counts
# ---------------------------------------------------------------------------------------
class CountsTable:
    """Sums per-line results into one row per (benchmark, model, id, temperature, min_p)."""

    def __init__(self, extra_fields=()):
        self.extra_fields = tuple(extra_fields)
        self.rows = OrderedDict()
        self.shard_hashes = {}
        self.warnings = []

    def add(self, cell, n, c, extra=None, texts_hash=None):
        key = cell_key(cell)
        if texts_hash is not None:
            seen = self.shard_hashes.setdefault(key, set())
            if texts_hash in seen and n > 1:
                self.warnings.append(f"{cell['source']}: the same completions already appeared for cell {key}; "
                                     f"if a shard was included twice, its samples are counted twice")
            seen.add(texts_hash)
        row = self.rows.get(key)
        if row is None:
            row = {"benchmark": key[0], "model": key[1], "id": key[2], "temperature": key[3], "min_p": key[4],
                   "n": 0, "c": 0}
            for f in self.extra_fields:
                row[f] = 0
            self.rows[key] = row
        row["n"] += int(n)
        row["c"] += int(c)
        for f in self.extra_fields:
            row[f] += int((extra or {}).get(f, 0))

    def sorted_rows(self):
        def sk(k):
            return (k[0], k[1], k[2], k[3], -1.0 if k[4] is None else k[4])
        return [self.rows[k] for k in sorted(self.rows, key=sk)]


def check_counts_row(row):
    """Problems with one counts.jsonl row ([] = valid), by the package's validator (onetemp.counts.validate_row),
    plus the graders' own rule that the benchmark is one of the three."""
    try:
        validate_row(row, "counts row")
    except ValueError as e:
        return [str(e)]
    if row["benchmark"] not in BENCHMARKS:
        return [f"benchmark {row['benchmark']!r}"]
    return []


def check_outputs(paths, force=False):
    """Called before any grading starts: refuse at once, not after hours of work, if an output already exists
    (without --force) or if two outputs are the same file. Returns the absolute paths (None stays None)."""
    out = [os.path.abspath(p) if p else None for p in paths]
    given = [p for p in out if p]
    if len(set(given)) != len(given):
        raise SystemExit("two outputs name the same file")
    for p in given:
        if os.path.exists(p) and not force:
            raise SystemExit(f"refusing to overwrite {p} (pass --force)")
    return out


def write_jsonl_atomic(path, rows, force=False):
    """Write rows as JSON lines through a temporary file in the same directory, then rename (never half-written)."""
    if os.path.exists(path) and not force:
        raise SystemExit(f"refusing to overwrite {path} (pass --force)")
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".jsonl", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json_atomic(path, obj, force=False):
    if os.path.exists(path) and not force:
        raise SystemExit(f"refusing to overwrite {path} (pass --force)")
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Summary:
    """Failure taxonomy per (model, setting): bucket counts over samples, plus truncation (finish_reason 'length').

    A truncated completion usually loses its answer, which is a measurement artifact rather than a model failure, so
    it is reported separately. Counts only: no answer, test case or completion ever enters the summary."""

    def __init__(self):
        self.by_setting = {}

    def add(self, cell, buckets, n_truncated, n_truncated_pass=0):
        k = json.dumps([cell["model"], cell["temperature"], cell["min_p"]])
        s = self.by_setting.setdefault(k, {"model": cell["model"], "temperature": cell["temperature"],
                                           "min_p": cell["min_p"], "cells": 0, "n": 0, "buckets": Counter(),
                                           "truncated": 0, "truncated_and_correct": 0})
        s["cells"] += 1
        s["n"] += sum(buckets.values())
        s["buckets"].update(buckets)
        s["truncated"] += n_truncated
        s["truncated_and_correct"] += n_truncated_pass

    def as_json(self):
        out = []
        for k in sorted(self.by_setting, key=lambda x: json.loads(x)[0:2] + [json.loads(x)[2] or -1.0]):
            s = dict(self.by_setting[k])
            s["buckets"] = dict(sorted(s["buckets"].items()))
            out.append(s)
        return out


# ---------------------------------------------------------------------------------------
# Public data sources, fetched at run time and checked against their sha256
# ---------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def default_cache_dir():
    """The download cache the graders share with the prompt builders (onetemp.paths)."""
    return _shared_cache_dir()


def fetch_pinned(url, sha256, cache_dir, filename, local_file=None, check=True):
    """Path to a verified local copy of a public file.

    local_file: use this copy instead of downloading. Otherwise the file is downloaded once into cache_dir. Either
    way its sha256 must equal the pinned value (the bytes the paper's runs used) unless check is False.
    """
    if local_file:
        path = local_file
    else:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            print(f"downloading {url}", file=sys.stderr)
            fd, tmp = tempfile.mkstemp(prefix=".dl_", dir=cache_dir)
            try:
                with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=120) as resp:
                    shutil.copyfileobj(resp, out, 1 << 20)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    if check:
        got = sha256_file(path)
        if got != sha256:
            raise SystemExit(f"{path}: sha256 {got} is not the pinned {sha256}. This is not the data the paper "
                             f"graded against; pass --no-verify-data to use it anyway.")
    return path


def package_versions(names):
    """{name: version or None} for the given distributions, recorded in every summary."""
    out = {}
    try:
        from importlib import metadata
    except ImportError:          # pragma: no cover
        return {n: None for n in names}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out
