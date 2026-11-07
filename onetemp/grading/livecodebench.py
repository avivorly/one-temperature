#!/usr/bin/env python3
"""LiveCodeBench grader of the paper: completions -> counts.jsonl, by running the code on the problem's tests.

    python -m onetemp.grading.livecodebench --allow-code-execution --inputs runs/mistral_lcb --out counts_lcb.jsonl \\
        --summary summary.json [--lcb-dir path/to/code_generation_lite] [--workers 6]

WARNING: this executes model-generated code. LiveCodeBench's reliability_guard() is not a sandbox. Run it only in an
isolated container or VM without network access; --allow-code-execution is required.

WHAT IT COMPUTES (Appendix C: LiveCodeBench is graded by unit tests). For every completion:
  1. extract_code(): the generators of the paper's runs primed the answer with ```python and stopped at
     "```\\n```", "```\\n\\n" or "\\n\\n\\n"; the code is the first ```python ... ``` block, else the first ``` ... ```
     block, else the whole text, stripped. Two variants ran (see EXTRACT_BY_MODEL): the generators of Qwen3-8B and
     GPT-OSS 20B also drop a closing ``` that ends the text; those of Mistral 7B Instruct and Qwen3-0.6B do not.
  2. The code runs on all of the problem's tests (public, then private) through LiveCodeBench's own harness
     (lcb_execution.py, vendored from LiveCodeBench with the two local fixes the paper's evaluators ran): one process
     per completion, killed after a global limit of (6 + 1) x tests + 5 s; a 6 s limit per test (wall clock); the
     solution's address space capped at 4 GiB (a local fix); stdin programs compared line by line (exact, else as
     Decimals), call-based (LeetCode) programs by ==.
  3. A completion passes when every test passes: all(x == True for x in result).
Per (question, setting) it writes n = samples and c = passing samples (instance-weighted), the estimate c/n of
P_q(tau) in Section 2. Completions are run in chunks of 500 with --workers processes at a time, as in the paper.

Every completion is executed, also when several are identical (as in the paper's runs). --dedup runs each distinct
extracted code once and counts its verdict for every sample with that code. It is much faster, and differs only
where the code alone does not decide the outcome: a wall-clock limit, the memory cap, or a program that uses
randomness or the clock (`random` is reseeded in every forked test process). The time limits are wall clock, so
give every worker its own core.

The tests come from the public dataset livecodebench/code_generation_lite on the Hugging Face hub, at a pinned
revision, downloaded at run time (or read from --lcb-dir) and checked by sha256; none is stored in this repository.
A question's id is the dataset's question_id ("1873_A", "abc301_a", "2848"). Release version vN is file test{N}.jsonl
(test.jsonl for v1). As in LiveCodeBench, private tests are stored compressed (base64, zlib, pickle); they are
decoded only after the file's sha256 matched the pinned value (unless --no-verify-data is given).
"""
import argparse
import base64
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import urllib.request
import zlib
from collections import Counter
from datetime import datetime

from . import completions_io as io_  # noqa: E402

LCB_REPO = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"   # the hub's main branch since 2025-06-05
# version -> (file, sha256, bytes): the six files the paper's evaluators read, checked against these hashes
LCB_FILES = {
    "v1": ("test.jsonl", "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f", 1252609773),
    "v2": ("test2.jsonl", "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3", 713377060),
    "v3": ("test3.jsonl", "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79", 623360766),
    "v4": ("test4.jsonl", "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da", 1204644685),
    "v5": ("test5.jsonl", "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214", 557699297),
    "v6": ("test6.jsonl", "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5", 134303240),
}
TIMEOUT_S = 6          # per test, as in the paper's runs
CHUNK = 500            # completions per codegen call, as in the paper's runs
HARNESS = ("LiveCodeBench 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 with the force-kill and 4 GiB fixes; "
           "extraction v1/v2 by model; pass = all tests")


# ---------------------------------------------------------------------------------------
# Step 1: code extraction, verbatim from the paper's generators
# ---------------------------------------------------------------------------------------
def extract_code_v1(text):
    """The generators of Mistral 7B Instruct and Qwen3-0.6B."""
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def extract_code_v2(text):
    """The generators of Qwen3-8B and GPT-OSS 20B: v1, plus a closing ``` at the end of the text is dropped."""
    for pattern in [r'```python\s*(.*?)```', r'```\s*(.*?)```']:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    text = text.strip()
    if text.endswith('```'):
        text = text[:-3].strip()
    return text


EXTRACTORS = {"v1": extract_code_v1, "v2": extract_code_v2}
# keyed by the model's name without its owner, lower-cased, so a local path such as /models/Qwen3-8B also matches
EXTRACT_BY_MODEL = {
    "mistral-7b-instruct-v0.3": "v1",
    "qwen3-0.6b": "v1",
    "qwen3-8b": "v2",
    "gpt-oss-20b": "v2",
}
_WARNED = set()


def extractor_for(model, choice="auto"):
    """The paper's extraction for this model; other models get v2 (the later, fixed variant) unless told."""
    if choice != "auto":
        return choice
    key = model.rstrip("/").split("/")[-1].lower()
    if key not in EXTRACT_BY_MODEL and model not in _WARNED:
        _WARNED.add(model)
        print(f"WARNING: {model!r} is not one of the paper's models; using extraction v2 (choose with --extract)",
              file=sys.stderr)
    return EXTRACT_BY_MODEL.get(key, "v2")


# ---------------------------------------------------------------------------------------
# The tests: public dataset rows, read at run time
# ---------------------------------------------------------------------------------------
class LcbData:
    """The six test*.jsonl files of livecodebench/code_generation_lite, located, verified and indexed by question_id.

    lcb_dir: a directory holding the files under their hub names (for example a clone of the dataset repository);
    otherwise each needed file is downloaded once into cache_dir. A file is hashed once and then indexed
    (question_id -> byte offset of its line); the index is cached next to the file's data."""

    def __init__(self, lcb_dir=None, cache_dir=None, verify=True):
        self.lcb_dir = os.path.abspath(lcb_dir) if lcb_dir else None
        self.cache_dir = os.path.join(os.path.abspath(cache_dir or io_.default_cache_dir()), "livecodebench",
                                      LCB_REVISION)
        self.verify = verify
        self._paths, self._index, self.checked = {}, {}, {}

    def path(self, version):
        if version in self._paths:
            return self._paths[version]
        name, sha, size = LCB_FILES[version]
        if self.lcb_dir:
            p = os.path.join(self.lcb_dir, name)
            if not os.path.exists(p):
                raise SystemExit(f"{p} not found (--lcb-dir must hold {name})")
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            p = os.path.join(self.cache_dir, name)
            if not os.path.exists(p):
                url = f"https://huggingface.co/datasets/{LCB_REPO}/resolve/{LCB_REVISION}/{name}"
                print(f"downloading {url} ({size / 1e9:.2f} GB)", file=sys.stderr)
                fd, tmp = tempfile.mkstemp(prefix=".dl_", dir=self.cache_dir)
                try:
                    with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=300) as resp:
                        shutil.copyfileobj(resp, out, 1 << 22)
                    os.replace(tmp, p)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
        if self.verify:
            if os.path.getsize(p) != size or io_.sha256_file(p) != sha:
                raise SystemExit(f"{p} is not the pinned {name} (sha256 {sha}); these are not the tests the paper "
                                 f"used. Pass --no-verify-data to use it anyway (its private tests are unpickled).")
            self.checked[version] = {"file": name, "sha256": sha, "verified": True}
        else:
            self.checked[version] = {"file": name, "sha256": None, "verified": False}
        self._paths[version] = p
        return p

    def index(self, version):
        """question_id -> [byte offset, row number] of its line. The top-level key is the first `"question_id": "`
        in a line (inside any string value the quotes are escaped), so the scan never parses the test strings."""
        if version in self._index:
            return self._index[version]
        p = self.path(version)
        cache = os.path.join(self.cache_dir, f"index_{version}.json")
        st = os.stat(p)
        stamp = [os.path.realpath(p), st.st_size, int(st.st_mtime)]
        if os.path.exists(cache):
            try:
                with open(cache, encoding="utf-8") as fh:
                    saved = json.load(fh)
                if saved.get("file") == stamp:
                    self._index[version] = saved["index"]
                    return saved["index"]
            except (OSError, ValueError, KeyError):
                pass
        needle = b'"question_id": "'
        idx, off = {}, 0
        with open(p, "rb") as fh:
            for rowno, line in enumerate(fh):
                k = line.find(needle)
                if k >= 0:
                    e = line.index(b'"', k + len(needle))
                    qid = line[k + len(needle):e].decode("utf-8")
                    if qid in idx:
                        raise SystemExit(f"{p}: question_id {qid} appears twice")
                    idx[qid] = [off, rowno]
                off += len(line)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump({"file": stamp, "index": idx}, fh)
        except OSError:
            pass
        self._index[version] = idx
        return idx

    def find(self, qid, version=None):
        """-> (version, row number, row dict) for question_id qid, searching the given version first, then v1..v6."""
        order = ([version] if version in LCB_FILES else []) + [v for v in LCB_FILES if v != version]
        for v in order:
            hit = self.index(v).get(qid)
            if hit is None:
                continue
            with open(self.path(v), "rb") as fh:
                fh.seek(hit[0])
                row = json.loads(fh.readline())
            if row.get("question_id") != qid:
                raise SystemExit(f"index of {v} is stale for {qid}; delete {self.cache_dir}/index_{v}.json")
            return v, hit[1], row
        raise SystemExit(f"question_id {qid!r} is in none of {', '.join(LCB_FILES)}")


def evaluation_sample(row):
    """A dataset row -> LiveCodeBench's evaluation sample, built exactly as lcb_runner's CodeGenerationProblem
    (benchmarks/code_generation.py) builds it: the tests are public + private, in that order; fn_name is the
    metadata's func_name (call-based LeetCode problems) or None (stdin programs). The checks mirror its enums."""
    assert row["platform"] in ("leetcode", "codeforces", "atcoder"), row["platform"]
    assert row["difficulty"] in ("easy", "medium", "hard"), row["difficulty"]
    datetime.fromisoformat(row["contest_date"])
    public = json.loads(row["public_test_cases"])
    try:
        private = json.loads(row["private_test_cases"])
    except Exception:  # noqa: BLE001  (LiveCodeBench: a bare except, then the compressed form)
        private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(row["private_test_cases"].encode("utf-8")))))
    for t in public + private:
        assert t["testtype"] in ("stdin", "functional"), t["testtype"]
    metadata = json.loads(row["metadata"])
    tests = public + private
    sample = {"input_output": json.dumps({"inputs": [t["input"] for t in tests],
                                          "outputs": [t["output"] for t in tests],
                                          "fn_name": metadata.get("func_name", None)})}
    info = {"n_tests": len(tests), "n_public": len(public), "call_based": metadata.get("func_name") is not None,
            "platform": row["platform"]}
    return sample, info


# ---------------------------------------------------------------------------------------
# Steps 2-3: run and judge
# ---------------------------------------------------------------------------------------
def outcome(result, meta):
    """Failure class of one completion, for the summary only (the verdict is all(x == True))."""
    if all(x == True for x in result):  # noqa: E712  (the paper's rule, verbatim)
        return "passed"
    code = (meta or {}).get("error_code")
    return {-2: "wrong_answer", -3: "time_limit", -4: "runtime_or_compile_error",
            -5: "harness_error_or_global_timeout"}.get(code, "failed_other")


def run_codes(sample, codes, workers, timeout=TIMEOUT_S, chunk=CHUNK):
    """Per-code (passed, outcome), running the codes through LiveCodeBench's harness in chunks, in order."""
    from . import lcb_execution as lx
    out = []
    for ci in range(0, len(codes), chunk):
        part = codes[ci:ci + chunk]
        results, metas = lx.codegen_results(samples_list=[sample], generations_list=[part],
                                            num_process_evaluate=workers, timeout=timeout, debug=False)
        res, mts = results[0], metas[0]
        assert len(res) == len(part) == len(mts)
        for r, m in zip(res, mts):
            out.append((all(x == True for x in r), outcome(r, m)))  # noqa: E712
    return out


def grade_cell(cell, sample, extract, workers, dedup=False, timeout=TIMEOUT_S, chunk=CHUNK):
    """-> (n, c, outcomes Counter, n_truncated, n_truncated_and_passed, distinct codes)."""
    codes = [EXTRACTORS[extract](t) for t in cell["completions"]]
    if dedup:
        uniq = list(dict.fromkeys(codes))
        verdict = dict(zip(uniq, run_codes(sample, uniq, workers, timeout, chunk)))
        per = [verdict[c] for c in codes]
    else:
        per = run_codes(sample, codes, workers, timeout, chunk)
    passed = [p for p, _ in per]
    fins = cell["finish_reasons"]
    trunc = [f == "length" for f in fins] if fins is not None else [False] * len(codes)
    return (len(codes), sum(passed), Counter(o for _, o in per), sum(trunc),
            sum(1 for p, t in zip(passed, trunc) if p and t), len(set(codes)))


def grade_inputs(paths, data, workers=6, model=None, extract="auto", dedup=False, timeout=TIMEOUT_S, chunk=CHUNK,
                 log=sys.stderr):
    counts = io_.CountsTable()
    summary = io_.Summary()
    totals = Counter()
    extractors = Counter()
    samples = {}
    for cell in io_.iter_cells(paths, "livecodebench", model=model):
        qid = cell["id"]
        if qid not in samples:
            version, rowno, row = data.find(qid, cell.get("lcb_version"))
            samples.clear()                      # keep one problem's tests in memory at a time
            samples[qid] = evaluation_sample(row) + (version, rowno)
            del row
        sample, info, version, rowno = samples[qid]
        if "lcb_version" in cell and cell["lcb_version"] != version:      # every line's hints are checked
            raise SystemExit(f"{cell['source']}: lcb_version {cell['lcb_version']} but {qid} is in {version}")
        if "lcb_row" in cell and cell["lcb_row"] != rowno:
            raise SystemExit(f"{cell['source']}: lcb_row {cell['lcb_row']} but {qid} is row {rowno}")
        ex = extractor_for(cell["model"], extract)
        extractors[f"{cell['model']} -> {ex}"] += 1
        h = io_.texts_sha256(cell["completions"])
        n, c, outs, nt, ntp, distinct = grade_cell(cell, sample, ex, workers, dedup, timeout, chunk)
        counts.add(cell, n, c, texts_hash=h)
        summary.add(cell, outs, nt, ntp)
        totals.update({"cells": 1, "samples": n, "passes": c, "distinct_codes": distinct})
        print(f"{qid} T={cell['temperature']:g} min_p={cell['min_p']}: {c}/{n} passed "
              f"({info['n_tests']} tests, {distinct} distinct codes)", file=log, flush=True)
    totals["extractors"] = dict(extractors)
    return counts, summary, totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-code-execution", action="store_true",
                    help="required: you run model-generated code (use an isolated container or VM, no network)")
    ap.add_argument("--inputs", nargs="+", help="completions files or directories (*.jsonl, *.jsonl.gz)")
    ap.add_argument("--out", help="counts.jsonl to write")
    ap.add_argument("--summary", help="optional JSON: outcome classes and truncation per (model, setting)")
    ap.add_argument("--lcb-dir", help="directory with test.jsonl ... test6.jsonl (default: download and cache)")
    ap.add_argument("--cache-dir", default=io_.default_cache_dir(), help="where downloaded data is kept")
    ap.add_argument("--no-verify-data", action="store_true", help="skip the sha256 check of the dataset files")
    ap.add_argument("--model", help="model id for lines that do not name one")
    ap.add_argument("--extract", choices=("auto", "v1", "v2"), default="auto",
                    help="code extraction; auto = the paper's variant for the model")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                    help="completions run at once (default min(6, cores)); one core each")
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S, help="seconds per test (the paper: 6)")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="completions per harness call (the paper: 500)")
    ap.add_argument("--dedup", action="store_true", help="run each distinct code once (not how the paper ran it)")
    ap.add_argument("--workdir", help="working directory for the solutions (default: a new temporary directory)")
    ap.add_argument("--force", action="store_true", help="overwrite --out / --summary")
    ap.add_argument("--try-code", metavar="FILE", help="run one Python file on the tests of --qid and print the result")
    ap.add_argument("--qid", help="question_id for --try-code")
    a = ap.parse_args(argv)
    if not a.allow_code_execution:
        ap.error("this runs model-generated code: pass --allow-code-execution (isolated machine, no network)")
    if a.try_code is None and (not a.inputs or not a.out):
        ap.error("--inputs and --out are required")
    if a.try_code is not None and not a.qid:
        ap.error("--try-code needs --qid")
    # every path is made absolute before the working directory changes; outputs are checked before any work
    out_path, summary_path = io_.check_outputs([a.out, a.summary], force=a.force)
    data = LcbData(a.lcb_dir, a.cache_dir, verify=not a.no_verify_data)
    inputs = [os.path.abspath(p) for p in (a.inputs or [])]
    try_code = os.path.abspath(a.try_code) if a.try_code else None
    own_workdir = a.workdir is None
    workdir = tempfile.mkdtemp(prefix="lcb_work_") if own_workdir else os.path.abspath(a.workdir)
    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)                     # solutions may write files; they land here
    try:
        if try_code is not None:
            _, _, row = data.find(a.qid)
            sample, info = evaluation_sample(row)
            with open(try_code, encoding="utf-8") as fh:
                code = fh.read()
            (passed, how), = run_codes(sample, [code], 1, a.timeout, a.chunk)
            print(json.dumps({"qid": a.qid, "passed": passed, "outcome": how, **info}))
            return 0
        counts, summary, totals = grade_inputs(inputs, data, workers=a.workers, model=a.model, extract=a.extract,
                                               dedup=a.dedup, timeout=a.timeout, chunk=a.chunk)
    finally:
        os.chdir(cwd)
        if own_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
    rows = counts.sorted_rows()
    io_.write_jsonl_atomic(out_path, rows, force=a.force)
    for w in counts.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if summary_path:
        harness = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_execution.py")
        io_.write_json_atomic(summary_path, {
            "benchmark": "livecodebench", "grader": HARNESS, "harness_sha256": io_.sha256_file(harness),
            "dataset": {"repo": LCB_REPO, "revision": LCB_REVISION, "files_read": data.checked},
            "timeout_s": a.timeout, "chunk": a.chunk, "workers": a.workers, "dedup": a.dedup,
            "python": sys.version.split()[0], "packages": io_.package_versions(["numpy"]),
            "totals": dict(totals), "by_setting": summary.as_json()}, force=a.force)
    print(f"{len(rows)} cells, {totals['samples']} samples, {totals['passes']} passing -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
