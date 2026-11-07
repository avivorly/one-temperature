#!/usr/bin/env python3
"""The three public benchmarks, fetched at run time at pinned revisions and checked by sha256 (Appendix C).

Only question text leaves this module. Reference answers, solutions and test cases are never kept: each loader
returns, per question, its id and the fields its prompt needs, and nothing else.

  MATH-500       HuggingFaceH4/MATH-500 test.jsonl at revision 6e4ed1a (the 500-question PRM800K test split of
                 MATH, byte-identical to openai/prm800k math_splits/test.jsonl). Id: unique_id. Field: problem.
  Hi-ToM         Hi-ToM_data.json of github.com/ying-hui-he/Hi-ToM_dataset (1,200 records). Id:
                 s{sample_id}_o{question_order}_{prompting_type}. Field: the dataset's own prompt.
  LiveCodeBench  livecodebench/code_generation_lite at revision 0fe84c3: release version vN is the file
                 test{N}.jsonl (test.jsonl for v1). The paper's 975 questions are listed, with their version and
                 row, in question_sets/livecodebench_975.json. Id: question_id. Fields: question_content,
                 starter_code.

Files are cached under --cache-dir (default ~/.cache/one-temperature, the graders' cache, so each file is
downloaded once) or read from a local copy given with --math500-file / --hitom-file / --lcb-dir.
"""
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request

from onetemp.paths import default_cache_dir as _shared_cache_dir

HERE = os.path.dirname(os.path.abspath(__file__))

MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_URL = f"https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{MATH500_REVISION}/test.jsonl"
MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"     # 446,564 bytes

HITOM_COMMIT = "459784fa8de09f35e9110e1e9454b41cd9b4f985"
HITOM_URL = (f"https://raw.githubusercontent.com/ying-hui-he/Hi-ToM_dataset/{HITOM_COMMIT}/"
             f"Hi-ToM_data/Hi-ToM_data.json")
HITOM_SHA256 = "088b6c54d21d1a0d3045812f36ac8b5b342346a6a996c003738976730ac256f4"       # 5,134,393 bytes

LCB_REPO = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILES = {   # version -> (file, sha256, bytes)
    "v1": ("test.jsonl", "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f", 1252609773),
    "v2": ("test2.jsonl", "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3", 713377060),
    "v3": ("test3.jsonl", "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79", 623360766),
    "v4": ("test4.jsonl", "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da", 1204644685),
    "v5": ("test5.jsonl", "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214", 557699297),
    "v6": ("test6.jsonl", "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5", 134303240),
}
LCB_QUESTIONS_FILE = os.path.join(HERE, "question_sets", "livecodebench_975.json")


def default_cache_dir():
    """The download cache the graders share with the prompt builders (onetemp.paths)."""
    return _shared_cache_dir()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, sha256, cache_dir, filename, local_file=None, verify=True):
    """Path to a verified copy of a public file: local_file if given, else a cached download."""
    if local_file:
        path = local_file
    else:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            print(f"downloading {url}", file=sys.stderr)
            fd, tmp = tempfile.mkstemp(prefix=".dl_", dir=cache_dir)
            try:
                with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=300) as resp:
                    shutil.copyfileobj(resp, out, 1 << 22)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    if verify:
        got = sha256_file(path)
        if got != sha256:
            raise SystemExit(f"{path}: sha256 {got} is not the pinned {sha256}; these are not the questions the "
                             f"paper sampled (pass --no-verify-data to use them anyway)")
    return path


# ---------------------------------------------------------------------------------------------------------------
def load_math500(cache_dir=None, local_file=None, verify=True):
    """[{"id": unique_id, "problem": str}] for the 500 questions, in file order. Answers are not read."""
    path = fetch(MATH500_URL, MATH500_SHA256, cache_dir or default_cache_dir(), "math500_test.jsonl",
                 local_file, verify)
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out.append({"id": r["unique_id"], "problem": r["problem"]})
    if verify:
        assert len(out) == 500 and len({q["id"] for q in out}) == 500, "MATH-500 must have 500 unique ids"
    return out, {"url": MATH500_URL, "sha256": sha256_file(path)}


def hitom_id(record):
    return f"s{record['sample_id']}_o{record['question_order']}_{record['prompting_type']}"


def load_hitom(cache_dir=None, local_file=None, verify=True):
    """[{"id", "prompt"}] for the 1,200 questions, in file order. The prompt is the dataset's own prompt field
    (story, question, the 15 options A-O, and a note); the answer field is not read."""
    path = fetch(HITOM_URL, HITOM_SHA256, cache_dir or default_cache_dir(), "Hi-ToM_data.json", local_file, verify)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)["data"]
    out = [{"id": hitom_id(r), "prompt": r["prompt"]} for r in data]
    if verify:
        assert len(out) == 1200 and len({q["id"] for q in out}) == 1200, "Hi-ToM must have 1,200 unique ids"
    return out, {"url": HITOM_URL, "sha256": sha256_file(path)}


def lcb_question_index(path=LCB_QUESTIONS_FILE):
    """The paper's 975 LiveCodeBench questions: [{"global_idx", "question_id", "version", "row"}] in paper order."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    rows = d["questions"]
    assert len(rows) == 975 and len({r["question_id"] for r in rows}) == 975
    assert [r["global_idx"] for r in rows] == list(range(975))
    return rows


def _lcb_path(version, lcb_dir, cache_dir, verify):
    name, sha, size = LCB_FILES[version]
    if lcb_dir:
        p = os.path.join(lcb_dir, name)
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found (--lcb-dir must hold {name})")
    else:
        d = os.path.join(cache_dir or default_cache_dir(), "livecodebench", LCB_REVISION)
        url = f"https://huggingface.co/datasets/{LCB_REPO}/resolve/{LCB_REVISION}/{name}"
        p = fetch(url, sha, d, name, None, verify=False)
    if verify and (os.path.getsize(p) != size or sha256_file(p) != sha):
        raise SystemExit(f"{p} is not the pinned {name} (sha256 {sha}); pass --no-verify-data to use it anyway")
    return p


def load_livecodebench(ids=None, lcb_dir=None, cache_dir=None, verify=True, index_file=LCB_QUESTIONS_FILE):
    """[{"id": question_id, "question_content", "starter_code", "lcb_version", "lcb_row", "global_idx"}].

    ids: question_ids to load (default: all 975), returned in paper order. Each needed file is streamed once and
    only the needed rows are parsed; their test cases are dropped as soon as the row is read."""
    index = lcb_question_index(index_file)
    want = None if ids is None else set(ids)
    if want is not None:
        unknown = want - {r["question_id"] for r in index}
        if unknown:
            raise SystemExit(f"{len(unknown)} ids are not among the paper's 975 LiveCodeBench questions, e.g. "
                             f"{sorted(unknown)[:3]}")
    rows = [r for r in index if want is None or r["question_id"] in want]
    by_version = {}
    for r in rows:
        by_version.setdefault(r["version"], {})[r["row"]] = r
    got, files = {}, {}
    for v in sorted(by_version):
        p = _lcb_path(v, lcb_dir, cache_dir, verify)
        files[v] = {"file": LCB_FILES[v][0], "sha256": LCB_FILES[v][1] if verify else sha256_file(p)}
        need = by_version[v]
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as fh:
            for rowno, line in enumerate(fh):
                if rowno not in need:
                    continue
                rec = json.loads(line)
                meta = need[rowno]
                if rec["question_id"] != meta["question_id"]:
                    raise SystemExit(f"{v} row {rowno} is {rec['question_id']}, expected {meta['question_id']}")
                got[meta["question_id"]] = {"id": meta["question_id"], "question_content": rec["question_content"],
                                            "starter_code": rec.get("starter_code") or "",
                                            "lcb_version": v, "lcb_row": rowno, "global_idx": meta["global_idx"]}
                del rec
        missing = [m["question_id"] for k, m in need.items() if m["question_id"] not in got]
        if missing:
            raise SystemExit(f"{v}: rows not found for {missing[:3]}")
    out = [got[r["question_id"]] for r in rows]
    return out, {"repo": LCB_REPO, "revision": LCB_REVISION, "files": files}


def load_questions(benchmark, ids=None, cache_dir=None, math500_file=None, hitom_file=None, lcb_dir=None,
                   verify=True):
    """Questions of one benchmark (optionally a subset of ids, kept in dataset order) and the data provenance."""
    if benchmark == "livecodebench":
        return load_livecodebench(ids, lcb_dir, cache_dir, verify)
    if benchmark == "math500":
        qs, prov = load_math500(cache_dir, math500_file, verify)
    elif benchmark == "hitom":
        qs, prov = load_hitom(cache_dir, hitom_file, verify)
    else:
        raise SystemExit(f"unknown benchmark {benchmark!r}")
    if ids is not None:
        want = set(ids)
        unknown = want - {q["id"] for q in qs}
        if unknown:
            raise SystemExit(f"{len(unknown)} ids are not {benchmark} question ids, e.g. {sorted(unknown)[:3]}")
        qs = [q for q in qs if q["id"] in want]
    return qs, prov


def read_ids_file(path):
    """Question ids from a JSON list, a JSON object with "ids", or a text file with one id per line."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    s = text.lstrip()
    if s.startswith("[") or s.startswith("{"):
        d = json.loads(text)
        ids = d["ids"] if isinstance(d, dict) else d
    else:
        ids = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not all(isinstance(i, str) and i for i in ids):
        raise SystemExit(f"{path}: ids must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate ids")
    return ids
