"""Check the released counts against their manifest, and prove they hold counts only.

    python -m onetemp.data.check                # data/manifest.json of the repository
    python -m onetemp.data.check --manifest PATH
    python -m onetemp.data.check --math500-test FILE --hitom-data FILE --lcb-meta FILE   # also the gold scan

What is checked, file by file (every check must pass; the exit code is 1 otherwise):

1. Integrity. The file exists; its sha256, byte size, row count, question count, settings, sample and
   correct totals, n range and rows with ``n_cell`` equal the manifest.
2. Schema. Every row passes ``onetemp.counts.validate_row`` in strict mode; one benchmark and one model per file, equal to
   the manifest; every question has every setting exactly once (a complete rectangle); the question
   order is the file's own, with no question split across non-adjacent lines.
3. Populations. Every file holds every question the paper's analysis code read for its study; its
   included questions (the paper's inclusion rule) are listed in ``included_ids.json``, in the file's
   order, with the count and sha256 recorded in the manifest. Every figure or table use points at an
   existing file, and the population it uses (all questions, or the included ones) has the size the
   paper prints (``n_printed``). The one known difference, the Appendix C count of an earlier Qwen3-8B
   data release, is printed with its reason.
4. Consistency. Files that share questions (for example a Hi-ToM screen inside its pooled study, or the two
   LiveCodeBench parses) agree on every shared cell:
   the same c and the same number of samples actually drawn. Where one file holds a part of the other's
   draws (the 400 second-stage samples inside the 500 of screen plus second stage), every shared cell
   satisfies c_a <= c_b, n_a < n_b and c_b - c_a <= n_b - n_a.
5. Counts only (the proof that no gold answer, test case or completion is in the export).
   a. Structure: a row has only the keys of the schema; the only strings are benchmark, model and id;
      benchmark and model come from fixed lists; every id matches its benchmark's identifier pattern
      (LiveCodeBench question_id, MATH-500 unique_id, Hi-ToM s<sample>_o<order>_<prompt type>), so no
      field can carry free text.
   b. Raw scan: no file contains a newline inside a JSON string, a backslash, or any of a list of
      markers of prompts, completions and code (``boxed``, ``def ``, ``return``, ``import``, ``Story``...).
   c. With the public datasets given (optional flags): every id is a public id of its benchmark, and no
      reference answer of MATH-500 or Hi-ToM occurs in any file, neither as a string value nor as a
      substring of the raw text (answers made only of digits, signs and dots are skipped in the raw
      scan, because they occur inside numbers such as temperatures).

It uses the Python standard library and this package's counts reader (onetemp.counts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from onetemp.counts import (ALLOWED_KEYS, DEFAULT_MANIFEST, iter_counts,  # noqa: E402
                         load_manifest, setting_of, sha256_file)

MODELS = {"mistralai/Mistral-7B-Instruct-v0.3", "openai/gpt-oss-20b", "Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B"}
ID_PATTERNS = {
    "livecodebench": re.compile(r"^([0-9]{1,6}|[0-9]{3,6}_[A-Z][0-9]?|(abc|arc|agc)[0-9]{3}_[a-z][0-9]?)$"),
    "math500": re.compile(r"^test/[a-z_]+/[0-9]{1,5}\.json$"),
    "hitom": re.compile(r"^s[0-9]{1,4}_o[0-4]_(CoTP|VP)$"),
}
RAW_MARKERS = ("\\", "\\n", "boxed", "def ", "return", "import ", "print(", "input(", "class ", "Story",
               "story", "Answer", "answer", "Question", "question", "assistant", "<|", "```", "lambda",
               "solution", "prompt", "completion", "stdin", "stdout", "expected_output", "test_case")


class Report:
    def __init__(self):
        self.fails: list[str] = []
        self.passes = 0

    def ok(self, cond: bool, msg: str) -> None:
        if cond:
            self.passes += 1
        else:
            self.fails.append(msg)


def check_file(base: Path, f: dict, rep: Report, cells_out: dict) -> None:
    path = base / f["path"]
    rep.ok(path.exists(), f"{f['path']}: missing")
    if not path.exists():
        return
    rep.ok(sha256_file(path) == f["sha256"], f"{f['path']}: sha256 differs from the manifest")
    rep.ok(path.stat().st_size == f["bytes"], f"{f['path']}: byte size differs")
    raw = path.read_bytes().decode("utf-8")
    for m in RAW_MARKERS:
        rep.ok(m not in raw, f"{f['path']}: raw text contains the marker {m!r}")
    rows = 0
    ids: list[str] = []
    per_q: dict[str, set] = defaultdict(set)
    settings: set = set()
    n_vals: set = set()
    tot_drawn = tot_c = ncell_rows = 0
    strings: set = set()
    last = None
    split = False
    for row in iter_counts(path, strict=True):      # validate_row (shipped schema) on every line
        rows += 1
        rep.ok(set(row) <= ALLOWED_KEYS, f"{f['path']}: extra keys")
        for k, v in row.items():
            if isinstance(v, str):
                strings.add((k, v))
            elif k in ("benchmark", "model", "id"):
                rep.fails.append(f"{f['path']}: {k} is not a string")
        bench = row["benchmark"]
        if row["id"] != last:
            if row["id"] in per_q:
                split = True
            ids.append(row["id"])
            last = row["id"]
        s = setting_of(row)
        if s in per_q[row["id"]]:
            rep.fails.append(f"{f['path']}: duplicate cell {row['id']} @ {s}")
        per_q[row["id"]].add(s)
        settings.add(s)
        n_vals.add(row["n"])
        drawn = row.get("n_cell", row["n"])
        tot_drawn += drawn
        tot_c += row["c"]
        ncell_rows += "n_cell" in row
        cells_out[(f["path"], row["id"], s)] = (row["c"], drawn)
    rep.ok(not split, f"{f['path']}: a question's rows are not adjacent")
    keys_seen = {k for k, _ in strings}
    rep.ok(keys_seen <= {"benchmark", "model", "id"}, f"{f['path']}: strings outside benchmark/model/id")
    rep.ok({v for k, v in strings if k == "benchmark"} == {f["benchmark"]}, f"{f['path']}: benchmark differs")
    rep.ok({v for k, v in strings if k == "model"} == {f["model"]}, f"{f['path']}: model differs")
    rep.ok(f["model"] in MODELS, f"{f['path']}: unknown model {f['model']}")
    pat = ID_PATTERNS[f["benchmark"]]
    bad = [v for k, v in strings if k == "id" and not pat.match(v)]
    rep.ok(not bad, f"{f['path']}: {len(bad)} ids do not match the {f['benchmark']} id pattern, e.g. {bad[:2]}")
    st = sorted(settings)
    rep.ok(all(per_q[q] == settings for q in ids), f"{f['path']}: not a complete question x setting rectangle")
    rep.ok(rows == f["rows"], f"{f['path']}: rows {rows} != manifest {f['rows']}")
    rep.ok(len(ids) == f["questions"], f"{f['path']}: questions {len(ids)} != manifest {f['questions']}")
    rep.ok(st == f["settings"], f"{f['path']}: settings differ from the manifest")
    rep.ok(min(n_vals) == f["n_min"] and max(n_vals) == f["n_max"], f"{f['path']}: n range differs")
    rep.ok(tot_drawn == f["samples_drawn"], f"{f['path']}: samples {tot_drawn} != {f['samples_drawn']}")
    rep.ok(tot_c == f["correct"], f"{f['path']}: correct {tot_c} != {f['correct']}")
    rep.ok(ncell_rows == f["rows_with_n_cell"], f"{f['path']}: rows with n_cell differ")
    f["_ids"] = ids
    f["_strings"] = {v for _, v in strings}


def gold_scan(base: Path, files: list[dict], a, rep: Report) -> dict:
    """Optional part 5c: public ids and reference answers, read from local copies of the public datasets."""
    info: dict = {}
    golds: list[str] = []
    public: dict[str, set] = {}
    if a.math500_test:
        ids, ans = set(), []
        with open(a.math500_test, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                ids.add(r["unique_id"])
                ans.append(str(r["answer"]))
        public["math500"] = ids
        golds += ans
        info["math500_answers"] = len(ans)
    if a.hitom_data:
        d = json.loads(Path(a.hitom_data).read_text(encoding="utf-8"))["data"]
        public["hitom"] = {f"s{x['sample_id']}_o{x['question_order']}_{x['prompting_type']}" for x in d}
        ans = [str(x["answer"]) for x in d]
        golds += ans
        info["hitom_answers"] = len(ans)
    if a.lcb_meta:
        m = json.loads(Path(a.lcb_meta).read_text(encoding="utf-8"))
        qs = m["questions"] if isinstance(m, dict) and "questions" in m else m
        public["livecodebench"] = {q["question_id"] for q in qs}
        info["lcb_public_ids"] = len(public["livecodebench"])
    for f in files:
        if f["benchmark"] in public and "_ids" in f:
            miss = [q for q in f["_ids"] if q not in public[f["benchmark"]]]
            rep.ok(not miss, f"{f['path']}: {len(miss)} ids are not public {f['benchmark']} ids")
    if golds:
        gold_set = {g.strip() for g in golds if g.strip()}
        raw_scan = sorted({g for g in gold_set if len(g) >= 3 and not re.fullmatch(r"[-+0-9.,]+", g)})
        info["distinct_answers"] = len(gold_set)
        info["answers_in_raw_scan"] = len(raw_scan)
        hits_value = 0
        for f in files:
            hv = len(gold_set & f.get("_strings", set()))
            hits_value += hv
            rep.ok(hv == 0, f"{f['path']}: {hv} string values equal a reference answer")
        # the raw scan covers every text file of the folder, not only the counts
        hits_raw, scanned = 0, 0
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix not in (".jsonl", ".json", ".md", ".py", ".txt", ".csv"):
                continue
            scanned += 1
            text = p.read_text(encoding="utf-8", errors="replace")
            found = [g for g in raw_scan if g in text]
            hits_raw += len(found)
            rep.ok(not found, f"{p.relative_to(base)}: reference answers occur in the raw text: {found[:5]}")
        info["string_value_hits"] = hits_value
        info["raw_scan_files"] = scanned
        info["raw_text_hits"] = hits_raw
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--math500-test", default=None, help="MATH-500 test.jsonl (HuggingFaceH4/MATH-500)")
    ap.add_argument("--hitom-data", default=None, help="Hi-ToM_data.json (the Hi-ToM dataset repository)")
    ap.add_argument("--lcb-meta", default=None,
                    help="JSON list (or {'questions': [...]}) of LiveCodeBench records with question_id")
    a = ap.parse_args(argv)
    man = load_manifest(a.manifest)
    base = Path(a.manifest).resolve().parent
    rep = Report()
    cells: dict = {}
    files = man["files"]
    paths = [f["path"] for f in files]
    rep.ok(len(paths) == len(set(paths)), "duplicate file paths in the manifest")
    for f in files:
        check_file(base, f, rep, cells)
    by_path = {f["path"]: f for f in files}
    inc_path = base / "included_ids.json"
    inc = json.loads(inc_path.read_text(encoding="utf-8"))["files"] if inc_path.exists() else {}
    for f in files:
        if "included" not in f:
            continue
        lst = inc.get(f["path"])
        rep.ok(lst is not None, f"{f['path']}: no included list in included_ids.json")
        if lst is None or "_ids" not in f:
            continue
        ids = lst["ids"]
        pos = {q: i for i, q in enumerate(f["_ids"])}
        rep.ok(all(q in pos for q in ids), f"{f['path']}: included ids not in the file")
        rep.ok([pos[q] for q in ids if q in pos] == sorted(pos[q] for q in ids if q in pos),
               f"{f['path']}: included ids are not in the file's order")
        rep.ok(len(ids) == lst["n"] == f["included"]["n"], f"{f['path']}: included n differs")
        rep.ok(hashlib.sha256("\n".join(ids).encode()).hexdigest() == f["included"]["ids_sha256"],
               f"{f['path']}: included ids sha256 differs from the manifest")
    explained = []
    for u in man["uses"]:
        f = by_path.get(u["file"])
        rep.ok(f is not None, f"use {u['label']}: file {u['file']} not in the manifest")
        if f is None or u.get("n_printed") is None:
            continue
        if u.get("population") == "all":
            have = f["questions"]
        elif "included" in f:
            have = f["included"]["n"]
        else:
            rep.ok(False, f"use {u['label']}: {u['file']} records no included questions")
            continue
        if u.get("expected_difference"):
            explained.append(f"{u['label']} ({u['file']}): paper {u['n_printed']}, file {have}: "
                             f"{u['expected_difference']}")
            continue
        rep.ok(have == u["n_printed"],
               f"use {u['label']}: {have} {u.get('population', 'included')} questions, the paper prints {u['n_printed']}")
    for c in man.get("consistency", []):
        fa, fb = by_path.get(c["a"]), by_path.get(c["b"])
        rep.ok(fa is not None and fb is not None, f"consistency: unknown file in {c}")
        if fa is None or fb is None or "_ids" not in fa or "_ids" not in fb:
            continue
        ia, ib = set(fa["_ids"]), set(fb["_ids"])
        if c["relation"] == "a_subset_of_b":
            rep.ok(ia <= ib, f"consistency: {c['a']} is not a subset of {c['b']}")
        elif c["relation"] == "b_subset_of_a":
            rep.ok(ib <= ia, f"consistency: {c['b']} is not a subset of {c['a']}")
        elif c["relation"] == "same_questions":
            rep.ok(ia == ib, f"consistency: {c['a']} and {c['b']} hold different questions")
        shared = ia & ib
        rep.ok(len(shared) == c["shared_questions"], f"consistency: {len(shared)} shared questions, "
                                                     f"manifest says {c['shared_questions']} ({c['a']} / {c['b']})")
        sa = {k[1:]: v for k, v in cells.items() if k[0] == c["a"] and k[1] in shared}
        sb = {k[1:]: v for k, v in cells.items() if k[0] == c["b"] and k[1] in shared}
        common = set(sa) & set(sb)
        if c.get("samples", "same") == "same":
            # the same draws: equal c and equal number of samples drawn
            diff = [k for k in common if sa[k] != sb[k]]
            rep.ok(not diff, f"consistency: {len(diff)} shared cells differ between {c['a']} and {c['b']}")
        else:
            # a's draws are a part of b's (for example the second stage inside screen + second stage):
            # c_a <= c_b, n_a < n_b, and b's extra draws hold at most their own number of correct ones
            bad = [k for k in common if not (sa[k][0] <= sb[k][0] and sa[k][1] < sb[k][1]
                                             and sb[k][0] - sa[k][0] <= sb[k][1] - sa[k][1])]
            rep.ok(not bad, f"consistency: {len(bad)} cells of {c['a']} are not part of {c['b']}")
        rep.ok(len(common) == c["shared_cells"], f"consistency: {len(common)} shared cells, manifest says "
                                                 f"{c['shared_cells']} ({c['a']} / {c['b']})")
    info = gold_scan(base, files, a, rep)
    n_rows = sum(f["rows"] for f in files)
    print(f"files: {len(files)}  rows: {n_rows:,}  figure/table uses: {len(man['uses'])}  "
          f"consistency pairs: {len(man.get('consistency', []))}")
    for e in explained:
        print("  known difference:", e)
    if info:
        print("public-data scan:", json.dumps(info))
    print(f"checks passed: {rep.passes}  failed: {len(rep.fails)}")
    for m in rep.fails[:50]:
        print("  FAIL", m)
    return 0 if not rep.fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
