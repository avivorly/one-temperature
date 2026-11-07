"""Synthetic smoke test of the data tools. No released count, gold answer or completion is used.

    python -m pytest tests/test_data.py

Covers: the converter's two parses and its refusals, the row validator, the question-by-setting table,
min_p files, the check script on a synthetic manifest (pass, then fail after tampering, then the
optional gold scan on a made-up dataset), and the unbiased pass@K estimator against brute force.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from onetemp.data import check as check_counts  # noqa: E402
from onetemp.counts import read_counts, sha256_file, table, validate_row  # noqa: E402
from onetemp.data.convert import records_to_rows, write_rows  # noqa: E402
from onetemp.data.fitfree import unbiased  # noqa: E402

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def synthetic_records():
    """Three made-up LiveCodeBench-like questions on a 4-temperature grid; q2 has uneven n."""
    temps = [0.1, 0.5, 1.0, 2.0]
    return [
        {"task_id": "t-a", "question_idx": 0, "n": 100,
         "results": [{"temperature": t, "n": 100, "passed": p} for t, p in zip(temps, (90, 70, 40, 2))]},
        {"task_id": "t-b", "question_idx": 1, "n": 120,
         "results": [{"temperature": t, "n": nn, "passed": p}
                     for t, nn, p in zip(temps, (100, 100, 120, 120), (0, 3, 9, 1))]},
        {"task_id": "t-c", "question_idx": 2, "n": 100,
         "results": [{"temperature": t, "n": 100, "passed": 0} for t in temps]},
    ]


ID_MAP = {"0": "1873_A", "1": "abc321_a", "2": "2777"}


class ConverterTest(unittest.TestCase):
    def test_two_parses(self):
        rq = records_to_rows(synthetic_records(), "livecodebench", MODEL, "question_n",
                             id_from="question_idx", id_map=ID_MAP)
        rc = records_to_rows(synthetic_records(), "livecodebench", MODEL, "cell_n",
                             id_from="question_idx", id_map=ID_MAP)
        self.assertEqual(len(rq), 12)
        self.assertEqual([r["id"] for r in rq[::4]], ["1873_A", "abc321_a", "2777"])
        b_q = [r for r in rq if r["id"] == "abc321_a"]
        b_c = [r for r in rc if r["id"] == "abc321_a"]
        self.assertEqual([r["n"] for r in b_q], [120] * 4)                  # question-level n everywhere
        self.assertEqual([r.get("n_cell") for r in b_q], [100, 100, None, None])
        self.assertEqual([r["n"] for r in b_c], [100, 100, 120, 120])       # each cell's own n
        self.assertEqual([r["c"] for r in b_q], [r["c"] for r in b_c])
        self.assertFalse(any("n_cell" in r for r in rc))
        for r in rq + rc:
            validate_row(r, strict=True)

    def test_keep_and_refusals(self):
        rows = records_to_rows(synthetic_records(), "livecodebench", MODEL, "cell_n", id_from="question_idx",
                               id_map=ID_MAP, keep=["2777", "1873_A"])
        self.assertEqual(list(dict.fromkeys(r["id"] for r in rows)), ["1873_A", "2777"])  # input order kept
        bad = synthetic_records()
        bad[0]["results"][0]["passed"] = 101
        with self.assertRaises(ValueError):
            records_to_rows(bad, "livecodebench", MODEL, "cell_n")
        dup = synthetic_records()
        dup[1]["task_id"] = "t-a"
        with self.assertRaises(ValueError):
            records_to_rows(dup, "livecodebench", MODEL, "cell_n")
        with self.assertRaises(ValueError):
            records_to_rows(synthetic_records(), "livecodebench", MODEL, "cell_n", keep=["not-there"])

    def test_min_p(self):
        recs = [{"task_id": "s1_o0_CoTP", "n": 50,
                 "results": [{"min_p": m, "temperature": 2.0, "n": 50, "passed": p}
                             for m, p in ((0.0056, 5), (0.178, 1))]}]
        rows = records_to_rows(recs, "hitom", "Qwen/Qwen3-0.6B", "cell_n")
        t = table(rows)
        self.assertEqual(t.axis, "min_p")
        self.assertEqual(t.fixed, {"temperature": 2.0})
        self.assertEqual(t.settings, [0.0056, 0.178])
        recs[0]["results"][1]["temperature"] = 1.5
        with self.assertRaises(ValueError):
            records_to_rows(recs, "hitom", "Qwen/Qwen3-0.6B", "cell_n")


class ValidatorTest(unittest.TestCase):
    def good(self):
        return {"benchmark": "math500", "model": MODEL, "id": "test/algebra/1.json", "temperature": 0.7,
                "min_p": None, "n": 10, "c": 3}

    def test_rejects(self):
        validate_row(self.good(), strict=True)
        for k, v in (("c", 11), ("n", 0), ("benchmark", "gsm8k"), ("min_p", 1.5), ("temperature", -1.0),
                     ("c", True), ("id", "")):
            r = self.good()
            r[k] = v
            with self.assertRaises(ValueError, msg=k):
                validate_row(r, strict=True)
        r = self.good()
        r["completion"] = "x"
        with self.assertRaises(ValueError):
            validate_row(r, strict=True)
        r = self.good()
        r["n_cell"] = 10            # n_cell must differ from (be below) n
        with self.assertRaises(ValueError):
            validate_row(r, strict=True)

    def test_default_mode_accepts_written_fields(self):
        """Outside the shipped files, the graders and the profile fit add documented fields."""
        r = dict(self.good(), c_strict=2, p_fit=0.25, peak_pct=31.0)
        validate_row(r)
        with self.assertRaises(ValueError):
            validate_row(r, strict=True)
        with self.assertRaises(ValueError):
            validate_row(dict(self.good(), c_strict=4))          # more exact-layer passes than passes
        with self.assertRaises(ValueError):
            validate_row(dict(self.good(), tau=0.7))            # tau only in a min_p row
        validate_row(dict(self.good(), min_p=0.1, tau=0.7))


class CheckScriptTest(unittest.TestCase):
    def build(self, d: Path):
        rows = records_to_rows(synthetic_records(), "livecodebench", MODEL, "question_n",
                               id_from="question_idx", id_map=ID_MAP)
        rel = "counts/livecodebench/demo__temperature__question_n.jsonl"
        (d / "counts/livecodebench").mkdir(parents=True)
        write_rows(rows, d / rel)
        t = table(read_counts(d / rel))
        # made-up inclusion rule for the test: questions with at least one correct sample
        inc = [q for q in t.ids if sum(t.c[t.ids.index(q)]) > 0]
        f = {"path": rel, "benchmark": "livecodebench", "model": MODEL, "rows": len(rows),
             "questions": len(t.ids), "settings": t.settings, "settings_count": len(t.settings),
             "n_min": 100, "n_max": 120, "rows_with_n_cell": 2,
             "samples_drawn": sum(r.get("n_cell", r["n"]) for r in rows), "correct": sum(r["c"] for r in rows),
             "included": {"n": len(inc), "rule": "at least one correct sample",
                          "ids_sha256": hashlib.sha256("\n".join(inc).encode()).hexdigest()},
             "bytes": (d / rel).stat().st_size, "sha256": sha256_file(d / rel)}
        man = {"files": [f], "labels": {}, "consistency": [],
               "uses": [{"label": "fig:demo", "file": rel, "population": "all", "n_printed": 3},
                        {"label": "fig:demo2", "file": rel, "population": "included", "n_printed": 2}]}
        (d / "manifest.json").write_text(json.dumps(man))
        (d / "included_ids.json").write_text(json.dumps({"files": {rel: {"n": len(inc), "ids": inc}}}))
        return rel

    def test_pass_then_fail(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as pub:
            d = Path(tmp)
            rel = self.build(d)
            self.assertEqual(check_counts.main(["--manifest", str(d / "manifest.json")]), 0)
            # a made-up public dataset (kept outside the scanned folder): its answer occurs nowhere in it
            fake = Path(pub) / "fake_math.jsonl"
            fake.write_text(json.dumps({"unique_id": "test/algebra/1.json", "answer": "\\frac{9973}{104729}"}) + "\n")
            self.assertEqual(check_counts.main(["--manifest", str(d / "manifest.json"),
                                                "--math500-test", str(fake)]), 0)
            # the same answer planted in a text file of the folder is found by the raw scan
            (d / "notes.md").write_text("an answer: \\frac{9973}{104729}\n")
            self.assertEqual(check_counts.main(["--manifest", str(d / "manifest.json"),
                                                "--math500-test", str(fake)]), 1)
            (d / "notes.md").unlink()
            # tampering with one count changes the sha256 and the totals
            text = (d / rel).read_text().replace('"c":90', '"c":91', 1)
            (d / rel).write_text(text)
            self.assertEqual(check_counts.main(["--manifest", str(d / "manifest.json")]), 1)

    def test_free_text_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rel = self.build(d)
            text = (d / rel).read_text().replace('"id":"2777"', '"id":"x = \\\\boxed{3}"')
            (d / rel).write_text(text)
            self.assertEqual(check_counts.main(["--manifest", str(d / "manifest.json")]), 1)


class EstimatorTest(unittest.TestCase):
    def test_unbiased_matches_brute_force(self):
        n = 7
        for c in range(n + 1):
            draws = [1] * c + [0] * (n - c)
            for K in range(1, n + 1):
                subsets = list(itertools.combinations(range(n), K))
                brute = sum(1 for s in subsets if any(draws[i] for i in s)) / len(subsets)
                self.assertAlmostEqual(unbiased(n, c, K), brute, places=12)
        self.assertAlmostEqual(unbiased(1500, 30, 1), 0.02, places=15)
        self.assertEqual(unbiased(10, 9, 2), 1.0)
        self.assertTrue(math.isclose(unbiased(1000, 1, 400), 0.4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
