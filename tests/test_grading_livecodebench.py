"""Tests of the LiveCodeBench grader on toy problems written for this test (no LiveCodeBench test case is used).

    python -m pytest tests/test_grading_livecodebench.py        (from the repository root; Linux, needs numpy)

These tests execute small, known-harmless programs through the vendored LiveCodeBench harness.
"""
import base64
import json
import os
import pickle
import sys
import tempfile
import unittest
import zlib

import importlib.util  # noqa: E402

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("numpy is not installed: pip install -e .")
if not sys.platform.startswith("linux"):
    raise unittest.SkipTest("the LiveCodeBench harness needs Linux (fork and SIGALRM)")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from onetemp.grading import completions_io  # noqa: E402
from onetemp.grading import livecodebench as L  # noqa: E402


def _compress(tests):
    """LiveCodeBench's storage of private tests: base64(zlib(pickle(json string)))."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode("utf-8")


def _row(qid, public, private, func_name=None, compressed=True, platform="atcoder"):
    meta = {"func_name": func_name} if func_name else {}
    return {"question_title": f"toy {qid}", "question_content": "(toy statement)", "platform": platform,
            "question_id": qid, "contest_id": "toy", "contest_date": "2025-01-01T00:00:00", "starter_code": "",
            "difficulty": "easy", "public_test_cases": json.dumps(public),
            "private_test_cases": _compress(private) if compressed else json.dumps(private),
            "metadata": json.dumps(meta)}


SUM_PUBLIC = [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
SUM_PRIVATE = [{"input": "10 -4\n", "output": "6\n", "testtype": "stdin"},
               {"input": "0 0\n", "output": "0\n", "testtype": "stdin"}]
ADD_PUBLIC = [{"input": "1\n2", "output": "3", "testtype": "functional"}]
ADD_PRIVATE = [{"input": "-5\n5", "output": "0", "testtype": "functional"}]
PAIR_PUBLIC = [{"input": "4", "output": "[2, 2]", "testtype": "functional"}]


class TestExtraction(unittest.TestCase):
    def test_variants(self):
        cases = [  # (raw completion text after the ```python priming, v1 result, v2 result)
            ("print(1)\n", "print(1)", "print(1)"),
            ("print(1)\n```", "print(1)\n```", "print(1)"),                 # only v2 drops a closing fence
            ("```python\nprint(2)\n```\ntext", "print(2)", "print(2)"),
            ("```\nprint(4)\n```", "print(4)", "print(4)"),
            ("x = 1\n```\nmore ```python\nprint(3)\n```", "print(3)", "print(3)"),
        ]
        for raw, v1, v2 in cases:
            self.assertEqual(L.extract_code_v1(raw), v1)
            self.assertEqual(L.extract_code_v2(raw), v2)

    def test_variant_by_model(self):
        self.assertEqual(L.extractor_for("mistralai/Mistral-7B-Instruct-v0.3"), "v1")
        self.assertEqual(L.extractor_for("Qwen/Qwen3-0.6B"), "v1")
        self.assertEqual(L.extractor_for("Qwen/Qwen3-8B"), "v2")
        self.assertEqual(L.extractor_for("openai/gpt-oss-20b"), "v2")
        self.assertEqual(L.extractor_for("org/other", "v1"), "v1")
        # a local model directory or another spelling of the same model gets the same variant
        self.assertEqual(L.extractor_for("/models/Mistral-7B-Instruct-v0.3/"), "v1")
        self.assertEqual(L.extractor_for("qwen/qwen3-0.6b"), "v1")
        self.assertEqual(L.extractor_for("org/unknown-model"), "v2")


class TestProblems(unittest.TestCase):
    def test_evaluation_sample_decodes_private_tests(self):
        for compressed in (True, False):
            sample, info = L.evaluation_sample(_row("toy_a", SUM_PUBLIC, SUM_PRIVATE, compressed=compressed))
            io = json.loads(sample["input_output"])
            self.assertEqual(io["inputs"], ["1 2\n", "10 -4\n", "0 0\n"])         # public first, then private
            self.assertEqual(io["outputs"], ["3\n", "6\n", "0\n"])
            self.assertIsNone(io["fn_name"])
            self.assertEqual((info["n_tests"], info["n_public"], info["call_based"]), (3, 1, False))
        sample, info = L.evaluation_sample(_row("toy_b", ADD_PUBLIC, ADD_PRIVATE, func_name="addTwo",
                                                platform="leetcode"))
        self.assertEqual(json.loads(sample["input_output"])["fn_name"], "addTwo")
        self.assertTrue(info["call_based"])

    def test_index_and_find(self):
        with tempfile.TemporaryDirectory() as d:
            _write_lcb_dir(d)
            data = L.LcbData(lcb_dir=d, cache_dir=os.path.join(d, "cache"), verify=False)
            v, rowno, row = data.find("toy_b")
            self.assertEqual((v, rowno, row["question_id"]), ("v3", 1, "toy_b"))
            v, rowno, row = data.find("toy_a", "v1")
            self.assertEqual((v, rowno), ("v1", 0))
            with self.assertRaises(SystemExit):
                data.find("missing")
            # a second instance reads the cached index
            data2 = L.LcbData(lcb_dir=d, cache_dir=os.path.join(d, "cache"), verify=False)
            self.assertEqual(data2.find("toy_c")[:2], ("v3", 2))

    def test_unverified_file_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _write_lcb_dir(d)
            with self.assertRaises(SystemExit):
                L.LcbData(lcb_dir=d, cache_dir=os.path.join(d, "cache"), verify=True).path("v1")


def _write_lcb_dir(d):
    rows = {"test.jsonl": [_row("toy_a", SUM_PUBLIC, SUM_PRIVATE)],
            "test3.jsonl": [_row("toy_z", SUM_PUBLIC, SUM_PRIVATE),
                            _row("toy_b", ADD_PUBLIC, ADD_PRIVATE, func_name="addTwo", platform="leetcode"),
                            _row("toy_c", PAIR_PUBLIC, [], func_name="halves", platform="leetcode")]}
    for v, (name, _, _) in L.LCB_FILES.items():
        with open(os.path.join(d, name), "w") as fh:
            for r in rows.get(name, []):
                fh.write(json.dumps(r) + "\n")


SUM_OK = "a, b = map(int, input().split())\nprint(a + b)"
SUM_MAIN = "def main():\n    a, b = map(int, input().split())\n    print(a + b)\n\nif __name__ == '__main__':\n    main()"
SUM_WRONG = "a, b = map(int, input().split())\nprint(a - b)"
SUM_CRASH = "a, b = map(int, input().split())\nprint(a // 0)"
SUM_SYNTAX = "def (:\n    pass"
SUM_LOOP = "while True:\n    pass"
SUM_FLOAT = "a, b = map(int, input().split())\nprint(f'{a + b:.3f}')"     # "3.000" equals "3" as Decimals
ADD_OK = "class Solution:\n    def addTwo(self, a: int, b: int) -> int:\n        return a + b"
ADD_WRONG = "class Solution:\n    def addTwo(self, a: int, b: int) -> int:\n        return a * b"
PAIR_TUPLE = "class Solution:\n    def halves(self, n):\n        return (n // 2, n // 2)"   # a tuple counts as a list


class TestExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stdin, _ = L.evaluation_sample(_row("toy_a", SUM_PUBLIC, SUM_PRIVATE))
        cls.call, _ = L.evaluation_sample(_row("toy_b", ADD_PUBLIC, ADD_PRIVATE, func_name="addTwo",
                                               platform="leetcode"))
        cls.pair, _ = L.evaluation_sample(_row("toy_c", PAIR_PUBLIC, [], func_name="halves", platform="leetcode"))
        cls.cwd = os.getcwd()
        cls.tmp = tempfile.TemporaryDirectory()
        os.chdir(cls.tmp.name)                       # solutions run here

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls.tmp.cleanup()

    def test_stdin_programs(self):
        got = L.run_codes(self.stdin, [SUM_OK, SUM_MAIN, SUM_WRONG, SUM_CRASH, SUM_SYNTAX, SUM_FLOAT],
                          workers=3, timeout=2)
        self.assertEqual(got, [(True, "passed"), (True, "passed"), (False, "wrong_answer"),
                               (False, "runtime_or_compile_error"), (False, "runtime_or_compile_error"),
                               (True, "passed")])

    def test_time_limit(self):
        self.assertEqual(L.run_codes(self.stdin, [SUM_LOOP], workers=1, timeout=1), [(False, "time_limit")])

    def test_call_based_programs(self):
        got = L.run_codes(self.call, [ADD_OK, ADD_WRONG], workers=2, timeout=2)
        self.assertEqual(got, [(True, "passed"), (False, "wrong_answer")])
        self.assertEqual(L.run_codes(self.pair, [PAIR_TUPLE], workers=1, timeout=2), [(True, "passed")])

    def test_order_is_kept_across_chunks(self):
        codes = [SUM_OK, SUM_WRONG, SUM_OK, SUM_OK, SUM_WRONG]
        got = L.run_codes(self.stdin, codes, workers=2, timeout=2, chunk=2)
        self.assertEqual([p for p, _ in got], [True, False, True, True, False])

    def test_global_timeout(self):
        # a loop inside one C call never returns to the interpreter, so the per-test alarm cannot fire; the
        # global limit, (1 + 1) x 1 test + 5 = 7 s, kills the process (about 7 s)
        hang = "class Solution:\n    def halves(self, n):\n        return sum(range(10 ** 15))"
        self.assertEqual(L.run_codes(self.pair, [hang], workers=1, timeout=1),
                         [(False, "harness_error_or_global_timeout")])


class TestCommandLine(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            lcb = os.path.join(d, "lcb")
            os.makedirs(lcb)
            _write_lcb_dir(lcb)
            runs = os.path.join(d, "runs")
            os.makedirs(runs)
            fence = "\n```"
            lines = [
                # Mistral: the v1 extraction keeps a closing fence, so that sample fails (as in the paper's runs)
                {"benchmark": "livecodebench", "model": "mistralai/Mistral-7B-Instruct-v0.3", "id": "toy_a",
                 "temperature": 0.2, "completions": [SUM_OK, SUM_OK + fence, SUM_WRONG, SUM_OK],
                 "finish_reasons": ["stop", "stop", "stop", "length"]},
                # Qwen3-8B: the v2 extraction drops it
                {"model": "Qwen/Qwen3-8B", "id": "toy_a", "temperature": 0.2, "lcb_version": "v1", "lcb_row": 0,
                 "completions": [SUM_OK, SUM_OK + fence]},
                {"model": "Qwen/Qwen3-8B", "id": "toy_b", "temperature": 1.3, "min_p": 0.1,
                 "completions": [ADD_OK, ADD_WRONG, ADD_OK]},
            ]
            with open(os.path.join(runs, "cells.jsonl"), "w") as fh:
                for r in lines:
                    fh.write(json.dumps(r) + "\n")
            want = {("mistralai/Mistral-7B-Instruct-v0.3", "toy_a", 0.2, None): (4, 2),
                    ("Qwen/Qwen3-8B", "toy_a", 0.2, None): (2, 2),
                    ("Qwen/Qwen3-8B", "toy_b", 1.3, 0.1): (3, 2)}
            for extra in ([], ["--dedup"]):
                out = os.path.join(d, f"counts{len(extra)}.jsonl")
                summ = os.path.join(d, f"summary{len(extra)}.json")
                rc = L.main(["--allow-code-execution", "--inputs", runs, "--out", out, "--summary", summ,
                             "--lcb-dir", lcb, "--cache-dir", os.path.join(d, "cache"), "--no-verify-data",
                             "--workers", "2", "--timeout", "2"] + extra)
                self.assertEqual(rc, 0)
                with open(out) as fh:
                    rows = [json.loads(x) for x in fh]
                for r in rows:
                    self.assertEqual(completions_io.check_counts_row(r), [])
                got = {(r["model"], r["id"], r["temperature"], r["min_p"]): (r["n"], r["c"]) for r in rows}
                self.assertEqual(got, want)
                with open(summ) as fh:
                    s = json.load(fh)
                mis = [x for x in s["by_setting"] if x["model"].startswith("mistralai")][0]
                self.assertEqual(mis["buckets"], {"passed": 2, "runtime_or_compile_error": 1, "wrong_answer": 1})
                self.assertEqual((mis["truncated"], mis["truncated_and_correct"]), (1, 1))
            # relative paths work although the solutions run in another directory
            cwd = os.getcwd()
            os.chdir(d)
            try:
                rc = L.main(["--allow-code-execution", "--inputs", "runs", "--out", "rel.jsonl", "--lcb-dir", "lcb",
                             "--cache-dir", "cache_rel", "--no-verify-data", "--workers", "2", "--timeout", "2",
                             "--dedup"])
            finally:
                os.chdir(cwd)
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "rel.jsonl")) as fh:
                self.assertEqual(len(fh.readlines()), 3)
            self.assertTrue(os.path.isdir(os.path.join(d, "cache_rel")))
            # an existing output is refused before any work, not after it
            with self.assertRaises(SystemExit) as cm:
                L.main(["--allow-code-execution", "--inputs", runs, "--out", os.path.join(d, "rel.jsonl"),
                        "--lcb-dir", os.path.join(d, "does-not-exist")])
            self.assertIn("refusing to overwrite", str(cm.exception))
            # code execution needs the explicit flag
            with self.assertRaises(SystemExit):
                L.main(["--inputs", runs, "--out", os.path.join(d, "x.jsonl"), "--lcb-dir", lcb])
            # a wrong row hint is refused
            bad = os.path.join(d, "bad.jsonl")
            with open(bad, "w") as fh:
                fh.write(json.dumps({"model": "m/x", "id": "toy_b", "temperature": 1.0, "lcb_row": 0,
                                     "completions": [ADD_OK]}) + "\n")
            with self.assertRaises(SystemExit):
                L.main(["--allow-code-execution", "--inputs", bad, "--out", os.path.join(d, "y.jsonl"),
                        "--lcb-dir", lcb, "--cache-dir", os.path.join(d, "cache"), "--no-verify-data"])


if __name__ == "__main__":
    unittest.main()
