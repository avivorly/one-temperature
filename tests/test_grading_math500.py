"""Tests of the MATH-500 grader on synthetic cases only (no reference answer of the benchmark is used).

    python -m pytest tests/test_grading_math500.py        (from the repository root)

Each case is (completion, reference, expected bucket, expected verdict, expected rule), labelled by hand from the
rules in the docstring of onetemp/grading/math500.py.
"""
import gzip
import hashlib
import json
import os
import sys
import tempfile
import unittest

import importlib.util  # noqa: E402

if importlib.util.find_spec("sympy") is None:
    raise unittest.SkipTest("sympy is not installed: pip install -e \".[dev]\"")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from onetemp.grading import completions_io  # noqa: E402
from onetemp.grading import math500 as G  # noqa: E402

B = "\\boxed"


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


CASES = [
    # extraction: last span, no-boxed bucket, empty box, substring guard
    ("The answer is 4137.", "4137", "no_boxed", False, None),
    ("Thus $" + B + "{}$.", "4137", "empty_after_normalize", False, None),
    ("So " + B + "{91}.", "913", "parsed_wrong", False, None),
    ("So " + B + "{913}.", "913", "exact_match", True, None),
    ("First " + B + "{912}, corrected: " + B + "{913}", "913", "exact_match", True, None),
    (B + "{913} ... no, actually " + B + "{912}", "913", "parsed_wrong", False, None),
    ("\\fbox{913}", "913", "no_boxed", False, None),
    # normalisation (exact layer)
    (B + "{\\dfrac{3}{7}}", "\\frac{3}{7}", "exact_match", True, None),
    (B + "{\\frac37}", "\\frac{3}{7}", "exact_match", True, None),
    (B + "{\\$4137}", "4137", "exact_match", True, None),
    (B + "{613\\%}", "613", "exact_match", True, None),
    (B + "{4137.}", "4137", "exact_match", True, None),
    (B + "{811.0}", "811", "exact_match", True, None),
    (B + "{127^\\circ}", "127", "exact_match", True, None),
    (B + "{\\left(2, -5\\right)}", "(2, -5)", "exact_match", True, None),
    (B + "{[0, 5)}", "[0,5)", "exact_match", True, None),
    # sympy equivalence (v1 layer)
    (B + "{0.453125}", "\\frac{29}{64}", "equivalence_match", True, None),
    (B + "{(x+1)^2}", "x^2+2x+1", "equivalence_match", True, None),
    (B + "{\\sqrt{8}}", "2\\sqrt{2}", "equivalence_match", True, None),
    (B + "{y+1}", "x+1", "parsed_wrong", False, None),
    # R1 units and degrees
    (B + "{923\\text{ cm}}", "923", "equivalence_match", True, "R1"),
    (B + "{127°}", "127", "equivalence_match", True, "R1"),
    # R2 prefixes
    (B + "{x = 619}", "619", "equivalence_match", True, "R2"),
    (B + "{y = 619}", "x = 619", "parsed_wrong", False, "R2"),
    # R3 lists and sets are multisets
    (B + "{4, 1}", "1, 4", "equivalence_match", True, "R3"),
    (B + "{\\{4, 1\\}}", "\\{1, 4\\}", "equivalence_match", True, "R3"),
    (B + "{1, 4, 4}", "1, 4", "parsed_wrong", False, "R3"),
    # R4 plus-minus
    (B + "{3 + \\sqrt{2}, 3 - \\sqrt{2}}", "3 \\pm \\sqrt{2}", "equivalence_match", True, "R4"),
    (B + "{\\pm 619}", "619", "parsed_wrong", False, "R4"),
    # R5 fail closed, and the layout whitelist
    (B + "{\\ln 619}", "619", "unparseable", False, "R5"),
    (B + "{Monday}", "\\text{Monday}", "unparseable", False, "R5"),
    (B + "{\\displaystyle \\frac{3}{7}}", "\\frac{3}{7}", "equivalence_match", True, "R5"),
    # MX mixed numbers
    (B + "{\\frac{7}{3}}", "2\\frac{1}{3}", "equivalence_match", True, "MX"),
    (B + "{4\\frac{1}{5}}", "\\frac{4}{5}", "parsed_wrong", False, None),
    # TS thousands
    (B + "{12,345}", "12345", "equivalence_match", True, "TS"),
    (B + "{3,100}", "3, 100", "parsed_wrong", False, "TS"),
    # BE braced exponents, PX prefixes, SS singletons, TP tuples
    (B + "{x^{2}+1}", "x^2+1", "equivalence_match", True, "BE"),
    (B + "{z_1 = 811}", "811", "equivalence_match", True, "PX"),
    (B + "{\\{811\\}}", "811", "equivalence_match", True, "SS"),
    (B + "{2, -5}", "(2, -5)", "equivalence_match", True, "TP"),
    (B + "{-5, 2}", "(2, -5)", "parsed_wrong", False, "TP"),
]


class TestVerdicts(unittest.TestCase):
    def test_hand_labelled_cases(self):
        for text, gold, bucket, lenient, rule in CASES:
            with self.subTest(text=text, gold=gold):
                r = G.grade_one(text, gold)
                self.assertEqual(r["bucket"], bucket)
                self.assertIs(r["lenient"], lenient)
                if lenient or rule is not None:
                    self.assertEqual(r["rule"], rule)

    def test_strict_is_the_exact_layer(self):
        self.assertTrue(G.grade_one(B + "{913}", "913")["strict"])
        self.assertFalse(G.grade_one(B + "{0.453125}", "\\frac{29}{64}")["strict"])
        # TS refuses one kind of exact match: "3,100" (a number) is not the list "3, 100"; strict keeps v1's verdict
        r = G.grade_one(B + "{3,100}", "3, 100")
        self.assertFalse(r["lenient"])
        self.assertTrue(r["strict"])

    def test_extract_sees_only_the_completion(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(G.extract_boxed).parameters), ("completion",))
        self.assertEqual(G.extract_boxed("a " + B + "{1} b " + B + "{\\frac{2}{3}}"), ("\\frac{2}{3}", 2))
        self.assertEqual(G.extract_boxed("no box"), (None, 0))
        self.assertEqual(G.extract_boxed(None), (None, 0))

    def test_timeout_fails_closed(self):
        old = G.TIMEOUT_S
        G.TIMEOUT_S = 1e-6
        try:
            r = G.grade_one(B + "{(x+1)^2}", "x^2+2x+1")
        finally:
            G.TIMEOUT_S = old
        self.assertEqual(r["bucket"], "grader_timeout")
        self.assertFalse(r["lenient"])

    def test_no_eval_of_model_output(self):
        # an expression that would run code under eval/sympify is refused by the character whitelist
        r = G.grade_one(B + "{__import__('os').getcwd()}", "811")
        self.assertFalse(r["lenient"])


class TestCellGrading(unittest.TestCase):
    def test_instance_weighted_and_cached(self):
        texts = [B + "{\\frac37}"] * 5 + [B + "{3/7}"] * 3 + ["no answer"] * 2
        r = G.grade_texts("\\frac{3}{7}", texts)
        self.assertEqual(r["pass"], "1" * 8 + "00")
        self.assertEqual(r["strict"], "1" * 5 + "0" * 5)
        self.assertEqual(r["bucket"], "e" * 5 + "q" * 3 + "nn")
        self.assertEqual(r["distinct_texts"], 3)
        self.assertEqual(r["cache_mismatches"], 0)

    def test_cache_recheck_runs(self):
        # 200 distinct texts with the same last span: the re-check fires on every 97th cached one
        texts = [f"attempt {i}: " + B + "{811}" for i in range(200)]
        r = G.grade_texts("811", texts)
        self.assertEqual(r["pass"], "1" * 200)
        self.assertEqual(r["distinct_spans"], 1)
        self.assertGreaterEqual(r["cache_checks"], 2)
        self.assertEqual(r["cache_mismatches"], 0)


class TestFrozenCode(unittest.TestCase):
    def test_frozen_block_is_unchanged(self):
        """The grading code between the two markers must stay byte-identical to the released version."""
        src = _read(G.__file__)
        a = src.index("# THE FROZEN GRADER (v2.5)")
        b = src.index("# END OF THE FROZEN GRADER")
        block = src[a:b]
        self.assertEqual(hashlib.sha256(block.encode("utf-8")).hexdigest(), FROZEN_BLOCK_SHA256)
        self.assertEqual(G.GRADER_VERSION, "v2.5")
        self.assertEqual(G.TIMEOUT_S, 2.0)

    def test_sympy_is_the_papers(self):
        import mpmath
        import sympy
        if (sympy.__version__, mpmath.__version__) != G.SYMPY_MPMATH:
            self.skipTest(f"sympy {sympy.__version__} / mpmath {mpmath.__version__}; the paper used "
                          f"{G.SYMPY_MPMATH} (the verdict tests above still ran)")
        from sympy.external.gmpy import GROUND_TYPES
        self.assertEqual(GROUND_TYPES, "python")


FROZEN_BLOCK_SHA256 = "cdfd4fcb04855213c6d7544c1da7b2a901e1820a0409139967bbea0f5e8b5b0f"


class TestCommandLine(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            data = os.path.join(d, "test.jsonl")
            with open(data, "w") as fh:
                for uid, ans in (("synthetic/1.json", "\\frac{29}{64}"), ("synthetic/2.json", "(2, -5)"),
                                 ("synthetic/3.json", "923")):
                    fh.write(json.dumps({"unique_id": uid, "problem": "...", "answer": ans}) + "\n")
            runs = os.path.join(d, "runs")
            os.makedirs(os.path.join(runs, "gpu1"))
            m = "org/synthetic-model"
            lines = [
                {"benchmark": "math500", "model": m, "id": "synthetic/1.json", "temperature": 0.7, "min_p": None,
                 "completions": [B + "{0.453125}", B + "{\\frac{29}{64}}", "none", B + "{2}"],
                 "finish_reasons": ["stop", "stop", "length", "stop"]},
                {"model": m, "id": "synthetic/2.json", "temperature": 0.7,
                 "completions": [B + "{2, -5}", B + "{(2,-5)}"]},
                {"model": m, "id": "synthetic/3.json", "temperature": 1.0, "min_p": 0.1,
                 "completions": [B + "{923\\text{ cm}}"] * 3},
            ]
            with open(os.path.join(runs, "a.jsonl"), "w") as fh:
                for r in lines:
                    fh.write(json.dumps(r) + "\n")
            # the same cell continued in a second, gzipped shard: its counts are summed
            with gzip.open(os.path.join(runs, "gpu1", "b.jsonl.gz"), "wt") as fh:
                fh.write(json.dumps({"model": m, "id": "synthetic/1.json", "temperature": 0.7,
                                     "completions": [B + "{29/64}"]}) + "\n")
            out, summ = os.path.join(d, "counts.jsonl"), os.path.join(d, "summary.json")
            rc = G.main(["--inputs", runs, "--out", out, "--summary", summ, "--dataset", data, "--no-verify-data",
                         "--any-sympy", "--jobs", "2"])
            self.assertEqual(rc, 0)
            rows = [json.loads(x) for x in _read(out).splitlines()]
            for r in rows:
                self.assertEqual(completions_io.check_counts_row(r), [])
            got = {(r["id"], r["temperature"], r["min_p"]): (r["n"], r["c"], r["c_strict"]) for r in rows}
            self.assertEqual(got, {("synthetic/1.json", 0.7, None): (5, 3, 1),
                                   ("synthetic/2.json", 0.7, None): (2, 2, 1),
                                   ("synthetic/3.json", 1.0, 0.1): (3, 3, 0)})
            s = json.loads(_read(summ))
            self.assertEqual(s["grader_version"], "v2.5")
            per = {(x["temperature"], x["min_p"]): x for x in s["by_setting"]}
            self.assertEqual(per[(0.7, None)]["truncated"], 1)
            self.assertEqual(per[(0.7, None)]["buckets"]["no_boxed"], 1)
            # refuses to overwrite, and refuses an id that is not in the dataset
            with self.assertRaises(SystemExit):
                G.main(["--inputs", runs, "--out", out, "--dataset", data, "--no-verify-data", "--any-sympy"])
            bad = os.path.join(d, "bad.jsonl")
            with open(bad, "w") as fh:
                fh.write(json.dumps({"model": m, "id": "synthetic/9.json", "temperature": 0.7,
                                     "completions": ["x"]}) + "\n")
            with self.assertRaises(SystemExit):
                G.main(["--inputs", bad, "--out", os.path.join(d, "c2.jsonl"), "--dataset", data,
                        "--no-verify-data", "--any-sympy"])


if __name__ == "__main__":
    unittest.main()
