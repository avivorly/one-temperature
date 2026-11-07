"""Tests of the shared input/output: reading completions files and writing counts.jsonl (synthetic data only).

    python -m pytest tests/test_grading_completions_io.py        (from the repository root)
"""
import gzip
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from onetemp.grading import completions_io as io_  # noqa: E402


def _write(path, rows, gz=False):
    opener = gzip.open if gz else open
    with opener(path, "wt") as fh:
        for r in rows:
            fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")


class TestReading(unittest.TestCase):
    def test_reads_plain_and_gzip_recursively(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "sub"))
            _write(os.path.join(d, "a.jsonl"), [{"model": "m/a", "id": "q1", "temperature": 0.5,
                                                 "completions": ["x", "y"]}, ""])
            _write(os.path.join(d, "sub", "b.jsonl.gz"), [{"id": "q2", "temperature": 1, "min_p": 0.05,
                                                           "provenance": {"model": "m/b"},
                                                           "completions": ["z"], "finish_reasons": ["length"]}],
                   gz=True)
            _write(os.path.join(d, "ignored.txt"), ["not read"])
            cells = list(io_.iter_cells([d], "hitom"))
            self.assertEqual([(c["model"], c["id"], c["temperature"], c["min_p"]) for c in cells],
                             [("m/a", "q1", 0.5, None), ("m/b", "q2", 1.0, 0.05)])
            self.assertEqual(cells[1]["finish_reasons"], ["length"])
            # --model fills a missing model id, and never overrides the one a line names
            self.assertEqual({c["model"] for c in io_.iter_cells([d], "hitom", model="m/z")}, {"m/a", "m/b"})
            _write(os.path.join(d, "c.jsonl"), [{"id": "q3", "temperature": 0.5, "completions": ["w"]}])
            self.assertEqual([c["model"] for c in io_.iter_cells([d], "hitom", model="m/z")], ["m/a", "m/z", "m/b"])

    def test_only_listed_keys_reach_a_grader(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jsonl")
            _write(p, [{"model": "m", "id": "q", "temperature": 0.1, "completions": ["c"],
                        "prompt": "a prompt that must not be read", "answer": "an answer that must not be read"}])
            cell = next(io_.iter_cells([p], "math500"))
            self.assertNotIn("prompt", cell)
            self.assertNotIn("answer", cell)

    def test_rejects_bad_lines(self):
        bad = [
            {"model": "m", "temperature": 0.1, "completions": []},                      # no id
            {"model": "m", "id": "q", "completions": []},                               # no temperature
            {"model": "m", "id": "q", "temperature": "hot", "completions": []},
            {"model": "m", "id": "q", "temperature": float("nan"), "completions": []},
            {"model": "m", "id": "q", "temperature": 0.1, "completions": "x"},
            {"model": "m", "id": "q", "temperature": 0.1, "completions": ["a"], "finish_reasons": []},
            {"id": "q", "temperature": 0.1, "completions": ["a"]},                       # no model
            {"benchmark": "hitom", "model": "m", "id": "q", "temperature": 0.1, "completions": ["a"]},
            {"model": "m", "id": "q", "temperature": 0.1, "n": 3, "completions": ["a"]},     # n disagrees
            "not json",
        ]
        with tempfile.TemporaryDirectory() as d:
            for i, row in enumerate(bad):
                p = os.path.join(d, f"bad{i}.jsonl")
                if isinstance(row, dict):
                    with open(p, "w") as fh:
                        fh.write(json.dumps(row, allow_nan=True) + "\n")
                else:
                    _write(p, [row])
                with self.subTest(i=i), self.assertRaises(SystemExit):
                    list(io_.iter_cells([p], "math500"))

    def test_same_file_twice_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jsonl")
            _write(p, [{"model": "m", "id": "q", "temperature": 0.1, "completions": ["c"]}])
            with self.assertRaises(SystemExit):
                list(io_.iter_cells([p, d], "math500"))


class TestCounts(unittest.TestCase):
    def test_shards_are_summed_and_sorted(self):
        t = io_.CountsTable(extra_fields=("c_strict",))
        cell = {"benchmark": "math500", "model": "m", "id": "q2", "temperature": 0.7, "min_p": None, "source": "x"}
        t.add(cell, 10, 4, {"c_strict": 2}, texts_hash="h1")
        t.add(dict(cell, source="y"), 5, 1, {"c_strict": 1}, texts_hash="h2")
        t.add(dict(cell, id="q1", temperature=0.1), 3, 3, {"c_strict": 0})
        rows = t.sorted_rows()
        self.assertEqual([(r["id"], r["n"], r["c"], r["c_strict"]) for r in rows], [("q1", 3, 3, 0), ("q2", 15, 5, 3)])
        self.assertEqual(t.warnings, [])
        t.add(dict(cell, source="z"), 10, 4, {"c_strict": 2}, texts_hash="h1")     # the first shard again
        self.assertEqual(len(t.warnings), 1)

    def test_min_p_zero_is_not_null(self):
        # a min_p sweep's value 0.0 at tau 1.5 is its own cell, apart from the temperature sweep's (1.5, null)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jsonl")
            _write(p, [{"model": "m", "id": "q", "temperature": 1.5, "min_p": None, "n": 2, "completions": ["a", "b"]},
                       {"model": "m", "id": "q", "temperature": 1.5, "min_p": 0.0, "n": 1, "completions": ["c"]},
                       {"model": "m", "id": "q", "temperature": 1.5, "min_p": 0.0, "n": 1, "completions": ["d"]}])
            t = io_.CountsTable()
            for cell in io_.iter_cells([p], "hitom"):
                t.add(cell, len(cell["completions"]), 0)
            rows = t.sorted_rows()
            self.assertEqual([(r["min_p"], r["n"]) for r in rows], [(None, 2), (0.0, 2)])

    def test_row_check(self):
        good = {"benchmark": "hitom", "model": "m", "id": "s1_o0_CoTP", "temperature": 0.075, "min_p": None,
                "n": 1500, "c": 3}
        self.assertEqual(io_.check_counts_row(good), [])
        self.assertEqual(io_.check_counts_row(dict(good, min_p=0.1)), [])
        self.assertTrue(any("c > n" in p for p in io_.check_counts_row(dict(good, c=1501))))
        self.assertIn("benchmark 'gsm8k'", io_.check_counts_row(dict(good, benchmark="gsm8k")))
        self.assertTrue(any("n must be an int" in p for p in io_.check_counts_row(dict(good, n=1.5))))
        self.assertTrue(any("missing keys ['min_p']" in p
                            for p in io_.check_counts_row({k: v for k, v in good.items() if k != "min_p"})))

    def test_outputs_are_checked_before_any_work(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.jsonl"), os.path.join(d, "b.json")
            self.assertEqual(io_.check_outputs([a, None]), [a, None])
            with open(a, "w") as fh:
                fh.write("x\n")
            with self.assertRaises(SystemExit):
                io_.check_outputs([a, b])
            self.assertEqual(io_.check_outputs([a, b], force=True), [a, b])
            with self.assertRaises(SystemExit):
                io_.check_outputs([b, b])

    def test_atomic_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "counts.jsonl")
            io_.write_jsonl_atomic(p, [{"a": 1}])
            with self.assertRaises(SystemExit):
                io_.write_jsonl_atomic(p, [{"a": 2}])
            io_.write_jsonl_atomic(p, [{"a": 3}], force=True)
            with open(p) as fh:
                self.assertEqual(fh.read(), '{"a": 3}\n')
            self.assertEqual(sorted(os.listdir(d)), ["counts.jsonl"])


class TestPinnedData(unittest.TestCase):
    def test_local_file_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.jsonl")
            with open(p, "w") as fh:
                fh.write("synthetic\n")
            sha = io_.sha256_file(p)
            self.assertEqual(io_.fetch_pinned("https://example.invalid/x", sha, d, "x", local_file=p), p)
            with self.assertRaises(SystemExit):
                io_.fetch_pinned("https://example.invalid/x", "0" * 64, d, "x", local_file=p)
            self.assertEqual(io_.fetch_pinned("https://example.invalid/x", "0" * 64, d, "x", local_file=p,
                                              check=False), p)


if __name__ == "__main__":
    unittest.main()
