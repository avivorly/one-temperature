"""Tests of the Hi-ToM grader on synthetic questions only (fictional containers; no Hi-ToM answer is used).

    python -m pytest tests/test_grading_hitom.py        (from the repository root)
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from onetemp.grading import completions_io  # noqa: E402
from onetemp.grading import hitom as H  # noqa: E402

CHOICES = ("A. violet_box, B. amber_crate, C. teal_drawer, D. teal_drawer_2, E. amber_basket, "
           "F. violet_treasure_chest")
GOLD = "teal_drawer"

# (completion, expected extracted value, expected bucket, expected ambiguity flag), labelled by hand
CASES = [
    ("C. teal_drawer", "teal_drawer", "correct", False),
    ("C\nBecause the last move put it there.", "teal_drawer", "correct", False),
    ("**C**", "teal_drawer", "correct", False),
    ("(C) teal_drawer", "teal_drawer", "correct", False),
    ("Answer: C", "teal_drawer", "correct", False),
    ("The answer is E.", "amber_basket", "wrong_valid_choice", False),
    ("option B", "amber_crate", "wrong_valid_choice", False),
    ("I think it is in the teal drawer.", "teal_drawer", "correct", False),
    ("A person moved it to the amber crate.", "amber_crate", "wrong_valid_choice", False),
    ("answer a question first: the teal drawer", "teal_drawer", "correct", False),
    ("teal_drawer_2", "teal_drawer_2", "wrong_valid_choice", False),       # never sliced to teal_drawer
    ("teal", None, "no_answer", False),                                     # a bare colour names nothing
    ("The teal cupboard.", None, "not_a_choice", False),                   # container-shaped, not on offer
    ("The teal cupboard, no: the teal drawer.", "teal_drawer", "correct", False),   # off-menu token skipped
    ("teal drawer or amber crate", "teal_drawer", "correct", True),         # first valid option decides
    ("teal-drawer", None, "no_answer", False),                              # hyphens are not folded
    ("Z. teal_drawer", "teal_drawer", "correct", False),                   # Z is not on offer; the name is
    ("Z.", None, "not_a_choice", False),
    ("", None, "empty", False),
    ("   \n  ", None, "empty", False),
]


class TestMatcher(unittest.TestCase):
    def test_hand_labelled_cases(self):
        for text, value, bucket, ambiguous in CASES:
            with self.subTest(text=text):
                got_value, route = H.extract(text, CHOICES)
                passed = H.grade(got_value, GOLD)
                self.assertEqual(got_value, value)
                self.assertEqual(H.bucket_for(route, passed), bucket)
                self.assertEqual(H.analyse(text, CHOICES)["ambiguous"], ambiguous)
                v = H.verdict_for(text, CHOICES, GOLD)
                self.assertIs(v["pass"], bucket == "correct")
                self.assertEqual(v["bucket"], bucket)

    def test_selection_rule_never_reads_the_answer(self):
        # the extracted value is the same whatever the reference answer is
        for text, *_ in CASES:
            a = H.verdict_for(text, CHOICES, "teal_drawer")["value"]
            b = H.verdict_for(text, CHOICES, "amber_crate")["value"]
            self.assertEqual(a, b)

    def test_contract_and_structure(self):
        self.assertEqual(H.grader_hash(), H.MATCHER_CONTRACT)
        self.assertEqual(H.MATCHER_CONTRACT, "bf137bfc")
        self.assertTrue(H.selftest(stream=io.StringIO()))


class TestAdapter(unittest.TestCase):
    def test_formats_by_model(self):
        self.assertEqual(H.answer_format_for("Qwen/Qwen3-8B"), "qwen3")
        self.assertEqual(H.answer_format_for("Qwen/Qwen3-0.6B"), "qwen3")
        self.assertEqual(H.answer_format_for("openai/gpt-oss-20b"), "gptoss")
        self.assertEqual(H.answer_format_for("mistralai/Mistral-7B-Instruct-v0.3"), "plain")

    def test_qwen3_end_tokens(self):
        text = "C<|im_end|>"
        # without the adapter the end token glued to the label hides it ...
        self.assertEqual(H.grade_completion(text, "plain", "M", CHOICES, GOLD)["bucket"], "no_answer")
        # ... with it, the label is read
        self.assertEqual(H.grade_completion(text, "qwen3", "M", CHOICES, GOLD)["bucket"], "correct")
        self.assertEqual(H.answer_text("teal drawer<|endoftext|>", "qwen3", "M"), ("teal drawer", "ok"))

    def test_gptoss_cut_at_first_stop(self):
        text = "C. teal_drawer<|end|><|start|>assistant<|channel|>final<|message|>E"
        self.assertEqual(H.answer_text(text, "gptoss", "M"), ("C. teal_drawer", "ok"))
        self.assertTrue(H.grade_completion(text, "gptoss", "M", CHOICES, GOLD)["pass"])
        self.assertEqual(H.answer_text("E<|return|>C", "gptoss", "M"), ("E", "ok"))

    def test_reasoning_arms(self):
        harmony = ("<|channel|>analysis<|message|>maybe E<|end|><|start|>assistant<|channel|>final"
                   "<|message|>C<|return|>")
        self.assertEqual(H.answer_text(harmony, "gptoss", "R"), ("C", "ok"))
        self.assertEqual(H.answer_text("<|channel|>analysis<|message|>E", "gptoss", "R"), (None, "no_final_channel"))
        self.assertEqual(H.answer_text("<think>E or A</think>\n\nC<|im_end|>", "qwen3", "R"), ("\n\nC", "ok"))
        r = H.grade_completion("<think>still thinking about E", "qwen3", "R", CHOICES, GOLD)
        self.assertEqual((r["pass"], r["bucket"], r["route"]), (False, "no_answer", "adapter"))
        with self.assertRaises(ValueError):
            H.answer_text("x", "qwen3", "X")


def _dataset(path, records):
    with open(path, "w") as fh:
        json.dump({"data": records}, fh)


def _record(sample_id, order, ptype, answer, choices=CHOICES):
    return {"sample_id": sample_id, "question_order": order, "prompting_type": ptype, "choices": choices,
            "answer": answer, "deception": False, "story_length": 1, "story": "(synthetic story)",
            "question": "(synthetic question)", "prompt": "(synthetic prompt)"}


class TestLoaderAndCommandLine(unittest.TestCase):
    def test_loader_ids_and_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "Hi-ToM_data.json")
            _dataset(p, [_record(0, 0, "CoTP", "teal_drawer"), _record(1, 3, "MC", "amber_crate"),
                         _record(2, 1, "CoTP", "not_on_offer")])
            choices, gold, meta, broken = H.load_answers(p)
            self.assertEqual(sorted(gold), ["s0_o0_CoTP", "s1_o3_MC"])
            self.assertEqual([b[0] for b in broken], ["s2_o1_CoTP"])
            self.assertEqual(meta["s1_o3_MC"]["question_order"], 3)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            data = os.path.join(d, "Hi-ToM_data.json")
            _dataset(data, [_record(0, 0, "CoTP", "teal_drawer"), _record(1, 2, "CoTP", "amber_crate")])
            runs = os.path.join(d, "runs")
            os.makedirs(runs)
            lines = [
                {"benchmark": "hitom", "model": "mistralai/Mistral-7B-Instruct-v0.3", "id": "s0_o0_CoTP",
                 "temperature": 0.75, "min_p": None,
                 "completions": ["C. teal_drawer", "E", "C. teal_drawer", "teal"],
                 "finish_reasons": ["stop", "stop", "stop", "length"]},
                {"model": "Qwen/Qwen3-0.6B", "id": "s0_o0_CoTP", "temperature": 1.5, "min_p": 0.05,
                 "completions": ["C<|im_end|>", "B<|im_end|>"]},
                {"model": "Qwen/Qwen3-0.6B", "id": "s1_o2_CoTP", "temperature": 1.5, "min_p": 0.05,
                 "completions": ["amber crate<|im_end|>"] * 3},
            ]
            with open(os.path.join(runs, "cells.jsonl"), "w") as fh:
                for r in lines:
                    fh.write(json.dumps(r) + "\n")
            out, summ = os.path.join(d, "counts.jsonl"), os.path.join(d, "summary.json")
            self.assertEqual(H.main(["--inputs", runs, "--out", out, "--summary", summ, "--dataset", data,
                                     "--no-verify-data", "--jobs", "2"]), 0)
            with open(out) as fh:
                rows = [json.loads(x) for x in fh]
            for r in rows:
                self.assertEqual(completions_io.check_counts_row(r), [])
            got = {(r["model"], r["id"], r["temperature"], r["min_p"]): (r["n"], r["c"]) for r in rows}
            self.assertEqual(got, {
                ("mistralai/Mistral-7B-Instruct-v0.3", "s0_o0_CoTP", 0.75, None): (4, 2),
                ("Qwen/Qwen3-0.6B", "s0_o0_CoTP", 1.5, 0.05): (2, 1),
                ("Qwen/Qwen3-0.6B", "s1_o2_CoTP", 1.5, 0.05): (3, 3)})
            with open(summ) as fh:
                s = json.load(fh)
            self.assertEqual(s["matcher_contract"], "bf137bfc")
            mis = [x for x in s["by_setting"] if x["model"].startswith("mistralai")][0]
            self.assertEqual(mis["buckets"], {"correct": 2, "no_answer": 1, "wrong_valid_choice": 1})
            self.assertEqual(mis["truncated"], 1)
            # an existing output is refused before any work
            with self.assertRaises(SystemExit) as cm:
                H.main(["--inputs", runs, "--out", out, "--dataset", data, "--no-verify-data"])
            self.assertIn("refusing to overwrite", str(cm.exception))
            # --try picks the answer format from --model
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                H.main(["--try", "C<|im_end|>", "--qid", "s0_o0_CoTP", "--model", "Qwen/Qwen3-8B",
                        "--dataset", data, "--no-verify-data"])
            self.assertEqual(json.loads(buf.getvalue())["bucket"], "correct")
            # a changed matcher is refused
            old = H.MATCHER_CONTRACT
            H.MATCHER_CONTRACT = "00000000"
            try:
                with self.assertRaises(SystemExit) as cm:
                    H.main(["--inputs", runs, "--out", os.path.join(d, "c2.jsonl"), "--dataset", data,
                            "--no-verify-data"])
                self.assertIn("refusing to grade", str(cm.exception))
            finally:
                H.MATCHER_CONTRACT = old


if __name__ == "__main__":
    unittest.main()
