#!/usr/bin/env python3
"""Synthetic smoke test of the sampling package: no GPU, no model, no network, standard library only.

    python -m pytest tests/test_sampling.py

It checks
  1. the prompt texts and builders (with a toy chat tokenizer that renders the three template layouts; the real
     templates are checked against the paper's prompt hashes by verify_prompts.py);
  2. the sampler end to end with the fake engine: output format, one file per cell, resume, the RUN.json guard,
     the clamp guard, min_p settings, decode_ids text mode;
  3. the multi-GPU launcher with two fake GPUs, in both split modes, and its by-the-files completion check;
  4. the screen rules (exact at the 0.30 boundary) and stage pooling.
"""
import gzip
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                          # the repository root
PKG = os.path.join(ROOT, "onetemp", "sampling")      # the sampling package (its question_sets/ live here)
from onetemp.sampling import paper_runs as pr      # noqa: E402
from onetemp.sampling import pool_counts as pc     # noqa: E402
from onetemp.sampling import prompts as P          # noqa: E402
from onetemp.sampling import screen as sc          # noqa: E402

PY = [sys.executable, "-B", "-m"]        # -B: the test leaves no __pycache__ in the package


# ---------------------------------------------------------------------------------------------------------------
class ToyTok:
    """Renders the Mistral v0.3, Qwen3 and GPT-OSS Harmony layouts as text; ids are UTF-8 bytes, with the
    Harmony markers as single 'special' ids (>= 1000) so that encode/decode round-trip."""
    SPECIAL = ["<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|im_start|>", "<|im_end|>", "<s>",
               "[INST]", "[/INST]"]

    def __init__(self, family):
        self.family = family

    def render(self, msgs, add_generation_prompt=False, enable_thinking=True, reasoning_effort="medium"):
        if self.family == "mistral":
            sys_ = [m["content"] for m in msgs if m["role"] == "system"]
            user = [m["content"] for m in msgs if m["role"] == "user"][-1]
            return "<s>[INST] " + (sys_[0] + "\n\n" if sys_ else "") + user + "[/INST]"
        if self.family == "qwen3":
            s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
            if add_generation_prompt:
                s += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
            return s
        s = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge "
             "cutoff: 2024-06\nCurrent date: 2031-01-01\n\nReasoning: " + reasoning_effort + "\n\n# Valid "
             "channels: analysis, commentary, final. Channel must be included for every message.<|end|>")
        for m in msgs:
            if m["role"] == "system":
                s += "<|start|>developer<|message|># Instructions\n\n" + m["content"] + "\n\n<|end|>"
            else:
                s += "<|start|>user<|message|>" + m["content"] + "<|end|>"
        return s + ("<|start|>assistant" if add_generation_prompt else "")

    def apply_chat_template(self, msgs, tokenize=True, **kw):
        text = self.render(msgs, **kw)
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for k, sp in enumerate(self.SPECIAL):
                if text.startswith(sp, i):
                    ids.append(1000 + k)
                    i += len(sp)
                    break
            else:
                ids.extend(text[i].encode("utf-8"))
                i += 1
        return ids

    def __call__(self, text):
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens=False):
        out, buf = [], bytearray()
        for t in ids:
            if t >= 1000:
                out.append(buf.decode("utf-8"))
                buf = bytearray()
                out.append(self.SPECIAL[t - 1000])
            else:
                buf.append(t)
        out.append(buf.decode("utf-8"))
        return "".join(out)


def test_prompts():
    assert P.sha256_text(P.MATH500_INSTRUCTION).startswith("468adbdf")
    assert P.MATH500_QWEN3_8B_INSTRUCTION.startswith(P.MATH500_INSTRUCTION + " Your answer can be at most 2,048")
    assert len(P.HITOM_CAP500_SUFFIX) == 301 and P.HITOM_CAP500_SUFFIX.count("—") == 2
    assert P.lcb_body("Do X.", "") == "Do X.\n\nRead from stdin, write to stdout. Enclose code in ```python``` " \
                                      "delimiters.\n\n"
    assert P.lcb_body("Do X.", "class Solution:\n    pass") == \
        "Do X.\n\nComplete this code:\n```python\nclass Solution:\n    pass\n```\n\n"
    mq = {"id": "test/x/1.json", "problem": "What is $1+1$?"}
    r = P.build("math500_mistral", mq, ToyTok("mistral"))
    assert r["prompt_text"] == "<s>[INST] " + P.MATH500_INSTRUCTION + "\n\nWhat is $1+1$?[/INST]"
    r = P.build("math500_qwen3_8b_concise", mq, ToyTok("qwen3"))
    assert P.MATH500_QWEN3_8B_SENTENCE in r["prompt_text"] and r["prompt_text"].endswith(P.QWEN3_NOTHINK_TAIL)
    r = P.build("math500_harmony", mq, ToyTok("gptoss"))
    assert "Current date: 2026-10-10\n" in r["prompt_text"] and "2031" not in r["prompt_text"]
    assert r["prompt_text"].endswith("<|start|>assistant<|channel|>final<|message|>")
    assert "<|start|>developer<|message|># Instructions\n\n" + P.MATH500_INSTRUCTION in r["prompt_text"]
    try:
        P.build("math500_qwen3", {"id": "q", "problem": "leak \\boxed{3}"}, ToyTok("qwen3"))
        raise AssertionError("a \\boxed{} in the problem must be refused")
    except ValueError:
        pass
    hq = {"id": "s0_o0_CoTP", "prompt": "Story...\nChoices: A. box, B. bag\n"}
    r = P.build("hitom_mistral_cap500", hq, ToyTok("mistral"))
    assert r["prompt_text"] == "<s>[INST] " + hq["prompt"] + P.HITOM_CAP500_SUFFIX + "[/INST]"
    r = P.build("hitom_harmony", hq, ToyTok("gptoss"))
    assert "developer" not in r["prompt_text"] and "Current date: 2026-10-10" in r["prompt_text"]
    r = P.build("hitom_qwen3", hq, ToyTok("qwen3"))
    assert r["prompt_text"].startswith("<|im_start|>user\n" + hq["prompt"])
    lq = {"id": "abc001_a", "question_content": "Print N.", "starter_code": ""}
    r = P.build("lcb_raw", lq, ToyTok("qwen3"))
    assert r["prompt_text"] == P.LCB_SYSTEM + "\n\n" + P.lcb_body("Print N.", "") + "```python\n"
    r = P.build("lcb_mistral", lq, ToyTok("mistral"))
    assert r["prompt_text"].endswith("[/INST]\n```python\n")
    r = P.build("lcb_qwen3", lq, ToyTok("qwen3"))
    assert r["prompt_text"].endswith("<think>\n\n</think>\n\n```python\n")
    r = P.build("lcb_harmony", lq, ToyTok("gptoss"))                       # render day kept (as run)
    assert "Current date: 2031-01-01" in r["prompt_text"]
    assert r["prompt_text"].endswith("<|start|>assistant<|channel|>final<|message|>```python\n")
    r = P.build("lcb_harmony", lq, ToyTok("gptoss"), harmony_date="2026-02-27")
    assert "Current date: 2026-02-27" in r["prompt_text"]
    assert r["prompt_ids_sha256"] == P.ids_sha256(r["prompt_token_ids"]) and r["n_prompt_tokens"] > 0


def test_profiles():
    assert pr.GRID_31[2] == 0.225 and pr.GRID_31[-5:] == (2.0, 2.2, 2.5, 3.0, 4.0)
    assert set(pr.GRID_16) <= set(pr.GRID_38) and len(pr.GRID_38_ROUND2) == 22
    for key, r in pr.RUNS.items():
        assert r["prompt_style"] in P.STYLES and P.STYLES[r["prompt_style"]][0] == r["benchmark"], key
        assert r["model"] in pr.MODELS and r["axis"] in ("temperature", "min_p"), key
        assert r["text"] in ("vllm_text", "decode_ids") and r["split"] in ("questions", "samples"), key
        for s in r["stages"]:
            q = s["questions"]
            assert q in ("all", "kept") or q.startswith("file:"), key
            if q.startswith("file:"):
                assert os.path.exists(os.path.join(PKG, q[5:])), (key, q)
        if r.get("kept_file"):
            assert os.path.exists(os.path.join(PKG, r["kept_file"])), key
    assert pr.stage_of(pr.RUNS["livecodebench/qwen3-0.6b"], "deep")["settings"] == pr.GRID_16
    assert "settings" not in pr.stage_of(pr.RUNS["livecodebench/qwen3-0.6b"], "main")
    assert pr.RUNS["math500/qwen3-8b"]["stages"][0]["n"] == 750
    assert pr.RUNS["hitom/mistral-7b"]["stages"][0]["n"] == 1000
    assert pr.RUNS["minp/qwen3-0.6b/tau1.5"]["settings"][0] == 0.0


# ---------------------------------------------------------------------------------------------------------------
def write_prompts(path, benchmark, model, style, ids):
    with open(path, "w") as fh:
        for k, qid in enumerate(ids):
            toks = [7, 11 + k, 13 + 2 * k]
            fh.write(json.dumps({"benchmark": benchmark, "model": model, "id": qid, "prompt_style": style,
                                 "prompt_token_ids": toks, "prompt_text": f"prompt {qid}",
                                 "n_prompt_tokens": len(toks), "prompt_ids_sha256": P.ids_sha256(toks),
                                 "prompt_text_sha256": P.sha256_text(f"prompt {qid}"),
                                 **({"lcb_version": "v1", "lcb_row": k} if benchmark == "livecodebench" else {})})
                     + "\n")


def run(cmd, expect=0):
    p = subprocess.run(PY + cmd, capture_output=True, text=True, cwd=ROOT)
    if p.returncode != expect:
        raise AssertionError(f"{cmd[0]} exited {p.returncode} (expected {expect}):\n{p.stdout}\n{p.stderr}")
    return p


def read_cells(root):
    out = []
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(d, f), "rt") as fh:
                    out.extend(json.loads(line) for line in fh if line.strip())
    return out


def toy_counts(cells, right):
    """A toy grader: c = completions that start with the 'right' answer."""
    agg = {}
    for c in cells:
        k = (c["benchmark"], c["model"], c["id"], c["temperature"], c["min_p"])
        n, s = agg.get(k, (0, 0))
        agg[k] = (n + len(c["completions"]), s + sum(t.startswith(right) for t in c["completions"]))
    return [{"benchmark": k[0], "model": k[1], "id": k[2], "temperature": k[3], "min_p": k[4], "n": v[0], "c": v[1]}
            for k, v in sorted(agg.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][4] or 0))]


def test_sampler(tmp):
    ids = [f"s{i}_o0_CoTP" for i in range(5)]
    pfile = os.path.join(tmp, "p_hitom.jsonl")
    write_prompts(pfile, "hitom", "Qwen/Qwen3-0.6B", "hitom_qwen3", ids)
    out = os.path.join(tmp, "hitom_screen")
    base = ["sample_vllm.py", "--run", "hitom/qwen3-0.6b", "--stage", "screen", "--prompts", pfile, "--out", out,
            "--engine", "fake", "--settings", "0.3,0.9,1.5", "--n", "40", "--seed", "1"]
    base[0] = "onetemp.sampling.sample_vllm"
    run(base)
    cells = read_cells(out)
    assert len(cells) == 15, len(cells)
    c = cells[0]
    for k in ("benchmark", "model", "id", "temperature", "min_p", "n", "completions", "finish_reasons",
              "stop_reasons", "num_output_tokens", "stage", "run", "part", "provenance"):
        assert k in c, k
    assert c["benchmark"] == "hitom" and c["model"] == "Qwen/Qwen3-0.6B" and c["min_p"] is None
    assert c["n"] == 40 == len(c["completions"]) == len(c["finish_reasons"])
    assert c["provenance"]["text_mode"] == "decode_ids" and c["provenance"]["batching"] == "question"
    assert c["provenance"]["engine"]["max_num_seqs"] == 256 and c["provenance"]["engine_seed"] == 1
    assert "answer" not in json.dumps(c["provenance"]).lower()
    # resume: nothing new
    p = run(base)
    assert "0 of 15 cells to do" in p.stderr, p.stderr
    # a different configuration in the same directory is refused
    run(base[:-2] + ["--n", "41", "--seed", "1"], expect=1)
    # screen + pooling on toy counts
    counts1 = toy_counts(cells, "A. red_box")
    cfile = os.path.join(tmp, "counts_screen.jsonl")
    with open(cfile, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in counts1)
    kfile = os.path.join(tmp, "kept.json")
    run(["onetemp.sampling.screen", "--rule", "hitom", "--counts", cfile, "--out", kfile])
    kept = json.load(open(kfile))
    assert kept["n_kept"] + kept["n_dropped"] == 5 and not kept["undecided"]
    for q, (peak, arg) in kept["peaks"].items():
        assert (q in kept["ids"]) == (peak < 0.30)
    if kept["ids"]:
        out2 = os.path.join(tmp, "hitom_topup")
        run(["onetemp.sampling.sample_vllm", "--run", "hitom/qwen3-0.6b", "--stage", "topup", "--prompts", pfile,
             "--ids-file", kfile, "--out", out2, "--engine", "fake", "--settings", "0.3,0.9,1.5", "--n", "60"])
        cells2 = read_cells(out2)
        assert len(cells2) == 3 * len(kept["ids"]) and all(x["n"] == 60 for x in cells2)
        pooled = pc.pool([counts1, toy_counts(cells2, "A. red_box")], "sum")
        assert all(r["n"] == (100 if r["id"] in kept["ids"] else 40) for r in pooled)
    # min_p run: temperature = tau, min_p varies, 0 is kept as 0.0 (not null)
    out3 = os.path.join(tmp, "minp")
    run(["onetemp.sampling.sample_vllm", "--run", "minp/qwen3-0.6b/tau1.5", "--stage", "screen", "--prompts",
         pfile, "--out", out3, "--engine", "fake", "--settings", "0,0.01", "--n", "8", "--max-cells", "4"], expect=3)
    cm = read_cells(out3)
    assert len(cm) == 4 and {x["temperature"] for x in cm} == {1.5} and {x["min_p"] for x in cm} == {0.0, 0.01}
    assert any("min_p=0.01" in x["provenance"]["sampling_params_repr"] for x in cm)
    # clamp guard: prompt + max_tokens > max_model_len is refused
    run(["onetemp.sampling.sample_vllm", "--run", "hitom/qwen3-0.6b", "--prompts", pfile, "--out",
         os.path.join(tmp, "clamp"), "--engine", "fake", "--engine-arg", "max_model_len=1001"], expect=1)


def test_launcher(tmp):
    ids = ["abc001_a", "abc001_b", "1873_A"]
    pfile = os.path.join(tmp, "p_lcb.jsonl")
    write_prompts(pfile, "livecodebench", "mistralai/Mistral-7B-Instruct-v0.3", "lcb_mistral", ids)
    out = os.path.join(tmp, "lcb_main")
    cmd = ["onetemp.sampling.launch", "--gpus", "0,1", "--stagger", "0", "--poll", "0.2",
           "--run", "livecodebench/mistral-7b", "--prompts", pfile, "--out", out, "--engine", "fake",
           "--settings", "0.1,1.0", "--n", "9"]
    p = run(cmd)
    chk = json.load(open(os.path.join(out, "LAUNCH_CHECK.json")))
    assert chk["complete"] and chk["cells_planned"] == 3 * 2 * 2, chk
    cells = read_cells(out)
    assert sorted(len(c["completions"]) for c in cells) == [4] * 6 + [5] * 6        # 9 = 5 + 4
    seeds = {c["part"][0]: c["provenance"]["engine_seed"] for c in cells}
    assert seeds == {0: 0, 1: 1}, seeds                                              # distinct seeds per part
    assert {c["lcb_version"] for c in cells} == {"v1"}
    # seeds: the first stage keeps the profile's (vLLM default = None on one worker), a later stage draws a fresh one
    plan = lambda stage: json.loads(run(["onetemp.sampling.sample_vllm", "--run", "livecodebench/mistral-7b",
                                         "--stage", stage, "--prompts", pfile, "--ids-file", ids_file,
                                         "--out", os.path.join(tmp, "plan"), "--engine", "fake", "--plan"]).stdout)
    ids_file = os.path.join(tmp, "one_id.txt")
    with open(ids_file, "w") as fh:
        fh.write("abc001_a\n")
    assert plan("main")["seed"] is None and isinstance(plan("deep")["seed"], int)
    # questions split: each cell from one worker, file names independent of the worker
    out2 = os.path.join(tmp, "lcb_q")
    run(["onetemp.sampling.launch", "--gpus", "0-1", "--stagger", "0", "--poll", "0.2", "--split",
         "questions", "--run", "livecodebench/mistral-7b", "--prompts", pfile, "--out", out2, "--engine", "fake",
         "--settings", "0.1,1.0", "--n", "9"])
    cells2 = read_cells(out2)
    assert len(cells2) == 6 and all(c["n"] == 9 and c["part"] is None for c in cells2)
    # an incomplete directory is reported as incomplete, never as done
    victim = sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.join(out2, "cells")) for f in fs)[0]
    with gzip.open(victim, "wt") as fh:
        fh.write(json.dumps({"id": "x"}) + "\n")
    p = run(["onetemp.sampling.launch", "--gpus", "0-1", "--stagger", "0", "--poll", "0.2", "--split",
             "questions", "--run", "livecodebench/mistral-7b", "--prompts", pfile, "--out", out2, "--engine",
             "fake", "--settings", "0.1,1.0", "--n", "9"], expect=3)
    assert "INCOMPLETE" in p.stderr


def test_screen_rules():
    rows = [{"id": "a", "temperature": 0.1, "n": 10, "c": 3},      # peak exactly 0.30
            {"id": "b", "temperature": 0.1, "n": 10, "c": 0},      # peak 0
            {"id": "c", "temperature": 0.1, "n": 1000, "c": 299}]  # just below
    r = sc.screen(rows, pr.SCREEN_RULES["hitom"], "temperature")
    assert r["ids"] == ["b", "c"]
    r = sc.screen(rows, pr.SCREEN_RULES["lcb-mistral"], "temperature")
    assert r["ids"] == ["c"]
    r = sc.screen(rows, pr.SCREEN_RULES["lcb"], "temperature")
    assert r["ids"] == ["a", "c"]
    rows2 = [{"benchmark": "x", "model": "m", "id": "q", "temperature": 1.0, "min_p": None, "n": 5, "c": 1}]
    rep = pc.pool([rows2, [dict(rows2[0], n=7, c=2)]], "replace")
    assert rep[0]["n"] == 7 and rep[0]["c"] == 2


def main():
    test_prompts()
    test_profiles()
    test_screen_rules()
    with tempfile.TemporaryDirectory() as tmp:
        test_sampler(tmp)
        test_launcher(tmp)
    print("sampling smoke test: OK")


if __name__ == "__main__":
    main()
