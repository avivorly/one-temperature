#!/usr/bin/env python3
"""Build the prompt file of one run: the exact prompt token ids of every question, rendered off the GPU.

    python -m onetemp.sampling.build_prompts --run math500/mistral-7b --out prompts/math500_mistral-7b.jsonl
    python -m onetemp.sampling.build_prompts --run hitom/gpt-oss-20b --out prompts/hitom_gpt-oss-20b.jsonl
    python -m onetemp.sampling.build_prompts --run livecodebench/qwen3-0.6b --lcb-dir path/to/code_generation_lite \\
        --out prompts/lcb_qwen3-0.6b.jsonl
    python -m onetemp.sampling.build_prompts --style math500_qwen3 --model Qwen/Qwen3-0.6B --out p.jsonl   # any style/model

The question text comes from the public datasets (public_data.py); the template from the model's tokenizer at
the revision the paper used (or --tokenizer <local dir>). Output: one JSON line per question

    {"benchmark", "model", "id", "prompt_style", "prompt_token_ids", "prompt_text", "n_prompt_tokens",
     "prompt_ids_sha256", "prompt_text_sha256"  [, "lcb_version", "lcb_row", "global_idx" for LiveCodeBench]}

and <out>.meta.json with the tokenizer, dataset hashes and the longest prompt. The prompts carry questions only:
no reference answer is ever loaded. verify_prompts.py compares such a file with the paper's prompt hashes.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from . import paper_runs as pr      # noqa: E402
from . import prompts as P          # noqa: E402
from . import public_data as pdata  # noqa: E402


def load_tokenizer(model_repo, revision=None, tokenizer=None):
    """The model's Hugging Face tokenizer (its chat template included), at the paper's revision."""
    from transformers import AutoTokenizer
    if tokenizer:
        return AutoTokenizer.from_pretrained(tokenizer), {"source": "local directory", "repo": model_repo}
    return (AutoTokenizer.from_pretrained(model_repo, revision=revision),
            {"source": "hub", "repo": model_repo, "revision": revision})


def resolve_question_ids(run, stage_name, ids_file=None):
    """The question ids a stage samples: --ids-file, or the stage's own list (None = all questions)."""
    if ids_file:
        return pdata.read_ids_file(ids_file), ids_file
    if run is None or stage_name is None:
        return None, "all"
    st = pr.stage_of(run, stage_name)
    q = st["questions"]
    if q == "all":
        return None, "all"
    if q.startswith("file:"):
        path = os.path.join(HERE, q[len("file:"):])
        return pdata.read_ids_file(path), q[len("file:"):]
    if q == "kept":
        raise SystemExit(f"stage {stage_name!r} samples the questions kept by the screen: pass --ids-file "
                         f"(screen.py writes it; the paper's list is {run.get('kept_file')})")
    raise SystemExit(f"bad questions spec {q!r}")


def build_records(benchmark, model_repo, style, questions, tok, harmony_date="default"):
    recs = []
    for q in questions:
        out = P.build(style, q, tok, harmony_date=harmony_date)
        rec = {"benchmark": benchmark, "model": model_repo, "id": q["id"], "prompt_style": style}
        rec.update(out)
        for k in ("lcb_version", "lcb_row", "global_idx"):
            if k in q:
                rec[k] = q[k]
        recs.append(rec)
    return recs


def add_common_args(ap):
    g = ap.add_argument_group("questions and prompts")
    g.add_argument("--run", help="a run profile of paper_runs.py, e.g. math500/mistral-7b or minp/qwen3-0.6b/tau1.5")
    g.add_argument("--style", help="prompt style (prompts.STYLES), instead of --run")
    g.add_argument("--model", help="HF model id (default: the run's model)")
    g.add_argument("--revision", help="model revision (default: the paper's, for the paper's models)")
    g.add_argument("--tokenizer", help="local tokenizer directory instead of the hub")
    g.add_argument("--stage", help="the run stage whose question list to use (default: the first stage)")
    g.add_argument("--ids-file", help="question ids to use (JSON list, {'ids': [...]}, or one per line)")
    g.add_argument("--harmony-date", default="default",
                   help="GPT-OSS date line: YYYY-MM-DD, 'none' for the render day, or 'default' (2026-10-10 on "
                        "MATH-500 and Hi-ToM, the render day on LiveCodeBench, as run)")
    g.add_argument("--cache-dir", help="download cache (default ~/.cache/one-temperature)")
    g.add_argument("--math500-file", help="local copy of MATH-500 test.jsonl")
    g.add_argument("--hitom-file", help="local copy of Hi-ToM_data.json")
    g.add_argument("--lcb-dir", help="directory holding code_generation_lite's test.jsonl ... test6.jsonl")
    g.add_argument("--no-verify-data", action="store_true", help="skip the sha256 checks of the datasets")


def resolve_style(a):
    """(run or None, benchmark, model_repo, revision, style) from --run / --style / --model."""
    run = pr.get_run(a.run) if a.run else None
    if run is not None:
        mk = run["model"]
        model_repo = a.model or pr.MODELS[mk]["repo"]
        revision = a.revision or (pr.MODELS[mk]["revision"] if model_repo == pr.MODELS[mk]["repo"] else None)
        style = a.style or run["prompt_style"]
    else:
        if not (a.style and a.model):
            raise SystemExit("give --run, or --style and --model")
        style, model_repo = a.style, a.model
        known = pr.REPO_TO_KEY.get(model_repo)
        revision = a.revision or (pr.MODELS[known]["revision"] if known else None)
    benchmark = P.STYLES[style][0] if style in P.STYLES else None
    if benchmark is None:
        raise SystemExit(f"unknown prompt style {style!r}; known: {sorted(P.STYLES)}")
    if run is not None and run["benchmark"] != benchmark:
        raise SystemExit(f"style {style} is for {benchmark}, the run is {run['benchmark']}")
    return run, benchmark, model_repo, revision, style


def harmony_date_arg(s):
    return None if s in ("none", "None") else s


def make_prompts(a):
    """Build prompt records from parsed arguments -> (records, meta)."""
    run, benchmark, model_repo, revision, style = resolve_style(a)
    stage_name = a.stage or (run["stages"][0]["name"] if run else None)
    ids, ids_src = resolve_question_ids(run, stage_name, a.ids_file)
    questions, data_prov = pdata.load_questions(benchmark, ids, a.cache_dir, a.math500_file, a.hitom_file,
                                                a.lcb_dir, verify=not a.no_verify_data)
    tok, tok_info = load_tokenizer(model_repo, revision, a.tokenizer)
    hdate = harmony_date_arg(a.harmony_date)
    recs = build_records(benchmark, model_repo, style, questions, tok, hdate)
    import transformers
    lens = [r["n_prompt_tokens"] for r in recs]
    meta = {"benchmark": benchmark, "model": model_repo, "revision": revision, "prompt_style": style,
            "run": a.run, "stage": stage_name, "questions": ids_src, "n_questions": len(recs),
            "harmony_date": (P.DEFAULT_HARMONY_DATE.get(style) if hdate == "default" else hdate),
            "tokenizer": tok_info, "transformers": transformers.__version__, "data": data_prov,
            "prompt_tokens": {"min": min(lens), "max": max(lens), "mean": round(sum(lens) / len(lens), 1)},
            "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return recs, meta


def write_prompts(path, recs, meta, force=False):
    if os.path.exists(path) and not force:
        raise SystemExit(f"refusing to overwrite {path} (pass --force)")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    meta = dict(meta, prompts_sha256=pdata.sha256_file(path))
    with open(path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    return meta


def read_prompts(path):
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    if len({r["id"] for r in recs}) != len(recs):
        raise SystemExit(f"{path}: duplicate question ids")
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", required=True, help="prompts JSONL to write")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args()
    recs, meta = make_prompts(a)
    meta = write_prompts(a.out, recs, meta, a.force)
    print(f"{len(recs)} prompts ({meta['prompt_style']}, {meta['model']}) -> {a.out}; prompt tokens "
          f"min {meta['prompt_tokens']['min']} max {meta['prompt_tokens']['max']}", file=sys.stderr)


if __name__ == "__main__":
    main()
