"""Step 1: sample n completions per question at every temperature of a grid, with vLLM (single GPU).

This is our generator, cleaned: one vLLM request per (question, temperature) with SamplingParams(n=...),
the model's chat template applied to the prompt, and a provenance record on every output line.

Output: one JSONL per temperature, <out_dir>/T{T:.4f}.jsonl (4 decimals so that grid points such as 0.075
and 0.15 never share a file), one line per question:
    {"id", "temperature", "completions": [...], "finish_reasons": [...], "provenance": {...}}
Use one output directory per sampling configuration (model, prompt, max_tokens, engine settings).
A rerun skips (question, temperature) cells already on disk, so an interrupted run can be resumed.

Multi-GPU: run one process per GPU on a disjoint subset of the temperatures (or of the questions), all
writing to the same --out-dir, e.g.
    CUDA_VISIBLE_DEVICES=0 python generate_vllm.py --temperatures 0.1,0.2,0.3,0.4 ... &
    CUDA_VISIBLE_DEVICES=1 python generate_vllm.py --temperatures 0.5,0.6,0.7,0.8 ... &
Simplified: our runs used a launcher script that did this split and restarted stopped workers.

Example (MATH-500, Mistral 7B Instruct, the paper's settings):
    python generate_vllm.py --benchmark math500 --model mistralai/Mistral-7B-Instruct-v0.3 --dtype half \
        --temperatures 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.5,1.7,2.0 \
        --n 1500 --max-tokens 4096 --out-dir runs/mistral7b_math500
"""
import argparse
import hashlib
import json
import os
import sys
import time

# The MATH-500 instruction, verbatim from our runs. It is the system turn; the problem is the user turn.
MATH500_SYSTEM = ("Solve the following math problem efficiently and clearly. "
                  "Reason step by step, and put your final answer within \\boxed{}.")


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_questions(args):
    """Return a list of {"id", "prompt", "system" (optional)}. Prompts only: answers never enter this script."""
    if args.benchmark == "math500":
        if args.math500_file:                      # a local copy of the HF test split (JSONL)
            rows = [json.loads(line) for line in open(args.math500_file) if line.strip()]
        else:
            from datasets import load_dataset
            rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
        assert len(rows) == 500 and len({r["unique_id"] for r in rows}) == 500
        return [{"id": r["unique_id"], "prompt": r["problem"], "system": MATH500_SYSTEM} for r in rows]
    text = open(args.questions).read()
    qs = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    for q in qs:
        assert set(q) <= {"id", "prompt", "system"}, f"question file must hold prompts only, got keys {sorted(q)}"
    return qs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--benchmark", choices=["math500"], help="build MATH-500 prompts from the public HF dataset")
    src.add_argument("--questions", help="JSON list or JSONL of {id, prompt[, system]}: prompts only, no answers")
    ap.add_argument("--math500-file", help="optional local JSONL copy of HuggingFaceH4/MATH-500 (split test)")
    ap.add_argument("--model", required=True, help="HF model id or local path, e.g. mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--dtype", default="half",
                    help="weights/activations dtype: half (fp16; Mistral 7B, Qwen3-0.6B) or bfloat16 (Qwen3-8B). "
                         "No quantization was used, except GPT-OSS 20B's native MXFP4 checkpoint")
    ap.add_argument("--temperatures", required=True, help="comma-separated temperature grid")
    ap.add_argument("--n", type=int, default=1500, help="samples per (question, temperature); paper: 1,500 (750 for Qwen3-8B on MATH-500)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="max new tokens per sample; paper: 4096 on MATH-500 (2048 Qwen3-8B), 1000 on Hi-ToM (500 Mistral)")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling; we kept vLLM's default 1.0 (off)")
    ap.add_argument("--seed", type=int, default=None,
                    help="optional per-request sampling seed (SamplingParams.seed); our runs left it unset by default")
    ap.add_argument("--no-thinking", action="store_true",
                    help="pass enable_thinking=False to the chat template (Qwen3 models, as in the paper)")
    # Engine settings. Hold dtype, max_model_len and max_num_seqs FIXED across all temperatures of a
    # model: they change how vLLM batches requests and therefore which samples come out.
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="context length; 0 = longest prompt + max_tokens. The script refuses to run if a "
                         "prompt + max_tokens does not fit, because vLLM would silently clamp it")
    ap.add_argument("--max-num-seqs", type=int, default=512, help="max concurrent sequences in a batch (ours: 512)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768, help="scheduler token budget (ours: 32768)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="fraction of GPU memory for vLLM (ours: 0.90)")
    ap.add_argument("--out-dir", required=True, help="output directory, one per sampling configuration")
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    temps = [float(t) for t in args.temperatures.split(",")]
    questions = load_questions(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenize every prompt with the model's chat template first, to size the context exactly.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    prompt_ids = {}
    for q in questions:
        msgs = ([{"role": "system", "content": q["system"]}] if q.get("system") else []) + \
               [{"role": "user", "content": q["prompt"]}]
        prompt_ids[q["id"]] = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **tmpl_kwargs)
    need = max(len(p) for p in prompt_ids.values()) + args.max_tokens
    max_model_len = args.max_model_len or need
    assert need <= max_model_len, f"max_model_len={max_model_len} < longest prompt + max_tokens = {need}"

    engine = dict(dtype=args.dtype, max_model_len=max_model_len, max_num_seqs=args.max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=True, enable_chunked_prefill=True)
    llm = LLM(model=args.model, disable_log_stats=True, **engine)

    # Resume: collect the (id, T) cells already written.
    done = set()
    for t in temps:
        path = os.path.join(args.out_dir, f"T{t:.4f}.jsonl")
        if os.path.exists(path):
            done |= {(json.loads(l)["id"], t) for l in open(path) if l.strip()}

    # Question-major order: a question gets all its temperatures before the next question starts, so an
    # interrupted run leaves complete temperature profiles for the questions it reached.
    files = {t: open(os.path.join(args.out_dir, f"T{t:.4f}.jsonl"), "a") for t in temps}
    for qi, q in enumerate(questions):
        for t in temps:
            if (q["id"], t) in done:
                continue
            sp = dict(n=args.n, temperature=t, top_p=args.top_p, max_tokens=args.max_tokens)
            if args.seed is not None:
                sp["seed"] = args.seed
            t0 = time.time()
            out = llm.generate([{"prompt_token_ids": prompt_ids[q["id"]]}], SamplingParams(**sp), use_tqdm=False)[0]
            rec = {
                "id": q["id"],
                "temperature": t,
                "completions": [c.text for c in out.outputs],
                "finish_reasons": [c.finish_reason for c in out.outputs],   # "length" = truncated at max_tokens
                "provenance": {
                    "model": args.model, "vllm_version": vllm.__version__, "engine": engine,
                    "sampling": sp, "chat_template_kwargs": tmpl_kwargs,
                    "prompt_sha256": sha256(q["prompt"]), "system_sha256": sha256(q.get("system") or ""),
                    "prompt_tokens": len(prompt_ids[q["id"]]), "gen_seconds": round(time.time() - t0, 2),
                },
            }
            files[t].write(json.dumps(rec) + "\n")
            files[t].flush()
        print(f"question {qi + 1}/{len(questions)} done at all {len(temps)} temperatures", file=sys.stderr)
    for f in files.values():
        f.close()


if __name__ == "__main__":
    main()
