#!/usr/bin/env python3
"""The sampling settings of every run in the paper, as data (Appendix C "Computational Resources" and Appendix I).

Each run profile says, for one (benchmark, model) or one min_p study:
  - which prompt builder to use (prompts.py),
  - the setting grid: temperatures (Appendix C tables) or min_p values at a fixed temperature tau (Appendix I),
  - the stages: how many samples per (question, setting) each stage draws, and on which questions,
  - max_tokens (the token limit of Appendix C),
  - the vLLM engine arguments and the extra SamplingParams arguments the paper's runs passed,
  - how the completion text was obtained (vLLM's text, or the token ids decoded with special tokens kept),
  - how requests were batched and split over GPUs, and which vLLM version ran.

The values are those the runs recorded about themselves (engine logs and per-file provenance), not those of
plans. Where a run's recorded value differs from the cleaner choice, the profile keeps the run's value and the
`notes` say so. The sampler (sample_vllm.py) reads a profile and lets every value be overridden on the command
line; the overrides are recorded in every output file.

Batch size (max_num_seqs), batching pattern and engine seed change WHICH samples come out (how the random draws
interleave), not their distribution. They are recorded so that a rerun is configured like the paper's runs, but a
rerun reproduces the distribution, not the paper's samples one for one.

    python -m onetemp.sampling.paper_runs                 # list the profiles
    python -m onetemp.sampling.paper_runs hitom/qwen3-0.6b   # print one profile as JSON
"""
import json
import sys

# ---------------------------------------------------------------------------------------------------------------
# Models (Appendix C; checklist "Licenses"). Revisions are the Hugging Face commits the runs loaded.
# ---------------------------------------------------------------------------------------------------------------
MODELS = {
    "mistral-7b": {"repo": "mistralai/Mistral-7B-Instruct-v0.3",
                   "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71", "family": "mistral"},
    "qwen3-8b": {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "family": "qwen3"},
    "qwen3-0.6b": {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                   "family": "qwen3"},
    "gpt-oss-20b": {"repo": "openai/gpt-oss-20b", "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
                    "family": "gptoss"},
}
REPO_TO_KEY = {v["repo"]: k for k, v in MODELS.items()}


def model_key(name):
    """'mistral-7b' or 'mistralai/Mistral-7B-Instruct-v0.3' -> 'mistral-7b'."""
    if name in MODELS:
        return name
    if name in REPO_TO_KEY:
        return REPO_TO_KEY[name]
    raise SystemExit(f"unknown model {name!r}; known: {sorted(MODELS)} or {sorted(REPO_TO_KEY)}")


# ---------------------------------------------------------------------------------------------------------------
# Setting grids (Appendix C tables; Appendix I "Setup").
# ---------------------------------------------------------------------------------------------------------------
# "16: 0.1, 0.2, ..., 1.2, 1.4, 1.5, 1.7, 2.0"
GRID_16 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0)
# "20: 0.1, 0.2, ..., 2.0"
GRID_20 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
# "38: 0.1, 0.2, ..., 3.5, 3.7, 4.0, 4.2" (Qwen3-8B on LiveCodeBench): the 16 values above, then 22 more
GRID_38 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
           2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 4.0, 4.2)
# The 22 temperatures of the second Qwen3-8B LiveCodeBench round (GRID_38 minus GRID_16)
GRID_38_ROUND2 = tuple(t for t in GRID_38 if t not in GRID_16)
# "31: 0.075, 0.15, ..., 1.95, 2.0, 2.2, 2.5, 3.0, 4.0" (Hi-ToM). Decimal literals as the runs used them, never
# computed as k * 0.075 (floating point would give 0.22499999999999998 for k = 3).
GRID_31 = (0.075, 0.15, 0.225, 0.3, 0.375, 0.45, 0.525, 0.6, 0.675, 0.75, 0.825, 0.9, 0.975, 1.05, 1.125, 1.2,
           1.275, 1.35, 1.425, 1.5, 1.575, 1.65, 1.725, 1.8, 1.875, 1.95, 2.0, 2.2, 2.5, 3.0, 4.0)
# Appendix I: "The grid at tau = 1.5 has the 10 values ..., and that at tau = 2.0 has the 5 values ..."
MINP_GRID_TAU15 = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7)
MINP_GRID_TAU20 = (0.0056, 0.0128, 0.0316, 0.0721, 0.178)

assert len(GRID_16) == 16 and len(GRID_20) == 20 and len(GRID_38) == 38 and len(GRID_31) == 31
assert len(GRID_38_ROUND2) == 22 and len(MINP_GRID_TAU15) == 10 and len(MINP_GRID_TAU20) == 5

# LiveCodeBench stop strings, identical in all four generators of the paper's runs
LCB_STOP = ["```\n```", "```\n\n", "\n\n\n"]
# gpt-oss: <|return|> <|call|> <|end|> <|endoftext|> (MATH-500 and Hi-ToM runs)
GPTOSS_STOP_IDS = [200002, 200012, 200007, 199999]

# The paper's first-stage keep rules (Appendix C "Datasets"; Appendix I "Setup"). peak = the highest measured
# pass rate c/n of a question over the stage's setting grid, on that stage's samples only.
#   keep iff low_exclusive < peak  and  peak < high (or <= high when high_inclusive)
SCREEN_RULES = {
    # Hi-ToM: "the questions whose best single-temperature pass rate was below 0.30 in a first stage"; no lower
    # bound (with multiple choice a guess can pass)
    "hitom": {"high": 0.30, "high_inclusive": False, "low_exclusive": None},
    # min_p screen of Qwen3-0.6B at tau = 1.5: "kept a question if and only if the pass rate of its best value was
    # below 0.30", over all 10 values including 0
    "minp": {"high": 0.30, "high_inclusive": False, "low_exclusive": None},
    # LiveCodeBench, Mistral: 0 < peak < 30% on the 1,500 first-stage samples
    "lcb-mistral": {"high": 0.30, "high_inclusive": False, "low_exclusive": 0.0},
    # LiveCodeBench, GPT-OSS 20B and Qwen3-8B: 0 < peak <= 30% (same sets as < on the paper's data)
    "lcb": {"high": 0.30, "high_inclusive": True, "low_exclusive": 0.0},
}

# ---------------------------------------------------------------------------------------------------------------
# Engine argument blocks. Every key is a vLLM LLM(...) argument.
# ---------------------------------------------------------------------------------------------------------------
_COMMON_019 = dict(enable_prefix_caching=True, enable_chunked_prefill=True, enforce_eager=False,
                   max_num_batched_tokens=32768, gpu_memory_utilization=0.90)


def _eng(**kw):
    d = dict(_COMMON_019)
    d.update(kw)
    return d


def _stage(name, n, questions, note="", settings=None):
    """A sampling stage: n samples per (question, setting) on `questions` ("all", "kept" = the screen's output,
    or "file:<ids file>"), at the run's settings unless the stage has its own."""
    s = {"name": name, "n": n, "questions": questions, "note": note}
    if settings is not None:
        s["settings"] = tuple(settings)
    return s


RUNS = {}

# ===================================== MATH-500 (Appendix C) =====================================================
# All 500 questions for every model, one stage (no deeper stage ran). vLLM 0.19.0. One generate call per
# (question, temperature) with SamplingParams(n = all samples), questions in order, temperatures ascending; the
# questions were split over single-GPU machines, so every cell comes from one engine.
RUNS["math500/mistral-7b"] = dict(
    benchmark="math500", model="mistral-7b", prompt_style="math500_mistral", axis="temperature",
    settings=GRID_16, max_tokens=4096,
    stages=[_stage("main", 1500, "all")],
    engine=_eng(dtype="half", max_model_len=5120, max_num_seqs=128, kv_cache_dtype="auto",
                generation_config="vllm", tokenizer_mode="mistral", config_format="mistral", load_format="mistral"),
    sampling={}, text="vllm_text", batching="cell", split="questions", engine_seed=None,
    vllm="0.19.0", gpus="NVIDIA A40",
    notes=["Weights, config and tokenizer loaded in Mistral's own format (consolidated.safetensors, params.json, "
           "tokenizer.model.v3); the prompt ids are built with the Hugging Face template and equal the "
           "mistral_common ids on all 500 questions.",
           "generation_config='vllm': no sampling default of the checkpoint applies; top_p=1, top_k off, min_p 0."])

RUNS["math500/qwen3-8b"] = dict(
    benchmark="math500", model="qwen3-8b", prompt_style="math500_qwen3_8b_concise", axis="temperature",
    settings=GRID_20, max_tokens=2048,
    stages=[_stage("main", 750, "all")],
    engine=_eng(dtype="bfloat16", max_model_len=3072, max_num_seqs=64, kv_cache_dtype="auto",
                generation_config="vllm", tokenizer_mode="hf", config_format="hf", load_format="safetensors"),
    sampling={}, text="vllm_text", batching="cell", split="questions", engine_seed=None,
    vllm="0.19.0", gpus="NVIDIA GeForce RTX 5090",
    notes=["750 samples and a 2,048-token limit, with the instruction followed by a sentence that states the limit "
           "(Appendix C).",
           "generation_config='vllm' keeps Qwen3's checkpoint defaults (T 0.6, top_p 0.95, top_k 20) from applying."])

RUNS["math500/qwen3-0.6b"] = dict(
    benchmark="math500", model="qwen3-0.6b", prompt_style="math500_qwen3", axis="temperature",
    settings=GRID_20, max_tokens=4096,
    stages=[_stage("main", 1500, "all")],
    engine=_eng(dtype="half", max_model_len=5120, max_num_seqs=256, kv_cache_dtype="auto",
                generation_config="vllm", tokenizer_mode="hf", config_format="hf", load_format="safetensors"),
    sampling={}, text="vllm_text", batching="cell", split="questions", engine_seed=None,
    vllm="0.19.0", gpus="NVIDIA GeForce RTX 5090",
    notes=["float16 on the bfloat16 checkpoint, as in Appendix C."])

RUNS["math500/gpt-oss-20b"] = dict(
    benchmark="math500", model="gpt-oss-20b", prompt_style="math500_harmony", axis="temperature",
    settings=GRID_16, max_tokens=4096,
    stages=[_stage("main", 1500, "all")],
    engine=_eng(dtype="bfloat16", max_model_len=5120, max_num_seqs=1024, kv_cache_dtype="auto"),
    sampling={"stop_token_ids": GPTOSS_STOP_IDS, "skip_special_tokens": True}, text="vllm_text",
    batching="cell", split="questions", engine_seed=None, vllm="0.19.0", gpus="NVIDIA L40S",
    notes=["bfloat16 activations with the checkpoint's native MXFP4 expert weights (detected by vLLM).",
           "Harmony prompt with the date line pinned to 2026-10-10 and the response primed with "
           "<|channel|>final<|message|> (prompts.py)."])

# ===================================== Hi-ToM (Appendix C) =======================================================
# First stage on all 1,200 questions, keep iff best single-temperature pass rate < 0.30, then a second stage on the
# kept questions; the paper pools both stages (500 + 1,000 = 1,500 per cell; Mistral 1,000 + 500).
# vLLM 0.19.0 on NVIDIA L40S for the three newer runs. These ran as one engine per GPU that took one question at a
# time and queued all 31 temperature cells of it together (batching="question"); each engine start drew a random
# engine seed and recorded it. The completion text is the generated token ids decoded with special tokens kept, so
# Qwen3 texts end in <|im_end|> and GPT-OSS texts in its Harmony stop token; the grader removes them.
_HITOM_DECODE = {"skip_special_tokens": False, "detokenize": False}

RUNS["hitom/qwen3-0.6b"] = dict(
    benchmark="hitom", model="qwen3-0.6b", prompt_style="hitom_qwen3", axis="temperature",
    settings=GRID_31, max_tokens=1000,
    stages=[_stage("screen", 500, "all"), _stage("topup", 1000, "kept")],
    screen_rule="hitom",
    engine=_eng(dtype="half", max_model_len=1830, max_num_seqs=256, kv_cache_dtype="auto",
                generation_config="vllm", tokenizer_mode="hf", config_format="hf", load_format="safetensors"),
    sampling=dict(_HITOM_DECODE), text="decode_ids", batching="question", split="questions",
    engine_seed="random", vllm="0.19.0", gpus="NVIDIA L40S",
    kept_file="question_sets/hitom_kept_qwen3-0.6b.json",
    notes=["max_model_len = longest prompt (830 tokens) + 1,000."])

RUNS["hitom/qwen3-8b"] = dict(
    benchmark="hitom", model="qwen3-8b", prompt_style="hitom_qwen3", axis="temperature",
    settings=GRID_31, max_tokens=1000,
    stages=[_stage("screen", 500, "all"), _stage("topup", 1000, "kept")],
    screen_rule="hitom",
    engine=_eng(dtype="bfloat16", max_model_len=1830, max_num_seqs=128, kv_cache_dtype="auto",
                generation_config="vllm", tokenizer_mode="hf", config_format="hf", load_format="safetensors"),
    sampling=dict(_HITOM_DECODE), text="decode_ids", batching="question", split="questions",
    engine_seed="random", vllm="0.19.0", gpus="NVIDIA L40S",
    kept_file="question_sets/hitom_kept_qwen3-8b.json",
    notes=["max_num_seqs 128 on L40S (256 and 512 preempted)."])

RUNS["hitom/gpt-oss-20b"] = dict(
    benchmark="hitom", model="gpt-oss-20b", prompt_style="hitom_harmony", axis="temperature",
    settings=GRID_31, max_tokens=1000,
    stages=[_stage("screen", 500, "all"), _stage("topup", 1000, "kept")],
    screen_rule="hitom",
    engine=_eng(dtype="bfloat16", max_model_len=1839, max_num_seqs=1024, kv_cache_dtype="auto"),
    sampling=dict(_HITOM_DECODE, stop_token_ids=GPTOSS_STOP_IDS), text="decode_ids", batching="question",
    split="questions", engine_seed="random", vllm="0.19.0", gpus="NVIDIA L40S",
    kept_file="question_sets/hitom_kept_gpt-oss-20b.json",
    notes=["max_model_len = longest prompt (839 tokens) + 1,000."])

RUNS["hitom/mistral-7b"] = dict(
    benchmark="hitom", model="mistral-7b", prompt_style="hitom_mistral_cap500", axis="temperature",
    settings=GRID_31, max_tokens=500,
    stages=[_stage("screen", 1000, "all"), _stage("topup", 500, "kept")],
    screen_rule="hitom",
    engine=_eng(dtype="half", max_model_len=2048, max_num_seqs=256, trust_remote_code=True,
                tokenizer_mode="hf", config_format="hf", load_format="safetensors"),
    sampling={}, text="vllm_text", batching="cell", split="questions", engine_seed=None,
    vllm="0.19.0", gpus="NVIDIA GeForce RTX 5090 (first stage); NVIDIA L40S (most of the second stage)",
    kept_file="question_sets/hitom_kept_mistral-7b.json",
    notes=["First stage: 8 engines (one per GPU), questions dealt round robin, one generate call per (question, "
           "temperature), vLLM's default engine seed 0, the Hugging Face tokenizer (the model directory held only "
           "the Hugging Face weight files; tokenizer_mode/config_format/load_format 'hf' is the explicit equivalent).",
           "Second stage (+500): mostly a run configured like the first stage on L40S GPUs with a random engine seed; "
           "for 5,734 of the 19,127 cells it is the first 500 samples of a 5,000-sample request.",
           "The prompt is the dataset prompt followed by an instruction that states the 500-token limit."])

# ===================================== min_p on Hi-ToM (Appendix I) ==============================================
# temperature = the fixed tau, min_p varies. top_p = 1 and top_k off (vLLM defaults). min_p is passed to vLLM only
# when it is above 0 (min_p = 0 is plain sampling, vLLM's default). Same engines and prompts as the Hi-ToM runs.
RUNS["minp/qwen3-0.6b/tau1.5"] = dict(
    benchmark="hitom", model="qwen3-0.6b", prompt_style="hitom_qwen3", axis="min_p", tau=1.5,
    settings=MINP_GRID_TAU15, max_tokens=1000,
    stages=[_stage("screen", 100, "all"), _stage("depth", 400, "kept")],
    screen_rule="minp",
    engine=dict(RUNS["hitom/qwen3-0.6b"]["engine"]),
    sampling=dict(_HITOM_DECODE), text="decode_ids", batching="question", split="questions",
    engine_seed="random", vllm="0.19.0",
    gpus="NVIDIA GeForce RTX 5090 (screen: 1,090 questions on RTX 5090, 110 on one L40S; second stage RTX 5090)",
    kept_file="question_sets/minp_kept_qwen3-0.6b.json",
    notes=["Pooled n = 500 per value (100 + 400). The tau = 2.0 study reuses this kept list (648 questions)."])

RUNS["minp/qwen3-0.6b/tau2.0"] = dict(
    benchmark="hitom", model="qwen3-0.6b", prompt_style="hitom_qwen3", axis="min_p", tau=2.0,
    settings=MINP_GRID_TAU20, max_tokens=1000,
    stages=[_stage("depth", 400, "file:question_sets/minp_kept_qwen3-0.6b.json")],
    engine=dict(RUNS["hitom/qwen3-0.6b"]["engine"]),
    sampling=dict(_HITOM_DECODE), text="decode_ids", batching="question", split="questions",
    engine_seed="random", vllm="0.19.0", gpus="RTX 5090 (351 questions), L40S (270), RTX 4090 (27)",
    notes=["All 5 values of a question ran on one GPU (here: one engine per question set, batching='question')."])

RUNS["minp/mistral-7b/tau2.0"] = dict(
    benchmark="hitom", model="mistral-7b", prompt_style="hitom_mistral_cap500", axis="min_p", tau=2.0,
    settings=MINP_GRID_TAU20, max_tokens=500,
    stages=[_stage("depth", 1500, "file:question_sets/hitom_kept_mistral-7b.json")],
    engine=dict(RUNS["hitom/mistral-7b"]["engine"]),
    sampling={}, text="vllm_text", batching="question", split="questions", engine_seed="random",
    vllm="0.19.0", gpus="RTX 5090 (330 questions), L40S (287)",
    notes=["The 617 questions kept by the first stage of hitom/mistral-7b, no further screen."])

# ===================================== LiveCodeBench (Appendix C) ================================================
# The 975 questions of question_sets/livecodebench_975.json (release_v6 = V1..V6 of code_generation_lite). Prompt:
# the problem statement, then the starter code or a stdin instruction; the response is primed to open a ```python
# block; stop strings LCB_STOP; max_tokens 500. Every GPU ran every question with its share of the samples
# (split="samples"): n per (question, T) = sum over GPUs. One generate call per (question, temperature, GPU).
# Engine seed: vLLM's default on every GPU. Stage 2 drew 10,000 more samples on the hard questions (screen rule).
_LCB_ENGINE_015 = dict(enable_prefix_caching=True, enable_chunked_prefill=True, enforce_eager=False,
                       max_num_batched_tokens=32768, gpu_memory_utilization=0.95, max_num_seqs=256,
                       trust_remote_code=True)

RUNS["livecodebench/qwen3-0.6b"] = dict(
    benchmark="livecodebench", model="qwen3-0.6b", prompt_style="lcb_raw", axis="temperature",
    settings=GRID_20, max_tokens=500,
    stages=[_stage("main", 9996, "file:question_sets/livecodebench_qwen3-0.6b_110.json",
                   "6 GPUs x 1,666 samples"),
            _stage("deep", 10000, "file:question_sets/livecodebench_qwen3-0.6b_deep30.json",
                   "+10,000 at the 16 temperatures of GRID_16 on 30 of the 110 (how the 30 were chosen is not "
                   "recorded)", settings=GRID_16),
            _stage("deep2", 100000, "file:question_sets/livecodebench_qwen3-0.6b_deep10.json",
                   "+100,000 (4 GPUs x 25,000) at GRID_16 on 10 of the 30", settings=GRID_16)],
    engine=dict(_LCB_ENGINE_015, dtype="half", max_model_len=1400),
    sampling={"stop": LCB_STOP}, text="vllm_text", batching="cell", split="samples", engine_seed=None,
    vllm="0.15.1 (the 100,000-sample stage; the base run's version was not logged)", gpus="not recorded; L40S",
    notes=["Raw text prompt, no chat template (Qwen3-0.6B was run as a completion model on LiveCodeBench).",
           "The 110 questions are the paper's; the rule that chose them is not recorded."])

RUNS["livecodebench/mistral-7b"] = dict(
    benchmark="livecodebench", model="mistral-7b", prompt_style="lcb_mistral", axis="temperature",
    settings=GRID_16, max_tokens=500,
    stages=[_stage("main", 1500, "all", "4 GPUs x 375 (first segment) or 6 x 250"),
            _stage("deep", 10000, "kept", "6 GPUs x 1,667 = 10,002, or 4 x 2,500; pooled with the 1,500")],
    screen_rule="lcb-mistral", pooling="sum",
    engine=dict(_LCB_ENGINE_015, dtype="half", max_model_len=1550, tokenizer_mode="mistral"),
    sampling={"stop": LCB_STOP}, text="vllm_text", batching="cell", split="samples", engine_seed=None,
    vllm="0.15.1", gpus="not recorded",
    kept_file="question_sets/livecodebench_deep_mistral-7b.json",
    notes=["vLLM 0.15.1 resolved tokenizer_mode='auto' to the Mistral tokenizer (mistral_common) for this hub "
           "repository; the prompt ids here are built with the Hugging Face template and equal it.",
           "A third, 20,000-sample run on 0 < peak < 1% questions is not in the paper's data."])

RUNS["livecodebench/qwen3-8b"] = dict(
    benchmark="livecodebench", model="qwen3-8b", prompt_style="lcb_qwen3", axis="temperature",
    settings=GRID_38, max_tokens=500,
    stages=[_stage("main", 1500, "all", "4 GPUs x 375; first the 16 temperatures of GRID_16, later the 22 of "
                                        "GRID_38_ROUND2"),
            _stage("deep", 10000, "kept", "4 GPUs x 2,500 per run; pooled with the 1,500; some cells got two or "
                                          "three runs (up to 19,000 in all)")],
    screen_rule="lcb", pooling="sum",
    engine=dict(enable_prefix_caching=True, enable_chunked_prefill=True, enforce_eager=False,
                max_num_batched_tokens=32768, gpu_memory_utilization=0.95, max_num_seqs=1300,
                trust_remote_code=True, dtype="bfloat16", max_model_len=1353),
    sampling={"stop": LCB_STOP}, text="vllm_text", batching="cell", split="samples", engine_seed=None,
    vllm="0.19.0 (most of the data; the first 16-temperature round ran 0.8.5.post1 with max_num_seqs 512, and one "
         "stage ran 0.18.1 on one RTX 4090)",
    gpus="NVIDIA L40S (and one RTX 4090)",
    kept_file="question_sets/livecodebench_deep_qwen3-8b.json",
    allow_clamp=True,
    notes=["max_model_len 1353 as run: 5 of the 975 prompts are longer than 853 tokens, so vLLM gave them fewer than "
           "500 new tokens (the sampler refuses this unless --allow-clamp, which this profile sets).",
           "max_num_seqs varied between runs (512, 64, 256, 1536, 1300); 1300 is the value of most of the data."])

RUNS["livecodebench/gpt-oss-20b"] = dict(
    benchmark="livecodebench", model="gpt-oss-20b", prompt_style="lcb_harmony", axis="temperature",
    settings=GRID_16, max_tokens=500,
    stages=[_stage("main", 1500, "all", "4 GPUs x 375"),
            _stage("deep", 10000, "kept", "4 GPUs x 2,500; REPLACES the 1,500 of these questions (not pooled)")],
    screen_rule="lcb", pooling="replace",
    engine=dict(_LCB_ENGINE_015, dtype="bfloat16", max_model_len=1450),
    sampling={"stop": LCB_STOP}, text="vllm_text", batching="cell", split="samples", engine_seed=None,
    vllm="0.15.1", gpus="not recorded",
    kept_file="question_sets/livecodebench_deep_gpt-oss-20b.json",
    notes=["max_model_len 1450 for the V5-V6 questions and 1410 for V1-V4 (both fit every prompt + 500).",
           "The Harmony template stamps the render day into the system message; the runs rendered on the GPU "
           "machine, so their prompts carry 2026-02-27 to 2026-03-06. Pin one with --harmony-date.",
           "No stop_token_ids beyond the checkpoint's own end-of-sequence ids."])


def get_run(key):
    if key not in RUNS:
        raise SystemExit(f"unknown run {key!r}; known runs:\n  " + "\n  ".join(sorted(RUNS)))
    return RUNS[key]


def stage_of(run, name):
    for s in run["stages"]:
        if s["name"] == name:
            return s
    raise SystemExit(f"run has no stage {name!r}; stages: {[s['name'] for s in run['stages']]}")


def main(argv):
    if len(argv) > 1:
        print(json.dumps(get_run(argv[1]), indent=1))
        return
    for k in sorted(RUNS):
        r = RUNS[k]
        st = ", ".join(f"{s['name']} n={s['n']} on {s['questions']}" for s in r["stages"])
        print(f"{k:28s} {r['axis']:11s} {len(r['settings']):2d} settings  max_tokens {r['max_tokens']:4d}  {st}")


if __name__ == "__main__":
    main(sys.argv)
