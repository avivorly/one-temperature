# One Temperature to Rule Them All? — code

Code for the NeurIPS 2026 paper *One Temperature to Rule Them All?*

The paper measures, for every question, how the single-attempt success probability P(τ) changes with the
sampling temperature τ, and asks when mixing temperatures helps under pass@K. It derives an efficient way
to compute the optimal multi-temperature allocation (a convex program over how many of the K samples go
to each temperature), and finds that its gain over the best single temperature *tuned for the budget K* is
modest.

## Pipeline

| step | what | script |
|---|---|---|
| 1 | Sample n completions per question at every temperature of a grid, with vLLM | `generate_vllm.py` |
| 2 | Grade every completion; write correct counts per (question, temperature) | `grade_math500.py`; Hi-ToM and LiveCodeBench: rules below |
| 3 | Per-question pass rates and pass@K per temperature | `passk.py` |
| 4 | Best single temperature vs uniform mixing vs optimal allocation, with held-out (leave-one-out or 10-fold) evaluation | `optimize_temperatures.py` |
| 5 | Robustness of the comparison to reweighting the question set by hardness | not included (see below) |

Everything after step 1 runs on a CPU. `smoke_test.py` runs steps 3 and 4 on synthetic counts and needs
no model, GPU or data.

The files passed between steps:

```
generation:  <out_dir>/T0.7000.jsonl one line per question:
             {"id", "temperature", "completions": [...], "finish_reasons": [...], "provenance": {...}}
counts:      counts.jsonl            one line per (question, temperature): {"id", "temperature", "n", "c"}
```

`c` is the number of the `n` completions graded correct. Keep one output directory per sampling
configuration (model, max_tokens, prompt); temperatures are separate files inside it.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # vLLM 0.19.0, as in the paper; only step 1 needs a GPU
python smoke_test.py                     # steps 3-4 on synthetic data
```

## Models and benchmarks

| model | link | dtype / chat format used |
|---|---|---|
| GPT-OSS 20B | <https://huggingface.co/openai/gpt-oss-20b> | bfloat16 with the checkpoint's native MXFP4; Harmony format, reasoning effort `medium`, response primed to open in the `final` channel (MATH-500, Hi-ToM) |
| Mistral 7B Instruct v0.3 | <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3> | float16, Mistral chat template |
| Qwen3-8B | <https://huggingface.co/Qwen/Qwen3-8B> | bfloat16, Qwen3 chat template, thinking disabled |
| Qwen3-0.6B | <https://huggingface.co/Qwen/Qwen3-0.6B> | float16, Qwen3 chat template, thinking disabled |

| benchmark | get it from | grading |
|---|---|---|
| LiveCodeBench V1-V6 (975 questions) | <https://github.com/LiveCodeBench/LiveCodeBench>, dataset `livecodebench/code_generation_lite` on the Hugging Face hub, `release_v1` ... `release_v6` | unit tests, official harness |
| MATH-500 (500 questions) | <https://huggingface.co/datasets/HuggingFaceH4/MATH-500> (split `test`); the PRM800K test split, <https://github.com/openai/prm800k> | last `\boxed{}` answer vs reference |
| Hi-ToM (1,200 questions) | <https://github.com/ying-hui-he/Hi-ToM_dataset> | first valid option named in the completion |

No benchmark data, reference answers, test cases or results are included in this repository; download each
benchmark from its source above.

Sampling settings of the paper (Appendix "Computational Resources"):

| | samples per (question, τ) | temperature grid | max new tokens |
|---|---|---|---|
| LiveCodeBench | about 1,500 (about 10,000 on harder questions) | 16: 0.1, 0.2, ..., 1.2, 1.4, 1.5, 1.7, 2.0 (Qwen3-8B: 38 up to 4.2; Qwen3-0.6B: 20 up to 2.0) | see the LiveCodeBench harness |
| MATH-500 | 1,500 (Qwen3-8B: 750) | the same 16 (Qwen3 models: 20, 0.1 ... 2.0) | 4096 (Qwen3-8B: 2048) |
| Hi-ToM | 1,500 | 31: 0.075, 0.15, ..., 1.95, 2.0, 2.2, 2.5, 3.0, 4.0 | 1000 (Mistral: 500) |

On Hi-ToM the paper analyzes the questions whose best single-temperature pass rate was below 0.30 in a first
stage of 500 samples per question and temperature (1,000 for Mistral 7B Instruct).

## Step 1: sampling with vLLM

```bash
# MATH-500, Mistral 7B Instruct, fp16, 1,500 samples per question at each of the 16 temperatures
python generate_vllm.py --benchmark math500 \
    --model mistralai/Mistral-7B-Instruct-v0.3 --dtype half \
    --temperatures 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.5,1.7,2.0 \
    --n 1500 --max-tokens 4096 --out-dir runs/mistral7b_math500

# Any other benchmark: a JSONL of {"id", "prompt"} (prompts only, never answers)
python generate_vllm.py --questions questions.jsonl --model ... --temperatures ... --out-dir runs/...
```

How we ran vLLM (see the comments in `generate_vllm.py`):

- One process per GPU. We split the temperature grid (or the question list) across GPUs with
  `CUDA_VISIBLE_DEVICES=<g> python generate_vllm.py --temperatures <subset> ...`; every process writes its
  own `T*.jsonl` files into the same output directory.
- `dtype`, `max_model_len` and `max_num_seqs` are held fixed across all temperatures of a model: they change
  vLLM's batching and therefore which samples come out.
- The prompt is the model's chat template applied to the question. On MATH-500 the instruction *"Solve the
  following math problem efficiently and clearly. Reason step by step, and put your final answer within
  \boxed{}."* is the system turn and the problem is the user turn. On Hi-ToM the user turn is the question's
  prompt exactly as the dataset gives it (for Mistral 7B Instruct, followed by a fixed instruction that states
  its token limit). GPT-OSS 20B's Harmony format is not built by this script.
- Every output line records its full sampling configuration and the sha256 of the exact prompt text.

## Step 2: grading

```bash
# MATH-500: save the HF test split as JSONL first, or omit --dataset to load it from the hub
python grade_math500.py --gen-dir runs/mistral7b_math500 --dataset math500_test.jsonl --out counts.jsonl
```

- **MATH-500** (`grade_math500.py`, the frozen grader of the paper's runs). The answer is the content of the
  *last* `\boxed{}` span of the completion; a completion with no `\boxed{}` fails. A sample passes if its answer
  equals the reference after normalization, or if an equivalence layer of generic rules and a time-bounded
  SymPy check accepts it. Model output is parsed with a restricted `parse_expr`, never `eval`/`sympify`; a
  timeout or a grader error is a fail. The counts file also carries `c_strict`, the exact layer alone.
- **Hi-ToM** (no script in this release). The prompt lists all options, so grade from the completion only,
  never from the prompt: the answer is the first valid option the completion mentions (one of the question's
  own options, by its name or by its letter given as an answer label), compared by exact match with the
  reference container. A completion that names no valid option fails.
- **LiveCodeBench** (no script in this release). Use the official evaluation of
  <https://github.com/LiveCodeBench/LiveCodeBench> (`lcb_runner`): run each completion's extracted code against
  the problem's tests; a completion passes only if every test passes.

For Hi-ToM and LiveCodeBench, write the same counts file: one line per (question, temperature) with `n` =
completions and `c` = completions that passed. Count every completion (instance-weighted), even when many
completions are identical.

## Steps 3-4: pass@K and the temperature-allocation comparison

```bash
python passk.py counts.jsonl                              # pass@K per temperature on the paper's K grid
python optimize_temperatures.py counts.jsonl              # leave-one-out (the paper's tables, red traces)
python optimize_temperatures.py counts.jsonl --folds 10   # 10-fold
python optimize_temperatures.py counts.jsonl --out results.json
```

For K in {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024},
`optimize_temperatures.py` prints, as ratios to the uniform scheme (K/l samples at each grid temperature):

- the best single temperature (chosen in-sample on all included questions, as in the paper's tables);
- the optimal allocation in-sample: the convex relaxation of
  `mean_i exp(sum_j K_j ln(1 - P_ij))`, `sum_j K_j = K`, solved by SLSQP with its analytic gradient and
  warm starts, followed by an integer search around the rounded optimum;
- the optimal allocation held-out: refit on the training questions, scored on the held-out ones
  (leave-one-out by default; `--folds 10` uses folds from `numpy.random.default_rng(0)`).

Questions whose peak pass rate is below `--min-peak` (default 1e-3, as in the paper) are excluded first.

## What is not included

- **Profile fitting.** The paper fits each question's temperature profile P(τ) with a non-negative, unimodal
  degree-10 Bernstein polynomial (weighted least squares with Clopper-Pearson weights, projected gradient
  descent) and evaluates the allocation on the fitted profiles. This package uses the measured rates
  c / n directly; with 1,500 samples per cell the two are close, but numbers can differ slightly from the paper.
- **The integer-optimality certificate.** The paper's code certifies the integer optimum by a KKT-bound
  branch-and-bound and cross-checks the relaxation with CLARABEL; here the integer search is a radius-2
  box around the rounded optimum plus 1-swap descent (marked `Simplified` in the code).
- **Bootstrap bands** on the ratio-to-uniform curves (paired bootstrap over questions).
- **Step 5, hardness reweighting** (Appendices on sensitivity to the hardness distribution and on reweighting
  outside a protected hardness window): the comparison of step 4 repeated on question sets reweighted by
  hardness. Not included in this release.
- Plotting code for the figures.

## License

MIT, see `LICENSE`.
