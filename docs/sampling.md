# Sampling: drawing the paper's samples with vLLM

**In short.** `onetemp.sampling` redraws the samples behind every experiment of the paper: four models
(Mistral-7B-Instruct-v0.3, Qwen3-8B, Qwen3-0.6B, gpt-oss-20b) on three benchmarks (LiveCodeBench, MATH-500,
Hi-ToM), at every temperature of the grids of Appendix C, and the min_p runs of Appendix I. It builds the exact
prompts from the public datasets, checks them against the prompt hashes shipped with the package, samples with
vLLM, and writes one completions file per (question, setting). The graders (`docs/grading.md`) turn those files
into `counts.jsonl`.

- Building and checking prompts needs no GPU.
- Sampling needs a GPU and vLLM 0.19.0.
- A rerun reproduces the paper's sampling distribution, not its samples one for one (see the last section).

## What is in the package

All modules run as `python -m onetemp.sampling.<name>` from the repository root.

| module | what it does |
|---|---|
| `paper_runs` | Every run of the paper as data: grid, samples per stage, questions, max_tokens, vLLM engine arguments, extra sampling arguments, text mode, batching, vLLM version. `python -m onetemp.sampling.paper_runs` lists the runs; add a run name to print its full profile as JSON. |
| `prompts` | The exact prompt of every (benchmark, model), as token ids: the MATH-500 instruction, the Hi-ToM prompt (plus Mistral's 500-token instruction), the LiveCodeBench body with the ```` ```python ```` priming, Qwen3 with thinking disabled, gpt-oss in the Harmony format primed to open in the final channel. |
| `public_data` | Loads the three benchmarks from their public sources at pinned revisions, checked by sha256. Only question text is kept, never answers or tests. |
| `build_prompts` | Renders a run's prompts off the GPU into a JSONL file of token ids, with hashes. |
| `verify_prompts` | Checks a prompt file against the paper's prompt hashes in `onetemp/sampling/expected_prompts/`. |
| `sample_vllm` | The sampler, one GPU. Resumable, one file per cell, refuses a silently clamped token budget. |
| `launch` | One sampler per GPU, questions or samples split between them; decides completion by the files. |
| `screen` | The first-stage screen: keep a question for the second stage by its best setting's pass rate. |
| `pool_counts` | Pools the counts of two stages as the paper did (sum, or replace); `--keep` writes only the questions a screen kept. |
| `fake_engine` | A deterministic stand-in for vLLM, used by the tests and by `--engine fake`. |
| `question_sets/` | The paper's question lists, ids only: the 975 LiveCodeBench questions with their release file and row, Qwen3-0.6B's 110, every second-stage list, the Hi-ToM kept lists, the min_p kept list. |
| `expected_prompts/` | Per question: what the paper's runs recorded about the prompt they sent (token count, sha256 of the ids, sha256 of the text). |

## Install

```bash
pip install -e ".[sampling,dev]"    # vllm==0.19.0, transformers==4.57.6 (GPU machine), and pytest for the tests
python -m pytest                    # synthetic tests, no GPU, no model, no network; tests/test_sampling.py covers this part
python -m onetemp.sampling.paper_runs    # the runs and their stages
```

Rebuilding and checking prompts needs no GPU: only `transformers` 4.57.6 with `jinja2` (the chat templates),
`sentencepiece` and `protobuf` (Mistral's tokenizer), and network access to the Hugging Face hub for the
tokenizers. On a machine without a GPU, install them with `pip install -e ".[prompts]"`.

## Where the questions come from

The package ships question ids, never question text. The text is downloaded at run time from the public sources,
at pinned revisions, and checked by sha256 before use. Downloads go to the cache shared with the graders,
`~/.cache/one-temperature` (`$XDG_CACHE_HOME/one-temperature` if set), so each file is fetched once.
`--cache-dir` changes the folder and `--no-verify-data` skips the hash check.

| benchmark | source | what is read |
|---|---|---|
| MATH-500 | `HuggingFaceH4/MATH-500` at revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, `test.jsonl` (446,564 bytes; sha256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`) | `unique_id` and `problem` |
| Hi-ToM | `github.com/ying-hui-he/Hi-ToM_dataset` at commit `459784fa8de09f35e9110e1e9454b41cd9b4f985`, `Hi-ToM_data/Hi-ToM_data.json` (5,134,393 bytes; sha256 `088b6c54d21d1a0d3045812f36ac8b5b342346a6a996c003738976730ac256f4`), 1,200 records | the id `s{sample_id}_o{question_order}_{prompting_type}` and the dataset's own `prompt` |
| LiveCodeBench | `livecodebench/code_generation_lite` at revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`; release version vN is file `test{N}.jsonl` (`test.jsonl` for v1); the six sha256 values are in `docs/grading.md` | `question_id`, `question_content`, `starter_code` |

- The 975 LiveCodeBench questions are listed with their release version and row in
  `onetemp/sampling/question_sets/livecodebench_975.json` (369, 107, 98, 87, 152 and 162 questions in v1 to v6).
- The six LiveCodeBench files hold the problem statements and the tests, 4.5 GB in all. They are downloaded into
  `<cache>/livecodebench/<revision>/` (the folder the LiveCodeBench grader also uses), or read from
  `--lcb-dir DIR` if the directory holds them under their hub names. Test cases are dropped as each row is read.
- MATH-500 answers and Hi-ToM answers are never loaded by the prompt builders. Hi-ToM is multiple choice, so the
  answer's name is inside every prompt among the 15 options; the graders therefore read only the completion.

## The run profiles

`python -m onetemp.sampling.paper_runs <run>` prints a run's full profile. There is one profile per
(benchmark, model) and one per min_p study. The values are those the paper's runs recorded about themselves
(engine logs and per-file provenance).

### Grids, stages and token limits

Samples are per (question, setting). A setting is a temperature, or on a min_p run a min_p value at the fixed
temperature tau.

| run | settings | stages: samples on which questions | max_tokens |
|---|---|---|---|
| `math500/mistral-7b` | 16 temperatures (0.1, ..., 1.2, 1.4, 1.5, 1.7, 2.0) | `main` 1,500 on all 500 | 4096 |
| `math500/qwen3-8b` | 20 (0.1, ..., 2.0) | `main` 750 on all 500 | 2048 |
| `math500/qwen3-0.6b` | 20 | `main` 1,500 on all 500 | 4096 |
| `math500/gpt-oss-20b` | 16 | `main` 1,500 on all 500 | 4096 |
| `hitom/qwen3-0.6b` | 31 (0.075, 0.15, ..., 1.95, 2.0, 2.2, 2.5, 3.0, 4.0) | `screen` 500 on all 1,200; `topup` 1,000 on the kept 743 | 1000 |
| `hitom/qwen3-8b` | 31 | `screen` 500 on all 1,200; `topup` 1,000 on the kept 471 | 1000 |
| `hitom/gpt-oss-20b` | 31 | `screen` 500 on all 1,200; `topup` 1,000 on the kept 678 | 1000 |
| `hitom/mistral-7b` | 31 | `screen` 1,000 on all 1,200; `topup` 500 on the kept 617 | 500 |
| `minp/qwen3-0.6b/tau1.5` | min_p 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7 at tau = 1.5 | `screen` 100 on all 1,200; `depth` 400 on the kept 648 | 1000 |
| `minp/qwen3-0.6b/tau2.0` | min_p 0.0056, 0.0128, 0.0316, 0.0721, 0.178 at tau = 2.0 | `depth` 400 on the same 648 | 1000 |
| `minp/mistral-7b/tau2.0` | the same 5 values at tau = 2.0 | `depth` 1,500 on Mistral's 617 | 500 |
| `livecodebench/mistral-7b` | 16 | `main` 1,500 on all 975; `deep` 10,000 on the 365 with 0 < peak < 30%, pooled (sum) with `main` | 500 |
| `livecodebench/qwen3-8b` | 38 (0.1, ..., 3.5, 3.7, 4.0, 4.2) | `main` 1,500 on all 975 (the first 16 temperatures first, the other 22 later); `deep` 10,000 per run on the questions with 0 < peak <= 30%, pooled (sum); some cells got two or three runs | 500 |
| `livecodebench/gpt-oss-20b` | 16 | `main` 1,500 on all 975; `deep` 10,000 on the 293 with 0 < peak <= 30%, replacing the 1,500 | 500 |
| `livecodebench/qwen3-0.6b` | 20 | `main` 9,996 (6 x 1,666) on its 110; `deep` +10,000 on 30 of them and `deep2` +100,000 on 10 of those 30, both at 16 temperatures | 500 |

The stage name is what `--stage` takes. The first stage is the default.

### Engine settings per run

| run | dtype | max_model_len | max_num_seqs | vLLM | GPU type |
|---|---|---|---|---|---|
| `math500/mistral-7b` | float16 | 5120 | 128 | 0.19.0 | NVIDIA A40 |
| `math500/qwen3-8b` | bfloat16 | 3072 | 64 | 0.19.0 | NVIDIA GeForce RTX 5090 |
| `math500/qwen3-0.6b` | float16 | 5120 | 256 | 0.19.0 | NVIDIA GeForce RTX 5090 |
| `math500/gpt-oss-20b` | bfloat16 | 5120 | 1024 | 0.19.0 | NVIDIA L40S |
| `hitom/qwen3-0.6b` | float16 | 1830 | 256 | 0.19.0 | NVIDIA L40S |
| `hitom/qwen3-8b` | bfloat16 | 1830 | 128 | 0.19.0 | NVIDIA L40S |
| `hitom/gpt-oss-20b` | bfloat16 | 1839 | 1024 | 0.19.0 | NVIDIA L40S |
| `hitom/mistral-7b` | float16 | 2048 | 256 | 0.19.0 | NVIDIA GeForce RTX 5090 (first stage); NVIDIA L40S (most of the second stage) |
| `minp/qwen3-0.6b/tau1.5` | float16 | 1830 | 256 | 0.19.0 | RTX 5090 (screen: 1,090 questions on RTX 5090, 110 on one L40S; second stage RTX 5090) |
| `minp/qwen3-0.6b/tau2.0` | float16 | 1830 | 256 | 0.19.0 | RTX 5090 (351 questions), L40S (270), RTX 4090 (27) |
| `minp/mistral-7b/tau2.0` | float16 | 2048 | 256 | 0.19.0 | RTX 5090 (330 questions), L40S (287) |
| `livecodebench/mistral-7b` | float16 | 1550 | 256 | 0.15.1 | not recorded |
| `livecodebench/qwen3-8b` | bfloat16 | 1353 | 1300 | 0.19.0 for most of the data; the first 16-temperature round ran 0.8.5.post1 with max_num_seqs 512; one stage ran 0.18.1 on one RTX 4090 | NVIDIA L40S (and one RTX 4090) |
| `livecodebench/gpt-oss-20b` | bfloat16 | 1450 (1410 for V1-V4) | 256 | 0.15.1 | not recorded |
| `livecodebench/qwen3-0.6b` | float16 | 1400 | 256 | 0.15.1 for the 100,000-sample stage; the base run's version was not logged | not recorded; L40S |

Recorded values worth knowing, as the profiles hold them:
- Qwen3-8B on LiveCodeBench ran with `max_num_seqs` 512, 64, 256, 1536 and 1300 in different runs; the profile
  carries 1300, the value of most of its data.
- Hi-ToM Qwen3-8B on L40S used `max_num_seqs` 128 (256 and 512 preempted).
- The MATH-500, Hi-ToM and min_p profiles use `gpu_memory_utilization` 0.90 and `kv_cache_dtype` auto; the
  LiveCodeBench profiles use 0.95 and `trust_remote_code` (as does Mistral on Hi-ToM). All use prefix caching,
  chunked prefill, `enforce_eager` off and `max_num_batched_tokens` 32768.

### What every profile fixes

- **Precision.** float16 for Mistral and Qwen3-0.6B, bfloat16 for Qwen3-8B, bfloat16 with the checkpoint's native
  MXFP4 expert weights for gpt-oss-20b. No other quantization.
- **Models.** Hugging Face repositories at the revisions in `paper_runs.MODELS`; the sampler and the prompt
  builder both load that revision.
- **Sampling.** top_p = 1, top_k off, no penalties, no per-request seed. Where the runs set vLLM's
  `generation_config="vllm"` (MATH-500 Mistral, Qwen3-8B and Qwen3-0.6B; Hi-ToM Qwen3 models), the checkpoints' own
  sampling defaults never apply. On LiveCodeBench the stop strings are ```` "```\n```" ````, ```` "```\n\n" ```` and
  `"\n\n\n"`. gpt-oss stops at the token ids of `<|return|>`, `<|call|>`, `<|end|>` and `<|endoftext|>` on
  MATH-500 and Hi-ToM.
- **Text.** On Hi-ToM the Qwen3 and gpt-oss runs decoded the generated token ids with special tokens kept
  (`text="decode_ids"`), so their completions end in `<|im_end|>` or a Harmony stop token; the Hi-ToM grader
  removes them. Elsewhere the completion is vLLM's text.
- **Batching.** MATH-500, LiveCodeBench and Mistral on Hi-ToM send one request per (question, setting) with
  `n` equal to the cell's samples (`--batching cell`). The other Hi-ToM runs and the min_p runs queued all
  settings of one question together (`--batching question`). Batch size (`max_num_seqs`), batching pattern and
  engine seed change which samples come out (how the random draws interleave), not their distribution. Keep them
  fixed within a run.
- **Split over GPUs.** MATH-500, Hi-ToM and min_p runs gave each question to one GPU (`--split questions`), so every
  cell comes from one engine. LiveCodeBench runs gave every GPU every question and a share of the samples
  (`--split samples`).
- **Engine seed.** vLLM's default seed (0) on MATH-500, LiveCodeBench and the first Mistral Hi-ToM stage; a random
  seed per engine start, recorded, on the other Hi-ToM runs and the min_p runs. The sampler keeps the profile's
  policy for a run's first stage and draws a fresh random seed for every later stage (a fresh engine with the same
  seed and the same first request can replay the same draws). `--seed N|random|none` overrides it. Every seed is
  recorded in every output line.

### Notes the profiles carry

- `math500/mistral-7b`: weights, config and tokenizer loaded in Mistral's own format; the prompt ids are built with
  the Hugging Face template and equal the `mistral_common` ids on all 500 questions.
- `math500/gpt-oss-20b`, `hitom/gpt-oss-20b`: the Harmony date line is pinned to 2026-10-10 in the prompts.
- `hitom/mistral-7b`: first stage on 8 engines (one per GPU), questions dealt round robin, one request per
  (question, temperature), the Hugging Face tokenizer (`tokenizer_mode`, `config_format` and `load_format` are
  `hf`). The second stage (+500) is mostly a run configured like the first on L40S GPUs with a random engine seed;
  for 5,734 of its 19,127 cells it is the first 500 samples of a 5,000-sample request.
- `minp/mistral-7b/tau2.0`: the 617 questions kept by the first stage of `hitom/mistral-7b`, no further screen.
- `livecodebench/mistral-7b`: vLLM 0.15.1 resolved `tokenizer_mode="auto"` to Mistral's tokenizer for this
  repository; the prompt ids are built with the Hugging Face template and equal it. A third, 20,000-sample run on
  questions with 0 < peak < 1% is not in the paper's data.
- `livecodebench/qwen3-0.6b`: a raw text prompt with no chat template (Qwen3-0.6B ran as a completion model on
  LiveCodeBench). The rule that chose its 110 questions, and the 30 and 10 with more samples, is not recorded; the
  lists are in `question_sets/`.
- `livecodebench/qwen3-8b`: `max_model_len` 1353 as run, so 5 of the 975 prompts (those longer than 853 tokens)
  get fewer than 500 new tokens. The sampler refuses such a run unless `--allow-clamp`, which this profile sets.
- `livecodebench/gpt-oss-20b`: the Harmony template stamps the render day into the system message; the runs
  rendered between 2026-02-27 and 2026-03-06. The default keeps that behaviour; `--harmony-date YYYY-MM-DD` pins a day.

## Prompts, and checking them against the paper

```bash
python -m onetemp.sampling.build_prompts --run hitom/gpt-oss-20b --out prompts/hitom_gpt-oss-20b.jsonl
python -m onetemp.sampling.verify_prompts --prompts prompts/hitom_gpt-oss-20b.jsonl \
    --expected onetemp/sampling/expected_prompts/hitom__gpt-oss-20b.json
```

`build_prompts` options: `--run` (a profile) or `--style` with `--model`; `--revision`, `--tokenizer DIR` (a local
tokenizer instead of the hub), `--stage`, `--ids-file`, `--harmony-date`, `--cache-dir`, `--math500-file`,
`--hitom-file`, `--lcb-dir`, `--no-verify-data`, `--out` (required) and `--force`. It writes one line per question:
`id`, `prompt_token_ids`, `prompt_text`, `n_prompt_tokens`, `prompt_ids_sha256` (sha256 of `json.dumps(ids)`) and
`prompt_text_sha256`, plus `<file>.meta.json` with the tokenizer, dataset hashes and the longest prompt. LiveCodeBench
lines also carry `lcb_version`, `lcb_row` and `global_idx`. The sampler sends the ids unchanged (`--prompts <file>`,
or it builds them itself), so every byte of every prompt is fixed before the GPU is touched.

`verify_prompts` compares a prompt file with what the paper's runs recorded, field by field. It prints how many
prompts are identical, different or missing, and exits 0 only if every expected id is present and identical. Each
expected file lists every question of its benchmark (397 for gpt-oss on LiveCodeBench), and a missing id counts as
a failure, so build all questions: use a run whose first stage is `all`. For `livecodebench/qwen3-0.6b` (first
stage: 110 questions) use `--style lcb_raw --model Qwen/Qwen3-0.6B`, which builds all 975.

| file in `expected_prompts/` | questions | fields compared |
|---|---|---|
| `math500__<model>.json`, 4 models | 500 each | token count, sha256 of the ids, sha256 of the text |
| `hitom__<model>.json`, 4 models | 1,200 each | the same |
| `livecodebench__qwen3-8b.json` | 975 | sha256 of the decoded prompt |
| `livecodebench__gpt-oss-20b.json` | 397 (those whose records survive) | sha256 of the decoded prompt with the date line masked |
| `livecodebench__mistral-7b.json`, `livecodebench__qwen3-0.6b.json` | 975 each | token count |

The prompts, in words (`onetemp/sampling/prompts.py` holds the code and the verbatim texts):
- **MATH-500.** System message: "Solve the following math problem efficiently and clearly. Reason step by step,
  and put your final answer within \boxed{}." (Qwen3-8B: followed by "Your answer can be at most 2,048 tokens long,
  so keep it very concise and end it with the final answer in \boxed{}."). User message: the problem.
- **Hi-ToM.** User message: the dataset's `prompt` field unchanged; Mistral's is followed by an instruction that
  its reply is capped at 500 tokens and should give the answer first.
- **LiveCodeBench.** System text "You are an expert Python programmer. Write a short, correct solution (under 500
  tokens)."; the problem statement; then "Complete this code:" with the starter code, or "Read from stdin, write to
  stdout. Enclose code in ```` ```python``` ```` delimiters."; the response is primed with ```` ```python ````.
  Qwen3-0.6B gets this as plain text with no chat template.
- **Chat formats.** Mistral's template (v0.3 joins the system text to the user turn); Qwen3's with thinking
  disabled (the prompt ends with an empty `<think>` block); gpt-oss's Harmony template with reasoning effort medium,
  the system text as its developer message (Hi-ToM has none), and `<|channel|>final<|message|>` appended so the
  model answers in the final channel directly.

## Sampling

Each output directory holds ONE sampling configuration (run and stage). `RUN.json` in it records the
configuration, and a worker or a rerun with a different one is refused (`--force` accepts it).

### MATH-500

```bash
python -m onetemp.sampling.sample_vllm --run math500/mistral-7b  --out runs/math500_mistral-7b
python -m onetemp.sampling.sample_vllm --run math500/qwen3-8b    --out runs/math500_qwen3-8b
python -m onetemp.sampling.sample_vllm --run math500/qwen3-0.6b  --out runs/math500_qwen3-0.6b
python -m onetemp.sampling.sample_vllm --run math500/gpt-oss-20b --out runs/math500_gpt-oss-20b
# on 8 GPUs, questions dealt round robin:
python -m onetemp.sampling.launch --gpus 0-7 --run math500/qwen3-0.6b --out runs/math500_qwen3-0.6b
```

### Hi-ToM: first stage, screen, second stage

```bash
python -m onetemp.sampling.launch --gpus 0-3 --run hitom/qwen3-0.6b --stage screen --out runs/hitom_qwen3-0.6b_screen
python -m onetemp.grading.hitom --inputs runs/hitom_qwen3-0.6b_screen/cells --out counts_screen.jsonl
python -m onetemp.sampling.screen --rule hitom --counts counts_screen.jsonl --out kept_qwen3-0.6b.json
python -m onetemp.sampling.launch --gpus 0-3 --run hitom/qwen3-0.6b --stage topup \
    --ids-file kept_qwen3-0.6b.json --out runs/hitom_qwen3-0.6b_topup
python -m onetemp.grading.hitom --inputs runs/hitom_qwen3-0.6b_topup/cells --out counts_topup.jsonl
# pool the two stages (500 + 1,000 = 1,500 per cell), keeping only the questions the screen kept:
python -m onetemp.sampling.pool_counts --policy sum --inputs counts_screen.jsonl counts_topup.jsonl \
    --keep kept_qwen3-0.6b.json --out counts_hitom_qwen3-0.6b.jsonl
```

Run on the paper's own stages (`data/selection/hitom/Qwen3-0.6B__temperature__screen500.jsonl`, then
`data/counts/hitom/Qwen3-0.6B__temperature__topup1000.jsonl`), these three steps (screen, pool, keep) give exactly the
23,033 rows of `data/counts/hitom/Qwen3-0.6B__temperature.jsonl`; the same with `--rule minp` on the min_p screen
and `Qwen3-0.6B__min_p_tau1.5__stage2_nu400.jsonl` gives the 6,480 rows of `Qwen3-0.6B__min_p_tau1.5__nu500.jsonl`.
The pooled files are sorted by id; the shipped files keep the paper's question order.

To sample exactly the paper's populations instead of re-screening, pass the paper's kept list:
`--ids-file onetemp/sampling/question_sets/hitom_kept_qwen3-0.6b.json` (also `_qwen3-8b`, `_gpt-oss-20b`,
`_mistral-7b`). Mistral's first stage is `--stage screen` with 1,000 samples and its second `--stage topup` with 500.

### min_p at a fixed temperature (Appendix I)

```bash
python -m onetemp.sampling.launch --gpus 0-3 --run minp/qwen3-0.6b/tau1.5 --stage screen --out runs/minp_q06_t1.5_screen
python -m onetemp.grading.hitom --inputs runs/minp_q06_t1.5_screen/cells --out counts_minp_screen.jsonl
python -m onetemp.sampling.screen --rule minp --counts counts_minp_screen.jsonl --out kept_minp.json
python -m onetemp.sampling.launch --gpus 0-3 --run minp/qwen3-0.6b/tau1.5 --stage depth --ids-file kept_minp.json \
    --out runs/minp_q06_t1.5_depth
# the tau = 2.0 studies; their question lists are in the profiles (Qwen3-0.6B: the 648; Mistral: its 617)
python -m onetemp.sampling.launch --gpus 0-3 --run minp/qwen3-0.6b/tau2.0 --out runs/minp_q06_t2.0
python -m onetemp.sampling.launch --gpus 0-3 --run minp/mistral-7b/tau2.0 --out runs/minp_mistral_t2.0
```

In the completions, `temperature` is the fixed tau and `min_p` is the setting. min_p = 0 is plain sampling: it is
recorded as `0.0` and not passed to vLLM. The temperature sweeps write `min_p: null`.

### LiveCodeBench

The runs split the samples of every cell over the GPUs (`--split samples`): 1,500 = 4 x 375, a second stage of
10,000 = 4 x 2,500. The launcher gives each GPU its own engine seed (`seed + worker`); pass `--same-seed` for the
runs' single seed.

```bash
python -m onetemp.sampling.launch --gpus 0-3 --run livecodebench/mistral-7b --out runs/lcb_mistral-7b_main
python -m onetemp.grading.livecodebench --allow-code-execution --inputs runs/lcb_mistral-7b_main/cells \
    --out counts_main.jsonl          # runs model-generated code: read docs/grading.md first
python -m onetemp.sampling.screen --rule lcb-mistral --counts counts_main.jsonl --out deep_mistral.json
python -m onetemp.sampling.launch --gpus 0-3 --run livecodebench/mistral-7b --stage deep \
    --ids-file deep_mistral.json --out runs/lcb_mistral-7b_deep
# grade the second stage to counts_deep.jsonl, then pool (1,500 + 10,000):
python -m onetemp.sampling.pool_counts --policy sum --inputs counts_main.jsonl counts_deep.jsonl \
    --out counts_lcb_mistral.jsonl
# gpt-oss-20b: --rule lcb, and the second stage REPLACES the first for those questions
python -m onetemp.sampling.pool_counts --policy replace --inputs counts_main.jsonl counts_deep.jsonl \
    --out counts_lcb_gptoss.jsonl
# Qwen3-0.6B: its 110 questions, 6 GPUs x 1,666 = 9,996 per cell
python -m onetemp.sampling.launch --gpus 0-5 --run livecodebench/qwen3-0.6b --out runs/lcb_qwen3-0.6b_main
```

The paper's second-stage lists are in `onetemp/sampling/question_sets/livecodebench_deep_*.json` and
`livecodebench_qwen3-0.6b_deep{30,10}.json`. Qwen3-8B's list (272 questions) holds the questions with more than
1,500 samples in some cell; the 257 of its first 16-temperature second stage are under `ids_16`.

### Overrides

Every profile value can be changed. The change is recorded in `RUN.json` and in every output line.

```bash
python -m onetemp.sampling.sample_vllm --run math500/qwen3-0.6b --settings 0.5,1.0 --n 64 --max-tokens 1024 \
    --engine-arg max_num_seqs=128 --ids-file my_ids.txt --out runs/test
python -m onetemp.sampling.sample_vllm --run math500/qwen3-0.6b --plan --out runs/test    # resolved plan only
python -m onetemp.sampling.sample_vllm --run math500/qwen3-0.6b --model-path /models/Qwen3-0.6B --out runs/local
```

Options of `sample_vllm` (the launcher passes all of them to every worker):

| option | meaning |
|---|---|
| `--run NAME` | the profile (required) |
| `--stage NAME` | which stage of the profile; default the first |
| `--out DIR` | output directory, one per configuration (required) |
| `--ids-file FILE` | question ids: a JSON list, `{"ids": [...]}`, or one id per line |
| `--settings A,B,...` | temperatures (or min_p values on a min_p run) instead of the profile's grid |
| `--tau X` | the fixed temperature of a min_p run |
| `--n N` | samples per (question, setting) in this stage, over all workers |
| `--max-tokens N` | token limit per sample |
| `--engine-arg KEY=VALUE` | a vLLM `LLM()` argument (JSON value); repeatable |
| `--model`, `--revision`, `--model-path DIR` | another model id or revision; weights from a local directory (the records keep `--model`) |
| `--seed`, `--same-seed` | engine seed policy (above) |
| `--batching cell\|question`, `--split questions\|samples` | request pattern and what `--num-workers` divides |
| `--worker I`, `--num-workers W` | this worker's index and the number of workers sharing `--out` (the launcher sets them) |
| `--allow-clamp` | run prompts for which prompt + `max_tokens` exceeds `max_model_len` (vLLM shortens them) |
| `--prompts FILE`, `--any-style` | prompts from `build_prompts` (default: built now); accept prompts of another style |
| `--harmony-date`, `--tokenizer`, `--cache-dir`, `--math500-file`, `--hitom-file`, `--lcb-dir`, `--no-verify-data` | as for `build_prompts` |
| `--engine vllm\|fake` | `fake` runs the whole pipeline with a deterministic stand-in, no GPU |
| `--max-cells N`, `--plan`, `--force` | stop after N new cells; print the plan and exit; accept a different `RUN.json` |

What the sampler does, in order:
1. Reads the profile, applies the overrides, and writes the resolved configuration to `<out>/RUN.json`.
2. Takes the prompts as token ids (`--prompts`, or builds them) and sends them unchanged.
3. Refuses to start if a prompt does not fit `max_model_len`, or if prompt + `max_tokens` exceeds it: vLLM silently
   shortens `max_tokens` in that case (`--allow-clamp` runs it anyway, as the Qwen3-8B LiveCodeBench profile does).
4. Visits cells question by question (all settings of a question before the next question). Each sample is an
   independent draw from the model at that setting.
5. Writes each finished cell atomically to its own file. A cell whose file exists is skipped, so a stopped run
   resumes where it stopped.
6. Exits 0 only when every cell of this worker is on disk. A worker stopped by SIGTERM or SIGINT exits 3.

## Several GPUs

```bash
python -m onetemp.sampling.launch --gpus 0,1,2,3 [--split questions|samples] <sampler arguments>
```

`--gpus` takes a list (`0,1,2,3`) or a range (`0-7`). The launcher also takes `--stagger SECONDS` (default 3, the
pause between engine starts) and `--poll SECONDS` (default 60). It:
1. builds the prompts once (`<out>/prompts.jsonl`) so every worker sends the same token ids;
2. plans every worker's cells with the workers' own code;
3. starts one sampler per GPU (`CUDA_VISIBLE_DEVICES`), logging to `<out>/logs/`;
4. passes SIGTERM to the workers on Ctrl-C or SIGTERM (never SIGKILL: a killed GPU process can leave memory held);
5. when no worker is alive, decides completion by the files: every planned cell file must exist, parse, and hold its
   planned number of samples. It writes `<out>/LAUNCH_CHECK.json` and exits 0 only if nothing is missing.

A worker that was stopped exits cleanly too, so a worker's exit code never counts as completion. Run the same
command again to resume.

`--split questions` (the default for MATH-500, Hi-ToM and min_p) deals the questions round robin, so every cell
comes from one engine. `--split samples` (the default for LiveCodeBench) has every GPU draw its share of every cell.

## The screen and pooling stages

The second stages sample only questions the first stage leaves open. `screen` computes, per question,

    peak = max over the stage's settings of c / n

on the first stage's samples only, and keeps the question by the paper's rule:

| `--rule` | keeps a question iff | used for |
|---|---|---|
| `hitom` | peak < 0.30 (no lower bound: with multiple choice a guess can pass) | Hi-ToM: 500 samples per temperature (Mistral 1,000) |
| `minp` | peak < 0.30 over the 10 min_p values, including 0 | min_p screen of Qwen3-0.6B at tau = 1.5: 100 samples per value |
| `lcb-mistral` | 0 < peak < 0.30 | LiveCodeBench, Mistral: 1,500 samples per temperature |
| `lcb` | 0 < peak <= 0.30 | LiveCodeBench, gpt-oss-20b and Qwen3-8B |

The comparisons are exact (integers: c / n < 0.30 is 10 c < 3 n), so no rounding moves a question across 0.30. A
question that lacks a setting of the grid, or has a different n somewhere, is reported as `undecided` and is neither
kept nor dropped, unless `--partial-ok`.

```bash
python -m onetemp.sampling.screen --rule hitom --counts counts_screen.jsonl --model Qwen/Qwen3-0.6B --out kept.json
```

Options: `--counts` (the first stage's `counts.jsonl`, required), `--rule` (required), `--out` (required),
`--benchmark`, `--model` (the Hugging Face id; required if the file holds several models), `--axis temperature|min_p`
(default: min_p for `--rule minp`), `--tau` (with min_p: only rows at this temperature), `--settings A,B,...` (the
stage's grid) and `--partial-ok`. The output JSON holds `ids` (kept ids, in input order), `n_kept`, `n_dropped`,
`undecided`, `grid`, `peaks` (`{id: [peak, argmax setting]}`), and the rule, axis, model and counts file. Pass it to
`--ids-file` for the second stage.

The first-stage counts of Hi-ToM for Qwen3-0.6B, Qwen3-8B and gpt-oss-20b, and of the min_p screen, ship in
`data/selection/hitom/` (`*__screen500.jsonl`, `*__screen100.jsonl`), so the screen can be run on the paper's own
first stage (section 2 under "What a rerun reproduces" lists what it keeps):

```bash
python -m onetemp.sampling.screen --rule hitom \
    --counts data/selection/hitom/Qwen3-0.6B__temperature__screen500.jsonl --out kept.json
```

Selecting on low observed rates biases the selected questions' first-stage pass rates down. The paper pools both
stages for its tables and figures and says so (Appendix I discusses the effect for Mistral 7B Instruct).

`pool_counts` merges the counts of two or more stages into one `counts.jsonl`:

```bash
python -m onetemp.sampling.pool_counts --policy sum     --inputs counts_stage1.jsonl counts_stage2.jsonl --out counts.jsonl
python -m onetemp.sampling.pool_counts --policy replace --inputs counts_stage1.jsonl counts_stage2.jsonl --out counts.jsonl
```

Inputs go earliest stage first. Rows are keyed by (benchmark, model, id, temperature, min_p). `sum` adds `n` and `c`
of equal keys (and any optional integer field present in every such row, such as `c_strict`): Hi-ToM
(500 + 1,000; Mistral 1,000 + 500), min_p at tau = 1.5 (100 + 400), LiveCodeBench Mistral and Qwen3-8B
(1,500 + 10,000). `replace` gives a question present in a later input all of its rows from the latest input that
has it: LiveCodeBench gpt-oss-20b, where a question's 10,000 second-stage samples replace its 1,500. Every sample
counts once. The output is written sorted; a pooled `c > n` stops the run. Without `--keep`, every question of every
input is written, including the ones the screen dropped, with their first-stage samples only. The Hi-ToM and min_p
data files hold only the kept questions, so pass the screen's output there: `--keep kept.json` (or any JSON list or
text file of ids; each must be in the inputs). On LiveCodeBench every question stays in the study, so no `--keep`.

## Output: the completions files the graders read

`<out>/cells/<setting>/<question>.jsonl.gz`, one JSON line per (question, setting):

```
{"benchmark": "hitom", "model": "Qwen/Qwen3-0.6B", "id": "s12_o3_CoTP", "temperature": 1.275, "min_p": null,
 "n": 500, "completions": ["...", ...], "finish_reasons": ["stop", "length", ...], "stop_reasons": [...],
 "num_output_tokens": [...], "stage": "screen", "run": "hitom/qwen3-0.6b", "part": null,
 "provenance": {"engine": {...}, "engine_seed": 1234, "sampling_params_repr": "SamplingParams(n=500, ...)",
                "vllm": "0.19.0", "gpu": "...", "prompt_ids_sha256": "...", ...}}
```

- `<setting>` is `t1.275` on temperature runs and `t2_mp0.0316` on min_p runs (`t1.5_mp0` for min_p 0).
- With `--split samples`, a cell is split into `<question>.part<w>of<W>.jsonl.gz`; `part` is `[w, W]`, and the
  graders add the parts up.
- LiveCodeBench lines also carry `lcb_version` and `lcb_row`, which the grader uses to find the problem's tests.
- `finish_reasons` tells truncation (`"length"`) apart from a finished answer. A truncated completion usually loses
  its answer, which is a measurement artifact, so the graders report truncation separately.
- **Provenance is written at sampling time, into every line**, never into a file name. It holds: the run, the
  stage and the model revision; the full engine arguments and the engine seed; the split (mode, worker, number of
  workers), the batching pattern and the text mode; `sampling_params_repr` (the exact vLLM `SamplingParams`); the
  sha256 of the prompt ids and of the prompt text, the prompt token count and style; the sha256 of the prompts file
  (when one was used), of the whole prompt set and of the sampler's source; timings per request; and the versions
  (vLLM, torch, transformers) and the GPU name.
- Besides the cells, the output directory holds `RUN.json` (the configuration), `workers/w<k>.json` (each worker's
  status), `logs/` and `prompts.jsonl` (launcher runs), and `LAUNCH_CHECK.json`.
- Point a grader at `<out>/cells`, not at `<out>`: the grader reads every `*.jsonl` and `*.jsonl.gz` file it finds,
  and `prompts.jsonl` is a prompt file, not a completions file.
- Each cell is written through a temporary file named `.tmp_<pid>_<cell>.jsonl.gz` in its folder and then renamed. A
  sampler killed in the middle of a write can leave such a file behind; delete it before grading.

The gold answers, tests and reference solutions never enter a sampling output: the prompts carry question text only.

## What a rerun reproduces

A rerun with these profiles reproduces the paper's sampling distribution: the same prompts (checked by hash), model
revisions, precision, sampler and token limits. It does not reproduce the paper's samples one for one. Which draws
come out also depends on the engine seed, the GPU, the batch composition and the vLLM version; the runs recorded
these but they cannot be replayed.

The checks below compare everything that decides which distribution the samples come from (prompts, question sets,
settings) with what the paper's runs recorded about themselves. No GPU was used for them, and the samples were
not redrawn. `sample_vllm` and `launch` were run end to end with the fake engine (`tests/test_sampling.py`), and
their output was read back by the graders' input reader and graded by the Hi-ToM grader.

### 1. Prompts: identical to the paper's, question by question

`build_prompts` (with the pinned datasets and the models' own tokenizers at the paper's revisions, transformers
4.57.6) rebuilt every prompt, and `verify_prompts` compared it with the records of the paper's runs.

| benchmark / model | questions | identical | compared | the paper's side |
|---|---|---|---|---|
| MATH-500 / Mistral 7B Instruct | 500 | 500 | token count, sha256 of the ids, sha256 of the text | the prompt ids the runs sent |
| MATH-500 / Qwen3-8B | 500 | 500 | same | same |
| MATH-500 / Qwen3-0.6B | 500 | 500 | same | same |
| MATH-500 / gpt-oss-20b | 500 | 500 | same | same |
| Hi-ToM / Mistral 7B Instruct | 1,200 | 1,200 | same | the runs' per-question prompt record (it also agrees with all 436 prompts recorded by the first stage's own output) |
| Hi-ToM / Qwen3-8B | 1,200 | 1,200 | same | same |
| Hi-ToM / Qwen3-0.6B | 1,200 | 1,200 | same | same |
| Hi-ToM / gpt-oss-20b | 1,200 | 1,200 | same | same |
| LiveCodeBench / Qwen3-8B | 975 | 975 | the decoded prompt text | the prompt recorded in the runs' output files |
| LiveCodeBench / gpt-oss-20b | 397 | 397 | the decoded prompt text, with each record's own date line | the prompts recorded in the surviving output files |
| LiveCodeBench / Mistral 7B Instruct | 975 | 975 | token count | the runs' per-question token counts |
| LiveCodeBench / Qwen3-0.6B | 975 | 975 | token count | the runs' per-question token counts (no prompt text survives) |
| LiveCodeBench / Mistral 7B Instruct, extra check | 105 | 105 | the prompt as SentencePiece pieces | the prompt recorded in the surviving output files |
| LiveCodeBench, the dataset path | 162 | 162 | problem text and starter code read by `public_data` from the pinned V6 file | the bodies of the recorded prompts |

How far the LiveCodeBench rows go. The problem texts for the LiveCodeBench comparison were taken from the recorded
Qwen3-8B prompts (all 975), because only the V6 file (134 MB) of the 4.5 GB dataset was downloaded for this check.
So for V1-V5 the comparison covers the system text, the body layout, the four chat formats, the priming and the
tokenization; the step from the dataset file to the problem text is checked on the 162 V6 questions, where it is
exact.

Prompt lengths against the token budgets of the runs (longest prompt + `max_tokens` <= `max_model_len`):

| run | prompt tokens (min / max) | max_tokens | max_model_len | fits |
|---|---|---|---|---|
| MATH-500 Mistral / Qwen3-8B / Qwen3-0.6B / gpt-oss | 41-910 / 80-863 / 49-832 / 107-856 | 4096 / 2048 / 4096 / 4096 | 5120 / 3072 / 5120 / 5120 | all |
| Hi-ToM Mistral / Qwen3 (both) / gpt-oss | 545-1046 / 408-830 / 457-839 | 500 / 1000 / 1000 | 2048 / 1830 / 1839 | all |
| LiveCodeBench Qwen3-0.6B / Mistral / gpt-oss | 168-842 / 211-1029 / 232-907 | 500 | 1400 / 1550 / 1450 (V1-V4 ran at 1410) | all |
| LiveCodeBench Qwen3-8B | 185-859 | 500 | 1353 | 5 prompts are longer than 853 tokens and get fewer than 500 new tokens, as in the runs |

### 2. Question populations: the screen reproduces the paper's kept sets

`screen`, run on the first-stage counts the runs recorded (first-stage samples only), against the paper's kept lists:

| run | first stage | rule | kept by `screen` | the paper's | same set |
|---|---|---|---|---|---|
| Hi-ToM / Qwen3-0.6B | 1,200 questions x 31 T x 500 | peak < 0.30 | 743 | 743 | yes |
| Hi-ToM / Qwen3-8B | 1,200 x 31 x 500 | peak < 0.30 | 471 | 471 | yes |
| Hi-ToM / gpt-oss-20b | 1,200 x 31 x 500 | peak < 0.30 | 678 | 678 | yes |
| min_p, Qwen3-0.6B, tau = 1.5 | 1,200 x 10 values x 100 | peak < 0.30 | 648 | 648 (Appendix I) | yes |
| LiveCodeBench / Mistral | 975 x 16 T x 1,500 | 0 < peak < 0.30 | 365 | 365 second-stage questions | yes |
| LiveCodeBench / gpt-oss-20b, V5-V6 | 314 x 16 T x 1,500 | 0 < peak <= 0.30 | 120 | 120 | yes |

- The Hi-ToM Mistral first-stage tally (1,000 samples per cell) was not available for this check, so its 617 were
  not recomputed. `question_sets/hitom_kept_mistral-7b.json` is the runs' list, and it equals the 617 questions of
  the paper's Mistral Hi-ToM data (Appendix I: 617).
- Every Hi-ToM kept list equals the question set of the paper's data for that model. Appendix C's "Included
  questions" for Hi-ToM (gpt-oss 678, Mistral 615, Qwen3-8B 466, Qwen3-0.6B 743) are the kept questions after the
  analyses' 0.1% peak filter, so they are subsets of these lists (equal for gpt-oss and Qwen3-0.6B).
- Every second-stage list in `question_sets/` was also checked against the cell sizes of the paper's data: Mistral's
  365 are exactly the questions with more than 1,500 samples; gpt-oss's 293 (120 + 173) exactly those with 10,000;
  Qwen3-0.6B's 30 and 10 exactly those with at least 19,996 and at least 94,996 samples in a cell (all at the 16
  temperatures of `GRID_16`). For Qwen3-8B, 272 questions have more than 1,500 samples in some cell; the 257 of its
  first 16-temperature second stage are among them.

### 3. Sampling settings against Appendix C and Appendix I

| Appendix statement | the profiles |
|---|---|
| grids of 16, 20, 38 and 31 temperatures (Appendix C tables) | `GRID_16`, `GRID_20`, `GRID_38`, `GRID_31` |
| min_p grids of 10 values at tau = 1.5 and 5 at tau = 2.0 | `MINP_GRID_TAU15`, `MINP_GRID_TAU20` |
| 1,500 samples (Qwen3-8B on MATH-500: 750) | MATH-500 stages of 1,500 / 750 |
| Hi-ToM: 1,500, first stage 500 (Mistral 1,000), keep below 0.30 | 500 + 1,000; Mistral 1,000 + 500; rule `hitom` |
| min_p: 100-sample screen, 400 per value in the second stage; nu = 500, 400, 1,500 | stages 100 + 400; 400; 1,500 |
| LiveCodeBench: about 1.5k, harder questions about 10k, sometimes more; Qwen3-0.6B approximately doubled on some | 1,500 + 10,000 (pooled or replacing); Qwen3-0.6B 9,996, +10,000 on 30, +100,000 on 10: the sizes and rules the runs recorded |
| token limits 4096 (Qwen3-8B 2048) on MATH-500, 1000 (Mistral 500) on Hi-ToM | the same |
| Qwen3-8B's instruction "followed by a sentence that states its limit"; Mistral's Hi-ToM instruction | verbatim in `prompts.py`, hash-checked (section 1) |
| gpt-oss: Harmony, reasoning effort medium, primed final channel, bfloat16 + native MXFP4 | `math500_harmony`, `hitom_harmony`, `lcb_harmony` |
| vLLM 0.19.0, top_p = 1, top_k = 0 (Appendix I) | the 0.19.0 profiles: the same; LiveCodeBench versions as in the engine table above |

`max_tokens` is 500 for all four models on LiveCodeBench.

### 4. Where this release differs from the runs, and why

- **Engine seeds when the samples of a cell are split over GPUs.** The LiveCodeBench runs gave every GPU vLLM's
  default seed. A fresh engine given the same first request can replay the same draws, so `launch` gives worker w
  the seed `seed + w`. `--same-seed` restores the runs' setting.
- **Engine seeds of second stages.** The LiveCodeBench second stages and part of the Mistral Hi-ToM second stage ran
  with vLLM's default seed, like their first stages. The sampler draws a fresh random seed for every stage after the
  first, so that a second stage cannot replay the first stage's draws; `--seed none` restores the runs' setting.
- **Batching on Hi-ToM and min_p.** Those runs used long-lived engines that queued all cells of one question together
  and refilled the queue as cells finished. `--batching question` sends all settings of a question in one generate
  call. Both change only how the random draws interleave.
- **The first Mistral Hi-ToM stage** loaded a local directory that held only the Hugging Face weight files, so vLLM
  used the Hugging Face tokenizer; the profile states `tokenizer_mode="hf"` (and the matching formats) explicitly,
  because vLLM 0.19 would pick Mistral's own tokenizer for the hub repository.
- **gpt-oss on LiveCodeBench** stamped the render day into its prompts (2026-02-27 to 2026-03-06 across the runs);
  the default keeps that behaviour and `--harmony-date` pins a day.

### 5. Not reproduced here

- **The samples.** No GPU job was run for this release; see the start of this section.
- **The LiveCodeBench dataset path for V1-V5.** Only V6 was downloaded for the check (section 1).
- **Two selection rules that were never recorded:** which 110 LiveCodeBench questions Qwen3-0.6B was run on, and which
  30 and 10 of them got more samples. The lists are in `question_sets/`.
- **The exact per-cell structure of Qwen3-8B's LiveCodeBench second stage** (some cells got two or three 10,000-sample
  runs, up to 19,000 samples in all): `question_sets/livecodebench_deep_qwen3-8b.json` lists its 272 questions; the
  per-cell sample counts are in `data/counts/livecodebench/Qwen3-8B__*.jsonl`.
- **The Hi-ToM Mistral screen** from its first-stage counts (not available; section 2).
