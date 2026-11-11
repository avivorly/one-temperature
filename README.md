# One Temperature to Rule Them All?

Code and data for the NeurIPS 2026 paper *One Temperature to Rule Them All?* ([arXiv:2611.13077](https://arxiv.org/abs/2611.13077)).

- For every question, the probability that one sample is correct depends on the sampling temperature: its
  **temperature profile**. We measured these profiles for four models on LiveCodeBench, MATH-500 and Hi-ToM,
  typically 1,500 samples at each of 16 to 38 temperatures per question.
- From the profiles we compute pass@K under a single temperature, under the uniform mix of temperatures and
  under the optimal multi-temperature allocation, held out by leave-one-out and 10-fold, and test how the best
  temperature depends on the mix of questions it is chosen on.
- This repository ships the per-question counts behind every figure and table (Figure 1 aside, see
  [What is not shipped](#what-is-not-shipped)), the code that draws the samples with vLLM, grades them and analyses
  them, and one command that recomputes 47 of the paper's 51 figures and tables, with most of its printed numbers,
  and compares them with the paper ([Paper map](#paper-map)).

To cite the paper, use the BibTeX entry under [Citation](#citation) or `CITATION.cff`.

**Contents:** [Install](#install) · [Quickstart: Figure 3](#quickstart-figure-3) · [Data](#data) ·
[Full pipeline](#full-pipeline) · [Paper map](#paper-map) · [Reproduce everything](#reproduce-everything) ·
[What is not included](#what-is-not-included) · [Citation](#citation) · [License](#license)

## Install

Python 3.10 or later (measured on 3.10.19 and 3.12; see "Package versions" below). From the repository root:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"     # the analysis (numpy, scipy, matplotlib) and the tests (pytest, sympy)
python -m pytest            # 87 tests, CPU only, no network, about 30 s
```

The run ends with `87 passed`.

| install | for |
|---|---|
| `pip install -e .` | the analysis: everything `python -m onetemp.reproduce` runs, on CPU |
| `pip install -e ".[dev]"` | the analysis and the test suite |
| `pip install -e ".[grading]"` | adds sympy 1.14.0 and mpmath 1.3.0, which the MATH-500 grader's verdicts are defined with |
| `pip install -e ".[prompts]"` | rebuilding the paper's prompts and checking them against its prompt hashes, without a GPU: transformers 4.57.6 with jinja2, sentencepiece and protobuf |
| `pip install -e ".[sampling]"` | drawing samples on a GPU: vllm 0.19.0 and transformers 4.57.6 |
| `pip install -r requirements-exact.txt && pip install --no-deps -e .` | the exact package versions the reproduction below ran with (Python 3.12 or later: contourpy 1.4.0 needs it) |

<details>
<summary><b>Package versions</b>: what depends on them, and what two other environments gave</summary>

The reproduction record (see [Reproduce everything](#reproduce-everything)) ran on Python 3.12.12 with the versions
of `requirements-exact.txt` (numpy 2.2.6, scipy 1.16.0, matplotlib 3.10.6, Pillow 11.3.0). Four things depend on
versions:

- **Python.** The profile analyses add floats with Python's own `sum`, which became compensated in Python 3.12, as
  in the paper's runs; on 3.10 or 3.11 a peak on a flat-topped profile could land on a neighbouring evaluation
  point. On Python 3.10.19 every compared number came out the same.
- **matplotlib and Pillow.** The paper's images were drawn with matplotlib 3.10.6 and Pillow 11.3.0. With other
  versions an image can differ in bytes; `python -m onetemp.reproduce profiles` then reports it as a note, and
  `python -m onetemp.profiles repro` as a known difference, not as a mismatch.
- **numpy, reweightings.** The cells of the random-reweighting figures (4(b), 17-28, 42-44) depend on numpy's random
  number streams and sort order; another numpy could move a single reweighting across a cell edge.
- **numpy, the min_p null.** The null simulations of the min_p checks (Appendix I) draw binomial counts, and numpy
  versions differ in the binomial draws they make from the same seed. The null's Monte Carlo values can then
  differ in the last printed digit: its quantiles, and with them whether its central 95% lies within the interval
  the paper prints, its median, and the counts of null replicates beyond a slope. With a numpy other than 2.2.6,
  `reproduce profiles` and `profiles repro` report such a difference as a known difference, not as a mismatch.
  The conclusions, which trends pass both checks, are always compared strictly.

Two other environments, each a fresh copy of this repository installed with the commands above, without the C
kernel:

| environment | `python -m pytest` | `python -m onetemp.reproduce all --jobs 3` |
|---|---|---|
| Python 3.12.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, Pillow 13.0.0 | `87 passed` | `58 results, 0 mismatches`; images 0 of 49 byte-identical (a note); numbers and checks 165 of 168 equal, 3 known differences: Figure 1 and two Monte Carlo values of the min_p null |
| Python 3.10.19, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9, Pillow 12.3.0 | `87 passed` | `58 results, 0 mismatches`, as in the record: images 49 of 49 byte-identical; numbers and checks 167 of 168 equal, 1 known difference (Figure 1) |

The two Monte Carlo values are those of the min_p study of Mistral 7B Instruct at tau = 2.0. The median slope of
its range-size null is -1.1944, which prints as -1.19 (numpy 2.2.6: -1.1960; the paper prints -1.20). The upper end
of its peak-slope null's central 95% is 0.5453, which prints as 0.55, outside the interval [-0.60, 0.54] the paper
prints for the three studies (numpy 2.2.6: 0.5433). Every reweighting cell, and which trends pass both checks, came
out the same.

</details>

**Optional: the C kernel of the profile fit.** The fit runs 50,000 gradient steps per question. Compiled once, it
takes about 5 s instead of about 60 s for 975 questions:

```bash
python -m onetemp.core.fit_all --build-kernel     # needs a C compiler (gcc, or $CC)
```

Without it, a numpy version of the same loops runs instead, and nothing changes but time: on the Figure 3 data
(975 questions, 16 temperatures), the numpy fallback gives coefficients bit-identical to the C kernel's portable
build. The paper's own build (`--build-flags paper`, compiled with `-march=native`) differs from the portable
build by at most 7.8e-12 in log(1 - P) and 1.4e-15 in P, with the same included questions; every printed number
is the same under both (`docs/core.md`, section 8).

## Quickstart: Figure 3

Figure 3 shows, for Mistral 7B Instruct on LiveCodeBench, the pass@K of the optimal allocation and of the fixed
temperatures it uses most (each the largest share of the optimal allocation at some K), divided by the pass@K of
the uniform mix of temperatures. This command reads the 975 questions, fits the profiles of the 446 that have at
least one correct sample, keeps the 359 whose fitted peak pass rate is at least 0.1%, and evaluates the optimal
allocation held out:

```bash
python -m onetemp.core.pipeline \
    --counts data/counts/livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl \
    --folds loo 10 --plot --name "Mistral 7B Instruct" --out-dir results/fig3
```

Output of a fresh install as above (Python 3.12, numpy 2.5.4, scipy 1.18.1), without the C kernel; with it, the
fit line says `backend c` and the fit takes about 5 s, and the numbers are the same:

```
[fit] 975 questions, 359 included (peak >= 0.1%), 16 settings, backend numpy, 62.52 s
[heldout] loo: 8.83 s, folds sha256 8648b71ed64c
[heldout] kf10: 0.32 s, folds sha256 12ee3719883b
[result] loo: optimal held-out / uniform at K=2 = 1.0869
[result] kf10: optimal held-out / uniform at K=2 = 1.0999
[done] 75.06 s -> results/fig3
```

- `kf10` is the 10-fold held-out optimum, the red curve of Figure 3: 1.0999 at K = 2. `loo` is the leave-one-out
  optimum, the red curve of Figure 14(b): 1.0869 at K = 2. Both fold lists hash to the paper's (`12ee3719883b`,
  `8648b71ed64c`, recorded in `onetemp/core/paper_numbers.json`).
- `results/fig3/rtu_kf10.png` redraws Figure 3 and `rtu_loo.png` Figure 14(b); their ±1σ bands differ from the
  paper's (see [Paper map](#paper-map)). The folder also holds the fitted profiles (`profiles.json`), the held-out
  results (`heldout_*.json`), the curves with their bootstrap bands (`rtu.json`), this model's pass@K table
  (Table 3) as `.md`, `.csv`, `.tex` and `.json`, and its row of the Appendix F.2 summary table (`summary.*`).
- `python -m onetemp.reproduce fig3` compares both red curves at all 20 values of K with the values the paper's
  Figures 3 and 14(b) draw (kept in `onetemp/core/paper_numbers.json`) and prints
  `Figure 3: 10-fold 20/20 K and leave-one-out 20/20 K equal at 4 decimals`. It runs this model's whole table
  check, which ends `livecodebench, mistralai/Mistral-7B-Instruct-v0.3: 50/50 checks match` (n, the 378 cells of
  Table 3, its row of the Appendix F.2 summary table and the 40 curve values).

The robustness panel, Figure 4(b), has its own fast check. It recomputes 50 of the 2,400 chunks of 10,000 random
reweightings, compares each with the paper's run of that chunk (`data/reference/fig4b_chunk_tallies.jsonl`), and
prints the panel merged from all 2,400 chunk tallies (about 4 minutes):

```bash
python -m onetemp.reproduce fig4b
```

```
[fig4b] recomputing 50 of the 2,400 chunks (about 4 minutes)
  Figure 4(b): 50 of 50 recomputed chunks equal the paper's run (W and KL hashes, band counts, every tally)

| K | 0–1 | 1–2 | 2–3 | 3–4 | 4–5 | 5–5.88 | KL > 3 | KL 1-2 |
|---|---|---|---|---|---|---|---|---|
| 512 | 99.99 | 99.85 | 98.75 | 97.12 | 93.38 | 87.97 | 93.00 | 99.85 |
| 128 | 100 | 99.79 | 98.39 | 96.34 | 91.31 | 84.03 | 90.80 | 99.79 |
| 32 | 100 | 99.91 | 98.63 | 96.59 | 92.01 | 85.16 | 91.48 | 99.91 |
| 4 | 99.92 | 98.04 | 93.47 | 87.19 | 78.76 | 69.40 | 78.80 | 98.04 |
| mean over K |  |  |  |  |  |  | 88.52 | 99.40 |

KL > 3, mean over K = 4, 32, 128, 512: 88.52%
KL > 3, per K: 78.80 / 91.48 / 90.80 / 93.00
KL 1-2, mean over K: 99.40%
K = 4, KL 5–5.88: 69.40

  Figure 4(b): 24 of 24 printed cells equal (K = 4, KL 5-5.88: 69.40)
  Figure 4(b): KL > 3 per K = 4, 32, 128, 512: 78.80 / 91.48 / 90.80 / 93.00, mean 88.52%, as in the paper's run

[reproduce] 3 results, 0 mismatches (243 s); outputs in results
```

This is the output of the fresh install above, without the C kernel. The paper prints the 24 cells of the panel;
Section 4.3.2 gives the KL > 3 mean as "on average roughly 90%" and the KL 1-2 share as "99%"
(`python -m onetemp.reproduce text` compares both); the per-K KL > 3 shares are those of the paper's run.

The full run of all 24,000,000 reweightings is under [Reproduce everything](#reproduce-everything).

## Data

`data/` holds, for every study of the paper, how many samples each question received at each setting and how
many of them were correct. It also holds a manifest, the lists of included questions, and the paper's own
analysis results as numbers (`data/reference/`). It holds no prompt, no completion, no reference answer and no
test case. The full description is `docs/data.md`.

| path | what |
|---|---|
| `data/counts/<benchmark>/*.jsonl` | 21 counts files: the 12 temperature sweeps (LiveCodeBench in two parses, below) and 3 min_p sweeps, plus 2 second-stage parts that Appendix I reads on their own; each in the question order the paper's analysis read |
| `data/selection/hitom/*.jsonl` | the 4 first-stage screens that chose the Hi-ToM questions |
| `data/included_ids.json` | the questions each file's inclusion rule keeps, for checking: fitted peak at least 0.1% (temperature sweeps), measured peak at least 0.1% (min_p sweeps), the kept questions (the screens and the top-up part) |
| `data/manifest.json` | per file: sha256, size, rows, questions, settings, samples; which figure and table uses which file |
| `data/reference/` | the paper's own runs as numbers: the heatmap cells, the beta* values, the 2,400 Figure 4(b) chunk tallies and the sha256 of the paper's image files, which the reproduction compares against |

### The counts.jsonl schema

One JSON object per line, one line per (benchmark, model, question, setting):

```json
{"benchmark":"livecodebench","model":"mistralai/Mistral-7B-Instruct-v0.3","id":"1873_A","temperature":0.1,"min_p":null,"n":11502,"c":0}
```

| key | type | meaning |
|---|---|---|
| `benchmark` | string | `livecodebench`, `math500` or `hitom` |
| `model` | string | the Hugging Face repo id: `openai/gpt-oss-20b`, `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen3-8B` or `Qwen/Qwen3-0.6B` |
| `id` | string | the benchmark's own question id: LiveCodeBench `question_id` (`1873_A`, `abc393_e`), MATH-500 `unique_id` (`test/precalculus/807.json`), Hi-ToM `s<sample_id>_o<question_order>_<prompting_type>` built from the fields of `Hi-ToM_data.json` (`s0_o0_CoTP`) |
| `temperature` | number >= 0 | the sampling temperature; in a min_p sweep, its fixed temperature |
| `min_p` | number in [0, 1], or `null` | the min_p value of a min_p sweep (Appendix I; `0.0` is plain sampling); `null` in a temperature sweep |
| `n` | integer >= 1 | the number of samples |
| `c` | integer, `0 <= c <= n` | the number of correct samples |
| `n_cell` | integer, optional | only in the LiveCodeBench `question_n` files: the cell's own sample count, on the rows where it differs from `n` |

Pass rates are weighted by instances: every sample counts, repeated completions included. Every file is a complete
rectangle (each question has each setting once, the rows of a question adjacent). The readers also accept
`.jsonl.gz` and four optional keys that the tools write (`tau`, `c_strict`, `p_fit`, `peak_pct`; `docs/data.md`,
section 2).

**Files that must not be combined.** LiveCodeBench has two files per model because the paper read its counts two
ways: `question_n` gives every temperature the question's largest per-temperature count, as Figures 1-3, Figure
4(b) and Appendices E-G read it; `cell_n` gives each temperature its own count, the "corrected parsing" of
Appendix H and Figure 4(a). Likewise `hitom/Qwen3-0.6B__temperature__topup1000.jsonl` and
`hitom/Qwen3-0.6B__min_p_tau1.5__stage2_nu400.jsonl` are the second-stage samples inside
`hitom/Qwen3-0.6B__temperature.jsonl` and `hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl`, for the Appendix I
comparisons. Give an analysis one of each pair, never both.

### The studies

Model order in the table below: GPT OSS 20B, Mistral 7B Instruct, Qwen3-8B, Qwen3-0.6B.

| benchmark | settings | samples per (question, setting) | questions | included (peak at least 0.1%: fitted; on the min_p row, measured) |
|---|---|---|---|---|
| LiveCodeBench V1-V6 | 16, 16, 38 and 20 temperatures | 1,500, and 10,000 or more on the harder questions; Qwen3-0.6B 9,996, and more on 30 of its questions | 975, 975, 975, 110 | 766, 359, 647, 85 (`question_n`); 766, 359, 677, 86 (`cell_n`) |
| MATH-500 | 16, 16, 20 and 20 temperatures | 1,500 (Qwen3-8B: 750) | 500 each | 500, 455, 496, 492 |
| Hi-ToM | 31 temperatures, 0.075 to 4.0 | 1,500 | the first-stage screen's kept questions: 678, 617, 471, 743 | 678, 615, 466, 743 |
| Hi-ToM, min_p (Appendix I) | Qwen3-0.6B: 10 min_p values at tau = 1.5 and 5 at tau = 2.0; Mistral 7B Instruct: 5 at tau = 2.0 | Qwen3-0.6B 500 and 400; Mistral 1,500 | 648, 648; 617 | 648, 648; 598 |

The per-file table with every count is in `docs/data.md`, section 3.

### Check the files

```bash
python -m onetemp.data.check       # sha256, schema, populations, consistency between files, counts only
python -m onetemp.data.fitfree     # the paper's numbers that need no profile fit, from the counts alone
```

`data/manifest.json` has sha256 `c51bfdded1de36644f14125192c325eb2a5c5f994f108f622034d5a38446c177`; it records the
sha256 of every counts and screen file (the 25 under `data/counts/` and `data/selection/`), and of each file's list
of included questions. `onetemp.data.check` verifies each file against it and ends with
`checks passed: 422174  failed: 0` (25 files, 420,597 rows). It also proves that the files hold counts only: every
row has only the schema's keys, and its only strings are the benchmark, the model and an id that matches its
benchmark's id pattern. Given local copies of the public datasets, `--math500-test`, `--hitom-data` and
`--lcb-meta` also check every id against them and scan the files for every reference answer (none is present).

### What is not shipped

Prompts, completions, reference answers and LiveCodeBench test cases. The prompt builders and the graders download
the questions, the reference answers and the test cases from the public sources at run time, at pinned revisions,
check them by sha256 and keep them in `~/.cache/one-temperature`; the prompts are rebuilt from the questions, and
the completions come from sampling. Also not shipped: the question metadata (join on `id` with the public
datasets), the first stage of Mistral 7B Instruct on Hi-ToM and the first-stage screens of the LiveCodeBench
studies. Figure 1 and the Appendix C count 646 come from an earlier release of the Qwen3-8B LiveCodeBench data, not
shipped; the redraw differs in the grey background.

### Sources, versions and licenses

The models and the datasets are not in this repository: the code downloads them at run time. The only code from
these projects in it is LiveCodeBench's test harness (`onetemp/grading/lcb_execution.py`) and the way
`evaluation_sample()` in `onetemp/grading/livecodebench.py` reads a problem's tests, both under LiveCodeBench's MIT
license (`THIRD_PARTY_NOTICES.md`).

| asset | source and version | license |
|---|---|---|
| Mistral 7B Instruct (Mistral AI) | <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3> | Apache-2.0, per its model card |
| Qwen3-8B, Qwen3-0.6B (the Qwen team) | <https://huggingface.co/Qwen/Qwen3-8B>, <https://huggingface.co/Qwen/Qwen3-0.6B> | Apache-2.0, per their model cards |
| gpt-oss-20b (OpenAI) | <https://huggingface.co/openai/gpt-oss-20b> | Apache-2.0, per its model card |
| LiveCodeBench | release_v6 (V1-V6) of <https://huggingface.co/datasets/livecodebench/code_generation_lite> (revision `0fe84c3`), with code at <https://github.com/LiveCodeBench/LiveCodeBench> (commit `28fef95`) | dataset card: cc; code: MIT |
| MATH-500 | <https://huggingface.co/datasets/HuggingFaceH4/MATH-500> at revision `6e4ed1a`, which is the PRM800K test split of MATH, from <https://github.com/openai/prm800k> and <https://github.com/hendrycks/math> | no license stated; PRM800K and MATH: both MIT |
| Hi-ToM | the 1,200-question `Hi-ToM_data.json` of <https://github.com/ying-hui-he/Hi-ToM_dataset> (commit `459784f`) | Apache-2.0 |
| vLLM | <https://github.com/vllm-project/vllm>: version 0.19.0 for the runs of Appendix I; each run's version is in the table under [The paper's runs](#the-papers-runs) | Apache-2.0 |

The model revisions the runs used are pinned in `onetemp/sampling/paper_runs.py`. LiveCodeBench states that it
collected its problems from LeetCode, AtCoder and Codeforces; we used its released files.

## Full pipeline

The paper's pipeline has four stages:

| stage | command | needs | document |
|---|---|---|---|
| 1. draw samples | `python -m onetemp.sampling.sample_vllm` (one GPU), `python -m onetemp.sampling.launch` (several) | a GPU, `.[sampling]`, access to the Hugging Face hub (and to GitHub for the Hi-ToM questions, or a local copy with `--hitom-file`) | `docs/sampling.md` |
| 2. grade them | `python -m onetemp.grading.math500`, `.hitom`, `.livecodebench` | CPU. MATH-500: `.[grading]`. LiveCodeBench: Linux, and it executes model code, so run it in a container without network on test files downloaded beforehand (`--lcb-dir`) | `docs/grading.md` |
| 3. screen and pool stages | `python -m onetemp.sampling.screen`, `python -m onetemp.sampling.pool_counts` | CPU | `docs/sampling.md` |
| 4. analyse | `onetemp.core` (profile fit, pass@K, allocation), `onetemp.profiles` (the Section 3 analyses), `onetemp.robustness` (Section 4.3.2, Appendices G-I) | CPU | `docs/core.md`, `docs/profiles.md`, `docs/robustness.md` |

Stage 2 writes `counts.jsonl` in the format above, which stage 4 reads; the shipped `data/counts/` let you start at
stage 4. Downloading models and tokenizers needs access to the Hugging Face hub; some model repositories ask you to
accept their terms and log in first (`huggingface-cli login`, or `HF_TOKEN`).

### The paper's runs

Every run is a named profile in `onetemp/sampling/paper_runs.py`: `python -m onetemp.sampling.paper_runs` lists
them, `python -m onetemp.sampling.paper_runs math500/qwen3-8b` prints one, and
`python -m onetemp.sampling.sample_vllm --run math500/qwen3-8b --out runs/x --plan` prints the whole resolved plan,
with the model revision. A profile fixes the model and its revision, the prompt, the grid, the samples per stage,
the question set, `max_tokens`, the dtype and the vLLM engine arguments, including `max_num_seqs`, which changes
which samples come out and so is never a free knob. The sampler passes vLLM only `n`, `max_tokens`, the temperature (and min_p),
the run's stop settings and its decoding switches (`skip_special_tokens`, `detokenize`), so top-p and top-k keep
vLLM's defaults: 1 and off.

| run (`--run`) | model | settings | samples per (question, setting) | max_tokens | dtype | vLLM |
|---|---|---|---|---|---|---|
| `math500/mistral-7b` | Mistral-7B-Instruct-v0.3 | 16 temperatures | 1,500 | 4096 | float16 | 0.19.0 |
| `math500/qwen3-8b` | Qwen3-8B | 20 | 750 | 2048 | bfloat16 | 0.19.0 |
| `math500/qwen3-0.6b` | Qwen3-0.6B | 20 | 1,500 | 4096 | float16 | 0.19.0 |
| `math500/gpt-oss-20b` | gpt-oss-20b | 16 | 1,500 | 4096 | bfloat16, native MXFP4 | 0.19.0 |
| `hitom/qwen3-0.6b` | Qwen3-0.6B | 31 | 500 on all 1,200 questions (screen), then 1,000 on the kept ones | 1000 | float16 | 0.19.0 |
| `hitom/qwen3-8b` | Qwen3-8B | 31 | 500, then 1,000 | 1000 | bfloat16 | 0.19.0 |
| `hitom/gpt-oss-20b` | gpt-oss-20b | 31 | 500, then 1,000 | 1000 | bfloat16, native MXFP4 | 0.19.0 |
| `hitom/mistral-7b` | Mistral-7B-Instruct-v0.3 | 31 | 1,000, then 500 | 500 | float16 | 0.19.0 |
| `minp/qwen3-0.6b/tau1.5` | Qwen3-0.6B | 10 min_p values at tau = 1.5 | 100 on all 1,200 (screen), then 400 on the 648 kept | 1000 | float16 | 0.19.0 |
| `minp/qwen3-0.6b/tau2.0` | Qwen3-0.6B | 5 min_p values at tau = 2.0 | 400 on the 648 | 1000 | float16 | 0.19.0 |
| `minp/mistral-7b/tau2.0` | Mistral-7B-Instruct-v0.3 | 5 min_p values at tau = 2.0 | 1,500 on its 617 Hi-ToM questions | 500 | float16 | 0.19.0 |
| `livecodebench/mistral-7b` | Mistral-7B-Instruct-v0.3 | 16 | 1,500, then 10,000 or more on 365 questions | 500 | float16 | 0.15.1 |
| `livecodebench/gpt-oss-20b` | gpt-oss-20b | 16 | 1,500; 10,000 in place of them on 293 questions | 500 | bfloat16, native MXFP4 | 0.15.1 |
| `livecodebench/qwen3-8b` | Qwen3-8B | 38 | 1,500, then 10,000 or more on 259 questions, at 12 to 38 of the temperatures | 500 | bfloat16 | 0.19.0 for most of the data |
| `livecodebench/qwen3-0.6b` | Qwen3-0.6B | 20 | 9,996 on its 110 questions, then, at 16 of the 20 temperatures, 10,000 more on 30 of them and 100,000 more on 10 of those | 500 | float16 | 0.15.1 for the 100,000-sample stage; not logged for the others |

The exact count of every cell is in `data/counts/livecodebench/*__cell_n.jsonl`.

The grids: 16 = 0.1, 0.2, ..., 1.2, 1.4, 1.5, 1.7, 2.0; 20 = 0.1, 0.2, ..., 2.0; 38 = 0.1, 0.2, ..., 3.5, 3.7, 4.0,
4.2; 31 = 0.075, 0.15, ..., 1.95, 2.0, 2.2, 2.5, 3.0, 4.0; min_p at tau = 1.5: 0, 0.001, 0.003, 0.01, 0.03, 0.1,
0.2, 0.3, 0.5, 0.7; at tau = 2.0: 0.0056, 0.0128, 0.0316, 0.0721, 0.178. The grids, the MATH-500 and Hi-ToM sample
counts, token limits and dtypes, the Hi-ToM screen and the min_p settings are those that Appendices C and I state
(`python -m onetemp.data.fitfree` checks the grids and the sample counts of the data against Appendix C). The
per-run engine settings, the GPU types and the recorded exceptions are in `docs/sampling.md`.

### Commands

```bash
# 1. sample: one GPU, or several (see "Several GPUs" below)
python -m onetemp.sampling.sample_vllm --run math500/mistral-7b --out runs/math500_mistral-7b
python -m onetemp.sampling.launch --gpus 0-7 --run math500/qwen3-0.6b --out runs/math500_qwen3-0.6b

# 2. grade: one counts line per (question, setting)
python -m onetemp.grading.math500 --inputs runs/math500_mistral-7b/cells --out counts_math500_mistral.jsonl
# the Hi-ToM and LiveCodeBench graders work the same way, on runs sampled as in docs/sampling.md:
#   python -m onetemp.grading.hitom --inputs runs/<run>/cells --out counts.jsonl
#   python -m onetemp.grading.livecodebench --allow-code-execution --lcb-dir <tests folder> \
#       --inputs runs/<run>/cells --out counts.jsonl
#   (the LiveCodeBench grader executes model-generated code: read docs/grading.md first)

# 3. screen and pool two sampling stages: see the Hi-ToM example below

# 4. analyse one counts file: profile fit, held-out comparison, bootstrap, tables, figure
python -m onetemp.core.pipeline --counts counts_math500_mistral.jsonl --folds loo 10 --plot --out-dir results/mine
```

**Hi-ToM: the first-stage screen.** Hi-ToM studies only the questions a model finds hard. A first stage draws 500
samples per (question, temperature) on all 1,200 questions (1,000 for Mistral 7B Instruct); a question is kept
when its best single-temperature pass rate is below 0.30; the kept questions get the second stage, and the two
stages are summed over the kept questions:

```bash
python -m onetemp.sampling.launch --gpus 0-3 --run hitom/qwen3-0.6b --stage screen --out runs/hitom_qwen3-0.6b_screen
python -m onetemp.grading.hitom --inputs runs/hitom_qwen3-0.6b_screen/cells --out counts_screen.jsonl
python -m onetemp.sampling.screen --rule hitom --counts counts_screen.jsonl --out kept_qwen3-0.6b.json
python -m onetemp.sampling.launch --gpus 0-3 --run hitom/qwen3-0.6b --stage topup \
    --ids-file kept_qwen3-0.6b.json --out runs/hitom_qwen3-0.6b_topup
python -m onetemp.grading.hitom --inputs runs/hitom_qwen3-0.6b_topup/cells --out counts_topup.jsonl
python -m onetemp.sampling.pool_counts --policy sum --inputs counts_screen.jsonl counts_topup.jsonl \
    --keep kept_qwen3-0.6b.json --out counts_hitom_qwen3-0.6b.jsonl
```

Run on the paper's own two stages, the screen and the pooling give exactly the rows of its data file: with
`data/selection/hitom/Qwen3-0.6B__temperature__screen500.jsonl` as `counts_screen.jsonl` and
`data/counts/hitom/Qwen3-0.6B__temperature__topup1000.jsonl` as `counts_topup.jsonl`, the screen keeps 743
questions and the pooled file has the 23,033 rows of `data/counts/hitom/Qwen3-0.6B__temperature.jsonl` (sorted by
id, where the shipped file keeps the paper's question order). The paper's kept lists are also shipped
(`onetemp/sampling/question_sets/`) for sampling exactly its populations with `--ids-file`. The min_p screen
(`--rule minp`) works the same way. On LiveCodeBench every question stays in its study: the second stage
(`--rule lcb`, `--rule lcb-mistral`) adds samples to the hard questions without `--keep`, and replaces the first
stage for gpt-oss-20b (`pool_counts --policy replace`).

**Several GPUs.** `launch` starts one sampler per GPU (`--gpus 0-7` or `--gpus 0,1,2,3`), deals the questions round
robin (`--split questions`) or splits every cell's samples (`--split samples`, as the LiveCodeBench runs did), and
decides completion by the output files, not by exit codes. With `--split samples` it gives each GPU its own engine
seed: the run's seed, or 0, plus the GPU's position in `--gpus`, counting from 0. The paper's LiveCodeBench runs
gave every GPU vLLM's default seed, which `--same-seed` restores (with `--seed none` on a second stage; see
`docs/sampling.md`, "Where this release differs from the runs, and why").

**Checking a rerun's prompts without a GPU.** `build_prompts` renders a run's prompts with the model's tokenizer and
`verify_prompts` compares them with what the paper's runs recorded about each prompt
(`onetemp/sampling/expected_prompts/`: the token count and two hashes; on LiveCodeBench, the token count or the
text hash alone). With `pip install -e ".[prompts]"`:

```bash
python -m onetemp.sampling.build_prompts --run math500/mistral-7b --out prompts/math500_mistral-7b.jsonl
python -m onetemp.sampling.verify_prompts --prompts prompts/math500_mistral-7b.jsonl \
    --expected onetemp/sampling/expected_prompts/math500__mistral-7b.json
# prints {"expected": 500, "identical": 500, "different": 0, "missing": 0, ...}
```

For all four models on MATH-500 and Hi-ToM, this gave 500 of 500 and 1,200 of 1,200 prompts identical to the
paper's (token count, sha256 of the token ids, sha256 of the text). A rerun reproduces the paper's sampling
distribution (prompts, models, precision, sampler, limits), not its individual draws, which also depend on seeds,
GPUs and batching.

**What Appendix C lists, and where it is.** The "Code and reproducibility" paragraph of the paper's Appendix C
lists four parts:
- a vLLM sampling script over a temperature grid: `onetemp/sampling/sample_vllm.py`, with `launch.py` and
  `paper_runs.py`;
- the MATH-500 grader: `onetemp/grading/math500.py`;
- the unbiased pass@K estimator: `passk_unbiased` in `onetemp/core/passk.py`;
- the comparison of the best single temperature, uniform mixing and the optimal allocation, with leave-one-out and
  10-fold held-out evaluation: `onetemp/core/allocation.py` and `onetemp/core/heldout.py`.

The exact commands are in this README, and the parts not included are under
[What is not included](#what-is-not-included).

## Paper map

Every figure and table of the paper, with the command that reproduces it and how closely. Each command runs from the
repository root on CPU without network, writes to `results/`, prints its comparison and exits with 0 when
everything matches. For Figures 3, 4, 17-31 and 42-45 the command runs only that figure's studies; for the other
items it runs the whole group the item belongs to. The groups and their run times are under
[Reproduce everything](#reproduce-everything).

"Reproduced exactly" means that every number compared is equal: the numbers the paper prints, at their printed
precision; the values of the paper's own runs in `data/reference/` (heatmap cells, beta*, the Figure 4(b) tallies),
exactly, with tau_A to 1e-9 and score_A to a relative 1e-9; and the values Figure 3 draws
(`onetemp/core/paper_numbers.json`), at 4 decimals. It also means that every compared image file has the paper's
sha256. The statuses are those of the record, with the versions of `requirements-exact.txt`; "Package versions"
under [Install](#install) says what two other environments gave.

**Coverage: 47 of 51** (45 figures and 6 tables; 4 not included, each with its reason).

| item | what it shows | command | status |
|---|---|---|---|
| Figure 1 | temperature profiles of Qwen3-8B questions on LiveCodeBench, three highlighted | `python -m onetemp.reproduce fig1` | reproduced, difference: Figure 1 comes from an earlier release of the Qwen3-8B LiveCodeBench data, not shipped; the redraw differs in the grey background |
| Figure 2 | peak temperature, efficient range size and maximal overlap against hardness, Qwen3-8B, LiveCodeBench | `python -m onetemp.reproduce fig2` | reproduced exactly (image sha256, n, OLS fit) |
| Figure 3 | pass@K over the uniform scheme's for the optimal allocation (10-fold) and the fixed temperatures it uses most, Mistral 7B Instruct, LiveCodeBench | `python -m onetemp.reproduce fig3` | reproduced, difference: the ±1σ bootstrap bands, which the paper drew without a seed; the optimal curve at all 20 values of K and the pass@K of every temperature (Table 3) are equal |
| Figure 4 | (a) beta* against D at a 10% loss; (b) the share of 24 million random reweightings that lose under 10%, Mistral 7B Instruct, LiveCodeBench | `python -m onetemp.reproduce fig4` | reproduced exactly ((a) every beta*; (b) all 24 printed cells, merged from the 2,400 shipped chunk tallies of the paper's run, of which the command recomputes 50 and the full run all 2,400) |
| Figures 5-7 | per-question peak temperature against hardness, LiveCodeBench, MATH-500, Hi-ToM (4 models each) | `python -m onetemp.reproduce fig5` (`fig6`, `fig7`) | reproduced exactly (image sha256 of all 12 panels; n and the end values of the sliding averages that Appendix E gives) |
| Figures 8-10 | per-question efficient range size against hardness, with the OLS fit | `python -m onetemp.reproduce fig8` (`fig9`, `fig10`) | reproduced exactly (image sha256, OLS slope, intercept and r²) |
| Figures 11-13 | maximal coverage R_Q(H) against hardness | `python -m onetemp.reproduce fig11` (`fig12`, `fig13`) | reproduced exactly (image sha256 of all 12 panels; the bar values that Appendix E gives, and the number of bars of each Figure 13 panel) |
| Figure 14 | pass@K over uniform for the leave-one-out optimum and the temperatures it uses most, LiveCodeBench, 4 models | `python -m onetemp.reproduce fig14` | reproduced, difference: the bootstrap bands, drawn without a seed in the paper; n of each panel, the pass@K of every temperature (Tables 2-6) and the optimum at K = 1, 16, 256, 1024 (the summary table of Appendix F.2) are equal |
| Figure 15 | the same on MATH-500 | `python -m onetemp.reproduce fig15` | reproduced exactly (n of each panel; the optimum and the best temperature at K = 1, 16, 256, 1024, the summary table of Appendix F.2). The bands, which the paper drew with a fixed seed, are redrawn by `pipeline --seed-scheme spawn --seed 33273348774934605029837111157444119385` but not compared |
| Figure 16 | the same on Hi-ToM | `python -m onetemp.reproduce fig16` | reproduced exactly (as Figure 15) |
| Figures 17-20 | score_B / score_A against KL(W ‖ uniform), 10,000 random reweightings, LiveCodeBench: Mistral, GPT OSS, Qwen3-8B, Qwen3-0.6B | `python -m onetemp.reproduce fig17` (`fig18` ... `fig20`) | reproduced exactly (every cell label and count; tau_A and score_A at each K) |
| Figures 21-24 | the same on MATH-500 | `python -m onetemp.reproduce fig21` (... `fig24`) | reproduced exactly (the same) |
| Figures 25-28 | the same on Hi-ToM | `python -m onetemp.reproduce fig25` (... `fig28`) | reproduced exactly (the same) |
| Figure 29 | beta* against D at 10% and 1% loss, LiveCodeBench, 4 models | `python -m onetemp.reproduce fig29` | reproduced exactly (every beta* after the sweep and after the exact recheck; tau_A, score_A) |
| Figure 30 | the same on MATH-500 | `python -m onetemp.reproduce fig30` | reproduced exactly (the same) |
| Figure 31 | the same on Hi-ToM | `python -m onetemp.reproduce fig31` | reproduced exactly (the same) |
| Figure 32 | the three simulated hardness shapes | `python -m onetemp.reproduce fig32` | not included: a drawing of the shapes; the numbers the text gives for the three pools are compared under Figure 33 |
| Figure 33 | beta* against D for the three simulated pools | `python -m onetemp.reproduce fig33` | reproduced exactly (every beta*; 109 templates; 966, 985 and 980 questions above H = 10^4) |
| Figure 34 | pass@K over uniform for each min_p value and the leave-one-out optimum, Qwen3-0.6B, Hi-ToM, tau = 1.5 | `python -m onetemp.reproduce fig34` | reproduced exactly (its table: 28 ratios, the optimum and uniform rows) |
| Figure 35 | pass@K of the tau = 2.0 min_p values over tau = 1.5 with min_p 0.003, Qwen3-0.6B | `python -m onetemp.reproduce fig35` | not included: the drawing; the numbers the text gives with it (the gains of min_p 0.0316 at tau = 2.0 over 0.003 at tau = 1.5) and the Appendix I table of the best values at tau = 2.0 and 1.5 are compared by `python -m onetemp.reproduce data` |
| Figure 36 | pass@K of min_p at tau = 2.0 over the best plain temperature, Mistral 7B Instruct, Hi-ToM | `python -m onetemp.reproduce fig36` | not included: the drawing; its table in Appendix I (the best value, 0.0056, against the best plain temperature at each K) is compared by `python -m onetemp.reproduce data` |
| Figure 37 | pass@K over uniform at tau = 2.0, measured rates, Qwen3-0.6B and Mistral 7B Instruct | `python -m onetemp.reproduce fig37` | reproduced exactly (the best single values, the ratios the text gives, n = 598 and the 19 questions below the floor) |
| Figure 38 | min_p profiles of Hi-ToM questions, 3 studies | `python -m onetemp.reproduce fig38` | reproduced exactly (image sha256) |
| Figures 39-41 | peak min_p, efficient range size and maximal coverage against hardness, 3 studies | `python -m onetemp.reproduce fig39` (`fig40`, `fig41`) | reproduced exactly (image sha256 and the numbers the text gives) |
| Figures 42-44 | score_B / score_A against KL for min_p: Qwen3-0.6B at tau = 1.5 and 2.0, Mistral at tau = 2.0 | `python -m onetemp.reproduce fig42` (`fig43`, `fig44`) | reproduced exactly (every cell label and count) |
| Figure 45 | beta* against D for min_p, 3 studies | `python -m onetemp.reproduce fig45` | reproduced exactly (every beta*) |
| Table 1 | the shape constraints of the profile fit, from function level to Bernstein coefficients | `python -m onetemp.reproduce tab1` | not included: a table of definitions with no computed entry; `onetemp/core/fit_all.py` applies the constraints |
| Table 2 | pass@K at every temperature and of each strategy, GPT OSS 20B, LiveCodeBench | `python -m onetemp.reproduce tab2` | reproduced exactly (378 of 378 cells, and the bold marks) |
| Table 3 | the same, Mistral 7B Instruct | `python -m onetemp.reproduce tab3` | reproduced exactly (378 of 378) |
| Tables 4-5 | the same, Qwen3-8B, low and high temperatures | `python -m onetemp.reproduce tab4` (`tab5`) | reproduced exactly (774 of 774) |
| Table 6 | the same, Qwen3-0.6B | `python -m onetemp.reproduce tab6` | reproduced exactly (450 of 450) |

The rows that group several items count each item once in the coverage: Figures 5-13 are 9 items, Figures 17-28 are
12, Figures 39-41 are 3, Figures 42-44 are 3 and Tables 4-5 are 2.

**The appendices' unnumbered tables and printed numbers** are compared by the same groups:
- `data` (no profile fit; 265 numbers): the sample counts and grids of the Appendix C tables, and the numbers of
  Appendices H and I that need no fit. It reads the included-question counts from `data/included_ids.json`;
  `tables` and `profiles` recompute them with this package's own fit.
- `tables`: the summary table of Appendix F.2. `heatmaps`: the Appendix G table (tau_A, score_A, the top-row
  shares). `profiles`: the numbers in the text of Appendices E and I.
- The further tests of Appendix H (the loss of the chosen temperature, one side at a time, tilting the mix,
  natural sub-datasets, the role of K) have their own commands and record in `docs/robustness.md`; they are not part
  of `python -m onetemp.reproduce`.

**Numbers in the main text:**

| where | the paper | this code | command | status |
|---|---|---|---|---|
| Section 2.3 | typically 1,500 samples at each of 16 to 38 temperatures between 0.075 and 4.2 | most common count 1,500; 16 to 38 temperatures; 0.075 to 4.2 | `python -m onetemp.reproduce text` | reproduced exactly |
| Section 4.2 | the optimal allocation has about a 10% advantage over each fixed temperature at some K | the curves of Figure 3 | `python -m onetemp.reproduce fig3` | drawn; no printed number to compare (a reading of Figure 3, whose curves are reproduced) |
| Section 4.3 | the separation of Section 3.2.2 is on the order of 10%-20% | Figures 2 and 11-13 | `python -m onetemp.reproduce fig11` | drawn; no printed number to compare (a reading of the coverage figures) |
| Section 4.3.2 | 0.16 ≲ α ≲ 3.15, roughly a factor of 20 | 0.1586 to 3.1462, a factor of 19.8 | `python -m onetemp.reproduce text` | reproduced exactly |
| Section 4.3.2 | KL > 3: on average roughly 90% of the reweightings lose under 10%; KL 1-2: this rises to 99% | 88.52% (mean over K = 4, 32, 128, 512); 99.40%; from the 2,400 shipped Figure 4(b) tallies | `python -m onetemp.reproduce text` | reproduced exactly |

## Reproduce everything

```bash
python -m onetemp.reproduce list                               # the groups, what each covers, run times
python -m onetemp.reproduce all --jobs 3 2>&1 | tee reproduce.log   # every group; outputs in results/<group>/
```

The comparison is printed, one line per result, and the last line counts the results and the mismatches; the
outputs and logs of every group are kept in `results/`. Run again with the same `--out`, it reuses every finished
result and redoes the rest. The record of this release, `all --jobs 3` with the versions of
`requirements-exact.txt`, Python 3.12 and the C kernel, on one machine:

| group | covers | result | time in the record |
|---|---|---|---|
| `data` | data files; Appendices C, H and I without a fit | `checks passed: 422174  failed: 0`; 265 of 265 numbers equal | 5 seconds |
| `text` | Sections 2.3 and 4.3.2 | 3 of 3 equal | 1.5 minutes, with `profiles` |
| `profiles` | Figures 1, 2, 5-13, 38-41 | 49 of 49 images byte-identical; 167 of 168 numbers equal, 1 known difference (Figure 1) | (above) |
| `fig4b` | Figure 4(b), Section 4.3.2 | 50 of 50 recomputed chunks equal; 24 of 24 printed cells equal; the KL > 3 shares per K equal the paper's run | 3 minutes |
| `heatmaps` | Figures 17-28, 42-44 | 2,976 of 2,976 cell labels and counts equal; tau_A and score_A at every K | 4 minutes |
| `beta` | Figures 4(a), 29-31, 45 | 2,760 of 2,760 beta* equal; every certificate proven | 3 minutes |
| `simulated` | Figure 33 | 312 of 312 beta* per pool; 109 templates; 966, 985 and 980 above H = 10^4 | 18 seconds |
| `tables` | Figure 3, Tables 2-6, the summary table of Appendix F.2, the Appendix I table | 189 of 189 checks | 14 minutes |
| `all` | every group | `[reproduce] 58 results, 0 mismatches` | 26 minutes (38 minutes of CPU time) |

With the default `--jobs 1`, `heatmaps` and `beta` run their studies one at a time and take about three times as
long; `python -m onetemp.reproduce list` gives each group's time on one core (about 45 minutes in all). Without
the C kernel, each profile fit takes about 14 times longer. Each process used at most about 0.5 GB of memory.

**The full Figure 4(b) run.** The fast check recomputes 50 of the 2,400 chunks. All 2,400 (24,000,000 random
reweightings, 4 to 5 s per chunk on one core, so about 3 core-hours) run in parallel; worker w takes the chunks w,
w + 8, ..., and a stopped run resumes:

```bash
python -m onetemp.core.fit_all \
    --counts data/counts/livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl \
    --n-mode question-max --out results/fig4b_full/profiles_qmax.json
for w in 0 1 2 3 4 5 6 7; do
  python -m onetemp.robustness.fig4b worker \
      --input data/counts/livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl \
      --profiles results/fig4b_full/profiles_qmax.json --tally-dir results/fig4b_full/tallies \
      --worker $w --n-workers 8 &
done; wait
python -m onetemp.robustness.fig4b merge --tally-dir results/fig4b_full/tallies --expect-chunks 2400 \
    --out results/fig4b_full/cells.json --compare-dir data/reference/fig4b_chunk_tallies.jsonl
```

`merge` prints the panel, compares every chunk with the paper's run and ends with `PASS` (exit code 0) or `FAIL`
(exit code 1); see `docs/robustness.md`, "Figure 4(b): 24 million W's".

## What is not included

- **Per-sample data** (prompts, completions, reference answers, LiveCodeBench test cases): the figures need counts
  only, and the public sources are downloaded at run time.
- **An earlier release of the Qwen3-8B LiveCodeBench data**: Figure 1 and the Appendix C count 646 come from an
  earlier release of the Qwen3-8B LiveCodeBench data, not shipped; the redraw differs in the grey background.
- **Table 1**: definitions with no computed entry; `onetemp/core/fit_all.py` applies them.
- **Figure 32**: a drawing of the three simulated shapes; the numbers given for them are compared.
- **Figures 35 and 36**: the drawings; the Appendix I numbers given with them are compared (for Figure 35 the gains
  of 0.0316 over 0.003 and the table of the best values at tau = 2.0 and 1.5, for Figure 36 its table).
- **The bootstrap bands of the LiveCodeBench ratio figures** (Figures 3 and 14): the paper drew them without a seed,
  so no seed recreates them; the code's bands are seeded and statistically the same.
- **Appendix F's cross-check with a second solver** (CLARABEL through CVXPY): cvxpy is outside the dependencies.
  `onetemp.core.allocation` reports the KKT residuals and the first-order gap bound instead, and its
  `--integer exact` option searches the whole integer simplex and returns a certificate of optimality.
- **Output lengths and truncation shares** of Appendix I (for Qwen3-0.6B, the mean output 35% to 46% shorter, 6.4%
  to at most 0.5% of samples at the token limit, and 251.1 against at most 222.8 tokens per sample; for Mistral
  7B Instruct, 449 to 238 tokens per sample and 76% to 3% at the limit): they need the completions.
- **Parts of Appendix H**: the resampling intervals, the group reweighting of whole sub-datasets, the adversarial
  pool, the theorem checks, the split of the beta* points above 10^6 with pass@K computed beyond double precision,
  and a few numbers that were not recomputed for this release (listed in `docs/robustness.md`, "Not reproduced
  here, and why").
- **The samples themselves**: a rerun with `onetemp.sampling` reproduces the sampling distribution, not the draws,
  which also depend on seeds, GPUs and batching.
- **The first-stage data of the LiveCodeBench studies and of Mistral 7B Instruct on Hi-ToM**, and two selection
  rules that were never recorded (which 110 LiveCodeBench questions Qwen3-0.6B ran on, and which 30 and 10 of them
  got more samples): the resulting question lists are in `onetemp/sampling/question_sets/`.
- **The question metadata** (LiveCodeBench platform and difficulty, MATH-500 level, Hi-ToM deception and story
  length): join on `id` with the public datasets. The MATH-500 subject and the Hi-ToM order and prompting type are
  part of the `id`.

## Citation

```bibtex
@inproceedings{orly2026onetemperature,
  title     = {One Temperature to Rule Them All?},
  author    = {Orly, Aviv and Shem-Ur, Ori and Oz, Yaron},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  eprint    = {2611.13077},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

The code and the data of this repository are under the MIT License (`LICENSE`). Two pieces come from
LiveCodeBench and keep its MIT license: `onetemp/grading/lcb_execution.py`, its test harness with the changes
listed in the file's header, and `evaluation_sample()` in `onetemp/grading/livecodebench.py`, which reads a
problem's tests the way LiveCodeBench's code does. The notice is kept in `lcb_execution.py` and in
`THIRD_PARTY_NOTICES.md`. The models and benchmarks are not included; their licenses are listed under
[Sources, versions and licenses](#sources-versions-and-licenses).
