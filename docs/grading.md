# Grading: completions to `counts.jsonl`

**In short.** The graders turn sampled completions into per-question success counts. For every question q and
setting tau they write n (samples) and c (samples graded correct). c/n is the estimate of the single-sample success
probability P_q(tau) that Section 2 of the paper builds on, and every analysis in the package reads these counts.
Given the paper's own completions, the graders return the paper's verdicts (see "Reproduction record" below).

There is one grader per benchmark. Each grades exactly as the paper's runs did (Appendix C.5, "Datasets"):

| benchmark | command | a sample passes when |
|---|---|---|
| MATH-500 | `python -m onetemp.grading.math500` | The content of the **last** `\boxed{}` span equals the reference answer after normalization, or a set of generic equivalence rules plus a time-bounded SymPy check accepts it. This is the frozen grader v2.5. |
| Hi-ToM | `python -m onetemp.grading.hitom` | The **first valid option** named in the completion (by name, or by letter used as an answer label) is the reference container. The answer is read from the completion only, never from the prompt. |
| LiveCodeBench | `python -m onetemp.grading.livecodebench` | The extracted code passes **every** public and private test in LiveCodeBench's own harness: 6 s per test, 4 GiB of memory. |

No reference answer, test case or completion is stored in this repository. Each grader downloads its public
dataset at run time, at a pinned revision, into the cache `~/.cache/one-temperature`
(`$XDG_CACHE_HOME/one-temperature` if set; the prompt builders of `docs/sampling.md` use the same folder), and checks
its sha256 before use.

## Install and test

```bash
pip install -e .                  # Hi-ToM and LiveCodeBench graders (numpy only; LiveCodeBench needs Linux)
pip install -e ".[grading]"       # adds sympy 1.14.0 and mpmath 1.3.0 for the MATH-500 grader
pip install -e ".[dev]"           # pytest, plus the same sympy for the tests
python -m pytest                  # the grading tests are tests/test_grading_*.py
python -m onetemp.grading.hitom --selftest
```

- The tests use only made-up cases: invented MATH-500-style answers, fictional Hi-ToM containers, and toy
  programming problems. No dataset is downloaded. The LiveCodeBench tests execute a few small, harmless programs.
- The Hi-ToM `--selftest` runs the structural checks and synthetic cases of the matcher (see "Integrity checks").

## Input: completions files

The sampling step (`docs/sampling.md`) writes one JSON object per line, one line per (question, setting) cell. The
files end in `.jsonl` or `.jsonl.gz`, and a grader reads every such file under the paths passed to `--inputs`.
For a sampler output directory, pass `<out>/cells` (the folder also holds a `prompts.jsonl` that is not a
completions file).

```json
{"benchmark": "math500", "model": "mistralai/Mistral-7B-Instruct-v0.3", "id": "test/algebra/101.json",
 "temperature": 0.7, "min_p": null, "completions": ["...", "..."], "finish_reasons": ["stop", "length"]}
```

- **`benchmark`** (optional) must match the grader.
- **`model`**: the Hugging Face repo id. If a line has none, the grader takes `provenance.model`, then `--model`.
- **`temperature`** is required. **`min_p`** is `null`, or absent, in the temperature sweeps. In the min_p runs
  (Appendix I), `temperature` is the fixed tau and `min_p` is the swept value.
- **`finish_reasons`** (optional, as long as `completions`) are used only to report truncation.
- **`n`** (optional): if present, it must equal the number of completions, so a truncated file is refused.
- **Several lines for one cell**, for example one per GPU or one per stage file, are summed. Giving the same file
  twice is refused, and a cell whose completions appear twice draws a warning.
- **Only these keys are read** (plus `lcb_version` and `lcb_row` for LiveCodeBench). A prompt or an answer stored in
  the same line never reaches a grader.
- **Pipes.** A path given explicitly may be any file or pipe, for example `--inputs /dev/stdin`.

**Question ids**, the same ones the shipped data use:

| benchmark | `id` | example |
|---|---|---|
| MATH-500 | `unique_id` of the test split | `test/precalculus/807.json` |
| Hi-ToM | `s{sample_id}_o{question_order}_{prompting_type}` | `s12_o3_CoTP` |
| LiveCodeBench | `question_id` of `code_generation_lite` | `1873_A`, `abc301_a`, `2848` |

For LiveCodeBench a line may also carry `"lcb_version"` (`"v1"` ... `"v6"`) and `"lcb_row"` (the 0-based line in that
version's file). The grader then checks them against the dataset and stops on a mismatch.

**The completion text must be what the paper graded.** The sampling scripts write it this way:

| benchmark, model | text in `completions` |
|---|---|
| MATH-500, all models | vLLM's `output.text` (`skip_special_tokens=True`) |
| LiveCodeBench, all models | vLLM's `output.text`: the raw text after the ```` ```python ```` priming. The grader extracts the code. |
| Hi-ToM, Mistral 7B Instruct | vLLM's `output.text` |
| Hi-ToM, Qwen3-0.6B, Qwen3-8B, gpt-oss-20b | The generated token ids decoded with `skip_special_tokens=False`, so that `<\|im_end\|>`, `<\|end\|>`, `<\|return\|>` etc. stay visible. See "Hi-ToM" below. |

## Output: `counts.jsonl`

One line per (benchmark, model, question, setting), sorted. An illustrative row (format only):

```json
{"benchmark": "hitom", "model": "Qwen/Qwen3-8B", "id": "s12_o3_CoTP", "temperature": 0.075, "min_p": null, "n": 1500, "c": 3}
```

- **Instance-weighted.** Every sample counts once, also when the model repeats itself. Verdicts are cached by text
  only to save time.
- **MATH-500 adds `"c_strict"`:** the samples that pass the exact layer alone, without the equivalence rules.
- **`--summary FILE`** writes, per (model, setting), the outcome classes over samples, the number of truncated samples
  (finish reason `length`) and the truncated samples that still passed. It also records the grader's version, the
  data it read (the sha256 of the MATH-500 and Hi-ToM files; for LiveCodeBench which files were read and whether
  their sha256 was checked) and totals. It holds counts only.
- The writes are atomic (a temporary file, then a rename), and an existing output is never overwritten without
  `--force`. Outputs are checked before any grading starts, so a refusal comes at once, not after hours of work.

The file feeds the analyses directly: it is the format of `data/counts/` (`python -m onetemp.counts` is the one
reader). Pool two sampling stages with `python -m onetemp.sampling.pool_counts` (`docs/sampling.md`).

## Commands

```bash
# MATH-500 (8 grading processes; the dataset is downloaded once and cached)
python -m onetemp.grading.math500 --inputs runs/math500_mistral/cells --out counts/math500_mistral.jsonl \
    --summary counts/math500_mistral.summary.json --jobs 8

# Hi-ToM (the answer format is chosen from the model id; --answer-format overrides it)
python -m onetemp.grading.hitom --inputs runs/hitom_qwen3-8b/cells --out counts/hitom_qwen3-8b.jsonl --jobs 8

# LiveCodeBench (runs model-generated code: use an isolated container or VM with no network)
python -m onetemp.grading.livecodebench --allow-code-execution --inputs runs/lcb_mistral/cells \
    --out counts/lcb_mistral.jsonl --summary counts/lcb_mistral.summary.json --workers 6

# One verdict, for checking a rule by hand
python -m onetemp.grading.math500 --try 'so $\boxed{\dfrac{3}{7}}$' --gold '\frac{3}{7}'
python -m onetemp.grading.hitom --try 'C. teal_drawer' --qid s12_o3_CoTP
python -m onetemp.grading.livecodebench --allow-code-execution --try-code solution.py --qid 1873_A
```

Common options:

| option | meaning |
|---|---|
| `--inputs PATH...`, `--out FILE` | completions files or folders; the `counts.jsonl` to write (both required for a full run) |
| `--summary FILE` | optional JSON with outcome classes, truncation and the data read |
| `--dataset FILE` (MATH-500, Hi-ToM) or `--lcb-dir DIR` (LiveCodeBench) | Use a local copy of the dataset instead of downloading it. `--lcb-dir` holds `test.jsonl` ... `test6.jsonl` under their hub names. |
| `--cache-dir DIR` | Where downloads are kept. The default is `~/.cache/one-temperature`. |
| `--no-verify-data` | Skip the sha256 check of the dataset. |
| `--model ID` | The model id for lines that do not name one. |
| `--jobs N` (MATH-500, Hi-ToM) / `--workers N` (LiveCodeBench) | Grading processes; LiveCodeBench defaults to min(6, cores). |
| `--force` | Overwrite an existing output. Without it, the graders refuse. |

Further options: MATH-500 `--rlimit-gb` and `--any-sympy`; Hi-ToM `--answer-format auto|plain|qwen3|gptoss`,
`--reasoning` and `--selftest`; LiveCodeBench `--allow-code-execution` (required), `--extract auto|v1|v2`,
`--timeout`, `--chunk`, `--dedup` and `--workdir`.

## MATH-500

The grader is the frozen version v2.5 that graded every MATH-500 sample in the paper. Its code is copied unchanged:
only comments and three docstrings were edited. The rules are fixed and generic, never per question; they were
frozen, with a hand-labelled fixture, before the data they graded existed, and were never made more lenient after
seeing results.

- **Extraction.** `extract_boxed(completion)` takes the completion only, and the **last** `\boxed{}` span counts.
  "Pass if any span matches" would be in-sample oracle selection. A completion with no `\boxed{}` is the
  `no_boxed` bucket and fails. Empty never matches empty. The number of spans is recorded (`multi_boxed` in the
  summary totals), never used to decide.
- **Exact layer.** Both sides are normalized (`\dfrac`, `\frac12`, `\left`/`\right`, `$`, `%`, a trailing period,
  `^\circ`, `3.0`), then compared by string equality: never `in`, never a regex search.
- **Equivalence layer.** Rule codes R1-R6, MX, TS, BE, PX, SS and TP; the full list, with each rule's text, is in the
  docstring of `onetemp/grading/math500.py`:
  - units and degree marks (R1);
  - `x =` prefixes (R2);
  - lists and sets as multisets, with tuples and intervals kept ordered (R3);
  - `\pm` (R4);
  - mixed numbers (MX: a reference mixed number means addition; a candidate's own mixed number must be equal under
    both readings);
  - thousands separators (TS);
  - braced exponents (BE);
  - subscripted prefixes (PX);
  - singleton sets (SS);
  - a bare list against a tuple (TP).

  Then SymPy equivalence. Model output is never passed to `eval` or `sympify` directly: only strings of digits,
  letters, `_`, `+-*/()`, `.` and spaces reach sympy's `parse_expr`, which evaluates them with a restricted namespace
  and empty `__builtins__`. Only a real `bool` from sympy is believed (an unevaluated `Eq` is truthy).
- **Fail closed.** A LaTeX command the translator does not handle makes the sample `unparseable` (R5). Every
  comparison is limited to 2 s. A timeout or an internal error fails the sample and has its own bucket (R6).
- **Caching.** A distinct last span is graded once per cell, and its verdict is reused for every sample with that
  span. As a check, whenever the number of distinct texts seen in the cell is a multiple of 97 and a new distinct text
  has an already-graded span, that text is graded again on its own. A disagreement that involves no timeout stops
  the run.
- **Limits.** Each grading process caps its own address space at 3 GB (`--rlimit-gb`): the paper's three-model grids
  ran with 3 GB, and the gpt-oss grading with 6 GB. A memory error becomes `grader_error`.
- **Environment.** The paper's verdicts were computed with sympy 1.14.0, mpmath 1.3.0 and pure-Python ground types.
  The module sets the ground types itself and refuses other versions unless you pass `--any-sympy` (verdicts may
  then differ from the paper's). The 2 s limit uses `SIGALRM`, so it needs Unix and the main thread of each grading
  process, which `--jobs` provides.

Buckets: `exact_match`, `equivalence_match` (both pass), `parsed_wrong`, `unparseable`, `no_boxed`,
`empty_after_normalize`, `grader_timeout`, `grader_error`.

## Hi-ToM

Every prompt lists all 15 options, so the reference container is inside every prompt. The grader therefore never
sees the prompt: it loads the option list and the answer from the dataset file and drops the story and the prompt as
it reads.

- **The matcher** is the paper's. Its 13 hashed functions and 4 constants are copied byte for byte; their contract
  hash, `grader_hash()`, is `bf137bfc`. `python -m onetemp.grading.hitom --selftest` checks it, and the grader refuses
  to run on any other hash. Only a docstring outside the hashed code was edited.
- **Signatures.** `extract(completion, choices)` sees the completion and the question's own option list, never the
  reference answer; the reference enters only in `grade(extracted, gold)`, which rejects a missing answer first.
- **Matching.** Exact, after case-folding, underscores read as spaces, and collapsed whitespace. `teal_drawer`
  never matches `teal_drawer_2`, and a bare colour matches nothing. Dict lookups only: no substring, prefix, fuzzy or
  regex match.
- **Letters** count only as answer labels: a leading `C`, `C.`, `(C)` or `**C**`, or a letter after
  `Answer:`, `The answer is` or `option`.
- **The first valid option decides.** A container-shaped token that is not on offer is skipped. A completion that
  names no valid option fails, as `not_a_choice` (it named something else) or `no_answer` (it named nothing). The
  last valid mention is recorded and never used ("pass if any mention matches" is not available).

**Answer formats.** The format is chosen from the model id, as in the paper's runs:
- **plain** (Mistral 7B Instruct and any other model): the whole completion goes to the matcher.
- **qwen3** (thinking off): every `<|im_end|>` and `<|endoftext|>` is removed first.
- **gptoss** (Harmony, with the final channel primed in the prompt): the text is cut at the first `<|return|>`,
  `<|end|>`, `<|call|>`, `<|start|>` or `<|endoftext|>`.

`--reasoning` selects a reasoning-on variant: the text after the last `</think>`, or the Harmony final channel. The
paper did not use it.

Buckets: `correct` (passes), `wrong_valid_choice`, `not_a_choice`, `no_answer`, `empty`. They partition the samples.

## LiveCodeBench

For each completion:

1. **Extraction.** The code is the first ```` ```python ... ``` ```` block, else the first ```` ``` ... ``` ````
   block, else the whole text, stripped. These are the paper's generators' own functions. The generators of
   Qwen3-8B and gpt-oss-20b also drop a closing ```` ``` ```` that ends the text (variant v2); those of Mistral 7B
   Instruct and Qwen3-0.6B do not (v1). The grader applies each model's own variant; `--extract v1|v2` overrides
   it, and a model outside the paper's four gets v2 with a warning.
2. **Execution.** The code runs on all of the problem's tests, public then private, through LiveCodeBench's own
   harness. `onetemp/grading/lcb_execution.py` is vendored from LiveCodeBench (https://github.com/LiveCodeBench/LiveCodeBench)
   commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` under its MIT license; the license text is in the file. As in the
   original:
   - one process per completion, killed after a global limit of (6 + 1) x tests + 5 s;
   - 6 s per test (`signal.alarm`, wall clock);
   - stdin programs compared line by line, exactly or else as Decimals; call-based (LeetCode) programs by `==`,
     with a returned tuple read as a list.

   Plus the two local fixes the paper's evaluators ran (the changes of LiveCodeBench pull request #103, "Fix:
   memory explosion issue", open and not merged):
   - a solution still alive after the global limit is also killed by its process id;
   - the solution's address space, data segment and stack are capped at 4 GiB by `reliability_guard` (the original
     has no cap).
3. **Verdict.** The completion passes if every test passes: `all(x == True for x in result)`.

As in the paper, every completion is executed, in chunks of 500 (`--chunk`) with `--workers` processes at a time.

- **`--dedup`** runs each distinct extracted code once and counts its verdict for every sample with that code. It is
  much faster and not how the paper ran it. It differs from running every sample only where the code alone does not
  decide the outcome: a wall-clock limit, the memory cap, or a program that uses randomness or the clock (`random` is
  reseeded in every forked test process).
- **Resources.** The limits are wall clock, so give each worker its own core and keep the machine otherwise idle.
  The 4 GiB cap counts the whole solution process, including what it inherits from the grader. A worker's solution
  may therefore use up to about 4 GiB, so provision that much RAM per worker.
- **Tests.** `livecodebench/code_generation_lite` at revision `0fe84c39`. Version vN is file `test{N}.jsonl`
  (`test.jsonl` for v1). As in LiveCodeBench, private tests are stored compressed (base64, zlib, pickle). They are
  unpickled only after the file matched its pinned sha256, unless `--no-verify-data` is given. The files are large
  (4.5 GB for the six); each is hashed once and indexed by `question_id` (the index is cached in the download
  cache).
- **Working directory.** Solutions run in a fresh temporary directory, removed afterwards (give `--workdir` to
  choose one instead; it is kept), so files a solution writes land there.
- **Outcome classes** in the summary: `passed`, `wrong_answer`, `time_limit`, `runtime_or_compile_error`,
  `harness_error_or_global_timeout` (in this harness a global timeout is recorded as a harness error), `failed_other`.

### Security: this grader executes model-generated code

- `--allow-code-execution` is required for every LiveCodeBench command, including `--try-code`. Without it the grader
  stops.
- LiveCodeBench's `reliability_guard` caps memory and disables some destructive functions (`os.remove`,
  `os.kill`, `subprocess.Popen` and others); it is **not a sandbox**. Run the grader inside a container or VM with
  no network access, and on nothing whose files you cannot lose. Without network it cannot download the tests:
  fetch `test.jsonl` ... `test6.jsonl` of `livecodebench/code_generation_lite` at revision `0fe84c39` beforehand and
  pass their folder with `--lcb-dir` (the sha256 check still runs).
- The memory cap is per solution process (4 GiB) and the time limits are wall clock; budget about 4 GiB of RAM and
  one core per `--workers` process.
- The private tests are decoded with `pickle`. Keep the sha256 check on (the default) so that only the pinned
  files are unpickled; `--no-verify-data` unpickles whatever file you point at.
- The harness needs Linux (it uses `fork` and `SIGALRM`); the start method is pinned to `fork`.

## Data sources (fetched at run time, checked by sha256)

| benchmark | source | sha256 |
|---|---|---|
| MATH-500 | `HuggingFaceH4/MATH-500` @ `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, `test.jsonl` (446,564 bytes; byte-identical to `openai/prm800k` `prm800k/math_splits/test.jsonl`) | `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` |
| Hi-ToM | `github.com/ying-hui-he/Hi-ToM_dataset` @ `459784fa8de09f35e9110e1e9454b41cd9b4f985`, `Hi-ToM_data/Hi-ToM_data.json` (5,134,393 bytes, 1,200 questions) | `088b6c54d21d1a0d3045812f36ac8b5b342346a6a996c003738976730ac256f4` |
| LiveCodeBench v1 | `livecodebench/code_generation_lite` @ `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `test.jsonl` (1,252,609,773 bytes) | `2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f` |
| LiveCodeBench v2 | same, `test2.jsonl` (713,377,060 bytes) | `095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3` |
| LiveCodeBench v3 | same, `test3.jsonl` (623,360,766 bytes) | `28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79` |
| LiveCodeBench v4 | same, `test4.jsonl` (1,204,644,685 bytes) | `d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da` |
| LiveCodeBench v5 | same, `test5.jsonl` (557,699,297 bytes) | `7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214` |
| LiveCodeBench v6 | same, `test6.jsonl` (134,303,240 bytes) | `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5` |

These are the bytes the paper's runs graded against:
- the MATH-500 answers, and the Hi-ToM options and answers, rebuilt from these files equal the paper's answer keys
  entry for entry (500 of 500 answers; 1,200 of 1,200 records, including the factorial labels);
- the LiveCodeBench files that the paper's code-evaluation runs downloaded were kept, and they hash to exactly these
  values (6 of 6).

## Integrity checks

| check | where it acts |
|---|---|
| Datasets are verified by sha256 (and size, for LiveCodeBench) before use; a mismatch stops the run. | all three graders; `--no-verify-data` skips it |
| The MATH-500 frozen block (the code between the `THE FROZEN GRADER (v2.5)` and `END OF THE FROZEN GRADER` marker comments of `onetemp/grading/math500.py`) is pinned by a sha256 in the test suite, so any edit to it fails `python -m pytest` (`tests/test_grading_math500.py`, `TestFrozenCode`). The same test pins the version string and the 2 s limit. | MATH-500 |
| sympy 1.14.0, mpmath 1.3.0 and pure-Python ground types are checked at the start; other versions are refused unless `--any-sympy`. | MATH-500 |
| Every reference answer must normalize to a non-empty string; an unknown question id stops the run. | MATH-500 |
| Cached verdicts are re-checked on every 97th new distinct text with a known span; a disagreement without a timeout stops the run. | MATH-500 |
| Fail closed: any grader exception, timeout or memory error counts as a fail, in its own bucket. | MATH-500 |
| The matcher's contract hash must be `bf137bfc`; otherwise the grader refuses to grade. `--selftest` also asserts that `extract` and `analyse` take exactly `(completion, choices)`, `grade` exactly `(extracted, gold)` and `answer_text` exactly `(text, family, arm)` (so no record, prompt or answer can reach the extraction), that `grade(None, gold)` rejects, that the module never imports `re`, and that synthetic cases land in the right buckets. | Hi-ToM |
| Every dataset answer must be one of its question's own options; duplicate ids and unparsable option lists stop the run. | Hi-ToM |
| `lcb_version` and `lcb_row` in a line must match the dataset; a `question_id` found in none of the six files stops the run. | LiveCodeBench |
| The summary records the sha256 of the vendored harness file and which dataset files were read and verified. | LiveCodeBench |
| `benchmark` in a line must match the grader; `n` must equal the number of completions; a file given twice is refused; the same completions seen twice for one cell draw a warning. | all three graders |
| An existing output is never overwritten without `--force`, and outputs are checked before any work. | all three graders |

## How the graders differ from the paper's runs

- **MATH-500:** the grading code is unchanged: its syntax tree, docstrings aside, is identical to the paper's grader.
  - Verdicts are cached per input line; the paper's runs cached per cell file. This is the same thing when a cell is
    one line, as the sampling scripts write it.
  - As in the paper, a verdict that the 2 s limit or the memory cap decides can depend on the machine.
- **Hi-ToM:** none. The matcher's hashed code and the adapter's `answer_text` are copied byte for byte. The matcher
  is deterministic.
- **LiveCodeBench:** the rule, the limits and the harness code are the paper's. The environment differs in three
  ways:
  - Solutions run in a fresh temporary directory (`--workdir`); the paper's ran in the LiveCodeBench checkout.
  - The multiprocessing start method is pinned to `fork`, the default of the paper's Python 3.10 on Linux.
  - The memory the grader process holds, which counts against the 4 GiB cap, is not the paper's evaluator's.

## Reproduction record

**Bottom line.** Given the paper's own completions, the released graders return the paper's verdicts.

- **MATH-500, every sample of the paper.** All four models were re-graded: 36,000 cells and 46,500,000 samples.
  - Every pass/fail verdict is identical.
  - All 36,000 counts equal the data files (`data/counts/math500/`).
  - Appendix C's "included questions" are reproduced exactly: 455, 496, 492 and 500.
  - Only 2 samples differ, and only in which failure bucket they land.
- **Hi-ToM, a random sample of 644,000 samples** from all four models. Every verdict and every bucket is
  identical, and 150 of 150 data cells of 1,500 samples are equal.
- **LiveCodeBench, a random sample of 94,000 executed samples** (Mistral 7B Instruct, Qwen3-8B, gpt-oss-20b).
  - 93,997 verdicts are identical, and 18 of 19 data cells are equal.
  - The 3 different samples were recorded as fails in the paper's run and pass in the re-run.
- **The command lines themselves**, fed real cells through a pipe:
  - MATH-500: 400 of 400 counts equal the data files;
  - Hi-ToM: 186 of 186, each cell summed from its two stages;
  - LiveCodeBench: 3 of 3, every completion executed.

### What was compared, and how

The paper's grading runs kept three kinds of records:
- **MATH-500 and Hi-ToM:** every cell's completions, with the verdict the paper's grader gave each sample.
- **LiveCodeBench:** the raw completions, with a pass flag for each sample.

The checks ran the released modules unchanged, on the same code paths as the command lines, with the stored
completions in memory:
- the MATH-500 per-cell worker function, with its 3 GB address-space cap and sympy cache clearing;
- the Hi-ToM `grade_completion`, with each model's answer format;
- the LiveCodeBench `run_codes` after the model's `extract_code`.

The reference answers and tests came from the pinned public files, as at run time. No completion, reference answer
or test case was copied into this repository.

Three comparisons:
1. **Per sample:** the released grader's verdict against the verdict the paper's run recorded. For MATH-500 also the
   strict (exact-layer) bit and the bucket; for Hi-ToM also the bucket.
2. **Per cell:** the released count c against the count in the data file that the paper's analyses read. These are
   the per-question counts behind the MATH-500, Hi-ToM and LiveCodeBench figures and tables.
3. **Code:** the parts copied from the paper's graders against their originals.

### Reproduction table

| paper number | ours | difference | where |
|---|---|---|---|
| MATH-500 counts c, every (question, tau) cell: Mistral 7B Instruct 8,000, Qwen3-0.6B 10,000, Qwen3-8B 10,000, gpt-oss-20b 8,000 | 36,000 of 36,000 equal | 0 | the per-question data behind every MATH-500 figure and table |
| MATH-500 per-sample pass/fail (46,500,000 samples) | 46,500,000 of 46,500,000 equal | 0 | the paper's records of its grading |
| MATH-500 per-sample strict (exact-layer) bit | 46,500,000 of 46,500,000 equal | 0 | the same |
| MATH-500 per-sample bucket | 46,499,998 of 46,500,000 equal | 2 samples, Mistral, both failing in both runs. A re-run of the 26 Mistral cells where the paper's run recorded time-outs found one such difference: `parsed_wrong` in the paper's run, `grader_timeout` here. The 2 s comparison limit is wall clock, so a heavily loaded machine can time out where an idle one does not. | the same |
| MATH-500 included questions, Mistral 7B Instruct: 455 | 455 | 0 | Appendix C.2 |
| MATH-500 included questions, Qwen3-8B: 496 | 496 | 0 | Appendix C.3 |
| MATH-500 included questions, Qwen3-0.6B: 492 | 492 | 0 | Appendix C.4 |
| MATH-500 included questions, gpt-oss-20b: 500 | 500 | 0 | Appendix C.1 |
| MATH-500 total passes (Mistral, 0.6B, 8B, gpt-oss) | 1,103,165; 4,784,482; 5,213,014; 7,194,409 | equal to the data files | the same data |
| Hi-ToM per-sample verdict and bucket, 980 random cells, 4 models (Qwen3-0.6B, Qwen3-8B, gpt-oss-20b with their adapters; Mistral 7B Instruct plain) | 644,000 of 644,000 equal | 0 | the paper's records of its grading |
| Hi-ToM counts of kept questions (first stage 500 + second stage 1,000 samples), 150 random (question, tau) cells of Qwen3-0.6B, Qwen3-8B and gpt-oss-20b | 150 of 150 equal | 0 | the per-question data behind the Hi-ToM figures and tables |
| LiveCodeBench per-sample verdicts, 57 random cells: Mistral 27, Qwen3-8B 21, gpt-oss-20b 9 (8 from its 1,500-sample run, 1 from its 10,000-sample run) | 93,997 of 94,000 equal | 3 samples (Mistral 1, Qwen3-8B 2), each recorded as a fail in the paper's run and a pass here | the paper's evaluation records |
| LiveCodeBench counts, the 19 sampled cells whose data cell is that one run (Mistral 10, gpt-oss-20b 9) | 18 of 19 equal | 1 cell: Mistral, question `3154`, tau 1.0; 428 in the data, 429 here | the per-question data behind the LiveCodeBench figures |
| Command line, MATH-500: `--inputs /dev/stdin`, fed 400 random cells (100 per model). The dataset was downloaded from the hub and checked by sha256. | 400 of 400 counts equal | 0 | the data files |
| Command line, Hi-ToM: `--inputs /dev/stdin`, fed every cell of 2 random kept questions per model (Qwen3-0.6B, Qwen3-8B, gpt-oss-20b). The dataset was downloaded from GitHub and checked by sha256. | 186 of 186 counts equal; each 1,500-sample cell is summed from its two stage lines | 0 | the data files |
| Command line, LiveCodeBench: `--inputs /dev/stdin`, fed 3 Mistral cells, every completion executed, test files checked by sha256 | 3 of 3 counts equal (487, 1,492 and 1,499 of 1,500) | 0 | the data file |

How the samples were drawn:
- **MATH-500:** no sampling; every cell was re-graded.
- **Hi-ToM:** two independent seeded random samples per model. A cell drawn by both would count twice; the expected
  number of such cells is below 0.1.
- **LiveCodeBench:** four seeded random samples of cells, deduplicated (one cell was drawn twice and is counted
  once). The cells had fewer than about 8 MB of tests, at most 1,500 distinct codes, and at least one pass and one
  fail recorded. Each distinct code ran once.

### Code identity

| part | check | result |
|---|---|---|
| MATH-500 grader v2.5, the frozen block | syntax tree, docstrings removed, against the paper's grader | identical (Python 3.12.3) |
| MATH-500 grader v2.5 | the paper grader's own hand-labelled fixture: 308 cases, written before the grader (the fixture is not shipped) | 308 of 308 verdicts identical to the paper's grader on every field (bucket, strict, lenient, boxed-span count, extracted text, layer, rule, reason), and 308 of 308 equal the fixture's labels |
| MATH-500 per-cell driver `grade_texts` | read against the paper's per-cell driver | the same caching, re-check and per-sample bits (independent review) |
| Hi-ToM matcher | `grader_hash()`: sha256 over the source of its 13 matching functions and 4 constants | `bf137bfc` (the paper's) |
| Hi-ToM adapter `answer_text` and helpers | source against the paper's adapter | byte-identical |
| LiveCodeBench harness (`lcb_execution.py`) | against LiveCodeBench commit `28fef95` plus the paper's two fixes (the changes of LiveCodeBench pull request #103) | `testing_util.py` verbatim; the execution functions verbatim except the listed edits |
| LiveCodeBench `extract_code` | against the paper's four generators | v1 = Mistral 7B Instruct and Qwen3-0.6B, v2 = Qwen3-8B and gpt-oss-20b, verbatim |
| LiveCodeBench extraction per model, from the records | Can a recorded pass exist only under the variant assigned to the model? | Qwen3-8B: the two variants differ on 11,997 of 12,000 sampled completions, and 8,961 of them are recorded passes. gpt-oss-20b: they differ on 15,271 of 22,000, with 5,401 recorded passes. Only v2 can pass these. Mistral 7B Instruct: they differ on 88 completions, with 0 recorded passes, as v1 predicts. |
| Answer keys | rebuilt from the pinned public files | MATH-500: 500 of 500 answers equal the paper's key. Hi-ToM: 1,200 of 1,200 records (options, answer, factorial labels) equal the paper's key, with ids `s{sample_id}_o{question_order}_{prompting_type}` |
| LiveCodeBench test files | sha256 of the files the paper's evaluators downloaded, against the hub's files at revision `0fe84c39` | 6 of 6 equal |

### Not reproduced here, and why

- **LiveCodeBench in full.** The LiveCodeBench counts files hold about 307 million samples, and each one runs against
  all of its question's tests:
  - Mistral 7B Instruct 81,955,793;
  - gpt-oss-20b 63,258,750;
  - Qwen3-8B 119,228,250 (the release every figure but Figure 1 uses; see `docs/data.md`, section 10);
  - Qwen3-0.6B 42,823,700.

  That is far beyond a single CPU machine, so a random sample was executed instead. To grade your own LiveCodeBench
  samples at this scale, `--dedup` is the practical route (see above for what it changes).
- **LiveCodeBench, Qwen3-0.6B:** per-completion records for it were not available for the check.
- **LiveCodeBench, Qwen3-8B, per cell.** Its sampled cells were compared per sample only (the table above), not
  against its counts file.
- **One LiveCodeBench run stopped early.** A model-generated solution exceeded the 3 GB memory limit of the
  container the checks ran in; LiveCodeBench itself allows each solution 4 GiB. The 38 cells finished before the stop
  are included above.
- **Hi-ToM in full.** The four models' records hold 121 million samples, and reading and grading take 2-4 ms per
  sample on these long completions, so the check is a random sample.
- **Hi-ToM, Mistral 7B Instruct.** The data cells combine a first stage of 1,000 samples, whose records were not
  available, with a second stage of 500. Its second-stage samples were compared per sample only.
- **Included questions.** The paper filters on the fitted peak of P_q(tau), from the profile fit (`onetemp.profiles`
  and `onetemp.core`). The counts above use the measured peak c/n >= 0.001; for MATH-500 they coincide with the
  paper's.

### Environment of these checks

- **MATH-500:** Python 3.12.3, sympy 1.14.0, mpmath 1.3.0, pure-Python ground types. This is the environment of the
  paper's MATH-500 grading. It ran as 4 processes, each capped at 3 GB of address space.
- **Hi-ToM:** Python 3.12.3, standard library only.
- **LiveCodeBench:**
  - Python 3.10.19 and numpy 2.2.6; the paper's evaluators ran Python 3.10.12 and numpy 2.2.6.
  - 2 workers, no network, solutions run in a temporary directory.
