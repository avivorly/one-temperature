# data: the per-question counts behind every figure and table

`data/` holds, for every study in the paper, how many samples each question received at each sampling setting
and how many of them were correct. Nothing else is stored: no prompt, no completion, no reference answer and no
test case. The analysis code reads these files; the graders write files in the same format.

**Bottom line.** 21 counts files and 4 first-stage screens (25 files, 420,597 rows) in one plain format, one
manifest that says which file each figure and table uses, and one command that proves the files are intact
and hold counts only:

```bash
python -m onetemp.data.check        # integrity, schema, populations, consistency, counts only
python -m onetemp.data.fitfree      # the paper's numbers that need no profile fit, from the counts alone
```

Both run from the repository root (the data folder is found next to the `onetemp` package; set
`ONETEMP_DATA` to use another one). `onetemp.counts` and `onetemp.data.check` need only the Python standard
library (3.10 or later); `onetemp.data.fitfree` also needs numpy, which `pip install -e .` installs.

## 1. What is shipped

| path | what |
|---|---|
| `data/counts/<benchmark>/` | one counts file per study (section 3): every question the paper's analysis code read for it, in the order it read them |
| `data/selection/hitom/` | the four first-stage screens that chose the Hi-ToM questions (inputs of a selection step only) |
| `data/included_ids.json` | the included questions of each file (the paper's inclusion rule), in the file's order, for checking |
| `data/manifest.json` | per file: rows, questions, settings, n range, samples, correct answers, the included questions, sha256; every figure and table that uses it; pairs of files that must agree (section 6) |
| `data/reference/` | the paper's own runs as numbers, which `python -m onetemp.reproduce` compares against (below) |

`data/reference/` holds:

| file | contents | used for |
|---|---|---|
| `heatmaps.json` | cell labels and counts of the heatmaps, with tau_A, score_A and the top-row shares | Figures 17-28 and 42-44 |
| `beta_star.json` | beta* against D, after the LP sweep and after the exact recheck | Figures 4(a), 29-31 and 45 |
| `simulated_pools.json` | beta* of the three simulated hardness pools | Figure 33 |
| `fig4b_chunk_tallies.jsonl` | the 2,400 chunk tallies of the 24 million random weightings | Figure 4(b) |
| `figure_sha256.json` | sha256 of the paper's image files, by figure number and panel | `python -m onetemp.profiles repro` |

The code that reads and checks the files:

| code | what |
|---|---|
| `onetemp/counts.py` | the one counts reader: rows, validation, question-by-setting tables, and a command line (section 7) |
| `onetemp/data/check.py` | `python -m onetemp.data.check` (section 6) |
| `onetemp/data/fitfree.py` | `python -m onetemp.data.fitfree` (section 8) |
| `onetemp/data/convert.py` | `python -m onetemp.data.convert`, the converter that wrote `counts/` (section 9) |

## 2. The counts.jsonl schema

One JSON object per line, one line per (benchmark, model, question, setting):

```json
{"benchmark":"livecodebench","model":"mistralai/Mistral-7B-Instruct-v0.3","id":"1873_A","temperature":0.1,"min_p":null,"n":11502,"c":0}
```

| key | type | meaning |
|---|---|---|
| `benchmark` | string, required | `livecodebench`, `math500` or `hitom` |
| `model` | string, required | the Hugging Face repo id: `openai/gpt-oss-20b`, `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-0.6B` |
| `id` | string, required | the benchmark's own question id: LiveCodeBench `question_id` (`livecodebench/code_generation_lite`, for example `1873_A`, `2727`, `abc393_e`), MATH-500 `unique_id` (`HuggingFaceH4/MATH-500`, for example `test/precalculus/807.json`), Hi-ToM `s<sample_id>_o<question_order>_<prompting_type>` built from the fields of `Hi-ToM_data.json` (for example `s0_o0_CoTP`) |
| `temperature` | number >= 0, required | the sampling temperature; in a min_p sweep, its fixed temperature |
| `min_p` | number in [0, 1] or `null`, required | the min_p value of a min_p sweep (Appendix I); `null` in a temperature sweep. Inside a min_p sweep, `0.0` is plain sampling |
| `n` | integer >= 1, required | the number of samples (see section 5 for the LiveCodeBench `question_n` files) |
| `c` | integer, required | the number of correct samples, `0 <= c <= n` |
| `n_cell` | integer, optional | only in the LiveCodeBench `question_n` files, on the rows where the cell's own sample count differs from `n`: `c <= n_cell < n` |

The readers also accept four optional keys that the tools write, never present in the shipped files:

| key | type | meaning |
|---|---|---|
| `tau` | number | in a min_p row, the fixed temperature; it must equal `temperature` |
| `c_strict` | integer | written by the MATH-500 grader: the samples that pass its exact layer alone, `0 <= c_strict <= c` |
| `p_fit` | number in [0, 1] | written by `python -m onetemp.core.fit_all --out-rows`: the fitted success probability at the row's setting |
| `peak_pct` | number | the same: the question's fitted peak, in percent |

Conventions of the shipped files:

- The **setting** of a row is its temperature in a temperature sweep and its min_p value in a min_p sweep, so
  the same analysis code serves both.
- Pass rates are **instance-weighted**: every sample counts, including repeated completions.
- Every file is a **complete rectangle**: each question has each setting exactly once. Rows of one question are
  adjacent, settings ascending. Lines are compact JSON with `\n` endings.
- **Question order matters.** Each file keeps the order of the source file the paper's analysis code loaded.
  The 10-fold split, the seeded random draws and the tie breaks (for example the stable sort of questions that
  tie at the hardness clamp) follow that order, so reordering a file can change printed values. One exception,
  recorded per use in the manifest (`question_order`): the Appendix I comparisons, profiles, hardness, KL and
  protection analyses read the min_p and Hi-ToM questions with their ids sorted as strings, while the min_p
  ratio-to-uniform figures read the file order.
- The reader also opens `.jsonl.gz`. The strict mode of `python -m onetemp.data.check` allows only the
  required keys and `n_cell`, a benchmark from the list above, and a string `id`.

## 3. The studies

"Questions" is every question the paper's analysis code read for the study. "Included" is the subset its
inclusion rule keeps, which is the n the paper prints (for the LiveCodeBench files: `question_n` / `cell_n`;
section 5). Paths are under `data/counts/`; the numbers are those of `data/manifest.json`.

| benchmark | model | file | questions | included | settings | n per row | rows | samples |
|---|---|---|---:|---:|---|---|---:|---:|
| LiveCodeBench | GPT OSS 20B | `livecodebench/gpt-oss-20b__temperature__{question_n,cell_n}.jsonl` | 975 | 766 / 766 | 16 temperatures, 0.1 to 2.0 | 1,500 to 12,500 | 15,600 | 63,258,750 |
| LiveCodeBench | Mistral 7B Instruct | `livecodebench/Mistral-7B-Instruct-v0.3__temperature__{question_n,cell_n}.jsonl` | 975 | 359 / 359 | the same 16 | 1,500 to 14,834 | 15,600 | 81,955,793 |
| LiveCodeBench | Qwen3-8B | `livecodebench/Qwen3-8B__temperature__{question_n,cell_n}.jsonl` | 975 | 647 / 677 | 38 temperatures, 0.1 to 4.2 | 1,500 to 19,000 | 37,050 | 119,228,250 |
| LiveCodeBench | Qwen3-0.6B | `livecodebench/Qwen3-0.6B__temperature__{question_n,cell_n}.jsonl` | 110 | 85 / 86 | 20 temperatures, 0.1 to 2.0 | 9,996 to 127,496 | 2,200 | 42,823,700 |
| MATH-500 | GPT OSS 20B | `math500/gpt-oss-20b__temperature.jsonl` | 500 | 500 | 16 | 1,500 | 8,000 | 12,000,000 |
| MATH-500 | Mistral 7B Instruct | `math500/Mistral-7B-Instruct-v0.3__temperature.jsonl` | 500 | 455 | 16 | 1,500 | 8,000 | 12,000,000 |
| MATH-500 | Qwen3-8B | `math500/Qwen3-8B__temperature.jsonl` | 500 | 496 | 20 | 750 | 10,000 | 7,500,000 |
| MATH-500 | Qwen3-0.6B | `math500/Qwen3-0.6B__temperature.jsonl` | 500 | 492 | 20 | 1,500 | 10,000 | 15,000,000 |
| Hi-ToM | GPT OSS 20B | `hitom/gpt-oss-20b__temperature.jsonl` | 678 | 678 | 31 temperatures, 0.075 to 4.0 | 1,500 | 21,018 | 31,527,000 |
| Hi-ToM | Mistral 7B Instruct | `hitom/Mistral-7B-Instruct-v0.3__temperature.jsonl` | 617 | 615 | 31 | 1,500 | 19,127 | 28,690,500 |
| Hi-ToM | Qwen3-8B | `hitom/Qwen3-8B__temperature.jsonl` | 471 | 466 | 31 | 1,500 | 14,601 | 21,901,500 |
| Hi-ToM | Qwen3-0.6B | `hitom/Qwen3-0.6B__temperature.jsonl` | 743 | 743 | 31 | 1,500 | 23,033 | 34,549,500 |
| Hi-ToM | Qwen3-0.6B | `hitom/Qwen3-0.6B__temperature__topup1000.jsonl` | 743 | 642 | 31 | 1,000 (second stage only) | 23,033 | 23,033,000 |
| Hi-ToM | Qwen3-0.6B | `hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl` | 648 | 648 | 10 min_p values at tau = 1.5 | 500 (100 screen + 400) | 6,480 | 3,240,000 |
| Hi-ToM | Qwen3-0.6B | `hitom/Qwen3-0.6B__min_p_tau1.5__stage2_nu400.jsonl` | 648 | 648 | the same 10 | 400 (second stage only) | 6,480 | 2,592,000 |
| Hi-ToM | Qwen3-0.6B | `hitom/Qwen3-0.6B__min_p_tau2.0__nu400.jsonl` | 648 | 648 | 5 min_p values at tau = 2.0 | 400 | 3,240 | 1,296,000 |
| Hi-ToM | Mistral 7B Instruct | `hitom/Mistral-7B-Instruct-v0.3__min_p_tau2.0__nu1500.jsonl` | 617 | 598 | the same 5 | 1,500 | 3,085 | 4,627,500 |

The first-stage screens (`data/selection/hitom/`, 500 samples per cell, and 100 per min_p value for the last):

| file | questions | kept by the screen | settings | rows |
|---|---:|---:|---|---:|
| `gpt-oss-20b__temperature__screen500.jsonl` | 1,200 | 678 | 31 temperatures | 37,200 |
| `Qwen3-8B__temperature__screen500.jsonl` | 1,200 | 471 | 31 temperatures | 37,200 |
| `Qwen3-0.6B__temperature__screen500.jsonl` | 1,200 | 743 | 31 temperatures | 37,200 |
| `Qwen3-0.6B__min_p_tau1.5__screen100.jsonl` | 1,200 | 648 | the 10 min_p values at tau = 1.5 | 12,000 |

The grids:

- LiveCodeBench Qwen3-8B: 0.1 to 3.5 in steps of 0.1, then 3.7, 4.0, 4.2.
- LiveCodeBench Qwen3-0.6B, and MATH-500 Qwen3-8B and Qwen3-0.6B: 0.1 to 2.0 in steps of 0.1.
- GPT OSS 20B and Mistral 7B Instruct on LiveCodeBench and MATH-500: 0.1 to 1.2 in steps of 0.1, then 1.4,
  1.5, 1.7, 2.0.
- Hi-ToM: 0.075 to 1.95 in steps of 0.075, then 2.0, 2.2, 2.5, 3.0, 4.0.
- min_p at tau = 1.5: 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7. At tau = 2.0: 0.0056, 0.0128,
  0.0316, 0.0721, 0.178.

## 4. Populations and inclusion rules

The analyses apply the inclusion rule themselves; `data/included_ids.json` lists the result for checking.

- **Temperature sweeps.** A question is included if the degree-10 Bernstein profile fitted to its counts peaks
  at a pass rate of at least 0.1 % (the paper's 1e-3 filter). This needs the profile fit
  (`python -m onetemp.core.fit_all`, see `docs/core.md`); the lists in `included_ids.json` are the output of the
  paper's own fitting code on these files, and `python -m onetemp.reproduce tables` recomputes the
  included counts of all 12 sweeps with this package's fit and compares them with the paper's.
- **min_p sweeps.** The same floor on measured pass rates (no fit): the best pass rate over the grid must be at
  least 0.1 %. For Mistral 7B Instruct this leaves 598 of 617; 19 fall below it, 8 of them with no correct
  sample.
- **Hi-ToM populations** are the questions kept by a first-stage screen: best single-setting pass rate below
  0.30 (Appendix C: 500 samples per cell, 1,000 for Mistral 7B Instruct; Appendix I: 100 samples per min_p
  value at tau = 1.5). The screens of GPT OSS 20B, Qwen3-8B and Qwen3-0.6B and the min_p screen are in
  `selection/`; for those, the rule applied to the screen gives exactly the questions of the study files:

  ```bash
  python -m onetemp.sampling.screen --rule hitom --counts data/selection/hitom/Qwen3-0.6B__temperature__screen500.jsonl \
      --model Qwen/Qwen3-0.6B --out kept.json
  ```

  (`--rule minp` for the min_p screen.)
- **The Appendix I comparison with the best plain temperature** (Qwen3-0.6B) uses the 642 questions in both
  the temperature file and the min_p files, sorted by id, and scores the temperature sweep on its 1,000
  second-stage samples (`topup1000`) and the min_p sweep on its 400 (`stage2_nu400`).
- **Mistral 7B Instruct on Hi-ToM:** the temperature analyses use the 615 included questions; the min_p
  comparison of Appendix I uses all 617 kept ones, with both files.
- The Hi-ToM Qwen3-0.6B file `Qwen3-0.6B__temperature.jsonl` equals the screen (500 samples) plus the
  top-up (1,000) cell by cell, and `topup1000` is the top-up part. Likewise `min_p_tau1.5__nu500` equals the
  screen (100) plus the second stage (400). `python -m onetemp.data.check` tests that the top-up, the second
  stage and the screens are parts of the larger files' draws (section 6).

## 5. Two parses on LiveCodeBench

Some LiveCodeBench questions have different sample counts at different temperatures: 4 of the 975 for
Mistral 7B Instruct, 20 for GPT OSS 20B, 268 for Qwen3-8B and 30 of the 110 for Qwen3-0.6B (50, 297, 5,197
and 237 rows). The pass-count files store, per question, one question-level `n` (its largest per-temperature
count) and, per temperature, that temperature's own `n`. The paper used both readings, each as one file:

| parse | `n` of a row | used by | included (GPT OSS / Mistral / Qwen3-8B / Qwen3-0.6B) |
|---|---|---|---|
| `question_n` | the question-level n at every temperature; `n_cell` keeps the cell's own count where it differs | Figures 1, 2, 3 and 4(b); Appendices E, F (the ratio-to-uniform figures, the summary table, Tables 2-6) and G | 766 / 359 / 647 / 85 |
| `cell_n` | each temperature's own n | the "corrected parsing" of Appendix H and Figure 4(a) | 766 / 359 / 677 / 86 |

On MATH-500 and Hi-ToM every cell has the question's `n`, so there is one file per model. To reproduce a
figure exactly, use the parse it used. The `question_n` files carry `n_cell`, so both parses can be rebuilt
from them (`CountsTable.cell_n(i, j)` returns the drawn count). In `docs/core.md`, `--n-mode question-max`
reads a `question_n` file as the paper did, and `--n-mode per-setting` gives each temperature its own n.

The two parse files of a LiveCodeBench model are two readings of one study, and the Hi-ToM Qwen3-0.6B
`topup1000` file is a part of the draws of `hitom/Qwen3-0.6B__temperature.jsonl`. Give an analysis one of
them, never both together.

## 6. Verify the files

```bash
python -m onetemp.data.check                 # data/manifest.json of the repository
python -m onetemp.data.check --manifest PATH
python -m onetemp.data.check --math500-test test.jsonl --hitom-data Hi-ToM_data.json --lcb-meta lcb_ids.json
```

Every check must pass; the exit code is 1 otherwise. The output starts with the totals (`files: 25 rows:
420,597`, the number of figure and table uses and of consistency pairs in the manifest), prints the known
difference with its reason, and ends with `checks passed: N failed: 0`. The checks, file by file:

1. **Integrity.** The file exists; its sha256, byte size, row count, question count, settings, sample and
   correct totals, n range and number of rows with `n_cell` equal the manifest.
2. **Schema.** Every row passes `onetemp.counts.validate_row` in strict mode; one benchmark and one model per
   file, equal to the manifest; every question has every setting exactly once (a complete rectangle); no
   question is split across non-adjacent lines.
3. **Populations.** Every file holds every question the paper's analysis code read for its study. The included
   questions are listed in `included_ids.json`, in the file's order, with the count and sha256 recorded in the
   manifest. Every figure or table use points at an existing file, and the population it uses (all questions,
   or the included ones) has the size the paper prints (`n_printed`). The one known difference, the Appendix C
   count of an earlier Qwen3-8B data release, is printed with its reason (section 10).
4. **Consistency.** The manifest lists 10 pairs of files that must agree. Files that hold the same draws (the
   two parses of a LiveCodeBench study) agree on every shared cell: the same `c` and the same number of samples
   actually drawn. Where one file holds a part of the other's draws (the 400 second-stage samples inside the
   500 of screen plus second stage; the top-up inside the Hi-ToM Qwen3-0.6B temperature file; each screen
   inside the study file it selected), every shared cell satisfies `c_a <= c_b`, `n_a < n_b` and
   `c_b - c_a <= n_b - n_a`, and the shared questions and cells are the numbers the manifest states.
5. **Counts only** (the proof that no gold answer, test case or completion is in the export):
   - *Structure.* A row has only the keys of the schema. Its only strings are `benchmark` (3 values), `model`
     (4 Hugging Face ids) and `id`, and every id matches its benchmark's identifier pattern, so no field can
     carry free text.
   - *Raw scan.* No file contains a newline inside a JSON string, a backslash, or any marker of prompts,
     completions and code (`boxed`, `def `, `return`, `import `, `Story`, `answer`, `prompt` and others).
   - *Public data (optional flags).* Every id is a public id of its benchmark, and no reference answer of
     MATH-500 or Hi-ToM occurs in any text file of `data/`, neither as a string value nor as a substring of the
     raw text. Answers made only of digits, signs and dots are skipped in the raw scan, because they occur
     inside numbers such as temperatures; numeric answers can only coincide with the numeric fields, which hold
     settings and counts.

The three optional flags take local copies of the public data:

- `--math500-test`: `test.jsonl`, the test split of `HuggingFaceH4/MATH-500`.
- `--hitom-data`: `Hi-ToM_data.json` from the Hi-ToM dataset repository (`ying-hui-he/Hi-ToM_dataset` on
  GitHub, `Hi-ToM_data/Hi-ToM_data.json`).
- `--lcb-meta`: any JSON list of objects with a `question_id` field from `livecodebench/code_generation_lite`
  (only the ids are read), for example the list written by
  `json.dump([{"question_id": q} for q in ds["question_id"]], open("lcb_ids.json", "w"))` after loading the
  dataset with the `datasets` library.

When the release was prepared, with the three flags the scan covered the 500 MATH-500 and 1,200 Hi-ToM
reference answers (331 distinct; 193 of them at least 3 characters long and not made only of digits, signs and
dots) and found them nowhere: 0 string values equal one, and 0 occurrences in the raw text of the text files
of `data/`. Run the command to repeat it.

### The manifest

`data/manifest.json` has these parts:

| key | contents |
|---|---|
| `format` | the schema in words: the keys, the setting, the weighting, the populations, the question order |
| `studies` | the 25 files with their question counts and included counts |
| `files` | per file: `path`, `benchmark`, `model`, `axis` (`temperature` or `min_p`), `fixed` (`{}` or `{"temperature": 1.5}`), `rows`, `questions`, `settings`, `settings_count`, `n_min`, `n_max`, `rows_with_n_cell`, `samples_drawn`, `correct`, `included` (`n`, the `rule` in words, `ids_sha256` of the list in `included_ids.json`), `question_order`, `source_sha256` (of the pass-count file it was converted from), `bytes`, `sha256`, `study`, `parse` (`question_n`, `cell_n`, or `both` where they coincide), `samples` |
| `uses` | one entry per figure or table use: `label`, `file`, `population` (`all` or `included`), `n_printed` (the n the paper prints, or null), `question_order` (`file` or `sorted ids`), and sometimes a `note` and an `expected_difference` |
| `labels` | the caption of each label |
| `consistency` | the pairs of files that must agree: `a`, `b`, `relation`, `samples`, `shared_questions`, `shared_cells` |

The hashes and sizes are in the file itself; read them there.

## 7. Read the files

```bash
python -m onetemp.counts summary data/counts/*/*.jsonl       # validate every file, print its shape and totals
python -m onetemp.counts table data/counts/math500/Qwen3-8B__temperature.jsonl --csv qwen3_8b_math500.csv
python -m onetemp.counts list-figures                        # every label, with its files and printed n
python -m onetemp.counts figure fig:TempsCom                 # one label: its files, population and printed n
```

- `summary` validates each file and prints its benchmark, model and axis, rows, questions, settings, the n
  range, the totals of samples and correct answers, and the number of rows with `n_cell`.
- `table` writes a wide CSV: `id`, then `n@<setting>` for each setting, then `c@<setting>`; `--csv` names the
  output (default: standard output).
- `figure` and `list-figures` read the manifest (`--manifest PATH` to use another).

In Python:

```python
from onetemp.counts import read_counts, table
t = table(read_counts("data/counts/hitom/Qwen3-0.6B__min_p_tau2.0__nu400.jsonl"))
t.axis, t.fixed, t.settings      # 'min_p', {'temperature': 2.0}, [0.0056, 0.0128, 0.0316, 0.0721, 0.178]
t.ids[0], t.n[0], t.c[0]         # the first question's counts at each setting
```

`read_counts`, `iter_counts`, `validate_row` and `table` are the package's one reader; `read_figure(label)`
returns the rows of every file a label uses. The analysis code builds its matrices from the same rows
(`python -m onetemp.core.counts_io --counts FILE --list` lists the groups of a file).

## 8. Which file each figure and table uses

`python -m onetemp.counts list-figures` prints every label with its files; `figure LABEL` explains one.
Figure and table numbers are those of the camera-ready build; tables without a LaTeX label are named by their
section's label plus a suffix, for example `zap:Multiple:Comp:res#summary-table`.

| paper | labels | files |
|---|---|---|
| Figure 1 | `fig:ManyQ` | `livecodebench/Qwen3-8B__temperature__question_n.jsonl` (see section 10) |
| Figure 2 | `fig:Seperation` | `livecodebench/Qwen3-8B__temperature__question_n.jsonl` |
| Figure 3 | `fig:TempsCom` | `livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl` |
| Figure 4 | `fig:Prot:D:Main` | (a) the Mistral `cell_n` file, (b) the Mistral `question_n` file |
| Appendix C tables | `zap:DataDis:gpt_oss_20b`, `zap:DataDis:mistral_7b_instruct`, `zap:DataDis:qwen3_8b`, `zap:DataDis:qwen3_0_6b` | the 12 temperature-sweep files (LiveCodeBench: `question_n`) |
| first-stage screens (Appendix C, Appendix I) | `zap:Datasets#hitom-screen`, `zap:MinP#screen` | `selection/hitom/*` |
| Figures 5-7, 8-10, 11-13 | `fig:AveTemp:*`, `fig:RangeSize:*`, `fig:relC:*` | the 12 temperature-sweep files, LiveCodeBench `question_n` |
| Figures 14-16, summary table (F.2), Tables 2-6 | `fig:rtu:*`, `zap:Multiple:Comp:res#summary-table`, `tab:passk_*` | the 12 temperature-sweep files, LiveCodeBench `question_n` (Tables 2-6: LiveCodeBench only) |
| Figures 17-28, Appendix G table | `fig:RandomH:KGrid*`, `zap:DataDis:RandomH:res#table` | the 12 temperature-sweep files, LiveCodeBench `question_n` (the table: LiveCodeBench only) |
| Figures 29-31, Appendix H text | `fig:Prot:D:LCB`, `fig:Prot:D:MATH`, `fig:Prot:D:HiToM`, `zap:DataDis:Prot#text` | the 12 temperature-sweep files, LiveCodeBench `cell_n` |
| Figure 33 | `fig:Prot:Sim` | `livecodebench/Mistral-7B-Instruct-v0.3__temperature__cell_n.jsonl` (Figure 32 has no data) |
| Figure 34 and the tau = 1.5 table | `fig:rtu:minp`, `zap:MinP#tau1.5-table` | `hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl` |
| Figure 35 and the tau = 2.0 against 1.5 table | `fig:minp:T2`, `zap:MinP#tau2-vs-tau1.5-table` | `hitom/Qwen3-0.6B__min_p_tau2.0__nu400.jsonl`, `hitom/Qwen3-0.6B__min_p_tau1.5__stage2_nu400.jsonl` |
| comparison with the best plain temperature | `zap:MinP#plain-T-table` | `hitom/Qwen3-0.6B__temperature__topup1000.jsonl` and the two files above |
| Figure 36 and the Mistral table | `fig:minp:mistral`, `zap:MinP#mistral-table` | `hitom/Mistral-7B-Instruct-v0.3__min_p_tau2.0__nu1500.jsonl`, `hitom/Mistral-7B-Instruct-v0.3__temperature.jsonl` |
| Figure 37 | `fig:minp:rtuT2` | `hitom/Qwen3-0.6B__min_p_tau2.0__nu400.jsonl`, the Mistral min_p file |
| Figures 38-41, 42-44, 45 | `fig:minp:profiles`, `fig:minp:Peak`, `fig:minp:RangeSize`, `fig:minp:relC`, `fig:minp:RandomH:q15` / `q20` / `mis`, `fig:minp:Prot` | the three min_p files `nu500`, `nu400` and `nu1500` (the included questions; each heatmap uses its own file) |
| Appendix I setup table and split-half checks | `zap:MinP#setup-table`, `zap:MinP#noise-checks` | the same three min_p files (all questions) |

## 9. The fit-free reproduction

```bash
python -m onetemp.data.fitfree                  # one line per number: paper | ours | difference | where
python -m onetemp.data.fitfree --json OUT.json  # also write every comparison to a JSON file
```

`fitfree` recomputes, from the counts alone and without a profile fit, the paper's numbers below. "Equal"
means within half a unit of the paper's last printed decimal (percentages and percentage points to 2
decimals, ratios to 3). Each line starts with `ok`, `OFF` (not equal) or `info` (shown for scale or as a known
difference, not a reproduction claim); the exit code is 1 if any line is `OFF`. The last line counts the
numbers reproduced. `python -m onetemp.reproduce data` runs this together with `onetemp.data.check`
(about a minute).

Pass@K is the unbiased estimator 1 - C(n-c, K) / C(n, K) (`onetemp.core.passk.passk_unbiased`) where the paper
uses it (Appendix I outside the ratio-to-uniform results), and the measured-rate pass@K 1 - (1 - c/n)^K, with
the uniform scheme spreading K equally over the grid, where the paper says it computes ratios from measured
pass rates. Paired standard errors are the sample standard deviation of the per-question differences over the
square root of the number of questions. Rates are instance-weighted.

| group | what is recomputed |
|---|---|
| Appendix C | the questions in the study (975 LiveCodeBench, 110 for Qwen3-0.6B, 500 MATH-500), the included questions, the temperatures in the grid, the samples per (question, temperature) (median and maximum where they differ), the grids themselves |
| Appendix G | the 119,228,250 samples of the later Qwen3-8B release |
| Appendix H | the included questions under the `cell_n` parse; the profiles the corrected parse changes (included under both parses, with a temperature of uneven n inside the fit bracket); the questions it adds or loses |
| Appendix I setup | the kept questions (648 / 648 / 617), nu (500 / 400 / 1,500) and the grids |
| Appendix I, tau = 1.5 | the ratio-to-uniform table (single values and the uniform row; the leave-one-out row needs the allocation optimiser and is checked by `python -m onetemp.core.paper_check`), pass@1 at min_p 0 and at the best value, the gains with their paired SEs |
| Appendix I, tau = 2.0 against 1.5 | the best value at each K, its pass@K, the gap and its paired SE, at K = 1, 4, 16, 64, 256, 400 |
| Appendix I, best plain temperature | the 642 questions both screens kept, the best plain temperature at each K, the gaps at tau = 2.0 and 1.5 with their paired SEs |
| Appendix I, Mistral | the best min_p value against the best plain temperature at K = 1, 4, 16, 64, 256, 1024, 1500; the relative gains |
| Appendix I, figures | the ratio-to-uniform statements at tau = 2.0, the peak counts of the profile figure, the inclusion floor of Mistral 7B Instruct (19 questions below it, 8 of them with no correct sample), the hardness bins of the coverage figure, the OLS slopes of the peak position on log10 hardness |
| for scale | the measured pass@1 of the included questions against the fitted pass@1 printed in Tables 3 and 4; they differ by design, because the tables use the fitted profiles |

Examples (paper | ours):

| where | what | paper | ours |
|---|---|---|---|
| Appendix C, Mistral 7B Instruct, LiveCodeBench | included questions | 359 | 359 |
| Appendix C, Qwen3-0.6B, Hi-ToM | included questions | 743 | 743 |
| Appendix G, Qwen3-8B | samples of the later release | 119,228,250 | 119,228,250 |
| Appendix H, Qwen3-8B | profiles changed by the corrected parse | 197 | 197 |
| Appendix H, Qwen3-8B | included questions added by the corrected parse | 30 | 30 |
| Appendix I, tau = 1.5 | uniform pass@1, % | 7.32 | 7.32098 |
| Appendix I, tau = 1.5 | pass@1 at min_p 0, % | 6.62 | 6.61944 |
| Appendix I, tau = 1.5 | gain of the best value (0.01) over min_p 0, pass@1, pp | 3.11 | 3.11049 |
| Appendix I, tau = 2.0 against 1.5 | gap at K = 1, pp | 0.71 | 0.706019 |
| Appendix I, best plain temperature | questions both stages kept | 642 | 642 |
| Appendix I, best plain temperature | best plain tau at K = 1 | 1.275 | 1.275 |
| Appendix I, Mistral | gap at K = 1, pp | 1.16 | 1.16175 |
| Appendix I, Mistral | gap at K = 1024, pp | -0.88 | -0.879405 |
| Appendix I, Mistral at tau = 2.0 | ratio of 0.0056 at K = 1 | 1.057 | 1.0571 |
| Appendix I, Mistral at tau = 2.0 | questions below the 0.1 % floor, of them with no correct sample | 19, 8 | 19, 8 |
| Appendix I, profiles | tau = 2.0, Qwen3-0.6B: questions peaking at 0.178 | 133 | 133 |
| Appendix I, peak text | Qwen3-0.6B, tau = 1.5: OLS slope of the peak position on log10 H | -3.1 | -3.10052 |

Run the command for the full list and the count of numbers reproduced.

## 10. What is not shipped, and why

- **Per-sample data:** prompts, completions, reference answers, LiveCodeBench test cases, output lengths and
  truncation counts. The files hold counts only, which is all that the figures, tables and printed numbers
  need. Reference answers, test cases and prompts belong to the benchmarks' public datasets, which the
  graders and the prompt builders download; they are not redistributed here, and
  `python -m onetemp.data.check` proves that none is present. Completions are large and are not needed to
  recompute any number.
- **Question metadata** (LiveCodeBench platform, difficulty and release; MATH-500 subject and level; Hi-ToM
  order, deception, story length and prompting type): join on `id` with the public datasets.
- **The earlier Qwen3-8B LiveCodeBench release** (100,568,250 samples). Figure 1 and the Appendix C count
  of 646 included questions come from that release, not shipped; the redraw differs in the grey background.
  The files here hold the later release (119,228,250 samples, 647 included questions on the `question_n`
  parse) that every other figure uses, as Appendix G states.
- **The first stage of Mistral 7B Instruct on Hi-ToM** (1,000 samples per cell on all 1,200 questions); its
  kept questions are the 617 of the Mistral files.
- **The first-stage screens of the LiveCodeBench studies** and the pass-count files the files were converted
  from.

## 11. How the files were made

`python -m onetemp.data.convert` converted, without any other change, the pass-count files that the paper's
analysis code read (the profile fit and the allocation comparison load them directly); the manifest records
each source's sha256 (`source_sha256`). The LiveCodeBench ids were mapped from the source files' question index
(the question's position in the list of 975) to the public `question_id`; every mapped title equals the
dataset's title for that id. The graders (`python -m onetemp.grading.math500`, `.hitom`, `.livecodebench`)
write files in the same format.

```bash
python -m onetemp.data.convert --input FILE --benchmark livecodebench \
    --model mistralai/Mistral-7B-Instruct-v0.3 --parse question_n \
    [--id-from task_id|question_idx] [--id-map MAP.json] [--keep IDS.txt] --out OUT.jsonl
```

- The input is a JSON list with one record per question: `{"task_id", "question_idx", "version", "n",
  "results": [{"temperature", "n", "passed"}, ...]}`; for a min_p sweep every result also has `min_p`, next
  to the fixed `temperature`. Only `task_id` or `question_idx`, `n` and the results' `temperature`, `min_p`,
  `n` and `passed` are read.
- `--parse question_n` gives every temperature the question-level `n` (and `n_cell` where the cell's own count
  differs); `--parse cell_n` gives each temperature its own `n`. On MATH-500 and Hi-ToM they give the same file.
- `--id-map` is a JSON object from the source id to the public question id; `--keep` is a text file with one
  public id per line (only those questions are written, in the input's order). `--out` must not exist.
- The converter refuses what the analysis would silently misread: a duplicate question id, a duplicate
  setting inside a question, a result without `n` or `passed`, `passed` outside `0..n`, a min_p sweep whose
  fixed temperature varies, a `--keep` id that is not in the input.

The paper's own profile-fitting code, run on the original pass-count files and on the same data rebuilt from
these counts files, gave bit-identical fitted peaks and log(1 - P) matrices, and exactly the included
questions the paper prints (table in section 3). `python -m onetemp.reproduce tables` repeats the included
counts and the printed numbers with this package's own fit; `docs/core.md` has the precision.
