# core: the paper's method, and how closely it reproduces

`onetemp.core` computes, from per-question sample counts, every quantity of the strategy comparison in
"One Temperature to Rule Them All?". Run every command from the repository root.

**Bottom line.** Every number of the paper that the core checks matches at its printed precision:
Figure 3, all pass@K tables (Tables 2 to 6), the whole summary table of Appendix F.2, the included-question
counts of Figures 14 to 16, and the Appendix I min_p table. One command repeats the quick check, another
the full one:

```bash
python -m onetemp.reproduce fig3      # Figure 3, 10-fold and leave-one-out, 20 values of K (about 1 minute)
python -m onetemp.reproduce tables    # Figure 3, Tables 2-6, summary table, Appendix I table (about 15 minutes)
```

## 1. What the core computes

| step | what | module | paper |
|---|---|---|---|
| profile fit | a smooth, non-negative, unimodal degree-10 Bernstein profile P(x) per question, fitted to the counts by weighted least squares with Clopper-Pearson weights; the peak; the inclusion rule (peak >= 0.1 percent) | `fit_all` | Appendix D |
| pass@K | the unbiased estimator 1 - C(n-c, K) / C(n, K) and the plug-in 1 - (1-P)^K | `passk` | Section 2.2 |
| allocation | the optimal split of K samples over the settings (Eq. 9): the convex relaxation and the integer allocation | `allocation` | Section 4.1, Appendix F |
| comparison | the best single setting, the uniform scheme, the ratio to uniform | `heldout`, `bootstrap` | Section 4.2 |
| held-out evaluation | leave-one-out and 10-fold, with the paper's fold rule | `heldout` | Appendix F |
| bootstrap | paired bootstrap bands on the ratio to uniform | `bootstrap` | Appendix F |
| tables | the pass@K tables (Tables 2-6) and the summary table | `tables` | Appendix F.2 |
| figure | the ratio-to-uniform figure (Figure 3, and Figures 14-16) | `plot_rtu` | Appendix F |
| pipeline | all of the above for one dataset, in one command | `pipeline` | |

The setting axis is generic: the temperature, or min_p at a fixed temperature (Appendix I). Pass
`--axis min_p --fixed-temperature TAU` for the second.

## 2. Install

```bash
pip install -e .            # numpy, scipy, matplotlib; Python 3.10 or later
pip install -e ".[dev]"     # also the test suite: python -m pytest
```

**Optional: the C kernel.** The profile fit runs 50,000 gradient steps per question. The paper's kernel
is C (`onetemp/core/bernstein_kernel.c`, unchanged, sha256
`064cac9ecaf0fc86fd2911d150165645cbf28f635ef4d6473b63e704e7cfe1f2`). Compile it once with any C compiler
(`gcc` by default, or the one named in the `CC` environment variable):

```bash
python -m onetemp.core.fit_all --build-kernel                   # portable build, the default
python -m onetemp.core.fit_all --build-kernel --build-flags paper   # the paper's own -march=native build
```

The library is written to `onetemp/core/_bernstein_kernel.so` (`.dylib` on macOS, `.dll` on Windows). It is
searched in this order: `--kernel PATH`, the environment variable `ONETEMP_BERNSTEIN_KERNEL`, the default
location. With `--backend auto` (the default) the fit uses the kernel when it finds one. `--backend c`
requires it, `--backend numpy` ignores it.

**Without a compiler, nothing is lost but time.** The numpy fallback repeats the C loops operation by
operation and gives the same coefficients as the portable C build, bit for bit (section 8). It takes about
60 s instead of about 4 s for 975 questions on one core.

## 3. Input: counts.jsonl

One JSON object per line, for each benchmark, model, question and setting:

```json
{"benchmark": "livecodebench", "model": "mistralai/Mistral-7B-Instruct-v0.3", "id": "1873_A",
 "temperature": 0.7, "min_p": null, "n": 11502, "c": 18}
```

`n` is the number of samples and `c` the number of correct ones. In a min_p sweep, `min_p` is set and
`temperature` is the fixed temperature. The full schema, and the files that ship, are in
`docs/data.md`. The counts of the paper are in `data/counts/`.

- Question ids must be unique within a benchmark and model.
- **Question order matters.** The 10-fold split permutes the included questions in file order, so a file in
  the paper's order gives the paper's folds. The shipped files keep that order.
- `onetemp.core.counts_io` builds the question-by-setting matrices from the one counts reader
  (`onetemp/counts.py`). `python -m onetemp.core.counts_io --counts FILE --list` lists the
  (benchmark, model, axis) groups of a file; `--summary` (with `--benchmark`, `--model`, `--axis`,
  `--fixed-temperature`, `--fixed-min-p`) prints the shape and totals of one group.

## 4. Commands

```bash
# everything for one model and benchmark: fit, leave-one-out and 10-fold, bootstrap, tables, figure
python -m onetemp.core.pipeline \
    --counts data/counts/livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl \
    --folds loo 10 --plot --name "Mistral 7B Instruct" --out-dir results/fig3

# check the paper's numbers (Figure 3, Tables 2-6, the summary table, the Appendix I table) for every
# study found in the given files; give one file per study
python -m onetemp.core.paper_check --counts FILE1 FILE2 ... --out results/paper_check.json

# min_p at a fixed temperature (Appendix I: measured rates, K up to 384, the paper's seed)
python -m onetemp.core.pipeline \
    --counts data/counts/hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl --benchmark hitom \
    --model Qwen/Qwen3-0.6B --axis min_p --fixed-temperature 1.5 --mode meas --max-k 384 \
    --seed-scheme spawn --seed 33273348774934605029837111157444119385 --out-dir results/minp_tau1.5
```

Each step also runs on its own:

```bash
python -m onetemp.core.fit_all   --counts COUNTS.jsonl --benchmark B --model M --out profiles.json \
                                 --out-rows profile_rows.jsonl   # each counts row + p_fit and peak_pct
python -m onetemp.core.heldout   --profiles profiles.json --folds loo --out heldout_loo.json
python -m onetemp.core.heldout   --profiles profiles.json --folds 10  --out heldout_kf10.json
python -m onetemp.core.bootstrap --profiles profiles.json --B 10000 --out rtu.json
python -m onetemp.core.tables    --profiles profiles.json --heldout heldout_loo.json --rtu rtu.json \
                                 --name "Mistral 7B Instruct" --out-prefix tables/livecodebench_mistral
python -m onetemp.core.plot_rtu  --rtu rtu.json --heldout heldout_kf10.json --out figure3.png
python -m onetemp.core.allocation --profiles profiles.json --k 2 16 256 1024 --integer exact
python -m onetemp.core.passk     --counts COUNTS.jsonl --benchmark B --model M --k 1 8 64   # unbiased estimator
```

Every command has `--help`. The options that matter most:

| option | where | meaning |
|---|---|---|
| `--n-mode question-max` (default) or `per-setting` | `fit_all`, `pipeline` | the sample count behind each rate (section 7, item 1) |
| `--mode fit` (default) or `meas` | `heldout`, `bootstrap`, `tables`, `allocation`, `pipeline` | fitted profiles (the paper) or measured rates c/n |
| `--min-peak` | the same | inclusion threshold in percent (default 0.1) |
| `--max-k` | `heldout`, `bootstrap`, `pipeline` | largest K of the paper's grid (default 1024; the grid also has 1536 and 2048) |
| `--folds loo` or a number | `heldout`; `--folds loo 10` in `pipeline` | the held-out split; `--fold-seed` (default 0) seeds the k-fold permutation |
| `--integer paper` (default) or `exact` | `heldout`, `pipeline`, `paper_check`; `allocation` defaults to `exact` and also takes `none` | the integer step (section 7, item 7) |
| `--relaxed slsqp` or `newton` | `allocation` | the relaxation solver |
| `--B`, `--seed`, `--seed-scheme plain\|spawn\|entropy` | `bootstrap`, `pipeline` | the bootstrap (section 7, item 8); `--method fast\|paper` in `bootstrap` |
| `--backend auto\|c\|numpy`, `--kernel PATH` | `fit_all`, `pipeline`, `paper_check` | the fit kernel |
| `--axis temperature\|min_p`, `--fixed-temperature` | `fit_all`, `pipeline`, `passk`, `counts_io` | the setting axis |

## 5. Modules

All are in `onetemp/core/`.

| module | what it computes |
|---|---|
| `counts_io.py` | Reads counts.jsonl into per-question count matrices on a setting axis. |
| `fit_all.py` | The profile fit of Appendix D, then each fitted P(x) on the allocation grid, its peak, and the inclusion rule (peak >= 1e-3). It has the paper's functions, a C kernel loader and builder, and the numpy fallback. |
| `bernstein_kernel.c` | The paper's projected-gradient kernel. |
| `passk.py` | The shared pass@K functions: `passk_unbiased`, `passk_plugin`, `log1m`, `unbiased_log_fail` (below), the single-setting, allocation and uniform scores, and the paper's K grids. |
| `allocation.py` | `solve_relaxed` is the paper's SLSQP relaxation (`method="newton"` gives an exactly feasible alternative); `kkt_report` gives its KKT residuals and first-order gap bound. `integer_paper` is the paper's integer step, ported unchanged. `integer_exact` searches the whole integer simplex by branch and bound and certifies the optimum to a tolerance (default 1e-9 expected unsolved questions). |
| `heldout.py` | The paper's held-out comparison: `make_folds` (the fold rule) and `compare`, which covers the optimal allocation and the best single setting, each in-sample and held-out, plus the uniform scheme and the delta-method standard errors. |
| `bootstrap.py` | The paired bootstrap of the ratio to uniform, in two forms that use the same draws: the paper's loop, and the count-then-matmul identity. |
| `tables.py` | The pass@K tables as LaTeX in the paper's layout and bold rule, CSV, Markdown and JSON, and the summary-table rows. |
| `plot_rtu.py` | The ratio-to-uniform figure in the paper's style. |
| `pipeline.py` | All of the above for one dataset, in one command. |
| `paper_check.py` | Compares the results with `paper_numbers.json`, the printed numbers of the paper. |

### The pass@K estimators (`passk.py`)

Two estimators of a question's pass@K at one setting:

- `passk_unbiased(n, c, K)` is the unbiased estimator 1 - C(n-c, K) / C(n, K) from c correct out of n
  samples, defined for K <= n (NaN otherwise). It is computed exactly in integers and rounded once to a
  double. `unbiased_log_fail(n, c, K)` is its companion, log C(n-c, K) / C(n, K): the estimator's chance
  that all K draws fail (minus infinity when n - c < K).
- `passk_plugin(P, K)` is the plug-in 1 - (1-P)^K of a single-sample success probability P. The paper
  evaluates every strategy this way on the fitted profiles. Its code used two floating-point forms of it,
  and each analysis keeps its own so that every printed number reproduces: `form="exp"`,
  1 - exp(K L) with L = log(1 - P) (Appendices F and G), and `form="expm1"`, -expm1(K log1p(-P))
  (Appendix H). The two agree to within a few units in the last place. `log1m(P)` is L, with 1 - P floored
  at 1e-15.

For an allocation q (sum q = K) over the settings, a question's failure probability is
exp(sum_j q_j L_j) (Eq. 7), so every strategy is scored from the matrix L = log(1 - P) alone:

- one setting j at budget K: 1 - exp(K L[:, j]);
- an allocation q: 1 - exp(L q);
- the uniform scheme: K/l samples at each of the l settings, 1 - exp((K/l) sum_j L[:, j]).

## 6. The paper's settings

| quantity | value |
|---|---|
| Clopper-Pearson level | 1 sigma, alpha = 0.31731050786291415 |
| sigma clamp | 1e-6 percent |
| basis | Bernstein, degree 10, on the bracket of settings with a correct sample, padded by one grid step |
| optimiser | projected gradient descent from 0 |
| iterations | 50,000 |
| step size | 0.99 / L, with L from 300 power-iteration steps, seed 42 |
| projection | every 20 iterations; soft-blended over the first 5% of iterations |
| peak | maximum of the fit on 1,001 points of the bracket |
| inclusion | peak >= 0.1 percent |
| log(1 - P) | 1 - P floored at 1e-15 |
| K grid | 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024 |
| table rows | K up to 512 |
| summary columns | K = 1, 16, 256, 1024 |
| relaxation | SLSQP: bounds 0 <= q_j <= K, sum q = K, analytic gradient, maxiter 100, ftol 1e-12. In-sample, the solve starts from uniform; each fold starts from the full-data optimum. |
| integer step | the paper's search: radius 2 around the relaxed optimum, radius 1 when 12 or more settings are active, with 1-swap descent |
| folds | leave-one-out: one question per fold. 10-fold: `numpy.random.default_rng(0).permutation(n)` cut by `numpy.array_split` (for n = 359: 9 folds of 36 and 1 of 35) |
| bootstrap | B = 10,000; SE = standard deviation of the ratio over resamples, ddof 1; band = ratio +/- SE |

## 7. The paper's behaviours kept on purpose

1. **The sample count behind each rate (`--n-mode`).** On LiveCodeBench a question can have different
   sample counts at different temperatures: in 4 questions for Mistral 7B Instruct, 20 for GPT OSS 20B,
   268 for Qwen3-8B and 30 for Qwen3-0.6B. The paper used two readings. `--n-mode question-max` (the
   default) divides every temperature of a question by that question's largest n: the "earlier parsing" of
   Appendix H, behind Figures 1-3 and 4(b), Appendices E-G and Tables 2-6. `--n-mode per-setting` uses each
   temperature's own n: the "corrected parsing" of Appendix H and Figure 4(a). The default keeps the
   paper's main results. Moving to `per-setting` n changes, on Mistral 7B Instruct, the Figure 3 ratios by
   up to 2.9e-4 and 41 of 378 cells of Table 3 by 0.001, and the included count of Qwen3-8B from 647 to
   677. On MATH-500 and Hi-ToM every cell of a question has the same n, so both modes agree.
2. **The uniform scheme puts K/l samples at each of the l settings,** fractional when l does not divide K.
3. **Held-out ratios divide the held-out total by the in-sample uniform total.**
4. **Table cells** are the expected solved count rounded to 6 decimals, divided by n, and then printed with
   3 decimals.
5. **The temperature axis drops T = 0** (greedy decoding) from the allocation grid. A T = 0 row, if
   present, is still a point of the fit. The min_p axis keeps min_p = 0, which is plain sampling.
6. **Boundary tightening.** At c = 0 or c = n, the fit substitutes the meaningful side's weight, and the
   kernel receives the substituted weights. The paper's code comment says only the step size sees them; we
   keep what the code does.
7. **The integer step defaults to the paper's search.** `--integer paper` (the default of `pipeline`,
   `heldout` and `paper_check`) is the paper's own step and gives every printed number. `--integer exact`
   searches the whole integer simplex by branch and bound and certifies optimality to a tolerance (`--tol`
   of `heldout` and `allocation`, default 1e-9 expected unsolved questions). The single-budget tool
   `allocation` defaults to `--integer exact`; pass `--integer paper` for the paper's step.
8. **Bootstrap seeds.** The paper's LiveCodeBench figures drew their bootstrap from fresh entropy, so their
   bands cannot be redrawn exactly (`--seed-scheme entropy` does the same). Its MATH-500, Hi-ToM and min_p
   figures used `default_rng(SeedSequence(33273348774934605029837111157444119385).spawn(1)[0])`, which
   `--seed-scheme spawn --seed 33273348774934605029837111157444119385` reproduces. `--seed-scheme plain`
   (the default) is `default_rng(seed)`.
9. **Floating point.** The default kernel build (`portable`) disables fused multiply-add, so the C kernel
   and the numpy fallback agree bit for bit. The paper's build used `-march=native` and fused multiply-add;
   `--build-flags paper` reproduces its fitted matrix bit for bit on x86-64 with FMA (section 8).

## 8. Floating point: the two builds and the numpy fallback

**Profile fit (Mistral 7B Instruct, LiveCodeBench: 975 questions x 16 temperatures).**

- **The paper's own matrix.** With the kernel built the paper's way (`--build-flags paper`, x86-64 with fused
  multiply-add), the fitted log(1 - P) matrix is bit-identical to the paper's, and the included set is the
  same: 359 questions.
- **Portable build.** With the default portable build (no fused multiply-add), log(1 - P) differs from the
  paper's by at most 7.8e-12, and P on the allocation grid by at most 1.4e-15. Every number in section 9 is
  unchanged, on all 12 datasets.
- **One near-tie.** On the other datasets, one question settles on a different fit under the two builds:
  MATH-500 Qwen3-0.6B, `test/intermediate_algebra/1197.json`. The two fits are mirror images with the same
  peak, 0.083%, but peaks at 0.778 and 1.322: a near-tie in the unimodal projection. The question is
  excluded under both, since its peak is below 0.1%. Every other coefficient on that dataset agrees within
  4.5e-14.
- **numpy fallback.** It is bit-identical to the portable C kernel on all 975 questions, in the synthetic
  test (`tests/test_core.py`, `test_c_kernel_equals_numpy`), and in further tests of the PAVA, projection
  and whole-kernel steps on synthetic and fitted questions. The one exception is an unreachable overflow
  case, described in `fit_all.py`.
- **Curve evaluation.** The vectorised evaluator repeats the compensated `sum()` of Python 3.12 and later,
  so its values equal the paper's scalar evaluator as run under Python 3.12 or later.

## 9. Reproduction

`python -m onetemp.core.paper_check` compares, number by number, with `onetemp/core/paper_numbers.json`
(the printed numbers of the paper) and prints one line per comparison: paper | ours | difference | where. It
exits with 0 when every number matches. Settings: the paper's own (section 6), with question-level n
(`--n-mode question-max`), the fitted profiles, the 0.1 percent peak filter, the paper's integer step and K
up to 1024. Input: the per-question counts of `data/counts/`, in the paper's question order.

### Figure 3 (Mistral 7B Instruct, LiveCodeBench, n = 359)

`python -m onetemp.reproduce fig3`. The red curve is the optimal allocation's held-out pass@K over the
uniform scheme's. Figure 3 draws the 10-fold curve; Figure 14(b) draws leave-one-out. The 10-fold split is
`default_rng(0).permutation(359)` cut by `array_split` into 10 folds. Its fold-list sha256 is
`12ee3719883b553060765fb98666bfb15b28caa4a355d340384fe6cc92234eed`, equal to the paper's split
(`paper_numbers.json` holds it, and the leave-one-out one).

| quantity | paper | ours | where |
|---|---|---|---|
| 10-fold, K = 2 | 1.0999 | 1.0999 (unrounded 1.099867) | Figure 3 |
| leave-one-out, K = 2 | 1.0869 | 1.0869 (unrounded 1.086936) | Figure 14(b) |
| 10-fold, all 20 values of K (1 to 1024), 4 decimals | 20 values | 20 equal | Figure 3 |
| leave-one-out, all 20 values of K, 4 decimals | 20 values | 20 equal | Figure 14(b) |
| integer allocations, in-sample and in every fold | | identical to the paper's | both |

Against the paper's own unrounded values, the held-out ratios agree to 1.7e-13 or better (the largest
difference is at K = 1).

The redrawn figure (`pipeline ... --folds loo 10 --plot`, file `rtu_kf10.png`) matches Figure 3 curve for
curve. The bands differ only in their random draws (section 10).

### pass@K tables (LiveCodeBench, Tables 2 to 6)

Every printed cell is compared as text: the value at each temperature and the five aggregate numbers of
every K row (Best tau in-sample and held-out, Uniform, Optimal in-sample and held-out), plus the bold marks.
There are 18 rows (K = 1 to 512).

| table | model | n | numbers compared and identical | where |
|---|---|---|---|---|
| Table 2 | GPT OSS 20B | 766 | 378 / 378 | Appendix F |
| Table 3 | Mistral 7B Instruct | 359 | 378 / 378 | Appendix F |
| Tables 4-5 | Qwen3-8B | 647 | 774 / 774 | Appendix F |
| Table 6 | Qwen3-0.6B | 85 | 450 / 450 | Appendix F |

Example cells from Table 3, all at K = 2: tau = 0.5 gives 0.238; Best tau gives 0.238 in-sample and
**0.238** held-out; Uniform gives 0.216; Optimal gives **0.238** in-sample and 0.235 held-out. Ours are
equal.

One table alone: `python -m onetemp.core.paper_check --counts data/counts/livecodebench/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl`.

### Summary table (Appendix F.2): all 12 rows

For each of the 12 (benchmark, model) rows and each K in {1, 16, 256, 1024}, `paper_check` compares
Optimal (LOO) / uniform and the best single temperature with its ratio, both at 3 decimals:

| numbers compared | identical | where |
|---|---|---|
| 48 Optimal (LOO) / uniform values | 48 / 48 | summary table |
| 48 best single temperatures with their ratios | 48 / 48 | summary table |

The included counts of all 12 rows equal the figure captions:

| benchmark | GPT OSS 20B | Mistral 7B Instruct | Qwen3-8B | Qwen3-0.6B |
|---|---:|---:|---:|---:|
| LiveCodeBench | 766 | 359 | 647 | 85 |
| MATH-500 | 500 | 455 | 496 | 492 |
| Hi-ToM | 678 | 615 | 466 | 743 |

Examples (the printed values), at K = 1, 16, 256, 1024:

| row | Optimal (LOO) / uniform | best single temperature (ratio) |
|---|---|---|
| LiveCodeBench, Mistral 7B Instruct | 1.119, 1.133, 1.082, 1.052 | 0.3 (1.119), 0.8 (1.135), 0.9 (1.083), 1.0 (1.052) |
| Hi-ToM, Mistral 7B Instruct | 1.727, 1.447, 1.153, 1.061 | 1.425 (1.758), 1.575 (1.447), 1.8 (1.153), 1.875 (1.062) |

### Ratio-to-uniform figures (Figures 14 to 16)

What the shipped commands check against the printed paper is the included-question count n of each of the 12
panels (the table above) and the summary-table values, which are the values of the red (leave-one-out) and
best-temperature curves at K = 1, 16, 256 and 1024. The bands are not compared (section 10 explains the
seeds). To redraw a panel, run `pipeline` with `--folds loo --plot` on its counts file; for the MATH-500 and
Hi-ToM panels add `--seed-scheme spawn --seed 33273348774934605029837111157444119385`.

### Appendix I, min_p at tau = 1.5 (Qwen3-0.6B, Hi-ToM, n = 648)

This runs through the generic setting axis (`--axis min_p --fixed-temperature 1.5`) on measured rates
(`--mode meas`), with K up to 384, as the appendix states. The input is
`data/counts/hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl`. All 28 ratio cells, the Optimal (LOO) row and the
uniform row are identical to the paper's table at 3 decimals (2 for the percentages):

| min_p | K = 1 | K = 16 | K = 64 | K = 384 |
|---|---:|---:|---:|---:|
| 0 | 0.904 | 0.979 | 1.024 | 1.008 |
| 0.001 | 1.284 | 1.169 | 1.081 | 1.014 |
| 0.003 | 1.318 | 1.174 | 1.076 | 1.013 |
| 0.01 | 1.329 | 1.166 | 1.065 | 1.006 |
| 0.03 | 1.291 | 1.132 | 1.037 | 0.983 |
| 0.1 | 1.151 | 1.015 | 0.940 | 0.914 |
| 0.2 | 0.987 | 0.873 | 0.808 | 0.780 |
| Optimal (LOO) / uniform | 1.329 | 1.173 | 1.082 | 1.014 |
| uniform pass@K, % | 7.32 | 58.38 | 85.32 | 97.94 |

The other Appendix I numbers, which need no profile fit, are recomputed by
`python -m onetemp.data.fitfree` (see `docs/data.md`).

## 10. Not reproduced exactly

- **LiveCodeBench bootstrap bands** (Figures 3 and 14). The paper drew them from fresh operating-system
  entropy (Figure 3 with 500 replicates), so no seed recreates them. Ours are seeded, with 10,000 replicates
  by default; statistically they are the same bands. The curves themselves (the ratios) do not depend on the
  bootstrap; the values compared are listed in section 9.
- **Appendix F's independent solver cross-check** (CLARABEL through CVXPY). It is not included, because
  cvxpy is outside this release's dependencies. `allocation.kkt_report` gives the KKT residuals and the
  first-order gap bound, and `integer_exact` certifies the integer optimum itself.

## 11. Outputs of `pipeline`

| file | contents |
|---|---|
| `profiles.json` | Per question: bracket, Legendre and Bernstein coefficients, R2, worst sigma deviation, peak and efficient range (the fit's and the measured), and P on the allocation grid. `fit_all --out-rows` also writes every counts row with `p_fit` (the fitted P at that setting) and `peak_pct` (the question's fitted peak, percent). |
| `heldout_loo.json`, `heldout_kf10.json` | Per K, in expected solved questions (divide by n for pass@K): the optimal allocation in-sample (`in_sample_solved`) and held-out (`mean_solved`); the best single setting in-sample (`case2_total`) and held-out (`case2_oos_total`); uniform (`case3_total`). Also the ratios to uniform, the in-sample integer allocation, the delta-method SEs, and the fold rule and its sha256. |
| `rtu.json` | Ratio to uniform of every setting at every K, with bootstrap SE and 2.5/97.5 percentiles, and the absolute pass@K of every setting and of uniform. |
| `passk_table.{tex,csv,md,json}` | The pass@K table. |
| `summary.{json,tex}` | The summary-table rows. |
| `rtu_loo.png`, `rtu_kf10.png` | The figure (with `--plot`). |
| `run.json` | The input's sha256, the options, the versions, n and timings. |

## 12. Run times (one core, approximate)

| task | time |
|---|---|
| fit, 975 questions x 16 settings | 4 s with the C kernel, 59 s with numpy |
| leave-one-out, n = 359, 20 values of K | 23 s with the paper's integer step, 87 s with the exact one |
| leave-one-out, n = 615, 31 settings | 66 s |
| leave-one-out, n = 647, 38 settings (the largest grid) | about 8 min with the paper's integer step, 61 min with the exact one |
| bootstrap, B = 10,000 | 2 s |
| `python -m onetemp.reproduce tables` (all 13 studies) | about 15 min |

`python -m onetemp.reproduce list` prints every group with its run time on one core.
