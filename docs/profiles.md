# Question profiles: Section 3, Appendices D, E and I

`onetemp.profiles` turns per-question counts into the question-profile results of the paper:
Figures 1, 2, 5-13 and the min_p versions in Figures 38-41. It reads the counts in `data/counts/`, fits
or reads each question's profile, and writes the per-question tables, the numbers behind the three
hardness views, and the figures.

Bottom line: `python -m onetemp.reproduce profiles` redraws every one of these figures from the shipped
counts in about 3 minutes and compares them with the paper. The printed numbers match, and so do the image
bytes (compared with `data/reference/figure_sha256.json`; the paper's images were drawn with matplotlib 3.10.6
and Pillow 11.3.0, see "Environment"). There is one known difference: the grey background of Figure 1 (see
"Figure 1" below). The command prints the counts of what matched.

| Paper | What | Command |
|---|---|---|
| Figure 1 | example temperature profiles of Qwen3-8B on LiveCodeBench, with P_E^1, P_E^2 and P_C | `example` |
| Figure 2 | peak temperature, efficient-range size and maximal coverage against hardness, Qwen3-8B on LiveCodeBench | `figure2` |
| Figures 5-7 | per-question peak temperature against hardness, with sliding averages (12 panels) | `figures` |
| Figures 8-10 | efficient-range size against hardness, with the OLS fit on log10 H (12 panels) | `figures` |
| Figures 11-13 | maximal coverage R_Q(H) (Eq. 5) in 17 log-uniform hardness bins (12 panels) | `figures` |
| Appendix E.2 text | sliding-average start and end values, OLS slopes and R^2, lowest coverage bars | `summary`, `repro` |
| Appendix D | the profile fit itself (degree-10 Bernstein fit); it lives in `onetemp.core` (see `docs/core.md`) | `fit` calls it |
| Appendix I, "Question profiles and hardness" | the same analyses over a min_p grid on measured rates, with the split-half and common-shape-null checks (Figures 38-41) | `minp` |
| everything above | all figures and printed numbers, compared with the paper | `repro` |

Requirements: Python 3.10 or newer, and `pip install -e .` from the repository root (numpy, scipy and
matplotlib; no GPU). Every command has `--help`. The tests of this package are in `tests/test_profiles.py`
and run with `python -m pytest` from the repository root.

## Definitions (as the paper's code computes them)

* **Fitted profile** (Appendix D): degree-10 Bernstein fit per question, P = 0 outside its bracket.
  Evaluated on 1,001 equally spaced points of the bracket.
* **Peak temperature** tau*_q: the first evaluation point with the largest fitted value.
* **Hardness** H_q = -1 / ln(1 - P_q(tau*_q)) (Eq. 3), with P clamped to [1e-9, 0.99999], so every
  question at or above 99.999% sits at H = 0.0869.
* **Inclusion** (Appendix D): fitted peak >= 0.1% (1e-3). Unsolvable questions are excluded.
* **Efficient range** E_q (Eq. 4) = [t_L, t_R], the half-maximum crossings next to the peak, found by
  linear interpolation between evaluation points (the bracket end if the curve never halves).
  |E_q| = t_R - t_L. All of these are rounded to 6 decimals, as the paper's code does.
* **Sliding averages** (Appendix E.1): symmetric window of 30 neighbours plus the anchor in the
  hardness-sorted list; near the ends the window shrinks symmetrically. Blue: all included questions.
  Red: those with tau*_q > 0.1. Each line drops the 15 questions at each end; on LiveCodeBench and
  MATH-500 it then starts at the first remaining question whose next question is at least 5% harder
  (`--advance auto`; Hi-ToM uses every full window).
* **Maximal coverage** (Eq. 5): 17 log-uniform bins between the smallest and largest included H. Per
  bin, the largest number of questions whose efficient ranges share one temperature (sweep line over
  the range ends, touching intervals count), divided by the questions in the bin. Empty bins have no
  bar. (Appendix E.1 also mentions floor(n/20) bins as a default; the paper's figures use 17, and
  `--bins auto` gives floor(n/20).)
* **Appendix I grids** (measured rates, no fit): the peak is the first grid value with the largest
  measured rate; E_q is the run of grid values around it whose rate is at least half the peak, and
  |E_q| counts grid values; coverage uses the same 17 log-uniform bins.

## Inputs

* `counts.jsonl`, the shared format (one line per benchmark, model, question and setting):
  `{"benchmark", "model", "id", "temperature", "min_p", "n", "c"}`. A min_p sweep has `min_p` set and
  `temperature` = the fixed tau. Extra fields (for example `n_cell`) are ignored. A study is the set of
  lines that share benchmark, model, setting axis and tau. The reader rejects c > n, a second line for
  the same question and setting, and two files that repeat a cell of the same study (for example both
  LiveCodeBench parses).
* The files in `data/counts/` are organised as `<benchmark>/<model>__temperature.jsonl`. LiveCodeBench has
  two parses: `__temperature__question_n.jsonl` (the "earlier parsing": one sample count per question,
  which the profile figures of the paper use) and `__temperature__cell_n.jsonl` (the "corrected parsing":
  each temperature's own count, used by Appendix H). Pass the `question_n` file to these commands.
  `--counts-dir data/counts` picks the 15 files the paper's profile analyses read (12 temperature sweeps
  and the three min_p files of Appendix I) and ignores the others.
* Question order is the order of first appearance in the file. Hardness ties are broken in that order
  (the sort is stable), which decides the first windows of the sliding averages next to the clamp
  H = 0.0869. Use the data files as shipped, which keep the paper's order: reversing the order of
  the MATH-500 GPT OSS 20B file moves the first red value from 0.7446 (printed 0.74) to 0.7634.
* Sample counts given to the fit (`--n-mode`): `question-max` divides every temperature of a question
  by its largest n, as the paper's LiveCodeBench profile figures do; `per-setting` uses each
  temperature's own n. `auto` (default) picks the paper's choice: `question-max` for LiveCodeBench,
  `per-setting` for MATH-500 and Hi-ToM, where every temperature has the same n and both agree.
* Fitted profiles: computed on the fly with `onetemp.core.fit_all`, or read from `--profiles FILE`, a
  `profiles.jsonl` written by `python -m onetemp.profiles fit`. This is not the `profiles.json` of
  `python -m onetemp.core.fit_all`, which is the input of the robustness analyses (`docs/robustness.md`).

## Commands

All examples run from the repository root.

```bash
C=data/counts

# Fit once and keep the fitted profiles (about 3-25 s per dataset with the C kernel)
python -m onetemp.profiles fit --counts $C/livecodebench/Qwen3-8B__temperature__question_n.jsonl \
    --benchmark livecodebench --model Qwen/Qwen3-8B --out profiles_lcb_qwen3_8b.jsonl

# Per-question table: id, included, H, peak_t, peak_val (percent), t_left, t_right, range, ...
python -m onetemp.profiles table --counts $C/livecodebench/Qwen3-8B__temperature__question_n.jsonl \
    --benchmark livecodebench --model Qwen/Qwen3-8B --profiles profiles_lcb_qwen3_8b.jsonl \
    --out table_lcb_qwen3_8b.csv

# The numbers of the three views (sliding means, OLS, coverage bars) as JSON
python -m onetemp.profiles summary --counts $C/math500/Mistral-7B-Instruct-v0.3__temperature.jsonl \
    --benchmark math500 --model mistralai/Mistral-7B-Instruct-v0.3 --out summary_math500_mistral.json

# Figures 5-13: AveTemp, RangeSize and relC images for one (benchmark, model)
python -m onetemp.profiles figures --counts $C/hitom/Qwen3-0.6B__temperature.jsonl \
    --benchmark hitom --model Qwen/Qwen3-0.6B --out-dir figs --stem Report_Qwen3.6B_HiToM

# Figure 2 (three panels side by side)
python -m onetemp.profiles figure2 --counts $C/livecodebench/Qwen3-8B__temperature__question_n.jsonl \
    --benchmark livecodebench --model Qwen/Qwen3-8B --profiles profiles_lcb_qwen3_8b.jsonl --out figure2.jpg

# Figure 1 (the paper's three questions abc393_e, abc394_c and 2730 by default; --highlight ID ID ID)
python -m onetemp.profiles example --counts $C/livecodebench/Qwen3-8B__temperature__question_n.jsonl \
    --benchmark livecodebench --model Qwen/Qwen3-8B --profiles profiles_lcb_qwen3_8b.jsonl --out figure1.png

# Appendix I: one min_p study, its four figures and the noise checks (seconds each)
python -m onetemp.profiles minp --counts $C/hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl \
    --model Qwen/Qwen3-0.6B --tau 1.5 --checks --seed-index 0 --out-dir minp_qwen06_t15
python -m onetemp.profiles minp --counts $C/hitom/Qwen3-0.6B__min_p_tau2.0__nu400.jsonl \
    --model Qwen/Qwen3-0.6B --tau 2.0 --checks --seed-index 1 --out-dir minp_qwen06_t20
python -m onetemp.profiles minp --counts $C/hitom/Mistral-7B-Instruct-v0.3__min_p_tau2.0__nu1500.jsonl \
    --model mistralai/Mistral-7B-Instruct-v0.3 --tau 2.0 --checks --seed-index 2 --out-dir minp_mistral_t20

# Everything: picks the 15 files the paper's profile analyses read, fits, draws, compares
python -m onetemp.profiles repro --counts-dir $C --out-dir repro_profiles
# the same through the one entry point (writes results/profiles/)
python -m onetemp.reproduce profiles
```

### Options

`fit`, `table`, `summary`, `figures`, `figure2` and `example` share these study options:

| option | meaning |
|---|---|
| `--counts FILE...` | counts file(s) or globs (required) |
| `--benchmark` | `livecodebench`, `math500` or `hitom` (required) |
| `--model` | HF repo id, or a unique case-insensitive part of it, for example `Qwen3-8B` (required) |
| `--profiles FILE` | fitted profiles from `fit` (default: fit now) |
| `--n-mode` | `auto` (default), `question-max` or `per-setting` |
| `--min-peak` | inclusion floor on the fitted peak, in percent (default 0.1, the paper's 1e-3) |
| `--source` | `fit` (default, the paper) or `measured`: measured rates c/n with linear interpolation between settings and no fit, which the paper's code offered as an option; `example` needs `fit` |

Per command:

| command | options | writes |
|---|---|---|
| `fit` | `--out` (required), `--verbose` | `profiles.jsonl`, one line per question (bracket and Legendre coefficients) |
| `table` | `--out` (required; CSV, or JSON if it ends in `.json`) | per-question table; prints how many questions are included |
| `summary` | `--out` (required), `--advance auto/on/off`, `--bins 17/auto` | JSON of the three views; prints the headline numbers |
| `figures` | `--out-dir` (required), `--stem`, `--no-titles`, `--advance`, `--bins` | `<stem>_AveTemp.jpg`, `<stem>_RangeSize.jpg`, `<stem>_relC.jpg` and `<stem>_summary.json`; the default stem is `<benchmark>_<model>` |
| `figure2` | `--out` (required), `--advance`, `--bins` | one JPEG with the three panels |
| `example` | `--out` (required), `--highlight ID ID ID`, `--labels TEXT TEXT TEXT`, `--alpha` (0.35), `--n-background` (60), `--seed` (0), `--record FILE` | the Figure 1 PNG; prints what was drawn (P_E^1, P_E^2, P_C, background counts) |
| `minp` | `--counts`, `--model`, `--tau` (required), `--benchmark` (default `hitom`), `--min-peak`, `--out-dir` (required), `--stem`, `--checks`, `--seed`, `--seed-index`, `--n-guard` (400), `--n-null` (400), `--n-fpr` (0) | `<stem>_Peak.jpg`, `_RangeSize.jpg`, `_relC.jpg`, `_profiles.jpg` and `<stem>_values.json` (default stem `<benchmark>_<model>_minp_T<tau>`) |
| `repro` | `--counts-dir` or `--counts`, `--profiles`, `--out-dir` (required), `--paper-hashes`, `--paper-figures`, `--paper-figure2`, `--skip-minp`, `--skip-checks` | see below |

`minp` works on the measured rates only (no fit, no `--profiles`) and needs n > 0 in every cell.

### What `repro` writes

* `repro_table.md` (quantity | paper | ours | difference | where | match) and `repro.json`.
* Under `<benchmark>/`: each dataset's `<stem>_table.csv`, `<stem>_summary.json` and the three images
  named as the paper's files; under `minp/`: the four images of each min_p study; in the output folder:
  `figure2_qwen3_8b_livecodebench.jpg` and `figure1_profiles_qwen3_8b_livecodebench.png`.
* The last line gives the counts, for example `N/M comparisons match, K known differences, 0 unexpected`.
  The exit status is 0 when there are no unexpected differences.
* `--paper-hashes` is the table of the paper's image checksums (default `data/reference/figure_sha256.json`;
  `none` skips the byte comparison). `--paper-figures` and `--paper-figure2` take a folder or a file of the
  paper's own images instead, if you have them; they are not needed.
* `--skip-checks` skips the min_p noise checks (they take under a minute).

Image files and the figures they belong to (a panel for each model, in the order GPT OSS 20B, Mistral 7B
Instruct, Qwen3-8B, Qwen3-0.6B):

| suffix | Figure for LiveCodeBench / MATH-500 / Hi-ToM |
|---|---|
| `_AveTemp.jpg` | 5 / 6 / 7 |
| `_RangeSize.jpg` | 8 / 9 / 10 |
| `_relC.jpg` | 11 / 12 / 13 |

Stems: `Report_GPT_oss_20B`, `Report_Mistral_7B`, `Report_Qwen3_8B_complete`, `Report_Qwen3.6B_filtered`
for LiveCodeBench; `Report_<model>_MATH500` and `Report_<model>_HiToM` with `GPT_oss_20B`, `Mistral_7B`,
`Qwen3_8B`, `Qwen3.6B` for the other two. The min_p images are `Report_<model>_HiToM_minp_T<tau>_<kind>.jpg`
with kind `profiles` (Figure 38), `Peak` (Figure 39), `RangeSize` (Figure 40) and `relC` (Figure 41), one panel
for each of the three studies. The full sha256 of every image is in `data/reference/figure_sha256.json`.

## The noise checks of Appendix I

On the min_p grids the peak and H come from the same noisy maximum, so a trend is stated only if it
passes both checks, fixed before any result was seen:

1. split-half guard: 400 splits of every cell's samples into halves by hypergeometric thinning; peak or
   range from one half, H and the bins from the other; the slope keeps its full-data sign in >= 95%;
2. common-shape null: p_q(v) = min(1, r_q g(v) / mean g), with g the pooled rate per grid value and r_q
   the question's mean rate, binomial draws at the same n, 400 replicates; the full-data slope must lie
   outside the null's central 95% on its own side.

The trend statistics are OLS slopes on log10 H: of the peak grid position (all questions, and the
questions whose peak is above the smallest value), of |E_q|, and of the coverage bars.

The seeds are those of the paper: `numpy.random.SeedSequence(20261104)` spawned once per study in the
order Qwen3-0.6B tau = 1.5, Qwen3-0.6B tau = 2.0, Mistral tau = 2.0 (`--seed-index 0/1/2`), and each
study's sequence spawned again into (guard, null, false-positive-rate) seeds. With the counts in
lexicographic id order (the analysis sorts them) and numpy 2.2.6 these reproduce the paper's check results
exactly. Other numpy versions can make different binomial draws from the same seed (numpy 2.5.4 does), and then
the null's Monte Carlo values can differ in the last printed digit: its quantiles, and with them whether its
central 95% lies within the interval the paper prints, its median, and the counts of null replicates beyond a
slope. With a numpy other than 2.2.6, `repro` reports such a difference as known. The split-half guard draws
hypergeometric counts, and numpy 2.5.4 drew the same ones (its 400 slopes per statistic agree to 1e-14); the
guard and the conclusions, which trends pass both checks, are compared strictly with every version.
`--n-fpr 100` adds the paper's diagnostic of how often the guard alone passes on null data (slow).

## Choices the paper's code made, kept here on purpose

They are the reason the outputs match the printed numbers and the paper's images.

* LiveCodeBench fits use one question-level n at every temperature (`question-max`), the earlier parsing the
  paper's profile figures use.
* The drawn sliding-average lines on LiveCodeBench and MATH-500 skip the questions at the clamp
  H = 0.0869 and at most two others (the 5% rule); Hi-ToM lines use every full window.
* Coverage figures use 17 bins whatever the number of questions.
* Figure 1: 60 background questions are drawn with `numpy.random.default_rng(0)`, 12 per peak-temperature
  band [0, 0.4), [0.4, 1), [1, 1.8), [1.8, 3), [3, 10); curves with a dip deeper than 5 points are
  dropped, then 3 of the 9 highest-peaked at random; grey alpha 0.35.
* Peak and range are read off the 1,001-point evaluation grid, and every number is rounded to 6 decimals.
* The sums inside the Legendre evaluation follow Python's own `sum`, which is compensated from Python 3.12 on,
  so peaks on flat tops land on the same evaluation point as in the paper's code. The reproduction below ran
  on Python 3.12.12; on Python 3.10.19 every compared number came out the same.
* Questions are kept in file order, and the hardness sort is stable (see "Inputs").
* Images are saved as the paper's were: Appendix E and I images as JPEG at 300 dpi with a tight bounding box,
  Figure 2 as one JPEG of three panels, Figure 1 as a PNG at 110 dpi.

## Reproduction record

### Environment

CPython 3.12.12, numpy 2.2.6, scipy 1.16.0, matplotlib 3.10.6 with Pillow 11.3.0, Linux x86-64.
**The paper's images were drawn with matplotlib 3.10.6 and Pillow 11.3.0.** Measured: with matplotlib 3.10.9 and
Pillow 12.3.0 all 49 images are byte-identical too; with matplotlib 3.11.2 and Pillow 13.0.0 none is, and the
numbers of that run differ only in two Monte Carlo values of the min_p null, from its numpy 2.5.4 (see "The noise
checks of Appendix I"). With any versions but 3.10.6 and 11.3.0, `repro` reports a differing byte row as
a known difference that names the installed versions, so the exit status depends on the numbers alone. With
matplotlib 3.10.6 and Pillow 11.3.0 installed, a byte difference counts as a mismatch. In the same way, with a
numpy other than 2.2.6 a differing Monte Carlo value of the min_p null is a known difference (see "The noise
checks of Appendix I"). Run
`python -m onetemp.profiles repro --counts-dir data/counts --out-dir OUT --paper-hashes none` to skip the byte
comparison.

The fit is the core package's Bernstein fit (`onetemp.core.fit_all`, Appendix D) with the paper's
sample-count rule per benchmark, with the C kernel built with the portable flags
(`python -m onetemp.core.fit_all --build-kernel`). The numpy fallback gave bit-identical coefficients on the
two datasets compared (LiveCodeBench Qwen3-0.6B, 110 questions; MATH-500 Mistral, 500). Fitting all 12 sweeps
takes 2.5-25 s each with the C kernel; with the drawing and the Appendix I noise checks (under a minute),
`python -m onetemp.reproduce profiles` takes about 3 minutes.

### Three independent checks of the analysis code

Made while preparing this release.

1. **Against the analysis code of the paper, question by question.** For all 12 (benchmark, model) sweeps,
   with profiles fitted by the paper's own fitter, every included question's H, peak temperature, peak
   pass rate, t_L and t_R equal the values the paper's code computes (exact float equality, 6,302
   included questions in total), and so do every drawn sliding-average point, OLS coefficient and
   coverage bar.
2. **Fit variants.** The portable build and the paper's build (which allows fused multiply-add) give
   Legendre coefficients that differ by at most 1.1e-13 on every question except one, MATH-500 Qwen3-0.6B
   `test/intermediate_algebra/1197.json`, whose two fits are mirror images (peak 0.083%, below the 0.1%
   floor, so excluded everywhere). No included question's numbers change, and every image is byte-identical
   with either fit.
3. **Included-only input.** A counts file holding only the 359 included Mistral LiveCodeBench questions
   gives the same three images, byte for byte, as the full 975-question file.

The included questions also equal the lists in `data/included_ids.json` for all 12 temperature files (the
same ids in the same order).

### Per-dataset numbers (ours, full precision; the paper prints 2-3 decimals of these)

| benchmark | model | n | n (peak > 0.1) | OLS slope | intercept | R^2 | red first | red last | lowest bar | bars |
|---|---|---|---|---|---|---|---|---|---|---|
| LiveCodeBench | GPT OSS 20B | 766 | 602 | -0.2493 | 1.1795 | 0.4939 | 0.2336 | 0.8712 | 0.7222 | 17 |
| LiveCodeBench | Mistral 7B Instruct | 359 | 326 | -0.1058 | 0.9269 | 0.2653 | 0.2636 | 0.9719 | 0.8276 | 17 |
| LiveCodeBench | Qwen3-8B | 647 | 542 | -0.4507 | 1.9148 | 0.6008 | 1.0029 | 1.4220 | 0.6111 | 17 |
| LiveCodeBench | Qwen3-0.6B | 85 | 82 | -0.0875 | 0.7498 | 0.1843 | 0.6376 | 0.7351 | 0.6000 | 17 |
| MATH-500 | GPT OSS 20B | 500 | 416 | -0.2340 | 1.1362 | 0.5491 | 0.7446 | 0.9048 | 0.5000 | 17 |
| MATH-500 | Mistral 7B Instruct | 455 | 351 | -0.0485 | 0.9645 | 0.0226 | 0.4946 | 0.7021 | 0.6667 | 17 |
| MATH-500 | Qwen3-8B | 496 | 464 | -0.3929 | 1.3323 | 0.7676 | 1.1447 | 0.9863 | 0.3333 | 17 |
| MATH-500 | Qwen3-0.6B | 492 | 351 | -0.1845 | 1.0922 | 0.3554 | 0.6066 | 0.8490 | 0.6250 | 17 |
| Hi-ToM | GPT OSS 20B | 678 | 672 | -0.1826 | 1.4331 | 0.2135 | 0.9338 | 1.4953 | 0.8333 | 16 |
| Hi-ToM | Mistral 7B Instruct | 615 | 614 | -0.2761 | 1.5285 | 0.5363 | 1.1711 | 1.8507 | 0.9565 | 17 |
| Hi-ToM | Qwen3-8B | 466 | 466 | -0.7534 | 2.8407 | 0.5481 | 1.8910 | 3.1885 | 0.8571 | 16 |
| Hi-ToM | Qwen3-0.6B | 743 | 739 | -0.0879 | 1.2595 | 0.0679 | 0.8014 | 1.5160 | 0.9825 | 17 |

"red first/last" is the first and last drawn value of the red sliding average (peak tau > 0.1);
"lowest bar" is the smallest maximal-coverage bar. Figure 2 shows the LiveCodeBench Qwen3-8B row.

### Appendix I (min_p), ours

| study | slope of peak position (all / above smallest value) | null central 95% (all) | range-size slope, null median, null replicates beyond it | coverage: lowest bar (bins >= 10), null at least as steep | trends claimed |
|---|---|---|---|---|---|
| Qwen3-0.6B, tau = 1.5 (648) | -3.1005 / -2.8647 | -0.163 .. 0.335 | -2.8812, -3.250, 398 of 400 steeper (the measured slope is flatter than 398 of 400) | 0.7273, 400 of 400 (guard 0.9375, fails) | peak only |
| Qwen3-0.6B, tau = 2.0 (648) | -1.9757 / -1.3544 | -0.599 .. -0.138 | -1.6877, -1.428, 2 of 400 at least as steep | 0.7000, 41 of 400 | peak, range |
| Mistral 7B Instruct, tau = 2.0 (598 of 617) | -1.5118 / -1.4805 | 0.251 .. 0.543 | -1.9554, -1.196, 0 of 400 at least as steep | 0.7000, 400 of 400 | peak, range |

Every peak slope keeps its sign in all 400 splits and no null replicate reaches it. Highlighted
questions of Figure 38 peak at 27.0%, 31.0%, 36.8% (tau = 1.5), 26.5%, 31.25%, 34.75% (Qwen3-0.6B,
tau = 2.0) and 33.2%, 33.13%, 37.13% (Mistral): the paper's "26.5% to 37.1%".

### Figure 1

Figure 1 and the Appendix C count 646 come from an earlier release of the Qwen3-8B LiveCodeBench data, not
shipped; the redraw differs in the grey background.

The grey background is a random sample of other questions, drawn from all questions of the Qwen3-8B
LiveCodeBench data with a fixed seed, by peak-temperature band, then thinned (a dip, and 3 of the 9
highest). On the shipped data (the later release, which also draws Figures 2, 5, 8 and 11) the redraw has
45 grey curves where the paper's legend reads 44; `repro` lists this as its one known difference for
Figure 1 and still exits with status 0. The figure's message, the two canonical profile shapes and the
P_E^1, P_E^2, P_C construction, rests on the three highlighted questions (LiveCodeBench `abc393_e`,
`abc394_c` and `2730`). Those coincide with the published image: the three curves, their measured points and
the P_E^1, P_E^2, P_C markers (pixel overlap of the coloured curves 0.991, 0.991 and 0.996, measured when
the release was prepared). The Figure 1 image is not part of the byte comparison.

### Every comparison (generated by `repro`)

`difference` is ours minus the paper's value at the printed precision. In every row below ours equals the
paper's value, except the one Figure 1 row marked as known. Run `python -m onetemp.reproduce profiles` for
the counts, or open `repro_table.md` for the full list with the match column.

**Figure and caption numbers of Figures 2 and 5-13** (paper value; ours is equal at the printed precision, difference 0):

| benchmark | model | n (captions) | n peak > 0.1 (legend) | OLS slope | OLS intercept | OLS R^2 | red first (E.2) | red last (E.2) | lowest coverage bar (E.2) | bars drawn (Fig. 13) |
|---|---|---|---|---|---|---|---|---|---|---|
| LiveCodeBench | GPT OSS 20B | 766 | 602 | -0.249 | 1.18 | 0.494 | | | | |
| LiveCodeBench | Mistral 7B Instruct | 359 | 326 | -0.106 | 0.927 | 0.265 | | | | |
| LiveCodeBench | Qwen3-8B | 647 | 542 | -0.451 | 1.915 | 0.601 | | | | |
| LiveCodeBench | Qwen3-0.6B | 85 | 82 | -0.088 | 0.75 | 0.184 | | | | |
| MATH-500 | GPT OSS 20B | 500 | 416 | -0.234 | 1.136 | 0.549 | 0.74 | 0.9 | 0.5 | |
| MATH-500 | Mistral 7B Instruct | 455 | 351 | -0.048 | 0.964 | 0.023 | 0.49 | 0.7 | 0.667 | |
| MATH-500 | Qwen3-8B | 496 | 464 | -0.393 | 1.332 | 0.768 | 1.14 | 0.99 | 0.333 | |
| MATH-500 | Qwen3-0.6B | 492 | 351 | -0.184 | 1.092 | 0.355 | 0.61 | 0.85 | 0.625 | |
| Hi-ToM | GPT OSS 20B | 678 | 672 | -0.183 | 1.433 | 0.213 | 0.93 | 1.5 | | 16 |
| Hi-ToM | Mistral 7B Instruct | 615 | 614 | -0.276 | 1.529 | 0.536 | 1.17 | 1.85 | | 17 |
| Hi-ToM | Qwen3-8B | 466 | 466 | -0.753 | 2.841 | 0.548 | 1.89 | 3.19 | | 16 |
| Hi-ToM | Qwen3-0.6B | 743 | 739 | -0.088 | 1.259 | 0.068 | 0.8 | 1.52 | | 17 |

Where: n is in the captions of Figures 5/8/11 (LiveCodeBench), 6/9/12 (MATH-500) and 7/10/13 (Hi-ToM); the
n with peak tau > 0.1 is in the legend of Figure 5, 6 or 7; the OLS numbers are in the legend of Figure 8, 9
or 10; the red start and end values and the MATH-500 lowest bar are in the Appendix E.2 text; the number of
Hi-ToM bars is in the Figure 13 caption. The Figure 2 legends (647 / 542 / -0.451, 1.915, 0.601) equal those
of the Qwen3-8B LiveCodeBench row.

**Yes/no checks of Appendix E** (paper "yes", ours "yes" in every case):

| check | datasets | where |
|---|---|---|
| blue line and red line each drop at most 2 non-clamp questions in the low-H advance | the 4 LiveCodeBench and 4 MATH-500 datasets | App. E.1 |
| the first coverage bar is 1 and the lowest bar is in one of the two highest-H bins | the 4 MATH-500 datasets | App. E.2 text |
| all coverage bars lie in [0.83, 1.0] | the 4 Hi-ToM datasets | App. E.2 text |
| questions at the clamp H = 0.0869: paper 0, ours 0 | the 4 Hi-ToM datasets | App. E.1 |

**The known difference**

| quantity | paper | ours | difference | where |
|---|---|---|---|---|
| Figure 1: grey background questions drawn | 44 | 45 | +1 | Fig. 1 image legend (known difference, see "Figure 1") |

**Appendix I** (paper value; ours is equal in every cell, difference 0; "-" means the paper prints no
number for that study):

| quantity | Qwen3-0.6B, tau = 1.5 | Qwen3-0.6B, tau = 2.0 | Mistral 7B Instruct, tau = 2.0 | where |
|---|---|---|---|---|
| included questions | 648 | 648 | 598 | App. I |
| excluded, no correct sample | - | - | 8 | App. I |
| excluded below 0.1% with some correct | - | - | 11 | App. I |
| questions peaking at the smallest value | 33 | - | - | App. I |
| questions peaking at the largest value | 24 | 133 | 119 | App. I |
| questions with no correct sample at the largest value | 576 | - | - | App. I |
| peak slope, all (2 dp) | -3.1 | -1.98 | -1.51 | App. I |
| peak slope, peak above smallest value (2 dp) | -2.86 | -1.35 | -1.48 | App. I |
| range-size slope (2 dp) | -2.88 | -1.69 | -1.96 | App. I |
| lowest coverage bar, bins with >= 10 questions (2 dp) | 0.73 | 0.7 | 0.7 | App. I |
| questions in bins wholly above H = 100 | 9 | 2 | 68 | Fig. 41 caption |
| peak slope keeps its sign in all 400 splits | yes | yes | yes | App. I |
| replicates of the null that reach the peak slope | 0 | 0 | 0 | App. I |
| null central 95% of the peak slope within [-0.6, 0.54] | yes | yes | yes | App. I |
| range-size null median (2 dp) | -3.25 | -1.43 | -1.2 | App. I |
| null replicates steeper than the range slope | 398 | - | - | App. I |
| null replicates at least as steep as the range slope | - | 2 | 0 | App. I |
| null replicates at least as steep as the coverage slope | 400 | 41 | 400 | App. I |
| peak trend passes both checks (all / above smallest value) | yes / yes | yes / yes | yes / yes | App. I |
| range narrows beyond noise | no | yes | yes | App. I |
| coverage trend beyond noise | no | no | no | App. I |
| coverage split-half check | fails | - | - | App. I |

Figure 38: the highlighted questions peak at 26.5% to 37.1% (paper) and 26.5% to 37.1% (ours).

### Image bytes

Each image that `repro` draws for Figures 2, 5-13 and 38-41 (the 36 panels of Figures 5-13, Figure 2, and the
12 panels of Figures 38-41) is compared by sha256 with the paper's image of the same figure panel, in
`data/reference/figure_sha256.json`. With matplotlib 3.10.6 and Pillow 11.3.0 every one is byte-identical. The
files are named as in the image-file table under "What `repro` writes", so a mismatch in `repro_table.md` names
the file and thus the figure and the panel.

## Not reproduced, and why

* **Figure 1's grey background** (one comparison above): drawn from an earlier release of the Qwen3-8B
  LiveCodeBench counts that is not shipped; the highlighted questions and markers do reproduce.
* **Section 3.1's "only a negligible fraction of the questions ... deviated from this pattern"**: the
  paper gives no number or test for it, and the fitted profiles are unimodal by construction, so there is
  nothing to compare.
* **The profile fit itself** (Appendix D) belongs to the core package and is checked there (`docs/core.md`);
  here it is checked only through its effect on these figures and numbers (see "Three independent checks").
