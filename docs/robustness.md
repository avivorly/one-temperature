# Robustness to the question mix: Section 4.3.2, Figure 4, Appendices G, H and I

The Goldilocks sampling effect (Proposition 4.2) says that, at budget K, the comparison between sampling
settings is driven by the questions whose hardness is near K. `onetemp.robustness` tests that claim on the
measured question profiles, for any sampling setting (temperature, or min_p at a fixed temperature). No GPU is
needed.

Bottom line: every number below is reproduced from the shipped counts, most of them bit for bit.
`python -m onetemp.reproduce fig4b` recomputes 50 of the 2,400 chunks of Figure 4(b) in about 4 minutes and
compares them with the paper's own run; `heatmaps`, `beta` and `simulated` redraw the rest (see
"Reproduce the paper's runs").

## The two tests

- **Random reweighting** (Appendix G, Figure 4(b), and the min_p version in Appendix I). Draw random
  per-question weights W from a Dirichlet sampler with a smooth base measure, choose the setting that is best
  under W, and score it back under uniform weights. The heatmaps show the ratio score_B / score_A against
  KL(W || uniform). The top row, ratio >= 0.9, is the share of W's whose choice loses under 10%.
- **Worst-case reweighting outside a protected window** (Appendix H, Figure 4(a), and the min_p version in
  Appendix I). Questions within D decades of K keep weight 1. beta* is the smallest bound beta on every other
  weight (W_q in [1/beta, beta]) at which a setting that loses at least 10% (or 1%) becomes the best. Every
  point with a question outside the window is certified in exact rational arithmetic (unless `--no-certify`).
- **The further tests of Appendix H:**
  - the exact worst-case loss R(beta, D) of the chosen setting;
  - reweighting one side of K at a time;
  - tilting the mix by hardness;
  - natural sub-datasets;
  - the role of K;
  - simulated hardness distributions up to K = 10^6.

| Paper | What | Module and command |
|---|---|---|
| Figures 17-20 | Appendix G heatmaps, LiveCodeBench | `heatmap --preset paper-g-lcb --row-bottom ...` |
| Figures 21-28 | Appendix G heatmaps, MATH-500 and Hi-ToM | `heatmap --preset paper-g` |
| Figure 4(b) | 24 million W's, Mistral 7B Instruct on LiveCodeBench | `fig4b check`, or `fig4b worker` and `fig4b merge` |
| Figures 42-44 | min_p heatmaps (Appendix I) | `heatmap --axis min_p --tau ... --preset paper-i` |
| Figure 4(a), Figures 29-31 | beta* against D (Appendix H) | `protected`, then `plot_prot` |
| Figure 45 | min_p beta* against D (Appendix I) | `protected --axis min_p --rates measured` |
| Figures 32-33 | the three simulated hardness shapes and their beta* | `simulate --shape peaked/reversed/flat` |
| Appendix H, further tests | R(beta, D); one side at a time; tilts; sub-datasets; role of K | `worst_loss`, `protected --window`, `tilt`, `subsets`, `prot_summary` |

## Requirements and layout

`pip install -e .` from the repository root (Python 3.10 or newer; numpy, scipy and matplotlib). The tests run
with `python -m pytest`; `tests/test_robustness.py` checks the whole package end to end on synthetic data.

| module (`onetemp/robustness/`) | what it computes |
|---|---|
| `data.py` | reads `counts.jsonl` and the fitted profiles into one study (a sweep of one setting axis) |
| `contexts.py` | pass@K, hardness, inclusion and the reference choice tau_A, in the conventions of Appendices G and H |
| `dirichlet.py` | the Dirichlet smooth-base sampler: calibration, shape draws, and three ways to draw W |
| `election.py` | the reweighted choice tau_B: `argmax`, the exact log-space election, and the even tie split |
| `heatmap.py` | KL x ratio heatmaps and the "% losing under 10%" tables (CLI) |
| `plot_heatmap.py` | draws a heatmap figure from its numbers file (CLI) |
| `fig4b.py` | the 24-million-W run behind Figure 4(b): `check`, `worker`, `merge` (CLI) |
| `protected.py` | beta* against D, for the symmetric window or one side (CLI) |
| `certify.py` | exact certificates of every beta* point, and the reachability of the "> 10^6" points |
| `worst_loss.py` | the exact worst-case loss R(beta, D) (CLI) |
| `tilt.py` | hardness tilts (CLI) |
| `subsets.py` | natural sub-datasets from question labels (CLI) |
| `simulate.py` | the three simulated hardness pools and their beta* curves (CLI) |
| `prot_summary.py` | Appendix H's aggregate statistics over many studies (CLI) |
| `plot_prot.py` | the beta*-against-D panels and the two-panel Figure 4 (CLI) |

Run any of them as `python -m onetemp.robustness.<module> --help`. Each CLI checks its output files before it
computes, and refuses to overwrite any of them. The one exception is the `fig4b` worker: it skips every chunk
whose tally already matches its inputs, and recomputes a tally that does not.

## Inputs

**Counts.** All analyses read the common per-question counts format, one JSON object per line, as in
`data/counts/`:

    {"benchmark": "math500", "model": "Qwen/Qwen3-8B", "id": "...", "temperature": 0.7, "min_p": null, "n": 750, "c": 412}

**Fitted profiles.** The temperature analyses use the fitted question profiles of Appendix D (the degree-10
unimodal Bernstein fit of Table 1). Run the profile fit of this release, `python -m onetemp.core.fit_all`, on the
same counts, and pass its output with `--profiles profiles.json`. Which counts file and which fit goes with which
figure:

| analysis | counts file | fit (`onetemp.core.fit_all --n-mode ...`) |
|---|---|---|
| Appendix G heatmaps and Figure 4(b), on LiveCodeBench | `livecodebench/<model>__temperature__question_n.jsonl` (the earlier parsing) | `question-max`, the default |
| Appendix H and Figure 4(a), on LiveCodeBench, and the simulated pools | `livecodebench/<model>__temperature__cell_n.jsonl` (the corrected parsing) | `per-setting` |
| everything on MATH-500 and Hi-ToM | `math500/<model>__temperature.jsonl`, `hitom/<model>__temperature.jsonl` | either mode: the two give the same profiles there |
| every min_p analysis (Figures 42-45) | `hitom/<model>__min_p_tau<tau>__nu<N>.jsonl` | none: measured rates, see below |

- **The two LiveCodeBench parses.** LiveCodeBench counts come in the earlier parsing (one sample count per
  question, the one Appendix G and Figure 4(b) use) and the corrected parsing (each temperature's own count, the
  one Appendix H and Figure 4(a) use). The number of included questions is 359, 766, 647 and 85 in Appendix G,
  and 359, 766, 677 and 86 in Appendix H (Mistral 7B Instruct, GPT OSS 20B, Qwen3-8B, Qwen3-0.6B).
- **The other route.** A `profiles.jsonl`, the counts file with two extra fields on every row, works too. Write
  it with `onetemp.core.fit_all --out-rows`:
  - `p_fit`: the fitted success probability at the row's setting;
  - `peak_pct`: the fitted peak of the question in percent.

**min_p sweeps** (rows with `min_p` set, and `temperature` = the fixed tau) use the measured counts directly
(`--rates measured`). There, pass@K is the unbiased estimator, as in Appendix I, and it needs K <= the number of
samples. Their grids of 10 and 5 values do not support the degree-10 fit.

**Rules applied on load:**
- **Question order** is the order of first appearance in the file, except with `--order id` (set by the preset
  `paper-i`), which sorts the questions by id. It matters only for the exact cells of the random reweighting
  (`data.py`): the sampler assigns its draws to questions in hardness order, and questions of equal hardness keep
  their input order. Any order gives the same distribution of results. The temperature files as shipped give the
  paper's cells of Figures 17-28 and 4(b); the paper's min_p heatmaps (Figures 42-44) read the questions sorted by
  id, which `paper-i` does.
- **Greedy decoding** (temperature 0) is dropped.
- **Grid:** a setting is kept only when every question has it. Dropped settings are reported.
- **Inclusion:** questions with a peak success probability below 10^-3 are left out everywhere (Appendix D,
  "Minimum peak success probability").

## Commands

The file names below are real names from `data/counts/`; for other studies change the model part. With one
study per file, `--benchmark` and `--model` can be left out. Keep each kind of run in its own folder, as below,
so that the summaries read the right files. Set up the two fits of the running example (Mistral 7B Instruct on
LiveCodeBench) first:

    C=data/counts/livecodebench
    python -m onetemp.core.fit_all --counts $C/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl \
        --n-mode question-max --out fits/lcb_mistral_qmax.json
    python -m onetemp.core.fit_all --counts $C/Mistral-7B-Instruct-v0.3__temperature__cell_n.jsonl \
        --n-mode per-setting --out fits/lcb_mistral_perset.json

### Appendix G: the twelve heatmaps (Figures 17-28) and their tables

    # LiveCodeBench (Figures 17-20): the floor rule for the ratio rows, with the bottom edge the paper printed
    python -m onetemp.robustness.heatmap --input $C/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl \
        --profiles fits/lcb_mistral_qmax.json --preset paper-g-lcb --row-bottom 0.4 --out out/G/lcb_mistral --plot
    #   --row-bottom: 0.7 for GPT OSS 20B and Qwen3-8B, 0.4 for Mistral 7B Instruct, 0.2 for Qwen3-0.6B
    # MATH-500 and Hi-ToM (Figures 21-28): ten 0.1 bands, equal cells
    python -m onetemp.robustness.heatmap --input data/counts/math500/Qwen3-8B__temperature.jsonl \
        --profiles fits/math500_qwen3-8b.json --preset paper-g --out out/G/math500_qwen3-8b --plot

Each run writes:
- `heatmap_numbers.json`: every cell's count and label, tau_A, score_A, the ratios per setting, the sampler
  record, and the numpy and scipy versions;
- `top_row_table.md`: the share losing under 10% per K and KL band, with KL > 3 and KL 1-2 pooled;
- `heatmap.jpg` (with `--plot`).

The paper's settings are 10,000 W's (`--n-w`), seed 20260606 (`--seed`), and K in {4, 32, 128, 512} (`--K`).
The presets set the rest:

| preset | figures | draw | election | rates | ratio rows | cells | question order |
|---|---|---|---|---|---|---|---|
| `paper-g-lcb` | 17-20 | `replay` | `exact` | `fit` | `script` (floor rule; set the bottom with `--row-bottom`) | drawn in nats x ratio units | file |
| `paper-g` | 21-28 | `replay` | `exact` | `fit` | `tenths0` (ten bands from 0) | equal size | file |
| `paper-i` | 42-44 | `linear` | `argmax` | `measured` | `tenths1` (nine bands from 0.1) | equal size | sorted by id |

Other options: `--draw`, `--election`, `--rates`, `--rows`, `--cells`, `--order` override a preset; `--axis` and `--tau`
select a min_p study; `--keep-all-zero` counts all-zero W's (only the linear draw makes them);
`--no-replay-check` skips the bit-for-bit check of the replayed draws. To redraw a figure from its numbers:
`python -m onetemp.robustness.plot_heatmap heatmap_numbers.json out.jpg [--cells equal]`.

### Figure 4(b): 24 million W's

The panel repeats the top row of Figure 17 (Mistral 7B Instruct on LiveCodeBench, 359 included questions) with
N = 24,000,000 W's instead of 10,000: 2,400 chunks of 10,000 W's each, drawn in log space from
`SeedSequence(20260606, spawn_key=(1 + i,))` for chunk i, and each elected at K = 4, 32, 128 and 512 with an
even split of exact ties.

**The fast check** (default 50 seeded chunks of the 2,400, about 4 s each on one core):

    python -m onetemp.robustness.fig4b check \
        --input $C/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl --profiles fits/lcb_mistral_qmax.json

- It recomputes a seeded sample of chunks (`--n-sample`, default 50; `--sample-seed`, default 0; or pick them with
  `--chunks 0,17,...`), compares each with the paper's run of the same chunk (the 2,400 tallies in
  `data/reference/fig4b_chunk_tallies.jsonl`; `--reference` names another file), and then prints the panel merged
  from all 2,400 shipped tallies, with the KL > 3 and KL 1-2 shares of Section 4.3.2.
- The comparison covers the hash of each chunk's 10,000 W's and of their KL values, the band counts and every
  tally. It ends with `PASS` (exit status 0) when every recomputed chunk is equal, `FAIL` otherwise.
- The calibration table and the band edges must equal the paper's, or the check stops. The "context hash" (it
  covers the fitted pass@K matrices to the last bit) equals the paper's only with the kernel built with
  `--build-flags paper`, the paper's `-march=native` build: build it with
  `python -m onetemp.core.fit_all --build-kernel --build-flags paper --kernel PATH`, fit with
  `ONETEMP_BERNSTEIN_KERNEL=PATH` set, and run the check on that fit. It is informational: the chunk tallies
  decide.
- `--out FILE` writes the comparison and the merged panel as JSON.

**The full run** (2,400 chunks, about 4 s each, so about 2.7 core-hours; about 180 MB of memory per process).
Run W workers in parallel, each `worker --worker w --n-workers W`; worker w takes the chunks w, w + W, ...:

    for w in 0 1 2 3 4 5 6 7; do
      python -m onetemp.robustness.fig4b worker \
          --input $C/Mistral-7B-Instruct-v0.3__temperature__question_n.jsonl \
          --profiles fits/lcb_mistral_qmax.json --tally-dir out/fig4b/tallies --worker $w --n-workers 8 &
    done; wait
    python -m onetemp.robustness.fig4b merge --tally-dir out/fig4b/tallies --expect-chunks 2400 \
        --out out/fig4b/cells.json --compare-dir data/reference/fig4b_chunk_tallies.jsonl

- **Worker:** each chunk's tally is written to its own file `chunk_NNNN.json` in `--tally-dir`. A worker skips
  every chunk whose tally already matches its inputs, so a stopped run resumes. Other options: `--K`,
  `--seed`, `--n-chunks` (2400), `--chunk-size` (10000), `--only-chunks`.
- **Merge:** sums the tallies into the panel's 24 cells and the "KL > 3" and "KL 1-2" shares quoted in
  Section 4.3.2, and requires exactly chunks 0..N-1 with `--expect-chunks`. `--compare-dir` (a folder of tallies
  or a `.jsonl` file) compares the run chunk by chunk with another run, for example the paper's.

### Appendix H: beta* against D (Figures 29-31), with exact certification, and Figure 4

    python -m onetemp.robustness.protected --input $C/Mistral-7B-Instruct-v0.3__temperature__cell_n.jsonl \
        --profiles fits/lcb_mistral_perset.json --out out/H/sym/lcb_mistral --plot
    python -m onetemp.robustness.plot_prot figure4 out/H/sym/lcb_mistral/beta_star.json \
        out/fig4b/cells.json out/figure4.pdf

**Defaults:**
- losses of 10% and 1% (`--thr 0.9,0.99`);
- K in {4, 32, 128, 512} (`--K`);
- the paper's 23 values of D from 0 to 6 (`--D`);
- beta on the grid 1.005^i, capped at 10^6.

**Outputs:**
- `beta_star.json`:
  - every point, with beta*, the winning tau_c, the window counts and the reason for each "> 10^6";
  - `beta_star_sweep` (after the LP sweep) and `beta_star` (after the exact recheck);
  - the certificate counts;
  - the reachability of each curve.
- `witness.npz`: the sweep's W at every finite point.
- `beta_star_vs_D.jpg` with `--plot`.

**How to read the arrays.** A beta* above 10^6 is written `Infinity`. A point after a curve's stop is `NaN`.
Python's json reads both. `c_empty` marks the points where no setting loses that much, which have no curve to
draw.

Other options of `protected`: `--window sym/hard/easy`, `--hardness best/tauA`, `--no-certify`, `--title`, and
the input options `--benchmark`, `--model`, `--axis`, `--tau` and `--rates fit/measured` (`fit`: the fitted
profiles; `measured`: c/n and the unbiased pass@K). `worst_loss`, `tilt` and `subsets` take the same input
options; `heatmap` takes them too (its `--rates` is set by the preset).

**Other tests of Appendix H:** (`C.jsonl` is a counts file, `F.json` its fit, `S` a study name)

    # one side at a time: only the harder (or easier) questions, more than d decades from K, are reweighted
    python -m onetemp.robustness.protected --input C.jsonl --profiles F.json --window hard \
        --D 0,0.25,0.5,0.75,1,1.5,2,2.5,3 --out out/H/hard/S
    python -m onetemp.robustness.protected --input C.jsonl --profiles F.json --window easy \
        --D 0,0.25,0.5,0.75,1,1.5,2,2.5,3 --out out/H/easy/S
    #   add --hardness tauA to measure each question's hardness at tau_A instead of at its own best setting
    # the role of K: beta* at D = 0 for every power of 2 from 1 to 1024 and four losses
    python -m onetemp.robustness.protected --input C.jsonl --profiles F.json \
        --K 1,2,4,8,16,32,64,128,256,512,1024 --thr 0.8,0.9,0.95,0.99 --D 0 --out out/H/roleK/S
    python -m onetemp.robustness.prot_summary role_of_K out/H/roleK/*/beta_star.json
    # counts, slopes and bends over the twelve symmetric runs
    python -m onetemp.robustness.prot_summary counts out/H/sym/*/beta_star.json
    python -m onetemp.robustness.prot_summary slopes out/H/sym/*/beta_star.json
    # the worst-case loss R(beta, D), exact (the no-window R is added)
    python -m onetemp.robustness.worst_loss --input C.jsonl --profiles F.json --beta 3/2,2,3,10,30,100,1000 \
        --D 0,0.25,0.5,1,1.5,2,3 --window sym,easy,hard --out out/H/R/S
    # tilts by hardness
    python -m onetemp.robustness.tilt --input C.jsonl --profiles F.json --out out/H/tilt/S
    # natural sub-datasets
    python -m onetemp.robustness.subsets --input C.jsonl --profiles F.json --labels labels.jsonl --out out/H/subsets/S
    # simulated hardness distributions (templates: Mistral 7B Instruct on LiveCodeBench); add --certify to certify
    python -m onetemp.robustness.simulate --input $C/Mistral-7B-Instruct-v0.3__temperature__cell_n.jsonl \
        --profiles fits/lcb_mistral_perset.json --shape peaked --out out/H/sim/peaked --plot

- **`worst_loss`:** `--K` (default 1, 4, 16, 32, 64, 128, 256, 512, 1024), `--beta` (rationals >= 1),
  `--D`, `--window` (comma-separated `sym`, `easy`, `hard`). Reported R is an exact rational, so "R <= 1%"
  and "R = 0" are decided exactly.
- **`tilt`:** every weight is tilted by 10^(gamma (log10 H_q - log10 K)); `--K` defaults to every power of 2
  from 1 to 1024.
- **`subsets`:** `--K` (default 1, 4, 16, 64, 256, 1024), `--n-min` (default 20 questions per sub-dataset).
- **`simulate`:** `--shape` is `peaked`, `reversed` or `flat` (1,000 questions each, hardness up to 500,000),
  `--K` defaults to 4, 32, 512, 10^4, 10^5, 10^6, and `--D` to the paper's grid up to 7.5 (26 values).
- **`prot_summary`:** `counts`, `slopes` and `role_of_K`. `role_of_K` reads each question's hardness from the
  runs themselves. For runs made before that field existed, `--x-from` takes a folder of `profiles.jsonl` files
  instead, one per study. `--out FILE` writes the full result as JSON.
- **The labels file of `subsets`** has one line per question:

      {"benchmark": ..., "id": ..., "labels": {"subject": "Algebra", "level": 3}}

  The labels come from the public datasets' own metadata, and carry no answers:
  - MATH-500: `subject`, `level`;
  - Hi-ToM: order, deception, story length, prompting type;
  - LiveCodeBench: difficulty, platform, release version.

### Appendix I: the min_p versions (Figures 42-45)

    P=data/counts/hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl
    python -m onetemp.robustness.heatmap --input $P --axis min_p --tau 1.5 \
        --preset paper-i --K 4,32,128,256 --out out/I/heat_q15 --plot
    python -m onetemp.robustness.protected --input $P --axis min_p --tau 1.5 \
        --rates measured --K 4,32,128,256 --out out/I/prot_q15
    python -m onetemp.robustness.plot_prot study out/I/prot_q15/beta_star.json out/I/prot_q15_cut.jpg --cut

- **Studies:** Qwen3-0.6B at tau = 1.5 (`__min_p_tau1.5__nu500.jsonl`) and at tau = 2.0
  (`__min_p_tau2.0__nu400.jsonl`), and Mistral 7B Instruct at tau = 2.0 (`__min_p_tau2.0__nu1500.jsonl`).
- **K:** Qwen3-0.6B uses K = 4, 32, 128, 256; Mistral 7B Instruct uses K = 4, 32, 128, 512.
- **Unbiased estimator:** it needs K <= the number of samples.
- **Cut panels:** `--cut` draws D up to the last value before any curve passes 10^6, as Figure 45 does.

## Reproduce the paper's runs

`python -m onetemp.reproduce` runs every command above with the paper's settings on the shipped counts and
compares the result with the paper's own runs in `data/reference/`:

    python -m onetemp.reproduce list                 # the groups, what each covers, and its run time
    python -m onetemp.reproduce fig4b                # Figure 4(b): 50 of the 2,400 chunks (--n-sample N for more)
    python -m onetemp.reproduce heatmaps --jobs 4    # Figures 17-28 and 42-44
    python -m onetemp.reproduce beta --jobs 4        # Figures 4(a), 29-31 and 45
    python -m onetemp.reproduce simulated            # Figure 33
    python -m onetemp.reproduce fig17                # one item: runs what it needs (fig17-fig28, fig42-fig45, fig4a, fig4)

Options: `--out DIR` (default `results/`), `--jobs N` (independent runs at a time), `--n-sample N`, `--data DIR`
(the counts folder, default `data/counts`). A result that already exists in `--out` is reused, so a stopped run
resumes; delete the folder to recompute. The exit status is 0 when every comparison matches.

| target | what it runs and compares | data/reference file |
|---|---|---|
| `fig4b` | the seeded check of `fig4b check`, then the 24 printed cells of Figure 4(b) and the per-K KL > 3 shares of the paper's run (the text group compares the two Section 4.3.2 numbers) | `fig4b_chunk_tallies.jsonl` |
| `heatmaps` | all twelve Appendix G heatmaps (presets `paper-g-lcb` with the row-bottom edges 0.7 / 0.4 / 0.7 / 0.2 for GPT OSS 20B / Mistral 7B Instruct / Qwen3-8B / Qwen3-0.6B, and `paper-g`) and the three min_p heatmaps (`paper-i`): every cell label and count, tau_A and score_A at every K | `heatmaps.json` |
| `beta` | the 12 temperature studies (corrected parsing, per-setting fits) and the three min_p studies (measured rates): beta* at every (loss, K, D), after the LP sweep and after the exact recheck, tau_A, score_A, and the certificate counts | `beta_star.json` |
| `simulated` | the three simulated pools: beta*, the 109 templates and the questions above H = 10^4 | `simulated_pools.json` |

The fits are made once per study and kept in `--out`. Run times on one core, as `list` prints them: `fig4b` about
4 minutes, `heatmaps` about 12, `beta` about 10, `simulated` about 1; with `--jobs 3`, `heatmaps` and `beta` run
about three times faster.

## What the paper's code did, kept here on purpose

Each item below is the paper's behaviour. It is the reason the outputs reproduce the printed numbers.

**Appendices G and H use slightly different conventions** (`contexts.py`).

| | G (random reweighting) | H (protected window) |
|---|---|---|
| pass@K | 1 - exp(K log(clip(1 - p, 1e-15, 1))) | -expm1(K log1p(-P)), with P = 1 - exp(log(clip(1 - p, 1e-15, 1))) capped at 1 - 10^-12 |
| hardness | the fitted peak (`peak_pct`) | the question's best setting on the grid |

**The random reweighting needs three draw modes.** Each one is needed for one set of figures:
- `linear`: the original Gamma draws, for Figures 42-44;
- `replay`: the same draws kept in log space, for Figures 17-28;
- `log`: an exact log-space draw on another stream, for Figure 4(b).

**It also needs three ways to choose tau_B:**
- `argmax`: for Figures 42-44;
- `exact`: in log space, with ties within 1e-12 broken to the lowest setting, for Figures 17-28;
- an even split of exact ties (`split`): for Figure 4(b).

**The paper fixed the ratio bands per family:**
- LiveCodeBench: the floor rule over every W's ratio, with the bottom edge the original sampler found (pass
  `--row-bottom`);
- MATH-500 and Hi-ToM: ten bands from 0;
- min_p: nine bands from 0.1.

**The "100" label and "99.99".** A cell reads "100" only when every W of its column is in it. A share that would
round to 100.00 while some W fails prints 99.99. Labels are rounded by largest remainders so that a column sums
to 100.00, so `top_row_table.md` (each share rounded on its own) can differ by 0.01 from the figure's label of
the same cell.

**KL bands.** [0,1), ..., [4,5), [5, ln n) with ln n the entropy ceiling; W's with KL >= ln n (near one-hot) are
not counted.

**Exactness and environment.**
- The bit-for-bit match of the heatmap cells depends on two things: numpy's random streams, and the order that
  numpy's default argsort gives to questions of equal hardness.
- The reproduction below ran on numpy 2.2.6 and scipy 1.16.0, on x86-64 Linux. Every output records the
  versions it ran with.
- Other versions or CPUs can move single W's across a cell edge, but not the distributions.
- The beta* sweep uses HiGHS through scipy. Its certificates are exact whatever the solver returns.

## What is not included

- **The resampling studies of Appendix H** (the 90% intervals over resampled questions or sample counts).
  - Resampling questions is supported: `Study.take(indices)` accepts repeated indices.
  - Resampling sample counts needs the profile fit after each resample, so it is not scripted here.
- **pass@K computed exactly from the fitted pass rates.** The code does not compute Appendix H's split of the
  "> 10^6" points that had no finite beta in double precision: 117 still have none, and 131 need beta > 10^12.
  The code classifies these points on the stored double values only, as Appendix H does first: 248 have no
  finite beta, and 125 have a finite beta above 10^6.
- **Other Appendix H studies:** the group reweighting of whole sub-datasets, the adversarial pool search, and the
  theorem checks of "One side at a time".

## Reproduction record

Every number below was recomputed with this package from the shipped counts. Each "ours" value comes from the
release code. Each "paper" value is the number printed in the paper. `python -m onetemp.reproduce` runs the
groups in the table of "Reproduce the paper's runs" (Figure 4(b), the heatmaps, beta*, the simulated pools); the
further tests of Appendix H were run with the commands in "Other tests of Appendix H" on the twelve studies.

**Inputs:**
- **Temperature studies:** the per-question sample counts of the twelve (benchmark, model) studies, with the
  paper's degree-10 profile fit of Appendix D.
  - For Appendix G (and Figure 4(b)), the earlier parsing (LiveCodeBench: 359, 766, 647 and 85 included
    questions).
  - For Appendix H, the corrected parsing (359, 766, 677 and 86).
- **min_p studies:** the counts of the three min_p studies of Appendix I.

**Environment:** Python 3.12.12, numpy 2.2.6, scipy 1.16.0, x86-64 Linux. Every run used one thread per process.

### Figure 4(b) and Section 4.3.2: 24 million random reweightings

Cells: the share of W's whose chosen temperature loses under 10%, for Mistral 7B Instruct on LiveCodeBench.
- **Draws:** 2,400 chunks of 10,000 W's each.
- **Tally check:** every chunk's tally equals the paper's run of the same chunk:
  - the hash of its 10,000 W's and of their KL values;
  - the band counts and every tally;
  - 2,400 of 2,400 chunks, with a largest difference of 0.

| K | 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | 5-5.88 | KL > 3 |
|---|---|---|---|---|---|---|---|
| 512 | 99.99 | 99.85 | 98.75 | 97.12 | 93.38 | 87.97 | 93.00 |
| 128 | 100 | 99.79 | 98.39 | 96.34 | 91.31 | 84.03 | 90.80 |
| 32 | 100 | 99.91 | 98.63 | 96.59 | 92.01 | 85.16 | 91.48 |
| 4 | 99.92 | 98.04 | 93.47 | 87.19 | 78.76 | **69.40** | 78.80 |

| the paper | ours | difference | where |
|---|---|---|---|
| the 24 printed cells of Figure 4(b) | the table above | 0 in all 24 | Figure 4(b) |
| K = 4, KL 5-5.88: 69.40 | 69.40 | 0 | Figure 4(b) |
| KL > 3, averaged over K: "on average roughly 90%" | 88.52% | rounds to the printed 90% | Section 4.3.2 |
| KL 1-2: "rises to 99%" | 99.40% | rounds to the printed 99% | Section 4.3.2 |
| KL > 3, per K (not printed; the paper's run): 78.80 / 91.48 / 90.80 / 93.00 | 78.80 / 91.48 / 90.80 / 93.00 | 0 | the paper's chunk tallies |

### Appendix G: the twelve heatmaps (Figures 17-28)

Each run used 10,000 W's at seed 20260606, with the log-space draws and the exact election.

| paper number | ours | difference | where |
|---|---|---|---|
| all 2,328 cell labels of Figures 17-28 | the same 2,328 labels and the same cell counts | 0 | Figures 17-28 |
| LiveCodeBench table: tau_A, score_A and "top" for 4 models x 4 K. Example, GPT OSS 20B at K = 4: 0.90, 505.86, 87.47 | all 16 rows equal. Example: 0.90, 505.86, 87.47 | 0 | Appendix G table |
| Mistral 7B Instruct, top row in the 5-5.88 band: 69.71, 84.18, 83.31, 87.87 | 69.71, 84.18, 83.31, 87.87 | 0 | Figure 17 caption, table |
| W's not drawn: GPT OSS 20B 97, Mistral 7B Instruct 167, Qwen3-0.6B 2,005 | 97, 167, 2,005 | 0 | "Experiment" |
| W's not drawn on Hi-ToM: 392 (GPT OSS 20B), 91 (Qwen3-0.6B) | 392, 91 | 0 | "Experiment" |
| MATH-500, lowest top row in the highest band: 92.31 | 92.31 (Mistral 7B Instruct, K = 128) | 0 | "Findings on MATH-500" |
| Mistral 7B Instruct on MATH-500, by K: 93.43, 97.48, 92.31, 95.36 | 93.43, 97.48, 92.31, 95.36 | 0 | same |
| Hi-ToM, K = 4: from 78.31 (Mistral) to 87.76 (Qwen3-8B) | 78.31 to 87.76 | 0 | same |
| Hi-ToM, K = 512: 99.94 (Mistral, GPT OSS 20B), 100 (Qwen3-8B, Qwen3-0.6B) | 99.94, 99.94, 100, 100 | 0 | same |

### Appendix I: the min_p heatmaps (Figures 42-44)

These runs used the original draws and election, and the unbiased pass@K.

| paper number | ours | difference | where |
|---|---|---|---|
| all 648 cell labels of Figures 42-44 | the same 648 labels and counts | 0 | Figures 42-44 |
| highest band at K = 4: 78.53, 78.40, 82.44 | 78.53, 78.40, 82.44 | 0 | "Reweighting the questions" |
| highest band at the largest K: 99.88, 100, 98.04 | 99.88, 100, 98.04 | 0 | same |
| empty cells: 163, 178 and 162 of 216 | 163, 178, 162 | 0 | same |
| lowest band holding a W: 0.1-0.2, 0.7-0.8, 0.5-0.6 | 0.1-0.2, 0.7-0.8, 0.5-0.6 | 0 | same |

### Appendix H and Figure 4(a): beta* against D

- **Grid:** 12 studies x 2 losses x 4 K x 23 values of D.
- **The LP sweep and the exact recheck:** every beta* equals the paper's value in `data/reference/beta_star.json`,
  at all 2,208 entries, including the "> 10^6" and stop entries. Figure 4(a) is the Mistral 7B Instruct
  LiveCodeBench subset at a 10% loss.

| paper number | ours | difference | where |
|---|---|---|---|
| all 1,129 finite points certified, both sides | 1,129 finite; every one has an exact W at beta* itself and a lambda proof one grid step below | 0 | "Exact certification" |
| all 373 points with beta* > 10^6 and questions outside proven | 373, all proven | 0 | same |
| of these, 248 with no finite beta (double precision) | 248, each with a lambda proof valid at every beta | 0 | "Points with beta* > 10^6" |
| and 125 with a finite beta above 10^6 | 125, each with an explicit W | 0 | same |
| GPT OSS 20B on MATH-500, 10% loss: no reachable loss at K = 32, 128 and 512 | the same three curves are unreachable | 0 | Figure 30 caption |
| 88 curves with >= 5 finite points; median slope per dataset and loss between 1.36 and 2.39 | 88 curves; medians 1.359 to 2.385 | 0 at the printed precision | "beta* against D" |
| slopes at K = 4 and a 10% loss between 0.9 and 2.5 | 0.923 to 2.499 | 0 at the printed precision | same |
| 42 of 88 bent curves, 35 of them steepening | 42 of 88, 35 steepening | 0 | same |

**The loss of the chosen temperature, R(beta, D).** Computed exactly; every point certified.

| paper number | ours | difference | where |
|---|---|---|---|
| R at beta = 10, D = 1.5, max over 12 studies and K = 4 ... 1024: 0.94% | 0.9361% (LiveCodeBench, Mistral 7B Instruct, K = 4) | 0 at the printed precision | "The loss of the chosen temperature" |
| no window: 18% to 36% in the median study at each K | 18.03% to 36.35% | 0 at the printed precision | same |
| no window, LiveCodeBench Qwen3-0.6B: more than 56% | at least 56.37% at every K >= 4 | 0 | same |
| D = 2.5: R exactly 0 at every K on the grid from 4 to 1024 | exactly 0 (rational) at all 96 points | 0 | same |
| K = 1, D = 1.5: R = 7.9% | 7.856% | 0 at the printed precision | same |

**One side at a time.** K in {4, 32, 128, 512}, d in {0, 0.25, ..., 3}.

| paper number | ours | difference | where |
|---|---|---|---|
| harder side: 97.5% of the 315 points with d <= 1 and >= 5 outside | 307 of 315 (97.46%), once the 4 curves with no reachable loss are left out, as the paper does | 0 at the printed precision | "One side at a time" |
| slope: median 1.20, IQR 0.87-1.44 over 67 curves | 1.20, IQR 0.87-1.44, 67 curves | 0 | same |
| easier side, d = 1, 10% loss: 22 of 43 flip | 22 of 43 | 0 | same |
| from d = 2.5, 10% loss: 0 of 34 | 0 of 34 | 0 | same |
| from d = 2.5, 1% loss: 1 of 34 | 1 of 34 | 0 | same |
| the 28 flips with d >= 1 at a 10% loss all promote a colder temperature | 28 flips, all colder | 0 | same |
| hardness at tau_A, more than 1.5 decades easier: never flips, at losses of 1, 5, 10 and 20%, for K = 1 to 1024 | 0 flips at each loss, over the 82 points where some question is reweighted | 0 | same |
| with 1 decade: holds at 101 of 102 points at a 10% loss; 60 of 102 flip at a 1% loss | 1 of 102 flips at 10% (101 hold); 60 of 102 flip at 1% | 0 | same |

**Tilting the mix** (132 curves: 12 studies x 11 K).

| paper number | ours | difference | where |
|---|---|---|---|
| gamma = 0.25 moves the best temperature colder in none of 132 | 0 | 0 | "Tilting the mix" |
| gamma = -0.25 moves it hotter in none of 132 | 0 | 0 | same |
| nondecreasing in gamma (from -1 to 1) in 124 of 132; the other 8 all on MATH-500 | 124; the 8 others all on MATH-500 | 0 | same |
| toward harder questions, tau_A first loses 1% at a median factor of 17 per decade | 17.2 | 0 at the printed precision | same |
| ... about 3 at K = 1 and about 2,100 at K = 1024 | 2.99 and 2,089 | 0 at the printed precision | same |
| toward easier questions, a 1% loss within a factor 10^4 in 23 of 132 | 23 of 132 | 0 | same |

**Natural sub-datasets.**

| paper number | ours | difference | where |
|---|---|---|---|
| 448 combinations of sub-dataset and model; 2,688 cases | 448; 2,688 | 0 | "Natural sub-datasets" |
| loses at least 10% in 15 cases (0.56%), all at K <= 4 | 15 (0.56%), all at K = 1 or 4 | 0 | same |
| loses at least 1% in 12.5% | 12.5% | 0 | same |
| median loss below 0.1% on every dataset | 0.067%, 0.079% and below 1e-6% | 0 | same |
| share losing >= 1%: 24.8% at K = 1, 4.5% at K = 64, 5.8% at K = 256, 5.1% at K = 1024 | 24.8%, 4.5%, 5.8%, 5.1% | 0 | same |

**The role of K.** beta*(D = 0) at K = 1, 2, ..., 1024, at losses of 1, 5, 10 and 20%.

| paper number | ours | difference | where |
|---|---|---|---|
| median Spearman correlation of beta*(D = 0) with K: 0.90, over 48 combinations | 0.902 over 48 | 0 at the printed precision | "The role of K" |
| never falls as K grows in only 7 of 48 | 7 of 48 | 0 | same |
| with the share of questions easier than K: Spearman 0.76 to 0.82 | 0.762 to 0.822 | 0 at the printed precision | same |
| with log10 K: 0.51 to 0.63 | 0.510 to 0.630 | 0 at the printed precision | same |

**Simulated hardness distributions** (Figures 32-33).

| paper number | ours | difference | where |
|---|---|---|---|
| 109 template questions with hardness between 10^2 and 10^3 | 109 | 0 | "Simulated hardness distributions" |
| 966, 985 and 980 of the 1000 questions above H = 10^4 | 966, 985, 980 | 0 | same |
| beta* of all three pools (2 losses x 6 K x 26 D) | identical to the paper's run in `data/reference/simulated_pools.json`, all 312 entries per pool | 0 | Figure 33 |
| beta*(D = 0) for K <= 512: 2.0 to 2.5 at a 10% loss, 1.1 to 1.5 at a 1% loss | 2.00 to 2.54 and 1.10 to 1.48 | 0 at the printed precision | same |

### Appendix I: beta* against D for min_p (Figure 45)

| paper number | ours | difference | where |
|---|---|---|---|
| beta* of the three studies (2 losses x 4 K x 23 D) | identical to the paper's arrays in `data/reference/beta_star.json` | 0 | Figure 45 |
| with no window, a factor of 1.78 to 237 for a 10% loss | 1.78 to 236.59 | 0 at the printed precision | "Reweighting outside a protected window" |
| ... and 1.11 to 10.27 for a 1% loss | 1.11 to 10.27 | 0 | same |
| beta* exceeds 10^6 from D = 0.8 to 1.75 (10% loss) and 1.25 to 2.0 (1% loss) | 0.8 to 1.75 and 1.25 to 2.0 | 0 | same |
| Qwen3-0.6B at tau = 2.0, 10% loss, K = 256: C is empty | C is empty at every D | 0 | Figure 45 caption |
| exact certification of every point (the same rule as Appendix H) | 93, 81 and 90 finite points, each with a W at beta* and a lambda proof below; 17, 8 and 22 "> 10^6" points proven | none unproven | Figure 45 |

## Not reproduced here, and why

- **The resampling intervals of Appendix H.** These are the 90% intervals over resampled questions and over
  resampled sample counts. Resampling the counts needs the profile fit after each resample.
- **pass@K computed exactly from the fitted pass rates.** These are the 117 / 131 split of the points with no
  finite beta, and the beta > 10^12 statements. Only the double-precision classification is implemented.
- **The 1.12 slope "over the 368 such curves of the larger grid".** The one-sided runs above use the paper's four
  K only.
- **Group reweighting of whole sub-datasets, the adversarial pool, and the theorem checks of Appendix H.** These
  are not implemented.
- **Other Appendix H numbers that were not recomputed for this release:** R between the grid points of K (1.34% at
  K = 5, and at most 0.18% at every integer K from 4 to 1024 with D = 2.5); the 339 of 3,219 grid points with
  beta < beta* where R reaches 10%; the one-sided check on simulated pools (4,177 points, and the flip at
  K = 4096); and the 807 of 1,526 sub-dataset cases whose best temperature moves the way the hardness shift
  predicts. These runs are not scripted here.
