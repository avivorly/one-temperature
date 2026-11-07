"""The paper's per-benchmark settings for the profile figures, and the numbers it prints (for ``repro``).

Everything here is read off the paper (text, captions, figure legends); nothing is fitted.
"""
from __future__ import annotations

#: Benchmarks as written in counts.jsonl.
BENCHMARKS = ("livecodebench", "math500", "hitom")

#: Models in panel order (a)-(d) of Figures 5-13: label, HF repo id, file-name stem of the paper's images.
MODELS = [
    ("GPT OSS 20B", "openai/gpt-oss-20b", "GPT_oss_20B"),
    ("Mistral 7B Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral_7B"),
    ("Qwen3-8B", "Qwen/Qwen3-8B", "Qwen3_8B"),
    ("Qwen3-0.6B", "Qwen/Qwen3-0.6B", "Qwen3.6B"),
]

#: Image-name stems of the paper's LiveCodeBench figures (they carry a dataset suffix).
LCB_IMAGE_STEM = {"GPT_oss_20B": "GPT_oss_20B", "Mistral_7B": "Mistral_7B",
                  "Qwen3_8B": "Qwen3_8B_complete", "Qwen3.6B": "Qwen3.6B_filtered"}
IMAGE_DATASET_TAG = {"livecodebench": "", "math500": "_MATH500", "hitom": "_HiToM"}

#: Low-hardness advance of the drawn sliding-mean lines (Appendix E.1): on for LiveCodeBench and MATH-500,
#: off for Hi-ToM ("The Hi-ToM lines use every full window").
ADVANCE_LOW_H = {"livecodebench": True, "math500": True, "hitom": False}

#: Sample-count rule fed to the profile fit for the figures (see fitting.py).  On MATH-500 and Hi-ToM every
#: setting has the same n, so both rules give the same fit.
N_MODE = {"livecodebench": "question-max", "math500": "per-setting", "hitom": "per-setting"}

#: Figure 1: the three highlighted Qwen3-8B LiveCodeBench questions, by LiveCodeBench question_id, and the
#: legend labels the figure prints for them (title and LiveCodeBench release).
FIG1_HIGHLIGHT = ["abc393_e", "abc394_c", "2730"]
FIG1_LABELS = ["GCD of Subset (v6)", "Debug (v6)", "maximum-or (v1)"]

#: The files of the data export (under its counts/ folder) that the paper's profile analyses read.
#: LiveCodeBench: the question_n parse (one question-level n at every temperature, as the paper's code).
#: Appendix I: the min_p files with 500 (tau = 1.5) and 400 / 1,500 (tau = 2.0) samples per cell.
DATA_FILES_TEMPERATURE = {
    "livecodebench": "livecodebench/{model}__temperature__question_n.jsonl",
    "math500": "math500/{model}__temperature.jsonl",
    "hitom": "hitom/{model}__temperature.jsonl",
}
DATA_FILES_MINP = [
    "hitom/Qwen3-0.6B__min_p_tau1.5__nu500.jsonl",
    "hitom/Qwen3-0.6B__min_p_tau2.0__nu400.jsonl",
    "hitom/Mistral-7B-Instruct-v0.3__min_p_tau2.0__nu1500.jsonl",
]


def data_files(counts_dir) -> list[str]:
    """The counts files the paper's profile analyses read, in the data export's layout (existing ones)."""
    from pathlib import Path
    root = Path(counts_dir)
    out = []
    for _, repo, _ in MODELS:
        for bench in BENCHMARKS:
            p = root / DATA_FILES_TEMPERATURE[bench].format(model=repo.split("/")[-1])
            if p.exists():
                out.append(str(p))
    out += [str(root / f) for f in DATA_FILES_MINP if (root / f).exists()]
    return out

#: Appendix I studies, in the order whose index seeds the noise checks (SeedSequence(root).spawn(3)).
MINP_ROOT_SEED = 20261104
MINP_STUDIES = [
    # (index, model repo id, tau, image stem)
    (0, "Qwen/Qwen3-0.6B", 1.5, "Qwen3.6B_HiToM_minp_T1.5"),
    (1, "Qwen/Qwen3-0.6B", 2.0, "Qwen3.6B_HiToM_minp_T2.0"),
    (2, "mistralai/Mistral-7B-Instruct-v0.3", 2.0, "Mistral_7B_HiToM_minp_T2.0"),
]


def image_stem(benchmark: str, model_stem: str) -> str:
    """The paper's image stem, e.g. Report_Qwen3_8B_MATH500 (+ _AveTemp.jpg)."""
    if benchmark == "livecodebench":
        return f"Report_{LCB_IMAGE_STEM[model_stem]}"
    return f"Report_{model_stem}{IMAGE_DATASET_TAG[benchmark]}"


# ----------------------------------------------------------------------------------------------------
# Printed numbers.  Keys: (benchmark, model stem).  Values: what the paper prints, as printed, and where.
# ----------------------------------------------------------------------------------------------------

#: Included questions n (captions of Figures 5-13; Figure 2 legend for Qwen3-8B on LiveCodeBench).
PRINTED_N = {
    ("livecodebench", "GPT_oss_20B"): 766, ("livecodebench", "Mistral_7B"): 359,
    ("livecodebench", "Qwen3_8B"): 647, ("livecodebench", "Qwen3.6B"): 85,
    ("math500", "GPT_oss_20B"): 500, ("math500", "Mistral_7B"): 455,
    ("math500", "Qwen3_8B"): 496, ("math500", "Qwen3.6B"): 492,
    ("hitom", "GPT_oss_20B"): 678, ("hitom", "Mistral_7B"): 615,
    ("hitom", "Qwen3_8B"): 466, ("hitom", "Qwen3.6B"): 743,
}

#: Figure legends (AveTemp: n with peak tau > 0.1; RangeSize: OLS slope, intercept, R^2 to 3 decimals).
PRINTED_LEGENDS = {
    ("livecodebench", "GPT_oss_20B"): {"n_red": 602, "slope": -0.249, "intercept": 1.180, "r2": 0.494},
    ("livecodebench", "Mistral_7B"): {"n_red": 326, "slope": -0.106, "intercept": 0.927, "r2": 0.265},
    ("livecodebench", "Qwen3_8B"): {"n_red": 542, "slope": -0.451, "intercept": 1.915, "r2": 0.601},
    ("livecodebench", "Qwen3.6B"): {"n_red": 82, "slope": -0.088, "intercept": 0.750, "r2": 0.184},
    ("math500", "GPT_oss_20B"): {"n_red": 416, "slope": -0.234, "intercept": 1.136, "r2": 0.549},
    ("math500", "Mistral_7B"): {"n_red": 351, "slope": -0.048, "intercept": 0.964, "r2": 0.023},
    ("math500", "Qwen3_8B"): {"n_red": 464, "slope": -0.393, "intercept": 1.332, "r2": 0.768},
    ("math500", "Qwen3.6B"): {"n_red": 351, "slope": -0.184, "intercept": 1.092, "r2": 0.355},
    ("hitom", "GPT_oss_20B"): {"n_red": 672, "slope": -0.183, "intercept": 1.433, "r2": 0.213},
    ("hitom", "Mistral_7B"): {"n_red": 614, "slope": -0.276, "intercept": 1.529, "r2": 0.536},
    ("hitom", "Qwen3_8B"): {"n_red": 466, "slope": -0.753, "intercept": 2.841, "r2": 0.548},
    ("hitom", "Qwen3.6B"): {"n_red": 739, "slope": -0.088, "intercept": 1.259, "r2": 0.068},
}

#: Appendix E.2 text: red sliding average, first and last drawn value (2 decimals).
PRINTED_RED_START_END = {
    ("math500", "GPT_oss_20B"): (0.74, 0.90), ("math500", "Mistral_7B"): (0.49, 0.70),
    ("math500", "Qwen3.6B"): (0.61, 0.85), ("math500", "Qwen3_8B"): (1.14, 0.99),
    ("hitom", "GPT_oss_20B"): (0.93, 1.50), ("hitom", "Mistral_7B"): (1.17, 1.85),
    ("hitom", "Qwen3_8B"): (1.89, 3.19), ("hitom", "Qwen3.6B"): (0.80, 1.52),
}

#: Appendix E.2 text: MATH-500 coverage bars start at 1 and reach their lowest value (3 decimals) in one of
#: the two highest-H bins.
PRINTED_MATH500_COVERAGE_MIN = {"GPT_oss_20B": 0.500, "Mistral_7B": 0.667, "Qwen3_8B": 0.333, "Qwen3.6B": 0.625}

#: Appendix E.2 text: every Hi-ToM coverage bar lies in [0.83, 1]; Figure 13 caption: GPT OSS 20B and
#: Qwen3-8B have 16 bars (one empty bin), the others 17.
PRINTED_HITOM_COVERAGE_RANGE = (0.83, 1.0)
PRINTED_HITOM_BARS = {"GPT_oss_20B": 16, "Mistral_7B": 17, "Qwen3_8B": 16, "Qwen3.6B": 17}

#: Appendix E.1: questions at the clamp H = 0.0869 exist on LiveCodeBench and MATH-500, none on Hi-ToM;
#: "This removes questions at H = 0.0869 and at most two other questions per line."
PRINTED_MAX_EXTRA_DROPPED = 2

#: Figure 1: the image legend reads "Other sampled questions (n=44)" next to the 3 highlighted questions.
PRINTED_FIG1 = {"legend_background": 44, "highlighted": 3}

#: Appendix I numbers (study index as MINP_STUDIES).
PRINTED_MINP = {
    "peak_slope_all": {0: -3.10, 1: -1.98, 2: -1.51},
    "peak_slope_subset": {0: -2.86, 1: -1.35, 2: -1.48},
    "peak_null_central95_within": (-0.60, 0.54),
    "range_slope": {0: -2.88, 1: -1.69, 2: -1.96},
    "range_null_median": {0: -3.25, 1: -1.43, 2: -1.20},
    "range_null_at_least_as_steep": {1: 2, 2: 0},
    "range_null_flatter_than": {0: 398},
    "relc_lowest_bar_bins_ge10": {0: 0.73, 1: 0.70, 2: 0.70},
    "relc_null_at_least_as_steep": {0: 400, 1: 41, 2: 400},
    "peak_at_smallest": {0: 33},
    "peak_at_largest": {0: 24, 1: 133, 2: 119},
    "no_correct_at_largest": {0: 576},
    "included": {0: 648, 1: 648, 2: 598},
    "excluded_no_correct": {2: 8},
    "excluded_single_correct": {2: 11},
    "bins_above_H100_questions": {0: 9, 1: 2, 2: 68},
    "highlighted_peak_percent_range": (26.5, 37.1),
}
