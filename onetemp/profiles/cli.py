"""Command-line interface: ``python -m onetemp.profiles <command> --help``.

Commands
--------
fit       fit every question's profile (core package's Bernstein fit) and write profiles.jsonl
table     per-question table: peak setting, peak pass rate, efficient range, hardness, included flag
summary   per-dataset numbers of the three hardness views (JSON)
figures   Figures 5-13 style images (AveTemp, RangeSize, relC) for one (benchmark, model)
figure2   the three-panel Figure 2 for one (benchmark, model)
example   Figure 1 (example profiles with P_E^1, P_E^2, P_C)
minp      Appendix I: the analyses on a min_p grid with the split-half and null checks, Figures 38-41
repro     run everything the paper shows and compare with its printed numbers (repro_table.md)

Inputs are the shared ``counts.jsonl`` files.  Fitted profiles are computed with the core package's fit,
or read from ``--profiles`` (a profiles.jsonl written by ``fit``).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from . import analysis, fitting, paper, plots
from . import io as qio


def _counts_files(patterns: list[str]) -> list[str]:
    files = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        files.extend(hits if hits else [p])
    if not files:
        raise SystemExit("no counts files given")
    return files


def _add_study_args(ap: argparse.ArgumentParser, need_model: bool = True) -> None:
    ap.add_argument("--counts", nargs="+", required=True, help="counts.jsonl file(s) or glob(s)")
    ap.add_argument("--benchmark", choices=paper.BENCHMARKS, required=True)
    ap.add_argument("--model", required=need_model,
                    help="HF repo id, or a unique case-insensitive part of it (e.g. 'Qwen3-8B')")
    ap.add_argument("--profiles", default=None, help="profiles.jsonl with fitted profiles (default: fit now)")
    ap.add_argument("--n-mode", default="auto", choices=["auto", "question-max", "per-setting"],
                    help="samples per setting given to the fit; auto = the paper's choice per benchmark")
    ap.add_argument("--min-peak", type=float, default=0.1,
                    help="inclusion floor on the fitted peak pass rate, in percent (paper: 0.1 = 1e-3)")
    ap.add_argument("--source", default="fit", choices=["fit", "measured"],
                    help="fit: the fitted profiles (the paper); measured: measured rates c/n with linear "
                         "interpolation between settings, no fit (the paper's code offered it as an option)")


def _load(args):
    studies = qio.load_counts(_counts_files(args.counts))
    st = qio.select_study(studies, benchmark=args.benchmark, model=args.model, axis="temperature")
    n_mode = paper.N_MODE[st.benchmark] if args.n_mode == "auto" else args.n_mode
    if getattr(args, "source", "fit") == "measured":
        return st, None, n_mode
    profs = fitting.profiles_for_study(st, args.profiles, n_mode=n_mode)
    return st, profs, n_mode


def _rows(st, profs, n_mode):
    """Per-question rows from the fitted profiles, or from measured rates when ``profs`` is None."""
    if profs is None:
        return analysis.question_table_measured(st, n_mode=n_mode)
    return analysis.question_table(st, profs)


def _advance(args, benchmark: str) -> bool:
    if args.advance == "auto":
        return paper.ADVANCE_LOW_H[benchmark]
    return args.advance == "on"


def cmd_fit(args) -> int:
    studies = qio.load_counts(_counts_files(args.counts))
    st = qio.select_study(studies, benchmark=args.benchmark, model=args.model, axis="temperature")
    n_mode = paper.N_MODE[st.benchmark] if args.n_mode == "auto" else args.n_mode
    profs = fitting.fit_study(st, n_mode=n_mode,
                              progress=(lambda i, n: print(f"\r  fitted {i}/{n}", end="", file=sys.stderr))
                              if args.verbose else None)
    qio.write_profiles(args.out, st, profs)
    print(f"\nwrote {args.out}: {len(profs)} profiles ({st.label}, n_mode={n_mode})", file=sys.stderr)
    return 0


def cmd_table(args) -> int:
    st, profs, n_mode = _load(args)
    rows = _rows(st, profs, n_mode)
    for r in rows:
        r["included"] = bool(r["peak_val"] is not None and r["peak_val"] > 0 and r["peak_val"] >= args.min_peak)
    if args.out.endswith(".json"):
        qio.write_json(args.out, rows)
    else:
        analysis.write_table_csv(args.out, rows)
    print(f"wrote {args.out}: {len(rows)} questions, {sum(r['included'] for r in rows)} included")
    return 0


def _summary(args):
    st, profs, n_mode = _load(args)
    rows = _rows(st, profs, n_mode)
    srt = analysis.included_sorted(rows, args.min_peak)
    s = analysis.summarize(srt, _advance(args, st.benchmark), n_bins=args.bins)
    s.update({"benchmark": st.benchmark, "model": st.model, "n_mode": n_mode, "source": args.source,
              "n_questions": len(rows), "min_peak_percent": args.min_peak})
    return st, srt, s


def cmd_summary(args) -> int:
    st, srt, s = _summary(args)
    qio.write_json(args.out, s)
    print(json.dumps(analysis.compact_summary(s), indent=1))
    return 0


def cmd_figures(args) -> int:
    st, srt, s = _summary(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"{st.benchmark}_{st.model.split('/')[-1]}"
    written = plots.draw_appendix_set(srt, s["advance_low_h"], str(out_dir / stem),
                                      n_bins=s["coverage"]["n_bins"], titles=not args.no_titles)
    qio.write_json(out_dir / f"{stem}_summary.json", s)
    for k, p in written.items():
        print(f"wrote {p}")
    return 0


def cmd_figure2(args) -> int:
    st, srt, s = _summary(args)
    out = plots.figure2_row(srt, s["advance_low_h"], args.out, n_bins=s["coverage"]["n_bins"])
    print(f"wrote {out}")
    return 0


def cmd_example(args) -> int:
    from . import profile_examples
    st, profs, n_mode = _load(args)
    if profs is None:
        raise SystemExit("example draws fitted profiles; use --source fit")
    data = profile_examples.build_records(st, profs, n_mode)
    if args.highlight:
        highlight, labels = args.highlight, args.labels
    else:
        highlight, labels = paper.FIG1_HIGHLIGHT, (args.labels or paper.FIG1_LABELS)
    rec = profile_examples.plot_profile_examples(data, highlight, args.out, bg_alpha=args.alpha,
                                                 n_background=args.n_background, seed=args.seed,
                                                 labels=labels)
    print(json.dumps({k: v for k, v in rec.items() if k != "background"}, indent=1))
    if args.record:
        qio.write_json(args.record, rec)
    return 0


def cmd_minp(args) -> int:
    from . import grid_profiles as gp
    studies = qio.load_counts(_counts_files(args.counts))
    st = qio.select_study(studies, benchmark=args.benchmark, model=args.model, axis="min_p", tau=args.tau)
    ids, grid, n, c = st.matrices(order="id")
    if (n <= 0).any():
        raise SystemExit("every (question, min_p) cell needs n > 0 for the grid analysis")
    grid = [float(v) for v in grid]
    nv = len(grid)
    pk = gp.peaks(c, n, args.min_peak)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"{st.benchmark}_{st.model.split('/')[-1]}_minp_T{st.tau:g}"
    result = {"study": st.label, "grid_min_p": grid, "facts": gp.study_facts(ids, grid, c, n, pk)}
    with plots.paper_style():
        _, result["peak_figure"] = gp.fig_grid_peak(ids, grid, pk, out_dir / f"{stem}_Peak.jpg")
        _, result["range_figure"] = gp.fig_grid_range(ids, grid, pk, out_dir / f"{stem}_RangeSize.jpg")
        _, result["coverage_figure"] = gp.fig_grid_coverage(ids, grid, pk, out_dir / f"{stem}_relC.jpg")
    _, result["profiles_figure"] = gp.fig_grid_profiles(ids, c, n, grid, pk, out_dir / f"{stem}_profiles.jpg")
    if args.checks:
        seeds = gp.paper_seeds(args.seed, args.seed_index)
        result["noise_checks"] = gp.noise_checks(c, n, nv, seeds, n_guard=args.n_guard, n_null=args.n_null,
                                                 n_fpr=args.n_fpr)
        result["noise_checks"]["_seeds"] = {"root": args.seed, "study_index": args.seed_index}
    qio.write_json(out_dir / f"{stem}_values.json", result)
    headline = {"facts": result["facts"], "range_ols": result["range_figure"]["ols"]}
    if args.checks:
        headline["trend"] = {k: {kk: result["noise_checks"][k][kk] for kk in ("full_data", "guard_pass", "null_pass",
                                                                            "trend_claimable")}
                          for k in gp.STATS}
        for k in gp.STATS:
            headline["trend"][k]["guard_frac_same_sign"] = result["noise_checks"][k]["guard"]["frac_same_sign_as_full"]
            headline["trend"][k]["null_q025_q975"] = [result["noise_checks"][k]["null"]["q025"],
                                                   result["noise_checks"][k]["null"]["q975"]]
            headline["trend"][k]["null_median"] = result["noise_checks"][k]["null"]["median"]
            headline["trend"][k]["null_at_least_as_extreme"] = \
                result["noise_checks"][k]["null"]["replicates_at_least_as_extreme"]
    print(json.dumps(headline, indent=1, default=float))
    return 0


def cmd_repro(args) -> int:
    from . import repro
    return repro.main(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m onetemp.profiles",
                                 description="Question-profile analyses of 'One Temperature to Rule Them All?' "
                                             "(Section 3; Appendices D, E, I).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fit", help="fit profiles with the core package's Bernstein fit -> profiles.jsonl")
    _add_study_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("table", help="per-question table (CSV, or JSON if --out ends in .json)")
    _add_study_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_table)

    for name, fn, hlp in (("summary", cmd_summary, "per-dataset numbers of the three hardness views"),
                          ("figures", cmd_figures, "AveTemp, RangeSize and relC images (Figures 5-13)"),
                          ("figure2", cmd_figure2, "three-panel Figure 2")):
        p = sub.add_parser(name, help=hlp)
        _add_study_args(p)
        p.add_argument("--advance", choices=["auto", "on", "off"], default="auto",
                       help="low-hardness advance of the drawn sliding means (auto: on for livecodebench and "
                            "math500, off for hitom, as the paper)")
        p.add_argument("--bins", default=17, type=lambda v: v if v == "auto" else int(v),
                       help="coverage bins: 17 (the paper's figures) or 'auto' = floor(n/20)")
        if name == "summary":
            p.add_argument("--out", required=True)
        elif name == "figures":
            p.add_argument("--out-dir", required=True)
            p.add_argument("--stem", default=None, help="file-name stem (default: <benchmark>_<model>)")
            p.add_argument("--no-titles", action="store_true", help="drop the in-image titles")
        else:
            p.add_argument("--out", required=True, help="output JPEG")
        p.set_defaults(func=fn)

    p = sub.add_parser("example", help="Figure 1: example profiles")
    _add_study_args(p)
    p.add_argument("--highlight", nargs=3, default=None, metavar="ID",
                   help=f"the three highlighted question ids (default: {paper.FIG1_HIGHLIGHT})")
    p.add_argument("--labels", nargs=3, default=None, metavar="TEXT",
                   help=f"legend texts of the highlighted questions (default with the default ids: {paper.FIG1_LABELS})")
    p.add_argument("--alpha", type=float, default=0.35,
                   help="alpha of the grey background curves (0.35 matches the published image; the original "
                        "script defaulted to 0.10)")
    p.add_argument("--n-background", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output image (PNG in the paper)")
    p.add_argument("--record", default=None, help="optional JSON with what was drawn")
    p.set_defaults(func=cmd_example)

    p = sub.add_parser("minp", help="Appendix I: profile analyses over a min_p grid (measured rates)")
    p.add_argument("--counts", nargs="+", required=True)
    p.add_argument("--benchmark", choices=paper.BENCHMARKS, default="hitom")
    p.add_argument("--model", required=True)
    p.add_argument("--tau", type=float, required=True, help="the fixed temperature of the min_p study")
    p.add_argument("--min-peak", type=float, default=0.1)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stem", default=None)
    p.add_argument("--checks", action="store_true", help="run the split-half guard and the common-shape null")
    p.add_argument("--seed", type=int, default=paper.MINP_ROOT_SEED)
    p.add_argument("--seed-index", type=int, default=0,
                   help="the study's index in the paper's order (0: Qwen3-0.6B tau 1.5, 1: Qwen3-0.6B tau 2.0, "
                        "2: Mistral tau 2.0)")
    p.add_argument("--n-guard", type=int, default=400)
    p.add_argument("--n-null", type=int, default=400)
    p.add_argument("--n-fpr", type=int, default=0,
                   help="null replicates for the guard's false-positive-rate diagnostic (paper: 100; slow)")
    p.set_defaults(func=cmd_minp)

    p = sub.add_parser("repro", help="reproduce the paper's profile figures and numbers; writes repro_table.md")
    p.add_argument("--counts-dir", default=None,
                   help="the data export's counts/ folder: picks the files the paper read (paper.data_files)")
    p.add_argument("--counts", nargs="+", default=[], help="counts.jsonl file(s) or glob(s), instead of --counts-dir")
    p.add_argument("--profiles", nargs="*", default=[], help="profiles.jsonl file(s) (default: fit now)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--paper-hashes", default=None,
                   help="sha256 table of the paper's images (default: data/reference/figure_sha256.json; 'none' "
                        "to skip the byte comparison)")
    p.add_argument("--paper-figures", default=None,
                   help="folder with the paper's Appendix E and I images (Report_*.jpg), to compare bytes")
    p.add_argument("--paper-figure2", default=None, help="the paper's Figure 2 image file, to compare bytes")
    p.add_argument("--skip-minp", action="store_true")
    p.add_argument("--skip-checks", action="store_true", help="skip the min_p noise checks (minutes)")
    p.set_defaults(func=cmd_repro)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
