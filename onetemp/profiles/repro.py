"""Reproduce every profile figure and printed profile number of the paper, and compare.

``python -m onetemp.profiles repro --counts-dir data/counts --out-dir OUT [--profiles P/*.jsonl]`` does, for
each of the 12 (benchmark, model) temperature sweeps it finds:

* fitted profiles (``--profiles`` files, else the core package's fit with the paper's n rule),
* the per-question table and the summary (``analysis``),
* the Appendix E images ``Report_<stem>_{AveTemp,RangeSize,relC}.jpg`` (Figures 5-13), and for Qwen3-8B on
  LiveCodeBench also Figure 2 (``figure2_qwen3_8b_livecodebench.jpg``) and Figure 1
  (``figure1_profiles_qwen3_8b_livecodebench.png``),

then the three min_p studies of Appendix I if their counts are present (Figures 38-41).  It writes
``repro_table.md`` (paper value | ours | difference | where) and ``repro.json``.  It compares the bytes (sha256)
of every image it draws with the paper's image of the same figure panel, from the hash table shipped in
``data/reference/figure_sha256.json`` (``--paper-hashes``), or from a folder of the paper's image files
(``--paper-figures``).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import analysis, fitting, paper, plots
from . import io as qio
from .hardness import H_CLAMP


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_hashes(path) -> dict:
    """{image file name: (sha256, figure)} from data/reference/figure_sha256.json (the paper's image files)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {v["file"]: (v["sha256"], fig) for fig, v in d["figures"].items() if "file" in v}


def _reference_sha(name: str, figdir, hashes):
    """The paper's sha256 of an image this package writes, and where it is: from a folder of the paper's image
    files when one is given (same file name), else from the shipped hash table; None when neither has it."""
    if figdir is not None and (Path(figdir) / name).exists():
        return _sha(Path(figdir) / name), "paper figure file"
    if hashes and name in hashes:
        sha, fig = hashes[name]
        return sha, f"{fig} image"
    return None


def _bytes_known() -> str | None:
    """Why an image may differ in bytes from the paper's: the paper's figures were drawn with matplotlib 3.10.6 and
    Pillow 11.3.0, and other versions can render differently. None when exactly those versions are installed, so that
    a difference then counts as one."""
    import matplotlib
    import PIL
    if matplotlib.__version__ == "3.10.6" and PIL.__version__ == "11.3.0":
        return None
    return (f"the paper's images were drawn with matplotlib 3.10.6 and Pillow 11.3.0 (here {matplotlib.__version__} "
            f"and {PIL.__version__}); the numbers decide")


def _draws_known() -> str | None:
    """Why a Monte Carlo value of the min_p null may differ from the paper's: the null draws binomial counts, and
    numpy versions differ in the binomial draws they make from the same seed (numpy 2.2.6, which computed the paper's
    values, and 2.5.4 do).  The Monte Carlo values are the null's quantiles, including whether its central 95% lies
    within the interval the paper prints, its median, and the counts of null replicates beyond a slope.  None under
    numpy 2.2.6, so that a difference then counts as one.  The conclusions, which trends pass both checks, and the
    split-half guard never get this reason."""
    import numpy as np
    if np.__version__ == "2.2.6":
        return None
    return (f"a Monte Carlo value of the null; the paper's come from numpy 2.2.6's binomial draws (here "
            f"{np.__version__}); which trends pass both checks is still compared strictly")


class Table:
    def __init__(self):
        self.rows = []

    def add(self, what: str, printed, ours, where: str, tol: float | None = None, exact: bool = False,
            known: str | None = None):
        """Record one comparison; the difference is ours - printed (numbers) or a match flag.

        ``known``: the reason for a difference that is expected and documented (it is reported, but does not
        make the run fail).
        """
        if isinstance(printed, (int, float)) and isinstance(ours, (int, float)) and not isinstance(printed, bool):
            diff = ours - printed
            if exact:
                ok = ours == printed
            else:
                ok = tol is None or abs(diff) <= tol
            self.rows.append({"what": what, "paper": printed, "ours": ours, "diff": diff, "ok": bool(ok),
                              "where": where, "known": known})
        else:
            self.rows.append({"what": what, "paper": printed, "ours": ours, "diff": None,
                              "ok": bool(printed == ours), "where": where, "known": known})

    @property
    def unexpected(self) -> list:
        return [r for r in self.rows if not r["ok"] and not r.get("known")]

    def markdown(self) -> str:
        out = ["| quantity | paper | ours | difference | where | match |", "|---|---|---|---|---|---|"]
        for r in self.rows:
            def f(v):
                if isinstance(v, float):
                    return f"{v:.6g}"
                return str(v)
            d = "" if r["diff"] is None else (f"{r['diff']:+.2g}" if isinstance(r["diff"], float) else str(r["diff"]))
            m = "yes" if r["ok"] else (f"no, known: {r['known']}" if r.get("known") else "NO")
            out.append(f"| {r['what']} | {f(r['paper'])} | {f(r['ours'])} | {d} | {r['where']} | {m} |")
        return "\n".join(out) + "\n"


def _round_half(x: float, nd: int) -> float:
    return float(f"{x:.{nd}f}")


def _profiles_index(paths) -> dict:
    idx = {}
    for p in paths or []:
        for key, d in qio.load_profiles(p).items():
            idx.setdefault(key, {}).update(d)
    return idx


def _study_profiles(st, pidx):
    key = (st.benchmark, st.model, st.axis, st.tau)
    if pidx:
        if key not in pidx:
            raise KeyError(f"no profiles for {key} in the given --profiles files")
        return pidx[key]
    return fitting.fit_study(st, n_mode=paper.N_MODE[st.benchmark])


def run_temperature(studies, pidx, out_dir: Path, figdir: Path | None, table: Table,
                    fig2_ref: str | None = None, hashes: dict | None = None) -> dict:
    results = {}
    for label, repo, mstem in paper.MODELS:
        for bench in paper.BENCHMARKS:
            try:
                st = qio.select_study(studies, benchmark=bench, model=repo, axis="temperature")
            except ValueError:
                continue
            profs = _study_profiles(st, pidx)
            rows = analysis.question_table(st, profs)
            srt = analysis.included_sorted(rows)
            adv = paper.ADVANCE_LOW_H[bench]
            s = analysis.summarize(srt, adv)
            cs = analysis.compact_summary(s)
            stem = paper.image_stem(bench, mstem)
            sub = out_dir / bench
            sub.mkdir(parents=True, exist_ok=True)
            analysis.write_table_csv(sub / f"{stem}_table.csv", rows)
            qio.write_json(sub / f"{stem}_summary.json", s)
            written = plots.draw_appendix_set(srt, adv, str(sub / stem))
            key = (bench, mstem)
            where_n = {"livecodebench": "Fig. 5/8/11 captions", "math500": "Fig. 6/9/12 captions",
                       "hitom": "Fig. 7/10/13 captions"}[bench]
            fig_ids = {"livecodebench": ("5", "8", "11"), "math500": ("6", "9", "12"), "hitom": ("7", "10", "13")}[bench]
            table.add(f"{bench} {label}: included questions n", paper.PRINTED_N[key], cs["n"], where_n, exact=True)
            lg = paper.PRINTED_LEGENDS[key]
            table.add(f"{bench} {label}: n with peak tau > 0.1", lg["n_red"], cs["n_red"],
                      f"Fig. {fig_ids[0]} legend", exact=True)
            for k, nm in (("slope", "ols_slope"), ("intercept", "ols_intercept"), ("r2", "ols_r2")):
                table.add(f"{bench} {label}: OLS {k} (3 dp)", lg[k], _round_half(cs[nm], 3),
                          f"Fig. {fig_ids[1]} legend", tol=0.0)
            if key in paper.PRINTED_RED_START_END:
                a, b = paper.PRINTED_RED_START_END[key]
                table.add(f"{bench} {label}: red sliding mean, first value (2 dp)", a, _round_half(cs["red_first"], 2),
                          "App. E.2 text", tol=0.0)
                table.add(f"{bench} {label}: red sliding mean, last value (2 dp)", b, _round_half(cs["red_last"], 2),
                          "App. E.2 text", tol=0.0)
            if bench == "math500":
                table.add(f"math500 {label}: lowest coverage bar (3 dp)",
                          paper.PRINTED_MATH500_COVERAGE_MIN[mstem], _round_half(cs["coverage_min"], 3),
                          "App. E.2 text", tol=0.0)
                heights = cs["coverage_heights"]
                in_top2 = min(heights[-2:]) == min(heights)
                table.add(f"math500 {label}: first bar = 1 and lowest bar in one of the two highest-H bins",
                          True, bool(heights[0] == 1.0 and in_top2), "App. E.2 text")
            if bench == "hitom":
                lo, hi = paper.PRINTED_HITOM_COVERAGE_RANGE
                table.add(f"hitom {label}: all coverage bars in [{lo}, {hi}]", True,
                          bool(all(lo <= _round_half(h, 2) <= hi for h in cs["coverage_heights"])), "App. E.2 text")
                table.add(f"hitom {label}: coverage bars drawn", paper.PRINTED_HITOM_BARS[mstem], cs["coverage_bars"],
                          "Fig. 13 caption", exact=True)
                table.add(f"hitom {label}: questions at the clamp H = 0.0869", 0, cs["n_at_clamp"], "App. E.1",
                          exact=True)
            if bench in ("livecodebench", "math500"):
                for line in ("blue", "red"):
                    ln = s[line]
                    if ln is None:
                        continue
                    hs = [r["H"] for r in srt] if line == "blue" else [r["H"] for r in srt if r["peak_t"] > 0.1]
                    # questions the advance skips after the 15-question trim, other than those at the clamp
                    dropped = hs[s["window"] // 2:ln["start_index"]]
                    non_clamp = sum(1 for h in dropped if abs(h - H_CLAMP) > 1e-12)
                    table.add(f"{bench} {label}: {line} line, non-clamp questions dropped by the low-H advance (<= 2)",
                              True, bool(non_clamp <= paper.PRINTED_MAX_EXTRA_DROPPED), "App. E.1")
            for kind, path in written.items():
                ref = _reference_sha(Path(path).name, figdir, hashes)
                if ref is not None:
                    table.add(f"{bench} {label}: {Path(path).name} bytes", ref[0][:16], _sha(path)[:16], ref[1],
                              known=_bytes_known())
            results[f"{bench}/{mstem}"] = cs

            if bench == "livecodebench" and mstem == "Qwen3_8B":
                f2 = out_dir / "figure2_qwen3_8b_livecodebench.jpg"
                plots.figure2_row(srt, adv, f2)
                table.add("Figure 2: n / n red / OLS (same data as Fig. 5c, 8c, 11c)",
                          "647 / 542 / -0.451, 1.915, 0.601",
                          f"{cs['n']} / {cs['n_red']} / {cs['ols_slope']:.3f}, {cs['ols_intercept']:.3f}, "
                          f"{cs['ols_r2']:.3f}", "Fig. 2 legends")
                if fig2_ref is not None and Path(fig2_ref).exists():
                    table.add("Figure 2 bytes", _sha(fig2_ref)[:16], _sha(f2)[:16], "paper figure file",
                              known=_bytes_known())
                elif hashes and f2.name in hashes:
                    table.add("Figure 2 bytes", hashes[f2.name][0][:16], _sha(f2)[:16], "Figure 2 image",
                              known=_bytes_known())
                results["figure1"] = _figure1(st, profs, out_dir, figdir, table)
    return results


def _figure1(st, profs, out_dir: Path, figdir, table: Table) -> dict:
    from . import profile_examples
    data = profile_examples.build_records(st, profs, paper.N_MODE[st.benchmark])
    names = [d["name"] for d in data]
    highlight = None
    for wanted in (paper.FIG1_HIGHLIGHT, paper.FIG1_LABELS):  # public question_ids, else "title (release)"
        try:
            highlight = profile_examples.resolve_names(wanted, names)
            break
        except KeyError:
            continue
    if highlight is None:
        table.add("Figure 1: highlighted questions found", True, False, "Fig. 1")
        return {"error": f"none of {paper.FIG1_HIGHLIGHT} or {paper.FIG1_LABELS} found"}
    out = out_dir / "figure1_profiles_qwen3_8b_livecodebench.png"
    rec = profile_examples.plot_profile_examples(data, highlight, str(out), bg_alpha=0.35,
                                                 labels=paper.FIG1_LABELS)
    why = ("Figure 1 was drawn from an earlier release of the Qwen3-8B LiveCodeBench data, not shipped; "
           "the redraw differs in the grey background")
    table.add("Figure 1: grey background questions drawn", paper.PRINTED_FIG1["legend_background"],
              rec["n_background"], "Fig. 1 image legend", exact=True, known=why)
    return {k: v for k, v in rec.items()}


def run_minp(studies, out_dir: Path, table: Table, checks: bool, figdir: Path | None = None,
             hashes: dict | None = None) -> dict:
    import numpy as np
    from . import grid_profiles as gp
    res = {}
    P = paper.PRINTED_MINP
    for idx, repo, tau, stem in paper.MINP_STUDIES:
        try:
            st = qio.select_study(studies, benchmark="hitom", model=repo, axis="min_p", tau=tau)
        except ValueError:
            continue
        ids, grid, n, c = st.matrices(order="id")
        grid = [float(v) for v in grid]
        nv = len(grid)
        pk = gp.peaks(c, n)
        sub = out_dir / "minp"
        sub.mkdir(parents=True, exist_ok=True)
        r = {"facts": gp.study_facts(ids, grid, c, n, pk)}
        paths = {k: sub / f"Report_{stem}_{k}.jpg" for k in ("Peak", "RangeSize", "relC", "profiles")}
        with plots.paper_style():
            _, r["peak"] = gp.fig_grid_peak(ids, grid, pk, paths["Peak"])
            _, r["range"] = gp.fig_grid_range(ids, grid, pk, paths["RangeSize"])
            _, r["coverage"] = gp.fig_grid_coverage(ids, grid, pk, paths["relC"])
        _, r["profiles"] = gp.fig_grid_profiles(ids, c, n, grid, pk, paths["profiles"])
        full = gp.trend(c, n, c, n, nv)
        lab = f"min_p {repo.split('/')[-1]} tau={tau:g}"
        for k, path in paths.items():
            ref = _reference_sha(path.name, figdir, hashes)
            if ref is not None:
                table.add(f"{lab}: {path.name} bytes", ref[0][:16], _sha(path)[:16], ref[1], known=_bytes_known())
        f = r["facts"]
        if idx in P["included"]:
            table.add(f"{lab}: included questions", P["included"][idx], f["included"], "App. I", exact=True)
        if idx in P["excluded_no_correct"]:
            table.add(f"{lab}: excluded, no correct sample", P["excluded_no_correct"][idx], f["no_correct_sample"],
                      "App. I", exact=True)
        if idx in P["excluded_single_correct"]:
            table.add(f"{lab}: excluded below 0.1% with some correct", P["excluded_single_correct"][idx],
                      f["below_floor_with_some_correct"], "App. I", exact=True)
        if idx in P["peak_at_smallest"]:
            table.add(f"{lab}: questions peaking at the smallest value", P["peak_at_smallest"][idx],
                      f["peak_at_smallest_value"], "App. I", exact=True)
        if idx in P["peak_at_largest"]:
            # tau = 1.5 counts all 648 (none excluded); Mistral counts the 598 included
            v = f["peak_at_largest_value_included"]
            table.add(f"{lab}: questions peaking at the largest value", P["peak_at_largest"][idx], v, "App. I",
                      exact=True)
        if idx in P["no_correct_at_largest"]:
            table.add(f"{lab}: questions with no correct sample at the largest value", P["no_correct_at_largest"][idx],
                      f["no_correct_at_largest_value"], "App. I", exact=True)
        table.add(f"{lab}: peak slope, all (2 dp)", P["peak_slope_all"][idx], _round_half(full["peak_slope_all"], 2),
                  "App. I", tol=0.0)
        table.add(f"{lab}: peak slope, peak above smallest value (2 dp)", P["peak_slope_subset"][idx],
                  _round_half(full["peak_slope_subset"], 2), "App. I", tol=0.0)
        table.add(f"{lab}: range-size slope (2 dp)", P["range_slope"][idx], _round_half(full["range_slope"], 2),
                  "App. I", tol=0.0)
        if idx in P["relc_lowest_bar_bins_ge10"]:
            table.add(f"{lab}: lowest coverage bar, bins with >= 10 questions (2 dp)",
                      P["relc_lowest_bar_bins_ge10"][idx], _round_half(f["lowest_bar_bins_ge10"], 2), "App. I",
                      tol=0.0)
        if idx in P["bins_above_H100_questions"]:
            table.add(f"{lab}: questions in bins wholly above H = 100", P["bins_above_H100_questions"][idx],
                      f["questions_in_bins_wholly_above_H100"], "Fig. 41 caption", exact=True)
        hl = [h["peak_rate_percent"] for h in r["profiles"]["highlighted"] if h.get("id")]
        r["highlighted_peak_percent"] = hl
        if checks:
            nc = gp.noise_checks(c, n, nv, gp.paper_seeds(paper.MINP_ROOT_SEED, idx), n_fpr=0)
            r["noise_checks"] = nc
            mc = _draws_known()  # the null's Monte Carlo values below; never the guard or the conclusions
            pa = nc["peak_slope_all"]
            table.add(f"{lab}: peak slope keeps its sign in all 400 splits", True,
                      bool(pa["guard"]["frac_same_sign_as_full"] == 1.0), "App. I")
            table.add(f"{lab}: no null replicate reaches the peak slope", 0,
                      pa["null"]["replicates_at_least_as_extreme"], "App. I", exact=True, known=mc)
            lo, hi = P["peak_null_central95_within"]
            table.add(f"{lab}: null central 95% of the peak slope within [{lo}, {hi}]", True,
                      bool(lo <= _round_half(pa["null"]["q025"], 2) and _round_half(pa["null"]["q975"], 2) <= hi),
                      "App. I", known=mc)
            rs = nc["range_slope"]
            table.add(f"{lab}: range-size null median (2 dp)", P["range_null_median"][idx],
                      _round_half(rs["null"]["median"], 2), "App. I", tol=0.0, known=mc)
            if idx in P["range_null_at_least_as_steep"]:
                table.add(f"{lab}: null replicates at least as steep as the range slope",
                          P["range_null_at_least_as_steep"][idx], rs["null"]["replicates_at_least_as_extreme"],
                          "App. I", exact=True, known=mc)
            if idx in P["range_null_flatter_than"]:
                steeper = int(np.sum(np.asarray(rs["null"]["per_replicate"]) < full["range_slope"]))
                table.add(f"{lab}: null replicates steeper than the range slope", P["range_null_flatter_than"][idx],
                          steeper, "App. I", exact=True, known=mc)
            rc = nc["relc_slope"]
            table.add(f"{lab}: null replicates at least as steep as the coverage slope",
                      P["relc_null_at_least_as_steep"][idx], rc["null"]["replicates_at_least_as_extreme"], "App. I",
                      exact=True, known=mc)
            # the claims in words: which trends pass both checks
            table.add(f"{lab}: peak trend passes both checks (all / above smallest value)", "yes / yes",
                      f"{'yes' if pa['trend_claimable'] else 'no'} / "
                      f"{'yes' if nc['peak_slope_subset']['trend_claimable'] else 'no'}", "App. I")
            table.add(f"{lab}: range narrows beyond noise", "no" if idx == 0 else "yes",
                      "yes" if rs["trend_claimable"] else "no", "App. I")
            table.add(f"{lab}: coverage trend beyond noise", "no", "yes" if rc["trend_claimable"] else "no",
                      "App. I")
            if idx == 0:
                table.add(f"{lab}: coverage split-half check fails", "fails",
                          "passes" if rc["guard_pass"] else "fails", "App. I")
        res[stem] = r
    if res:
        allhl = [v for r in res.values() for v in r.get("highlighted_peak_percent", [])]
        if allhl:
            lo, hi = P["highlighted_peak_percent_range"]
            table.add("min_p Figure 38: highlighted questions peak at 26.5% to 37.1%", f"{lo} to {hi}",
                      f"{min(allhl):.1f} to {max(allhl):.1f}", "Fig. 38 / App. I")
    return res


def main(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    import glob
    if getattr(args, "counts_dir", None):
        files.extend(paper.data_files(args.counts_dir))
    for p in args.counts:
        files.extend(sorted(glob.glob(p)) or [p])
    if not files:
        raise SystemExit("give --counts-dir (the data export's counts/ folder) or --counts FILE ...")
    print("counts files:\n  " + "\n  ".join(files))
    studies = qio.load_counts(files)
    pidx = _profiles_index(args.profiles)
    figdir = Path(args.paper_figures) if args.paper_figures else None
    hp = getattr(args, "paper_hashes", None)
    if hp is None:
        from onetemp.paths import reference_dir
        default = reference_dir() / "figure_sha256.json"
        hp = str(default) if default.exists() else "none"
    hashes = None if hp == "none" else _reference_hashes(hp)
    table = Table()
    results = {"temperature": run_temperature(studies, pidx, out_dir, figdir, table,
                                              getattr(args, "paper_figure2", None), hashes)}
    if not args.skip_minp:
        results["minp"] = run_minp(studies, out_dir, table, checks=not args.skip_checks, figdir=figdir,
                                   hashes=hashes)
    (out_dir / "repro_table.md").write_text(table.markdown(), encoding="utf-8")
    qio.write_json(out_dir / "repro.json", {"comparisons": table.rows, "results": results})
    n_ok = sum(r["ok"] for r in table.rows)
    n_known = sum(1 for r in table.rows if not r["ok"] and r.get("known"))
    print(table.markdown())
    print(f"{n_ok}/{len(table.rows)} comparisons match, {n_known} known differences, "
          f"{len(table.unexpected)} unexpected; table in {out_dir / 'repro_table.md'}")
    return 0 if not table.unexpected else 1
