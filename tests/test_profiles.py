"""Synthetic smoke tests for question_profiles (no real data).

Run with ``python -m pytest tests/test_profiles.py``.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from numpy.polynomial import legendre as npleg  # noqa: E402

from onetemp.profiles import analysis, curves, fitting, grid_profiles, hardness, plots, profile_examples, stats  # noqa: E402
from onetemp.profiles import io as qio  # noqa: E402
from onetemp.profiles.cli import main as cli_main  # noqa: E402


# ----------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------

def _paper_eval_legendre(coeffs, x_lo, x_hi, t):
    """Scalar reference, written the way the paper's analysis code evaluates a profile."""
    if t < x_lo or t > x_hi:
        return None
    u = 2.0 * (t - x_lo) / (x_hi - x_lo) - 1.0
    P = [1.0, u]
    for k in range(2, 11):
        P.append(((2 * k - 1) * u * P[k - 1] - (k - 1) * P[k - 2]) / k)
    return sum(c * p for c, p in zip(coeffs, P))


def _paper_fwhm(coeffs, x_lo, x_hi, n_pts=1000):
    """Scalar reference of the peak / half-maximum search (same loops as the paper's code)."""
    ts, vs = [], []
    for j in range(n_pts + 1):
        t = x_lo + (x_hi - x_lo) * j / n_pts
        ts.append(t)
        vs.append(_paper_eval_legendre(coeffs, x_lo, x_hi, t))
    peak_idx = max(range(n_pts + 1), key=lambda j: vs[j] if vs[j] is not None else -math.inf)
    peak_val = vs[peak_idx]
    if peak_val is None or peak_val <= 0:
        return None
    half_val = peak_val / 2.0
    t_left = None
    for j in range(peak_idx, 0, -1):
        a, b = vs[j - 1], vs[j]
        if a is not None and b is not None and a <= half_val < b:
            t_left = ts[j - 1] + (half_val - a) / (b - a) * (ts[j] - ts[j - 1])
            break
    if t_left is None:
        t_left = x_lo
    t_right = None
    for j in range(peak_idx, n_pts):
        a, b = vs[j], vs[j + 1]
        if a is not None and b is not None and a > half_val >= b:
            t_right = ts[j] + (half_val - a) / (b - a) * (ts[j + 1] - ts[j])
            break
    if t_right is None:
        t_right = x_hi
    return {"half_val": round(half_val, 6), "t_left": round(t_left, 6), "t_right": round(t_right, 6),
            "fwhm": round(t_right - t_left, 6), "peak_t": round(ts[peak_idx], 6), "peak_val": round(peak_val, 6)}


def _bump_profile(qid, peak_t, width, height_pct, x_lo=0.1, x_hi=2.0):
    """A synthetic fitted profile: a Gaussian bump (percent) approximated by a degree-10 Legendre series."""
    ts = np.linspace(x_lo, x_hi, 200)
    u = 2 * (ts - x_lo) / (x_hi - x_lo) - 1
    ys = height_pct * np.exp(-0.5 * ((ts - peak_t) / width) ** 2)
    coef = npleg.legfit(u, ys, 10)
    return qio.FittedProfile(id=qid, x_lo=x_lo, x_hi=x_hi, legendre=coef)


def _synthetic_study(nq=120, seed=1):
    """A synthetic temperature study and matching profiles: harder questions peak higher, narrower."""
    rng = np.random.default_rng(seed)
    grid = np.round(np.arange(0.1, 2.01, 0.1), 6)
    qs, profs = [], {}
    for i in range(nq):
        height = float(10 ** rng.uniform(-2.5, 2.0))  # percent: 0.003% .. 100%
        height = min(height, 99.0)
        peak_t = float(np.clip(0.3 + 0.4 * (2.0 - math.log10(height)) / 4.5 + rng.normal(0, 0.1), 0.15, 1.9))
        width = float(np.clip(0.5 - 0.08 * (2.0 - math.log10(height)), 0.08, 0.6))
        qid = f"q{i:03d}"
        p = _bump_profile(qid, peak_t, width, height)
        profs[qid] = p
        rate = curves.profile_probability(p, grid)
        n = np.full(len(grid), 400, dtype=np.int64)
        c = rng.binomial(n, rate)
        qs.append(qio.Question(id=qid, settings=grid.copy(), n=n, c=c.astype(np.int64)))
    st = qio.Study(benchmark="math500", model="example/model", axis="temperature", tau=None, questions=qs)
    return st, profs


def _write_counts(path, study):
    with open(path, "w") as fh:
        for q in study.questions:
            for s, n, c in zip(q.settings, q.n, q.c):
                fh.write(json.dumps({"benchmark": study.benchmark, "model": study.model, "id": q.id,
                                     "temperature": float(s), "min_p": None, "n": int(n), "c": int(c)}) + "\n")


# ----------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------

def test_legendre_matches_scalar_reference():
    # The reference uses plain Python floats, as the paper's code (its coefficients come from JSON lists):
    # Python's sum() treats float and numpy-scalar terms differently.
    rng = np.random.default_rng(0)
    for _ in range(20):
        coef = [float(v) for v in rng.normal(0, 20, 11)]
        x_lo, x_hi = (float(v) for v in sorted(rng.uniform(0.05, 4.0, 2)))
        ts = np.linspace(x_lo - 0.1, x_hi + 0.1, 57)
        got = curves.legendre_percent(coef, x_lo, x_hi, ts)
        for t, g in zip(ts, got):
            ref = _paper_eval_legendre(coef, x_lo, x_hi, float(t))
            if ref is None:
                assert math.isnan(g)
            else:
                assert g == ref, (g, ref)


def test_fitted_summary_matches_scalar_reference():
    rng = np.random.default_rng(1)
    for k in range(30):
        p = _bump_profile("x", rng.uniform(0.2, 1.8), rng.uniform(0.05, 0.6), rng.uniform(0.05, 100))
        got = curves.fitted_summary(p)
        ref = _paper_fwhm(p.legendre, p.x_lo, p.x_hi)
        assert got == ref, (got, ref)
    # a flat zero profile has no peak
    z = qio.FittedProfile("z", 0.1, 2.0, np.zeros(11))
    assert curves.fitted_summary(z) is None


def test_hardness():
    assert abs(hardness.H_CLAMP - 0.0868588963806) < 1e-10
    assert abs(hardness.hardness_from_probability(1 - math.exp(-1)) - 1.0) < 1e-12
    hs = [hardness.hardness_from_percent(p) for p in (90, 50, 10, 1, 0.1)]
    assert all(a < b for a, b in zip(hs, hs[1:]))
    assert hardness.is_included(0.1) and not hardness.is_included(0.099) and not hardness.is_included(None)


def test_sliding_mean_and_interior():
    y = np.arange(10, dtype=float)
    m = stats.sliding_mean_symmetric(y, win=4)  # radius 2, shrinking symmetrically at the ends
    assert m[0] == 0.0 and m[1] == 1.0 and m[5] == 5.0 and m[9] == 9.0
    H = np.array([0.0869] * 5 + [0.0870, 0.09, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
    xs, ys, start, stop = stats.drawn_interior(H, np.ones_like(H), win=4, advance_low_h=True)
    # trim 2 at each end, then skip while H[i+1]/H[i] < 1.05: from index 2 to index 5 (0.0870 -> 0.09 is +3.4%)
    assert (start, stop) == (6, 13), (start, stop)
    xs2, _, start2, _ = stats.drawn_interior(H, np.ones_like(H), win=4, advance_low_h=False)
    assert start2 == 2
    short = stats.drawn_interior(H[:4], np.ones(4), win=30)
    assert short[2] == 0 and short[3] == 4


def test_ols_and_coverage():
    H = np.logspace(-1, 3, 50)
    o = stats.ols_log10(H, 2.0 - 0.4 * np.log10(H))
    assert abs(o["slope"] + 0.4) < 1e-12 and abs(o["intercept"] - 2.0) < 1e-12 and abs(o["r2"] - 1) < 1e-12
    # 4 questions in one hardness bin: [0,1], [0.5,1.5], [1,2] share t = 1 -> 3 of 4; [3,4] is apart
    Hs = np.array([1.0, 1.0, 1.0, 1.0, 1000.0])
    tl = np.array([0.0, 0.5, 1.0, 3.0, 0.0])
    tr = np.array([1.0, 1.5, 2.0, 4.0, 1.0])
    bins = stats.coverage_bins(Hs, tl, tr, n_bins=3)
    assert bins[0]["M"] == 4 and bins[0]["max_count"] == 3 and bins[0]["height"] == 0.75
    assert bins[-1]["M"] == 1 and bins[-1]["height"] == 1.0
    assert len(bins) == 2  # the middle bin is empty and gets no bar
    assert stats.max_overlap(tl[:4], tr[:4]) == 3


def test_io_roundtrip_and_validation():
    st, _ = _synthetic_study(nq=6)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "counts.jsonl"
        _write_counts(p, st)
        # add a min_p study for another model
        with open(p, "a") as fh:
            for q in ("b", "a"):
                for mp in (0.0, 0.01, 0.1):
                    fh.write(json.dumps({"benchmark": "hitom", "model": "m2", "id": q, "temperature": 1.5,
                                         "min_p": mp, "n": 10, "c": 3}) + "\n")
        studies = qio.load_counts(p)
        assert len(studies) == 2
        t = qio.select_study(studies, benchmark="math500", model="example", axis="temperature")
        assert [q.id for q in t.questions] == [q.id for q in st.questions]
        m = qio.select_study(studies, benchmark="hitom", model="m2", axis="min_p", tau=1.5)
        ids, grid, n, c = m.matrices(order="id")
        assert ids == ["a", "b"] and list(grid) == [0.0, 0.01, 0.1] and n.shape == (2, 3)
        assert [q.id for q in m.questions] == ["b", "a"]  # file order kept
        bad = Path(d) / "bad.jsonl"
        bad.write_text(json.dumps({"benchmark": "math500", "model": "m", "id": "q", "temperature": 1.0,
                                   "min_p": None, "n": 5, "c": 6}) + "\n")
        try:
            qio.load_counts(bad)
            raise AssertionError("c > n must be rejected")
        except ValueError:
            pass
        dup = Path(d) / "dup.jsonl"
        row = json.dumps({"benchmark": "math500", "model": "m", "id": "q", "temperature": 1.0, "min_p": None,
                          "n": 5, "c": 1})
        dup.write_text(row + "\n" + row + "\n")
        try:
            qio.load_counts(dup)
            raise AssertionError("duplicate rows must be rejected")
        except ValueError:
            pass


def test_two_files_for_one_study_are_rejected_and_data_files_layout():
    from onetemp.profiles import paper
    st, _ = _synthetic_study(nq=3)
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.jsonl", Path(d) / "b.jsonl"
        _write_counts(a, st)
        _write_counts(b, st)
        try:
            qio.load_counts([a, b])
            raise AssertionError("two files holding the same study must be rejected")
        except ValueError as exc:
            assert "same study" in str(exc)
        # the data export layout: only existing files are returned, LiveCodeBench as the question_n parse
        (Path(d) / "livecodebench").mkdir()
        (Path(d) / "hitom").mkdir()
        (Path(d) / "livecodebench" / "Qwen3-8B__temperature__question_n.jsonl").write_text("")
        (Path(d) / "livecodebench" / "Qwen3-8B__temperature__cell_n.jsonl").write_text("")
        (Path(d) / "hitom" / "Qwen3-0.6B__min_p_tau2.0__nu400.jsonl").write_text("")
        got = [Path(p).name for p in paper.data_files(d)]
        assert got == ["Qwen3-8B__temperature__question_n.jsonl", "Qwen3-0.6B__min_p_tau2.0__nu400.jsonl"], got
    assert profile_examples.resolve_names(["GCD of Subset"], ["x", "GCD of Subset (v6)"]) == ["GCD of Subset (v6)"]


def test_fitting_rules_with_a_stand_in_backend():
    st, _ = _synthetic_study(nq=5)
    st.questions[0].c[:] = 0  # never solved -> zero profile on the whole grid
    calls = []

    def backend(grid, n, c):
        calls.append(int(n.max()))
        rate = 100.0 * c / np.maximum(n, 1)
        u = 2 * (grid - grid[0]) / (grid[-1] - grid[0]) - 1
        return float(grid[0]), float(grid[-1]), npleg.legfit(u, rate, 10)

    profs = fitting.fit_study(st, n_mode="question-max", backend=backend)
    assert profs[st.questions[0].id].extra.get("zero_profile") == "no correct sample"
    assert len(calls) == sum(1 for q in st.questions if q.c.sum() > 0)
    assert all(v == 400 for v in calls)  # question-max: one n per question
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "profiles.jsonl"
        qio.write_profiles(p, st, profs)
        back = fitting.profiles_for_study(st, str(p))
        for k, v in profs.items():
            assert np.array_equal(back[k].legendre, v.legendre) and back[k].x_lo == v.x_lo


def test_analysis_summary_invariants():
    st, profs = _synthetic_study(nq=150)
    rows = analysis.question_table(st, profs)
    srt = analysis.included_sorted(rows)
    assert all(r["peak_val"] >= 0.1 for r in srt)
    assert all(a["H"] <= b["H"] for a, b in zip(srt, srt[1:]))
    s = analysis.summarize(srt, advance_low_h=True)
    assert s["n"] == len(srt) and s["n_red"] <= s["n"]
    assert s["blue"]["n_points"] <= s["n"] - 30
    assert all(0 < b["height"] <= 1 for b in s["coverage"]["bins"])
    assert sum(b["M"] for b in s["coverage"]["bins"]) == s["n"]
    assert s["ols"]["slope"] < 0  # narrower ranges for harder synthetic questions
    c = analysis.compact_summary(s)
    assert c["coverage_bars"] == len(s["coverage"]["bins"]) <= 17
    meas = analysis.question_table_measured(st)
    assert len(meas) == len(rows)


def test_plots_and_figure2():
    st, profs = _synthetic_study(nq=120)
    srt = analysis.included_sorted(analysis.question_table(st, profs))
    with tempfile.TemporaryDirectory() as d:
        w = plots.draw_appendix_set(srt, True, str(Path(d) / "Report_synthetic"))
        for k in ("AveTemp", "RangeSize", "relC"):
            assert Path(w[k]).stat().st_size > 10_000
        f2 = plots.figure2_row(srt, True, Path(d) / "fig2.jpg", dpi=100)
        img = plots.plt.imread(f2)
        assert img.shape[0] == 450 and img.shape[1] == 3 * 700  # three 7.0 x 4.5 in panels at dpi 100


def test_profile_examples_figure():
    st, profs = _synthetic_study(nq=80)
    data = []
    for q in st.questions:
        p = profs[q.id]
        cx, cy = curves.curve_points(p)
        mt, mr = curves.measured_rates_percent(q.settings, q.n, q.c)
        data.append({"name": q.id, "curve_x": cx, "curve_y": cy, "fwhm": curves.fitted_summary(p),
                     "meas_t": mt, "meas_rate": mr})
    peaks = sorted(((curves.fitted_summary(profs[q.id]) or {"peak_val": 0})["peak_val"], q.id) for q in st.questions)
    hl = [peaks[-1][1], peaks[-5][1], peaks[-10][1]]
    with tempfile.TemporaryDirectory() as d:
        rec = profile_examples.plot_profile_examples(data, hl, str(Path(d) / "fig1.png"), bg_alpha=0.35)
        assert rec["n_highlighted"] == 3 and 0 < rec["n_background"] <= 60
        assert Path(rec["out"]).stat().st_size > 10_000


def _synthetic_minp(nq=90, seed=3):
    rng = np.random.default_rng(seed)
    grid = [0.0, 0.003, 0.01, 0.03, 0.1]
    rows = []
    for i in range(nq):
        base = 10 ** rng.uniform(-2.5, -0.5)
        pk = int(rng.integers(0, 5))
        shape = np.exp(-0.5 * ((np.arange(5) - pk) / 1.2) ** 2)
        p = np.clip(base * shape, 0, 1)
        n = 200
        c = rng.binomial(n, p)
        for v, ci in zip(grid, c):
            rows.append({"benchmark": "hitom", "model": "example/minp", "id": f"s{i}_o0_X", "temperature": 2.0,
                         "min_p": v, "n": n, "c": int(ci)})
    return rows


def test_grid_profiles_and_checks():
    rows = _synthetic_minp()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "minp.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        st = qio.select_study(qio.load_counts(p), benchmark="hitom", axis="min_p", tau=2.0)
        ids, grid, n, c = st.matrices(order="id")
        pk = grid_profiles.peaks(c, n)
        assert np.all(pk["L"] <= pk["j"]) and np.all(pk["j"] <= pk["R"])
        assert np.all(pk["size"] == pk["R"] - pk["L"] + 1)
        P = c / n
        for i in range(len(ids)):  # every value in E_q is at least half the peak, its neighbours are not
            if pk["pk"][i] > 0:
                assert np.all(P[i, pk["L"][i]:pk["R"][i] + 1] >= pk["pk"][i] / 2)
        full = grid_profiles.trend(c, n, c, n, len(grid))
        assert full["n"] == int(pk["inc"].sum())
        nc = grid_profiles.noise_checks(c, n, len(grid), grid_profiles.paper_seeds(1, 0), n_guard=20, n_null=20,
                                        n_fpr=2, s_fpr=5)
        for k in grid_profiles.STATS:
            assert set(("full_data", "guard_pass", "null_pass", "trend_claimable")) <= set(nc[k])
        # the CLI end to end on the same file
        out = Path(d) / "out"
        assert cli_main(["minp", "--counts", str(p), "--model", "example/minp", "--tau", "2.0",
                         "--out-dir", str(out), "--checks", "--n-guard", "10", "--n-null", "10"]) == 0
        assert len(list(out.glob("*.jpg"))) == 4 and len(list(out.glob("*_values.json"))) == 1


def test_cli_temperature_commands():
    st, profs = _synthetic_study(nq=100)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_counts(d / "counts.jsonl", st)
        qio.write_profiles(d / "profiles.jsonl", st, profs)
        common = ["--counts", str(d / "counts.jsonl"), "--benchmark", "math500", "--model", "example/model",
                  "--profiles", str(d / "profiles.jsonl")]
        assert cli_main(["table", *common, "--out", str(d / "t.csv")]) == 0
        assert cli_main(["summary", *common, "--out", str(d / "s.json")]) == 0
        s = json.loads((d / "s.json").read_text())
        assert s["n"] > 0 and s["coverage"]["n_bins"] == 17
        assert cli_main(["summary", *common, "--bins", "auto", "--out", str(d / "s2.json")]) == 0
        assert json.loads((d / "s2.json").read_text())["coverage"]["n_bins"] == max(2, s["n"] // 20)
        assert cli_main(["figures", *common, "--out-dir", str(d / "figs")]) == 0
        assert len(list((d / "figs").glob("*.jpg"))) == 3
        assert cli_main(["figure2", *common, "--out", str(d / "f2.jpg")]) == 0
        top = sorted(((curves.fitted_summary(profs[q.id]) or {"peak_val": 0})["peak_val"], q.id)
                     for q in st.questions)
        assert cli_main(["example", *common, "--highlight", top[-1][1], top[-3][1], top[-6][1],
                         "--out", str(d / "f1.png")]) == 0


def _run_all():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
