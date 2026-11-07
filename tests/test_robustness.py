"""Synthetic smoke tests of the robustness package (no data from the paper: every number here is made up).

    python -m pytest tests/test_robustness.py

Builds a synthetic temperature sweep (unimodal profiles over a 1000-fold range of hardness) and a synthetic min_p
sweep, writes them in the common formats, and runs every module on them with small sizes (simulate through its
builder only: its full pool of 1000 questions takes about a minute). Each check is an assert.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from onetemp.robustness import (certify, contexts, dirichlet, election, fig4b, heatmap, plot_heatmap, plot_prot,  # noqa: E402
                        prot_summary, protected, simulate,
                        subsets, tilt, worst_loss)
from onetemp.robustness.data import attach_core_profiles, load_study, write_jsonl  # noqa: E402

TEMPS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
MODEL = "synthetic/model"


def synthetic_profiles(path: Path, n_q: int = 60, seed: int = 7):
    """Unimodal profiles: peak height from 10^-3.5 to ~0.9, peak temperature drifting with hardness."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        peak = 10 ** rng.uniform(-3.5, -0.05)
        t0 = rng.uniform(0.4, 1.4)
        width = rng.uniform(0.3, 0.9)
        p = [peak * math.exp(-((t - t0) / width) ** 2) for t in TEMPS]
        n = 1000
        for t, pt in zip(TEMPS, p):
            rows.append({"benchmark": "math500", "model": MODEL, "id": f"q{q}", "temperature": t, "min_p": None,
                         "n": n, "c": int(rng.binomial(n, pt)), "p_fit": float(pt), "peak_pct": round(100 * peak, 6)})
    write_jsonl(path, rows)


def synthetic_minp(path: Path, n_q: int = 50, seed: int = 11):
    rng = np.random.default_rng(seed)
    vals = [0.0, 0.01, 0.03, 0.1, 0.3]
    rows = []
    for q in range(n_q):
        base = 10 ** rng.uniform(-2.5, -0.6)
        best = rng.integers(0, len(vals))
        for j, v in enumerate(vals):
            p = base * math.exp(-abs(j - best) * 0.4)
            rows.append({"benchmark": "hitom", "model": MODEL, "id": f"h{q}", "temperature": 1.5, "min_p": v,
                         "tau": 1.5, "n": 200, "c": int(rng.binomial(200, p))})
    write_jsonl(path, rows)


def check_data_and_contexts(tmp: Path):
    p = tmp / "prof.jsonl"
    synthetic_profiles(p)
    st = load_study(p, require_fit=True)
    assert st.axis == "temperature" and st.n.shape == (60, len(TEMPS)) and st.p_fit is not None
    ctx = contexts.kl_context(st, [1, 4, 16], rates="fit")
    assert ctx["n_inc"] <= 60 and np.all(np.diff(ctx["H_sorted"]) >= 0)
    for K, ck in ctx["per_k"].items():
        assert abs(ck["ratio_per_t"].max() - 1.0) < 1e-15 and ck["ratio_per_t"][ck["t_a_idx"]] == 1.0
        assert np.allclose(1 - np.exp(ck["log_fail"]), ck["pass_mat"], atol=1e-15)
    inp = contexts.prot_inputs(st, [1, 4, 16], rates="fit")
    assert np.isfinite(inp["x"]).all() and inp["n_inc"] == ctx["n_inc"]
    # unbiased estimator against its definition
    for n, c, K in [(10, 3, 4), (20, 0, 5), (20, 20, 7), (50, 1, 50)]:
        brute = 1 - math.comb(n - c, K) / math.comb(n, K)
        assert abs(contexts.unbiased_passk(n, c, K) - brute) < 1e-15
    # the small-n KL band rule
    assert heatmap.kl_col_edges(85).tolist()[:5] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert abs(heatmap.kl_col_edges(85)[-1] - math.log(85)) < 1e-15 and len(heatmap.kl_col_edges(359)) == 7
    return p


def check_dirichlet_modes():
    u = np.linspace(-1, 1, 40)
    a = dirichlet.sample(u, 300, "linear", seed=3)
    b = dirichlet.sample(u, 300, "replay", seed=3)      # replay checks itself against numpy bit for bit
    c = dirichlet.sample(u, 300, "log", seed=3)
    ok = ~(a["W"] == 0).any(axis=1)                       # rows without an underflowed entry
    assert np.allclose(a["W"][ok], b["W"][ok], rtol=1e-12, atol=1e-300)
    assert np.allclose(a["kl"][ok], b["kl"][ok], rtol=1e-9, atol=1e-12)
    for d in (b, c):
        assert np.allclose(d["W"].sum(axis=1), 1.0, atol=1e-12) and np.all(d["kl"] >= -1e-12)
        assert np.all(d["kl"] <= math.log(40) + 1e-9)
    assert np.array_equal(a["c_grid"], b["c_grid"])


def check_heatmap_presets(prof: Path, minp: Path, tmp: Path):
    st = load_study(prof, require_fit=True)
    for preset in ("paper-g-lcb", "paper-g"):
        cfg = heatmap.PRESETS[preset]
        res, arr = heatmap.run(st, [1, 4, 16], n_w=400, seed=5, draw=cfg["draw"], rule=cfg["election"],
                               rates=cfg["rates"], rows=cfg["rows"])
        for K in res["K"]:
            pk = res["per_k"][str(K)]
            for col, tot in zip(pk["displayed_pcts"], pk["col_totals"]):
                if tot:
                    assert abs(sum(col) - 100.0) < 1e-9
            assert pk["n_counted"] + pk["n_not_counted"] == 400
            assert all(0 <= v <= 100 for v in pk["top_row_pcts"] if v == v)
    # the exact election agrees with the plain argmax whenever the argmax has no near-tie
    ctx = contexts.kl_context(st, [4], rates="fit")
    dr = dirichlet.sample(ctx["u_sorted"], 500, "replay", seed=9)
    ck = ctx["per_k"][4]
    S = dr["W"] @ ck["pass_mat"]
    srt = np.sort(S, axis=1)
    clear = (srt[:, -1] - srt[:, -2]) > 1e-9 * srt[:, -1]
    el_a = election.elect_argmax(dr["W"], ck["pass_mat"])
    el_d, _ = election.elect_exact(dr["logW"], ck["log_fail"], ck["log_pass"])
    assert np.array_equal(el_a[clear], el_d[clear])
    # min_p preset: measured rates, unbiased pass@K
    sm = load_study(minp)
    assert sm.axis == "min_p" and sm.tau == 1.5
    cfg = heatmap.PRESETS["paper-i"]
    res, _ = heatmap.run(sm, [1, 4, 32], n_w=300, seed=5, draw=cfg["draw"], rule=cfg["election"],
                         rates=cfg["rates"], rows=cfg["rows"])
    assert res["axis"] == "min_p" and len(res["settings"]) == 5
    plot_heatmap.draw_figure(res, tmp / "minp_heat.jpg", cells="equal")      # three K panels
    out = tmp / "heat_cli"
    heatmap.main(["--input", str(minp), "--preset", "paper-i", "--K", "1,4,16,32", "--n-w", "200", "--out", str(out),
                  "--plot"])
    assert (out / "heatmap.jpg").exists() and (out / "top_row_table.md").exists()


def check_fig4b(prof: Path, tmp: Path):
    td = tmp / "tallies"
    args = ["worker", "--input", str(prof), "--benchmark", "math500", "--model", MODEL, "--K", "1,4",
            "--n-chunks", "2", "--chunk-size", "300", "--tally-dir", str(td)]
    fig4b.main(args)
    fig4b.main(args)                                      # a second run skips the verified tallies
    out = tmp / "cells.json"
    fig4b.main(["merge", "--tally-dir", str(td), "--out", str(out), "--expect-chunks", "2"])
    cells = json.loads(out.read_text())
    assert cells["n_chunks"] == 2 and cells["n_W"] == 600
    for K in ("1", "4"):
        assert all(0 <= c["share_pct"] <= 100 for c in cells["cells"][K] if c["n"])
    # the same tallies as one .jsonl file (the format of the shipped reference) read, merge and compare alike
    ref = tmp / "tallies.jsonl"
    with open(ref, "w", encoding="utf-8") as fh:
        for f in sorted(td.glob("chunk_*.json")):
            fh.write(json.dumps(json.loads(f.read_text())) + "\n")
    tl = fig4b.load_tallies(ref)
    assert sorted(tl) == [0, 1] and tl == fig4b.load_tallies(td)
    assert fig4b.main(["merge", "--tally-dir", str(td), "--out", str(tmp / "cells_cmp.json"),
                       "--compare-dir", str(ref)]) == 0           # merge gives a verdict: 0 when every chunk is equal
    rows = [json.loads(line) for line in ref.read_text().splitlines()]
    rows[1]["per_K"]["4"]["yd"][0] += 1.0
    bad = tmp / "tallies_changed.jsonl"
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert fig4b.main(["merge", "--tally-dir", str(td), "--out", str(tmp / "cells_bad.json"),
                       "--compare-dir", str(bad)]) == 1           # and 1 when a chunk differs
    assert json.loads(json.dumps(fig4b.merge_tallies(list(tl.values()), 2)["cells"])) == cells["cells"]
    cmp = fig4b.compare(list(tl.values()), fig4b.load_tallies(td))
    assert cmp["n_chunks_equal"] == 2 and cmp["n_tally_mismatch"] == 0 and cmp["max_abs_tally_diff"] == 0.0
    tl[1]["per_K"]["4"]["yd"][0] += 1.0                  # one changed tally is found
    cmp = fig4b.compare(list(tl.values()), fig4b.load_tallies(td))
    assert cmp["n_chunks_equal"] == 1 and cmp["n_tally_mismatch"] == 1
    dup = tmp / "tallies_dup"                            # a chunk in two files of a folder is refused
    dup.mkdir()
    first = sorted(td.glob("chunk_*.json"))[0]
    (dup / first.name).write_text(first.read_text())
    (dup / "chunk_copy.json").write_text(first.read_text())
    try:
        fig4b.load_tallies(dup)
    except ValueError as e:
        assert "appears in two files" in str(e)
    else:
        raise AssertionError("a duplicate chunk was not refused")


def check_protected_and_certify(prof: Path, tmp: Path):
    st = load_study(prof, require_fit=True)
    inp = contexts.prot_inputs(st, [4, 16], rates="fit")
    D = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    curves = protected.sweep(inp, [4, 16], (0.9, 0.99), D)
    arr = protected.beta_array(curves, [4, 16], (0.9, 0.99), D)
    assert not protected.monotone_violations(arr, D)
    cert = certify.certify_study(inp, curves, [4, 16], (0.9, 0.99), D)
    c = cert["summary"]["counts"]
    assert c["attain_none"] == 0 and c["lower_unproven"] == 0 and c["inf_unproven"] == 0
    n_final = int(np.isfinite(cert["beta_star_final"]).sum())
    assert c["attain_at_beta_star"] + c["attain_at_next_step"] == c["finite"] == n_final
    assert c["finite_sweep"] == int(np.isfinite(arr).sum())
    assert cert["summary"]["kinds_of_beyond_cap_points"]["inconsistent"] == 0
    for w in ("hard", "easy"):                            # one-sided windows run end to end
        cv = protected.sweep(inp, [4], (0.9,), [0.0, 1.0], window=w)
        assert len(cv[(0.9, 4)]) >= 1
    # the CLI, its plot, the summaries and the two-panel figure
    out = tmp / "prot"
    protected.main(["--input", str(prof), "--K", "4,16", "--D", "0,0.5,1,2", "--out", str(out), "--plot"])
    res = json.loads((out / "beta_star.json").read_text())
    assert (out / "beta_star_vs_D.jpg").exists() and "certify" in res and len(res["c_empty"]) == 2
    try:                                                  # outputs are never overwritten
        protected.main(["--input", str(prof), "--K", "4", "--D", "0", "--out", str(out)])
        raise AssertionError("the second run should have refused to overwrite")
    except SystemExit:
        pass
    assert prot_summary.counts(prot_summary._load([out / "beta_star.json"]))["finite"] == res["certify"]["counts"]["finite"]
    prot_summary.slopes(prot_summary._load([out / "beta_star.json"]), min_finite=2)
    roleK = tmp / "roleK"
    protected.main(["--input", str(prof), "--K", "1,2,4,8", "--thr", "0.9", "--D", "0", "--no-certify",
                    "--out", str(roleK)])
    rk = prot_summary.role_of_K(prot_summary._load([roleK / "beta_star.json"]), {})
    assert rk["n_curves"] == 1
    plot_prot.draw_study(res, tmp / "cut.jpg", cut=True)


def check_worst_loss_bruteforce():
    rng = np.random.default_rng(1)
    for trial in range(40):
        n, T = 9, 4
        s = np.round(rng.uniform(0, 1, size=(n, T)), 3)
        x = rng.uniform(-1, 3, size=n)
        A = int(np.argmax(s.sum(axis=0)))
        blk = worst_loss.ExactBlock(s)
        sF = [[Fraction(float(v)) for v in row] for row in s]
        for beta in (Fraction(3, 2), Fraction(10)):
            for window, D in (("sym", 0.0), ("sym", 0.5), ("easy", 0.0), ("hard", 0.0), ("all", None)):
                r = worst_loss.worst_loss(blk, x, 4, A, window, D, beta)
                out = worst_loss.outside_mask(x, 4, window, D)
                oi = np.flatnonzero(out)
                best = Fraction(0)
                for vert in itertools.product((1 / beta, beta), repeat=len(oi)):
                    W = [Fraction(1)] * n
                    for q, wv in zip(oi, vert):
                        W[q] = wv
                    SW = [sum(W[q] * sF[q][t] for q in range(n)) for t in range(T)]
                    if max(SW) > 0:
                        best = max(best, 1 - SW[A] / max(SW))
                assert r["R"] == best and r["certified"], (trial, beta, window, D, r["R"], best)


def check_tilt_and_subsets(prof: Path, tmp: Path):
    st = load_study(prof, require_fit=True)
    inp = contexts.prot_inputs(st, [4], rates="fit")
    r = tilt.curve(inp["x"], inp["s"][4], inp["t_a_idx"][4], 4)
    assert r["tau_star_G11"]["0"] == [inp["t_a_idx"][4]]
    lab = tmp / "labels.jsonl"
    write_jsonl(lab, [{"benchmark": "math500", "id": q, "labels": {"level": i % 3, "kind": "ab"[i % 2]}}
                      for i, q in enumerate(st.ids)])
    subs = subsets.subsets_of(inp["ids"], subsets.read_labels(lab), n_min=5)
    rows = subsets.natural_losses(inp["s"][4], subs)
    assert subs and all(0 <= row["loss"] <= 1 for row in rows)


def check_simulate_small(prof: Path):
    st = load_study(prof, require_fit=True)
    base = contexts.prot_inputs(st, [1], rates="fit")
    P_m = 1.0 - np.exp(np.log(np.clip(1.0 - st.p_fit, 1e-15, 1.0)))[[st.ids.index(q) for q in base["ids"]]]
    x_m = base["x"].copy()
    x_m[: max(3, len(x_m) // 3)] = np.linspace(2.0, 2.9, max(3, len(x_m) // 3))   # make templates exist
    sim = simulate.build(P_m, x_m, np.arange(len(x_m)), "peaked", n_q=200, n_fine=20001)
    assert sim["P"].shape == (200, len(TEMPS)) and np.all(np.diff(sim["H"]) > 0) and sim["P"].max() < 1


def check_core_profiles_and_resampling(prof: Path, tmp: Path):
    """The --profiles route (the profile fit's profiles.json) gives the same study as p_fit/peak_pct rows."""
    st = load_study(prof, require_fit=True)
    rows = [json.loads(l) for l in prof.read_text().splitlines()]
    counts = tmp / "counts_only.jsonl"
    write_jsonl(counts, [{k: v for k, v in r.items() if k not in ("p_fit", "peak_pct")} for r in rows])
    core = {"format": "core.fit_all/profiles-v1", "axis": "temperature", "settings": TEMPS, "alloc_settings": TEMPS,
            "questions": [{"id": q, "P_fit": st.p_fit[i].tolist(), "peak_fit_pct": float(st.peak_pct[i])}
                          for i, q in enumerate(st.ids)]}
    cp = tmp / "profiles.json"
    cp.write_text(json.dumps(core))
    st2 = attach_core_profiles(load_study(counts), cp)
    assert np.array_equal(st2.p_fit, st.p_fit) and np.array_equal(st2.peak_pct, st.peak_pct)
    boot = st.take(np.array([0, 0, 3]))                  # a resample with replacement keeps ids unique
    assert len(set(boot.ids)) == 3 and boot.n.shape[0] == 3
    assert fig4b.share_label(0.0, 0, 0.0) == "" and fig4b.share_label(9999.6, 10000, 1.0) == "99.99"


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        prof = check_data_and_contexts(tmp)
        minp = tmp / "minp.jsonl"
        synthetic_minp(minp)
        check_dirichlet_modes()
        check_heatmap_presets(prof, minp, tmp)
        check_fig4b(prof, tmp)
        check_protected_and_certify(prof, tmp)
        check_worst_loss_bruteforce()
        check_tilt_and_subsets(prof, tmp)
        check_simulate_small(prof)
        check_core_profiles_and_resampling(prof, tmp)
    print("all robustness smoke tests passed")


def test_all():
    """Entry point for pytest."""
    main()


if __name__ == "__main__":
    main()
