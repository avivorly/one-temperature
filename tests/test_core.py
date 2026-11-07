"""Synthetic smoke tests of the core method. No data of the paper is used: every count below is
drawn from made-up unimodal profiles.

    python -m pytest tests/test_core.py          (from the repository root)

The C kernel tests compile bernstein_kernel.c into a temporary directory when a C compiler is
found, and are skipped otherwise.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from onetemp.core import allocation as al
from onetemp.core import bootstrap as bs
from onetemp.core import counts_io as cio
from onetemp.core import fit_all as fa
from onetemp.core import heldout as ho
from onetemp.core import passk as pk
from onetemp.core import pipeline as pl
from onetemp.core import tables as tb

SETTINGS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 2.0]
FIT_ITERS = 3000          # the paper uses 50,000; a short run keeps the smoke test fast


def synthetic_rows(n_q=14, n=300, seed=0, benchmark="synthbench", model="org/synthetic-model"):
    """Made-up counts: Gaussian-shaped profiles over SETTINGS, one unsolved and one easy question."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        if q == 0:
            P = np.zeros(len(SETTINGS))
        elif q == 1:
            P = np.full(len(SETTINGS), 0.97)
        else:
            mu = rng.uniform(0.1, 2.0)
            w = rng.uniform(0.2, 0.8)
            h = 10 ** rng.uniform(-2.5, -0.2)
            P = h * np.exp(-0.5 * ((np.asarray(SETTINGS) - mu) / w) ** 2)
        c = rng.binomial(n, P)
        for t, ci in zip(SETTINGS, c):
            rows.append({"benchmark": benchmark, "model": model, "id": f"q{q:02d}", "temperature": t,
                         "min_p": None, "n": int(n), "c": int(ci)})
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def compositions(K, l):
    if l == 1:
        yield (K,)
        return
    for v in range(K + 1):
        for rest in compositions(K - v, l - 1):
            yield (v,) + rest


def brute_force(L, K):
    Q = np.array(list(compositions(K, L.shape[1])), dtype=np.float64)
    F = np.sum(np.exp(L @ Q.T), axis=0)
    return float(F.min())


def random_L(rng, n, l):
    peaks = rng.uniform(0, l - 1, n)
    width = rng.uniform(0.5, 3, n)
    height = 10 ** rng.uniform(-3, -0.05, n)
    j = np.arange(l)
    P = height[:, None] * np.exp(-0.5 * ((j[None, :] - peaks[:, None]) / width[:, None]) ** 2)
    return np.log(np.clip(1 - P, 1e-15, 1))


def _compiler():
    return shutil.which(os.environ.get("CC", "gcc")) or shutil.which("cc")


class TestCountsIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "counts.jsonl")
        rows = synthetic_rows()
        # a min_p sweep at temperature 1.5 for the same model
        for q in range(3):
            for mp in (0.0, 0.05, 0.1):
                rows.append({"benchmark": "synthbench", "model": "org/synthetic-model", "id": f"q{q:02d}",
                             "temperature": 1.5, "min_p": mp, "n": 100, "c": 10 + q, "tau": 1.5})
        write_jsonl(self.path, rows)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_load_and_axes(self):
        c = cio.load_counts(self.path)
        self.assertEqual(c.n.shape, (14, len(SETTINGS)))
        self.assertEqual(c.ids[:3], ["q00", "q01", "q02"])          # first-appearance order
        self.assertEqual(c.settings, SETTINGS)
        m = cio.load_counts(self.path, axis="min_p")
        self.assertEqual(m.settings, [0.0, 0.05, 0.1])
        self.assertEqual(m.fixed, {"temperature": 1.5})
        groups = cio.list_groups(self.path)
        self.assertEqual(len(groups), 2)

    def test_n_modes(self):
        c = cio.counts_from_arrays([0.5, 1.0], [[10, 20]], [[1, 2]])
        self.assertTrue(np.array_equal(c.n_for_fit("question-max"), [[20, 20]]))
        self.assertTrue(np.array_equal(c.n_for_fit("per-setting"), [[10, 20]]))

    def test_rejects_bad_rows(self):
        bad = os.path.join(self.tmp, "bad.jsonl")
        write_jsonl(bad, [{"benchmark": "b", "model": "m", "id": "x", "temperature": 1.0, "min_p": None, "n": 5, "c": 6}])
        with self.assertRaises(ValueError):
            cio.load_counts(bad)
        dup = os.path.join(self.tmp, "dup.jsonl")
        r = {"benchmark": "b", "model": "m", "id": "x", "temperature": 1.0, "min_p": None, "n": 5, "c": 1}
        write_jsonl(dup, [r, r])
        with self.assertRaises(ValueError):
            cio.load_counts(dup)


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        path = os.path.join(cls.tmp, "counts.jsonl")
        write_jsonl(path, synthetic_rows())
        cls.counts = cio.load_counts(path)
        cls.prof_np = fa.fit_profiles(cls.counts, backend="numpy", iterations=FIT_ITERS)
        cls.kernel = None
        if _compiler():
            try:
                cls.kernel = fa.build_kernel(os.path.join(cls.tmp, "kernel.so"), "portable", cc=_compiler(),
                                             verbose=False)
            except (subprocess.CalledProcessError, OSError):
                cls.kernel = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_shape_constraints(self):
        p = self.prof_np
        for i in np.where(p.fitted)[0]:
            b = p.bern[i]
            self.assertTrue(np.all(b >= 0))
            k = int(np.argmax(b))
            self.assertTrue(np.all(np.diff(b[:k + 1]) >= -1e-12) and np.all(np.diff(b[k:]) <= 1e-12))
        self.assertTrue(np.all((p.P_fit >= 0) & (p.P_fit <= 1)))
        self.assertEqual(p.peak_fit[0], 0.0)                    # the unsolved question
        self.assertFalse(p.included()[0])
        self.assertTrue(p.included()[1])                        # the easy question
        self.assertGreater(p.peak_fit[1], 90.0)

    def test_c_kernel_equals_numpy(self):
        if self.kernel is None:
            self.skipTest("no C compiler")
        p_c = fa.fit_profiles(self.counts, backend="c", kernel_path=self.kernel, iterations=FIT_ITERS)
        self.assertTrue(np.array_equal(p_c.bern, self.prof_np.bern), "C and numpy kernels differ")
        self.assertTrue(np.array_equal(p_c.P_fit, self.prof_np.P_fit))

    def test_vector_eval_matches_scalar(self):
        rng = np.random.default_rng(3)
        for i in np.where(self.prof_np.fitted)[0]:
            c = [float(v) for v in self.prof_np.coeffs[i]]
            lo, hi = float(self.prof_np.x_lo[i]), float(self.prof_np.x_hi[i])
            ts = np.concatenate([rng.uniform(lo, hi, 50), [lo, hi]])
            vec, inside = fa.eval_legendre_array(c, lo, hi, ts)
            sca = np.array([fa.eval_legendre(c, lo, hi, float(t)) for t in ts])
            if sys.version_info >= (3, 12):
                self.assertTrue(np.array_equal(vec, sca))     # compensated sum() since Python 3.12
            else:
                self.assertTrue(np.allclose(vec, sca, rtol=1e-14, atol=1e-12))

    def test_save_load_roundtrip(self):
        path = os.path.join(self.tmp, "profiles.json")
        self.prof_np.save(path)
        q = fa.load_profiles(path)
        self.assertTrue(np.array_equal(q.P_fit, self.prof_np.P_fit))
        self.assertEqual(q.alloc_settings, self.prof_np.alloc_settings)
        d = self.prof_np.to_json()
        d["questions"] = []
        empty = fa.Profiles.from_json(d)
        self.assertEqual(empty.P_fit.shape, (0, len(self.prof_np.alloc_settings)))

    def test_profile_rows(self):
        rows = fa.profile_rows(self.prof_np, self.counts)
        self.assertEqual(len(rows), int(self.counts.measured.sum()))
        i, k = 3, 2
        r = next(r for r in rows if r["id"] == self.counts.ids[i] and r["temperature"] == SETTINGS[k])
        self.assertEqual(r["p_fit"], float(self.prof_np.P_fit[i, k]))
        self.assertEqual(r["peak_pct"], float(self.prof_np.peak_fit[i]))
        self.assertEqual((r["n"], r["c"]), (int(self.counts.n[i, k]), int(self.counts.c[i, k])))
        path = os.path.join(self.tmp, "rows.jsonl")
        fa.write_profile_rows(self.prof_np, self.counts, path)
        back = [json.loads(line) for line in open(path, encoding="utf-8")]
        self.assertEqual(back, rows)                                   # floats round-trip exactly

    def test_single_setting_dataset(self):
        c = cio.counts_from_arrays([1.0], [[100], [100]], [[0], [5]])
        p = fa.fit_profiles(c, backend="numpy", iterations=50)
        self.assertTrue(np.all(np.isfinite(p.P_fit)) and np.all(np.isfinite(p.peak_fit)))
        self.assertFalse(p.included().any())          # nothing can be fitted on one point


class TestPassK(unittest.TestCase):
    def test_unbiased_against_combinatorics(self):
        for n in range(1, 13):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    want = 1.0 - math.comb(n - c, k) / math.comb(n, k)
                    self.assertAlmostEqual(pk.passk_unbiased(n, c, k), want, places=12)
        self.assertTrue(np.isnan(pk.passk_unbiased(3, 1, 4)))

    def test_unbiased_large_n(self):
        from fractions import Fraction
        n = np.array([11502, 1500, 750])
        c = np.array([25, 0, 749])
        for k in (1, 2, 64, 512):
            exact = [float(1 - Fraction(math.comb(int(a) - int(b), k), math.comb(int(a), k))) for a, b in zip(n, c)]
            self.assertTrue(np.allclose(pk.passk_unbiased(n, c, k), exact, rtol=0, atol=1e-15))
        self.assertAlmostEqual(float(pk.passk_unbiased(11502, 25, 1)), 25 / 11502, places=15)

    def test_plugin_and_scores(self):
        P = np.array([[0.1, 0.2], [0.0, 0.5]])
        L = pk.log1m(P)
        self.assertTrue(np.allclose(pk.passk_plugin(P, 3), 1 - (1 - P) ** 3))
        self.assertTrue(np.allclose(pk.solved_single(L, 2), (1 - (1 - P) ** 2).sum(axis=0)))
        self.assertAlmostEqual(pk.solved_allocation(L, [1, 1]), float(np.sum(1 - (1 - P[:, 0]) * (1 - P[:, 1]))))
        self.assertAlmostEqual(pk.solved_uniform(L, 2), pk.solved_allocation(L, [1, 1]))


class TestAllocation(unittest.TestCase):
    def test_exact_against_brute_force(self):
        rng = np.random.default_rng(7)
        for _ in range(120):
            l = int(rng.integers(2, 6))
            n = int(rng.integers(3, 30))
            K = int(rng.integers(1, 14 if l <= 4 else 10))
            L = random_L(rng, n, l)
            Fb = brute_force(L, K)
            x = al.solve_relaxed(L, K)
            q_p, s_p = al.integer_paper(x, K, L, n)
            q_e, s_e, cert = al.integer_exact(L, K, x_relaxed=x, incumbent=q_p, tol=1e-12)
            self.assertTrue(cert["proven"])
            self.assertEqual(int(q_e.sum()), K)
            self.assertTrue(np.all(q_e >= 0))
            self.assertLessEqual(n - s_e, Fb + 1e-10)              # exact = brute-force optimum
            self.assertGreaterEqual(n - s_p, Fb - 1e-10)           # nothing beats the optimum
            self.assertLessEqual(al.objective(L, x), Fb + 1e-9)    # the relaxation is a bound

    def test_exact_rejects_bad_inputs(self):
        L = np.log([[0.5, 0.5]])
        q, s, cert = al.integer_exact(L, 1, incumbent=[2, 0])        # not an allocation of K = 1
        self.assertTrue(cert["incumbent_rejected"])
        self.assertEqual(int(q.sum()), 1)
        self.assertAlmostEqual(1 - s, 0.5)
        with self.assertRaises(ValueError):
            al.integer_exact(L, 1, tol=-1e-9)

    def test_box_solver_and_kkt(self):
        rng = np.random.default_rng(11)
        L = random_L(rng, 40, 6)
        lo = np.array([0, 1, 0, 0, 2, 0])
        hi = np.array([5, 6, 3, 9, 8, 4])
        x = al.solve_box_simplex(L, 12, lo, hi)
        self.assertAlmostEqual(float(x.sum()), 12.0, places=9)
        self.assertTrue(np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12))
        xr = al.solve_relaxed(L, 12)
        rep = al.kkt_report(L, 12, xr)
        self.assertLess(rep["eq_gap"], 1e-8)
        self.assertGreaterEqual(rep["first_order_gap_bound"], -1e-9)
        xn = al.solve_relaxed(L, 12, method="newton")
        rep_n = al.kkt_report(L, 12, xn)
        self.assertLess(rep_n["eq_gap"], 1e-12)
        self.assertLess(rep_n["first_order_gap_bound"], 1e-8)
        self.assertLessEqual(al.objective(L, xn), al.objective(L, xr) + 1e-9)
        res = al.optimal_allocation(L, 12, "exact")
        self.assertTrue(res["certificate"]["proven"])
        self.assertGreaterEqual(res["solved_relaxed"], res["solved_exact"] - 1e-9)


class TestHeldout(unittest.TestCase):
    def test_paper_fold_rule(self):
        folds, mode = ho.make_folds(359, 10)
        self.assertEqual(mode, "k=10")
        self.assertEqual(sorted(len(f) for f in folds), [35] + [36] * 9)
        # the 10-fold split of the paper's Figure 3 (n = 359 included questions)
        self.assertEqual(ho.folds_sha256(folds), "12ee3719883b553060765fb98666bfb15b28caa4a355d340384fe6cc92234eed")
        loo, mode = ho.make_folds(5, None)
        self.assertEqual(mode, "loo")
        self.assertEqual([f.tolist() for f in loo], [[0], [1], [2], [3], [4]])

    def test_compare_invariants(self):
        rng = np.random.default_rng(5)
        L = random_L(rng, 30, 5)
        ks = [1, 2, 4, 8, 16]
        for k_folds in (None, 3):
            for integer in ("paper", "exact"):
                r = ho.compare(L, ks, k_folds, integer, ref_index=2)
                for e in r["per_k"]:
                    self.assertGreaterEqual(e["in_sample_solved"], e["case2_total"] - 1e-9)
                    self.assertGreaterEqual(e["in_sample_solved"], e["case3_total"] - 1e-9)
                    self.assertEqual(sum(e["q_int_full"]), e["k"])
                    self.assertAlmostEqual(e["ratio_optimal_heldout"], e["mean_solved"] / e["case3_total"])
                if integer == "exact":
                    self.assertTrue(r["exact_checks"]["exact_proven_all"])


class TestBootstrap(unittest.TestCase):
    def test_fast_equals_paper_loop(self):
        rng = np.random.default_rng(9)
        L = random_L(rng, 25, 4)
        a = bs.ratio_to_uniform(L, [1, 4, 16], B=300, seed=1, method="fast")
        b = bs.ratio_to_uniform(L, [1, 4, 16], B=300, seed=1, method="paper")
        self.assertTrue(np.allclose(a["ratio"], b["ratio"], rtol=0, atol=1e-14))
        self.assertTrue(np.allclose(a["ratio_se"], b["ratio_se"], rtol=1e-9, atol=1e-14))
        c = bs.ratio_to_uniform(L, [1, 4, 16], B=300, seed=1, method="fast")
        self.assertEqual(a["ratio_se"], c["ratio_se"])
        d = bs.ratio_to_uniform(L, [1, 4, 16], B=300, seed=1, seed_scheme="spawn")
        self.assertNotEqual(a["ratio_se"], d["ratio_se"])
        r = bs.bootstrap_ratio(1 - np.exp(4 * L[:, 0]), 1 - np.exp(L.mean(axis=1) * 4), B=200, seed=0)
        self.assertGreater(r["se"][0], 0)


class TestTablesAndPipeline(unittest.TestCase):
    def test_tables_bold_rule(self):
        row = {"K": 2, "per_setting": {"0.5": 0.2}, "uniform": 0.19, "best_in": 0.238, "best_out": 0.2381,
               "optimal_in": 0.2386, "optimal_out": 0.2347}
        best, uni, opt = tb._aggregate_cells(row)
        self.assertIn("\\textbf{0.239}", opt.split("\\\\")[0])      # optimal wins in-sample
        self.assertIn("\\textbf{0.238}", best.split("\\\\")[-1])    # best single wins held-out
        self.assertEqual(uni, "0.190")

    def test_pipeline_end_to_end(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "counts.jsonl")
            write_jsonl(path, synthetic_rows(n_q=10))
            out = os.path.join(tmp, "out")
            res = pl.run(path, out, max_k=64, folds=("loo", 3), B=100, iterations=FIT_ITERS, backend="numpy",
                         name="Synthetic", plot=True, log=lambda *a: None)
            for fn in ("profiles.json", "heldout_loo.json", "heldout_kf3.json", "rtu.json", "passk_table.tex",
                       "passk_table.csv", "passk_table.md", "summary.json", "run.json", "rtu_loo.png"):
                self.assertTrue(os.path.exists(os.path.join(out, fn)), fn)
            self.assertEqual(res["info"]["n_included"], 9)
            tex = open(os.path.join(out, "passk_table.tex"), encoding="utf-8").read()
            self.assertIn("\\begin{tabular}", tex)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
