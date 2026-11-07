"""Figure 4(b): the share of random reweightings W that lose under 10%, per K and KL band, from 24 million W's.

Panel (b) of Figure 4 repeats the top row of Figure 17 (Mistral 7B Instruct on LiveCodeBench, 359 included questions)
with N = 24,000,000 W's instead of 10,000 (Appendix G, "Experiment"):
  - the calibration table of the Dirichlet smooth-base sampler is computed once, from SeedSequence(20260606) child 0
    (the table of the 10,000-W figure);
  - chunk i = 0 ... 2399 draws 10,000 W's from SeedSequence(20260606, spawn_key=(1 + i,)), in log space
    (dirichlet mode "log"), so that no W underflows;
  - each W is elected at K = 4, 32, 128 and 512 by rule "split" (election.py): argmin_t log F_t, with the settings
    whose log F is within 1e-12 of the minimum counted as an even tie-break;
  - a W "loses under 10%" when its tau_B has ratio S_U(tau_B) / score_A >= 0.9; the cell is the share of the W's of a
    KL band that do, ties counted by their expected share;
  - KL bands [0,1), ..., [4,5), [5, ln 359); W's with KL >= ln 359 are not counted.
The text of Section 4.3.2 quotes the KL > 3 share (bands 3-4, 4-5, 5-5.88 pooled), averaged over the four K, and the
KL 1-2 share.

Three commands:
  - ``check``: the fast check. It recomputes a seeded sample of chunks (default 50, about 4 s each on one core) from
    the counts and the profile fit, compares each with the paper's run of the same chunk (the 2,400 tallies shipped in
    data/reference/fig4b_chunk_tallies.jsonl: the hashes of the 10,000 W's and of their KL values, the band counts and
    every tally), then prints the panel merged from all 2,400 shipped tallies.
  - ``worker``: the full run. Chunks are independent, so it splits over processes: worker w of W takes the chunks
    i = w, w + W, ...; each chunk's tally is written to its own file (one writer per path).
  - ``merge``: sums a run's tallies into the panel, and compares them chunk by chunk with another run's.

    python -m onetemp.robustness.fig4b check --input counts.jsonl --profiles profiles.json
    python -m onetemp.robustness.fig4b worker --input counts.jsonl --profiles profiles.json \
        --tally-dir tallies --worker 0 --n-workers 8
    python -m onetemp.robustness.fig4b merge --tally-dir tallies --expect-chunks 2400 --out fig4b_cells.json \
        --compare-dir data/reference/fig4b_chunk_tallies.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from . import contexts, dirichlet, election
from .data import attach_core_profiles, environment, guard_outputs, load_study
from .heatmap import kl_col_edges, kl_col_labels

K_PAPER = (4, 32, 128, 512)
N_CHUNKS = 2400
CHUNK = 10000
THR = 0.9
TALLY_KEYS = ("yd", "fail_d", "tied_d")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def setup(study, Ks, seed):
    """The figure's context and its calibration table, with the identity hashes every tally carries."""
    ctx = contexts.kl_context(study, Ks, rates="fit")
    calib_rng, _ = dirichlet.figure_streams(seed)
    c_grid, kl_med = dirichlet.calibrate(ctx["n_inc"], calib_rng)
    cal_sha = _sha(np.ascontiguousarray(c_grid).tobytes() + np.ascontiguousarray(kl_med).tobytes())
    ctx_sha = _sha(b"".join(np.ascontiguousarray(ctx["per_k"][int(K)]["pass_mat"]).tobytes() for K in Ks)
                   + np.ascontiguousarray(ctx["L_sorted"]).tobytes() + np.ascontiguousarray(ctx["u_sorted"]).tobytes())
    return ctx, c_grid, kl_med, cal_sha, ctx_sha


def make_tally(i: int, ctx, c_grid, kl_med, Ks, seed: int, chunk: int, col_edges) -> dict:
    """Draw chunk i, elect it at every K, and tally the bands."""
    t0 = time.perf_counter()
    nb = len(col_edges) - 1
    ln_n = math.log(ctx["n_inc"])
    dr = dirichlet.draw(ctx["u_sorted"], chunk, "log", dirichlet.chunk_stream(i, seed), c_grid, kl_med)
    W, logW, kl = dr["W"], dr["logW"], dr["kl"]
    t1 = time.perf_counter()
    finite = np.isfinite(kl)
    col = np.searchsorted(col_edges, kl, side="right") - 1
    valid = (col >= 0) & (col < nb) & finite
    cv = col[valid]
    wsum = W.sum(axis=1)
    T = {"chunk": i, "seed_entropy": seed, "spawn_key": [1 + i], "chunk_size": chunk,
         "W_sha256": _sha(np.ascontiguousarray(W).tobytes()), "kl_sha256": _sha(np.ascontiguousarray(kl).tobytes()),
         "n_band": np.bincount(cv, minlength=nb).tolist(),
         "n_kl_ge_ln_n": int((finite & (kl >= ln_n)).sum()), "n_kl_lt_0": int((finite & (kl < 0)).sum()),
         "n_nonfinite": int((~finite).sum()), "n_all_zero": int((wsum == 0).sum()),
         "n_sum_off_1e-9": int((np.abs(wsum - 1.0) > 1e-9).sum()), "per_K": {}}
    for K in Ks:
        ck = ctx["per_k"][int(K)]
        good = ck["ratio_per_t"] >= THR
        tied = election.tied_split(logW, ck["log_fail"])
        yd = election.split_share(tied, good)
        o = {"yd": yd, "fail_d": (yd < 1.0).astype(float), "tied_d": (tied.sum(axis=1) > 1).astype(float)}
        T["per_K"][str(int(K))] = {k: np.bincount(cv, weights=o[k][valid], minlength=nb).tolist() for k in TALLY_KEYS}
    T["seconds"] = {"draw": round(t1 - t0, 3), "elect": round(time.perf_counter() - t1, 3)}
    return T


def worker(a):
    study = load_study(a.input, a.benchmark, a.model, "temperature", require_fit=not a.profiles)
    if a.profiles:
        study = attach_core_profiles(study, a.profiles)
    Ks = [int(k) for k in a.K.split(",")]
    ctx, c_grid, kl_med, cal_sha, ctx_sha = setup(study, Ks, a.seed)
    col_edges = kl_col_edges(ctx["n_inc"])
    tdir = Path(a.tally_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    mine = [i for i in range(a.n_chunks) if i % a.n_workers == a.worker]
    if a.only_chunks:
        mine = [int(x) for x in a.only_chunks.split(",") if int(x) % a.n_workers == a.worker]
    ident = {"seed_entropy": a.seed, "chunk_size": a.chunk_size, "calibration_sha256": cal_sha, "ctx_sha256": ctx_sha,
             "K": Ks, "col_edges": col_edges.tolist(), "n_inc": ctx["n_inc"]}
    print(f"[fig4b] worker {a.worker}/{a.n_workers}: {len(mine)} chunks; n_inc {ctx['n_inc']}; calibration "
          f"{cal_sha[:12]}; ctx {ctx_sha[:12]}", flush=True)
    made = skipped = 0
    t_all = time.perf_counter()
    for i in mine:
        path = tdir / f"chunk_{i:04d}.json"
        if path.exists():
            try:
                old = json.loads(path.read_text())
                if all(old.get(k) == v for k, v in ident.items()) and old.get("chunk") == i:
                    skipped += 1
                    continue
            except (OSError, ValueError):
                pass
        T = make_tally(i, ctx, c_grid, kl_med, Ks, a.seed, a.chunk_size, col_edges) | ident | {"worker": a.worker}
        tmp = tdir / f".chunk_{i:04d}.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(T, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        made += 1
        if made == 1 or made % 25 == 0:
            print(f"[fig4b] w{a.worker} chunk {i} ({made} made, {skipped} skipped) {T['seconds']}", flush=True)
    summ = {"worker": a.worker, "n_workers": a.n_workers, "made": made, "skipped": skipped, "assigned": len(mine),
            "seconds": round(time.perf_counter() - t_all, 1),
            "max_rss_MiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)} | ident
    (tdir / f"worker_{a.worker}_of_{a.n_workers}.summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[fig4b] done {json.dumps(summ)}", flush=True)


def clopper_pearson(k: float, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    k = int(round(k))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return 100 * lo, 100 * hi


IDENT_KEYS = ("seed_entropy", "chunk_size", "calibration_sha256", "ctx_sha256", "K", "col_edges", "n_inc")
COUNTER_KEYS = ("n_kl_ge_ln_n", "n_kl_lt_0", "n_nonfinite", "n_all_zero", "n_sum_off_1e-9")


def load_tallies(src) -> dict:
    """Tallies by chunk index, from a folder of chunk_*.json files (a worker's output) or from one .jsonl file with
    one tally per line (the shipped reference, data/reference/fig4b_chunk_tallies.jsonl)."""
    src = Path(src)
    out = {}
    if src.is_dir():
        for f in sorted(src.glob("chunk_*.json")):
            t = json.loads(f.read_text())
            if int(t["chunk"]) in out:
                raise ValueError(f"{src}: chunk {t['chunk']} appears in two files ({f.name} and an earlier one)")
            out[int(t["chunk"])] = t
        return out
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                t = json.loads(line)
                if int(t["chunk"]) in out:
                    raise ValueError(f"{src}: chunk {t['chunk']} appears twice")
                out[int(t["chunk"])] = t
    return out


def merge_tallies(tallies: list, expect_chunks: int = 0) -> dict:
    """Sum the chunk tallies into the panel's cells, the KL > 3 and KL 1-2 shares, and their means over K."""
    if not tallies:
        raise ValueError("no tallies to merge")
    first = tallies[0]
    for t in tallies:
        for k in IDENT_KEYS:
            if t.get(k) != first.get(k):
                raise ValueError(f"chunk {t['chunk']}: {k} differs from chunk {first['chunk']}")
    chunks = sorted(int(t["chunk"]) for t in tallies)
    if len(set(chunks)) != len(chunks):
        raise ValueError("duplicate chunks")
    if expect_chunks and chunks != list(range(expect_chunks)):
        raise ValueError(f"expected chunks 0..{expect_chunks - 1}, found {len(chunks)} ({chunks[:3]}...)")
    Ks, col_edges = first["K"], first["col_edges"]
    labels = kl_col_labels(col_edges)
    nb = len(col_edges) - 1
    n_band = np.zeros(nb)
    sums = {str(K): {k: np.zeros(nb) for k in TALLY_KEYS} for K in Ks}
    totals = {k: 0 for k in COUNTER_KEYS}
    for t in tallies:
        n_band += np.asarray(t["n_band"], dtype=float)
        for k in totals:
            totals[k] += int(t[k])
        for K in Ks:
            for k in TALLY_KEYS:
                sums[str(K)][k] += np.asarray(t["per_K"][str(K)][k], dtype=float)
    n_w = int(n_band.sum()) + totals["n_kl_ge_ln_n"] + totals["n_kl_lt_0"] + totals["n_nonfinite"]
    cells, kl3, kl12 = {}, {}, {}
    hi_bands = [c for c in range(nb) if col_edges[c] >= 3.0 - 1e-12]
    b12 = [c for c in range(nb) if abs(col_edges[c] - 1.0) < 1e-12 and abs(col_edges[c + 1] - 2.0) < 1e-12]
    for K in Ks:
        yd = sums[str(K)]["yd"]
        cells[str(K)] = [{"band": labels[c], "n": int(n_band[c]), "share_pct": (100 * yd[c] / n_band[c]) if n_band[c] else None,
                          "label": share_label(yd[c], n_band[c], sums[str(K)]["fail_d"][c]),
                          "n_failing_W": int(sums[str(K)]["fail_d"][c]), "n_tied_W": int(sums[str(K)]["tied_d"][c]),
                          "cp95_pct": clopper_pearson(yd[c], int(n_band[c]))} for c in range(nb)]
        kl3[str(K)] = 100 * yd[hi_bands].sum() / n_band[hi_bands].sum()
        kl12[str(K)] = 100 * yd[b12].sum() / n_band[b12].sum() if b12 else float("nan")
    return {"n_chunks": len(chunks), "n_W": n_w, "K": Ks, "bands": labels, "not_counted": totals, "cells": cells,
            "kl_over_3_pct": kl3, "kl_over_3_mean_over_K": float(np.mean(list(kl3.values()))),
            "kl_1_2_pct": kl12, "kl_1_2_mean_over_K": float(np.mean(list(kl12.values()))),
            "identity": {k: first[k] for k in IDENT_KEYS}}


def merge(a):
    try:
        tallies = list(load_tallies(a.tally_dir).values())
        if not tallies:
            sys.exit(f"no tallies in {a.tally_dir}")
        guard_outputs([a.out])
        out = merge_tallies(tallies, a.expect_chunks)
    except ValueError as e:
        sys.exit(str(e))
    out["environment"] = environment()
    if a.compare_dir:
        out["compare"] = compare(tallies, load_tallies(a.compare_dir))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(panel_markdown(out))
    print()
    print(headline(out))
    if not a.compare_dir:
        return 0
    cmp = out["compare"]
    print(json.dumps(cmp, indent=1))
    ok = cmp["compared"] == len(tallies) and cmp["n_hash_mismatch"] == 0 and cmp["n_tally_mismatch"] == 0
    print(f"\n[fig4b merge] {'PASS' if ok else 'FAIL'}: {cmp['n_chunks_equal']} of {len(tallies)} chunks equal to "
          f"{a.compare_dir} (W and KL hashes, band counts, every tally); {cmp['missing_in_reference']} not in it")
    return 0 if ok else 1


def share_label(yd: float, n: float, fail: float) -> str:
    """Two decimals; "100" only when no W fails, and a share that would round to 100.00 while some W fails prints 99.99."""
    if n == 0:
        return ""
    if fail == 0:
        return "100"
    v = 100 * yd / n
    s = f"{v:.2f}"
    return "99.99" if s == "100.00" else s


def panel_markdown(out: dict) -> str:
    lines = ["| K | " + " | ".join(out["bands"]) + " | KL > 3 | KL 1-2 |", "|" + "---|" * (len(out["bands"]) + 3)]
    for K in sorted(out["K"], reverse=True):
        lines.append(f"| {K} | " + " | ".join(c["label"] for c in out["cells"][str(K)])
                     + f" | {out['kl_over_3_pct'][str(K)]:.2f} | {out['kl_1_2_pct'][str(K)]:.2f} |")
    lines.append(f"| mean over K | {' | '.join('' for _ in out['bands'])} | {out['kl_over_3_mean_over_K']:.2f} | "
                 f"{out['kl_1_2_mean_over_K']:.2f} |")
    return "\n".join(lines)


def compare(tallies, ref: dict) -> dict:
    """Chunk-by-chunk comparison with another run's tallies (``ref``: tallies by chunk, see ``load_tallies``): the
    hashes of the W's and of their KL values, the band counts, the counters of W's not counted, and every tally."""
    res = {"compared": 0, "missing_in_reference": 0, "hash_mismatch": [], "tally_mismatch": [], "max_abs_tally_diff": 0.0}
    bad = set()
    for t in tallies:
        r = ref.get(int(t["chunk"]))
        if r is None:
            res["missing_in_reference"] += 1
            continue
        res["compared"] += 1
        if r.get("W_sha256") != t["W_sha256"] or r.get("kl_sha256") != t["kl_sha256"]:
            res["hash_mismatch"].append(t["chunk"])
            bad.add(int(t["chunk"]))
        for k in ("n_band",) + COUNTER_KEYS:
            if r.get(k) != t[k]:
                res["tally_mismatch"].append([t["chunk"], k])
                bad.add(int(t["chunk"]))
        for K, v in t["per_K"].items():
            for k in TALLY_KEYS:
                a_, b_ = np.asarray(v[k]), np.asarray(r["per_K"][K][k])
                d = float(np.abs(a_ - b_).max())
                res["max_abs_tally_diff"] = max(res["max_abs_tally_diff"], d)
                if d != 0.0:
                    res["tally_mismatch"].append([t["chunk"], K, k, d])
                    bad.add(int(t["chunk"]))
    res["n_chunks_equal"] = res["compared"] - len(bad)
    res["n_hash_mismatch"] = len(res["hash_mismatch"])
    res["n_tally_mismatch"] = len(res["tally_mismatch"])
    res["hash_mismatch"] = res["hash_mismatch"][:20]
    res["tally_mismatch"] = res["tally_mismatch"][:20]
    return res


def headline(out: dict) -> str:
    """The summary lines of a merged panel: the KL > 3 share per K and its mean over K (Section 4.3.2 prints the mean
    as "on average roughly 90%"), the KL 1-2 mean ("rises to 99%"), and the K = 4 cell of the last band."""
    k3 = out["kl_over_3_pct"]
    last = out["cells"]["4"][-1] if "4" in out["cells"] else None
    lines = [f"KL > 3, mean over K = {', '.join(str(k) for k in out['K'])}: {out['kl_over_3_mean_over_K']:.2f}%",
             "KL > 3, per K: " + " / ".join(f"{k3[str(k)]:.2f}" for k in out["K"]),
             f"KL 1-2, mean over K: {out['kl_1_2_mean_over_K']:.2f}%"]
    if last is not None:
        lines.append(f"K = 4, KL {last['band']}: {last['label']}")
    return "\n".join(lines)


def check(a):
    """The fast check: recompute a seeded sample of chunks from the counts and the profile fit, compare each with the
    paper's run (the shipped tallies), then print the panel merged from all 2,400 shipped tallies."""
    from onetemp.paths import reference_dir
    ref_path = Path(a.reference) if a.reference else reference_dir() / "fig4b_chunk_tallies.jsonl"
    ref = load_tallies(ref_path)
    if a.out:
        guard_outputs([a.out])
    if sorted(ref) != list(range(N_CHUNKS)):
        sys.exit(f"{ref_path}: expected the {N_CHUNKS} chunks of the paper's run, found {len(ref)}")
    r0 = ref[0]
    study = load_study(a.input, a.benchmark, a.model, "temperature", require_fit=not a.profiles)
    if a.profiles:
        study = attach_core_profiles(study, a.profiles)
    Ks = [int(k) for k in r0["K"]]
    ctx, c_grid, kl_med, cal_sha, ctx_sha = setup(study, Ks, int(r0["seed_entropy"]))
    col_edges = kl_col_edges(ctx["n_inc"])
    ident = {"n_inc": ctx["n_inc"] == r0["n_inc"], "col_edges": col_edges.tolist() == r0["col_edges"],
             "calibration_sha256": cal_sha == r0["calibration_sha256"]}
    print(f"[fig4b check] {ctx['n_inc']} included questions (reference {r0['n_inc']}); calibration table "
          f"{'equal to' if ident['calibration_sha256'] else 'DIFFERENT from'} the paper's run; context hash "
          f"{'equal to' if ctx_sha == r0['ctx_sha256'] else 'not equal to'} the paper's run", flush=True)
    if ctx_sha != r0["ctx_sha256"]:
        print("[fig4b check]   (the context hash covers the fitted pass@K matrices to the last bit: it equals the "
              "paper's only with the kernel built --build-flags paper; the chunk tallies below decide)", flush=True)
    if not all(ident.values()):
        sys.exit(f"[fig4b check] inputs differ from the paper's run: {ident}")
    if a.chunks:
        chunks = sorted({int(x) for x in a.chunks.split(",")})
    else:
        rng = np.random.default_rng(a.sample_seed)
        chunks = sorted(int(i) for i in rng.choice(N_CHUNKS, size=min(a.n_sample, N_CHUNKS), replace=False))
    print(f"[fig4b check] recomputing {len(chunks)} of {N_CHUNKS} chunks (10,000 W's each; sample seed "
          f"{a.sample_seed}): {chunks[:8]}{' ...' if len(chunks) > 8 else ''}", flush=True)
    t0 = time.perf_counter()
    mine = []
    for n_done, i in enumerate(chunks, 1):
        T = make_tally(i, ctx, c_grid, kl_med, Ks, int(r0["seed_entropy"]), int(r0["chunk_size"]), col_edges)
        mine.append(T)
        if n_done == 1 or n_done % 10 == 0 or n_done == len(chunks):
            print(f"[fig4b check]   {n_done}/{len(chunks)} chunks, {time.perf_counter() - t0:.0f} s", flush=True)
    cmp = compare(mine, ref)
    ok = cmp["compared"] == len(chunks) and cmp["n_hash_mismatch"] == 0 and cmp["n_tally_mismatch"] == 0
    print(f"[fig4b check] {cmp['n_chunks_equal']}"
          f" of {len(chunks)} chunks equal to the paper's run (W and KL hashes, band counts, every tally); "
          f"largest tally difference {cmp['max_abs_tally_diff']:g}", flush=True)
    out = merge_tallies(list(ref.values()), N_CHUNKS)
    print(f"\nFigure 4(b), merged from the {out['n_chunks']} chunk tallies of the paper's run ({out['n_W']:,} W's; "
          f"{out['not_counted']['n_kl_ge_ln_n']:,} with KL >= ln {out['identity']['n_inc']} not counted):\n")
    print(panel_markdown(out))
    print()
    print(headline(out))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"chunks_checked": chunks, "compare": cmp, "all_equal": ok,
                                           "identity_equal": ident, "ctx_sha256_equal": ctx_sha == r0["ctx_sha256"],
                                           "panel": out, "environment": environment()}, indent=1))
    print(f"\n[fig4b check] {'PASS' if ok else 'FAIL'}: {len(chunks)} recomputed chunks "
          f"{'all equal' if ok else 'differ from'} the paper's tallies")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker", help="draw, elect and tally this worker's chunks")
    w.add_argument("--input", required=True)
    w.add_argument("--profiles", default=None, help="profiles.json of the profile fit (python -m onetemp.core.fit_all)")
    w.add_argument("--benchmark", default=None)
    w.add_argument("--model", default=None)
    w.add_argument("--K", default="4,32,128,512")
    w.add_argument("--seed", type=int, default=dirichlet.SEED)
    w.add_argument("--n-chunks", type=int, default=N_CHUNKS)
    w.add_argument("--chunk-size", type=int, default=CHUNK)
    w.add_argument("--worker", type=int, default=0)
    w.add_argument("--n-workers", type=int, default=1)
    w.add_argument("--only-chunks", default=None, help="comma-separated chunk indices")
    w.add_argument("--tally-dir", required=True)
    m = sub.add_parser("merge", help="sum the tallies into the panel's cells")
    m.add_argument("--tally-dir", required=True, help="a worker's tally folder, or a .jsonl file of tallies")
    m.add_argument("--out", required=True)
    m.add_argument("--expect-chunks", type=int, default=0, help="require exactly chunks 0..N-1")
    m.add_argument("--compare-dir", default=None,
                   help="another run's tallies (folder or .jsonl, e.g. the shipped reference), compared chunk by chunk")
    c = sub.add_parser("check", help="recompute a seeded sample of chunks and compare with the paper's run")
    c.add_argument("--input", required=True, help="the counts file (LiveCodeBench, Mistral 7B Instruct, question_n)")
    c.add_argument("--profiles", default=None, help="profiles.json of the fit with --n-mode question-max")
    c.add_argument("--benchmark", default=None)
    c.add_argument("--model", default=None)
    c.add_argument("--reference", default=None, help="default: data/reference/fig4b_chunk_tallies.jsonl")
    c.add_argument("--n-sample", type=int, default=50, help="chunks to recompute (default 50, about 4 s each)")
    c.add_argument("--sample-seed", type=int, default=0)
    c.add_argument("--chunks", default=None, help="comma-separated chunk indices instead of a sample")
    c.add_argument("--out", default=None, help="write the comparison and the merged panel as JSON")
    a = ap.parse_args(argv)
    if a.cmd == "worker":
        return worker(a)
    if a.cmd == "merge":
        return merge(a)
    return check(a)


if __name__ == "__main__":
    sys.exit(main())
