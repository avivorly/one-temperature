"""Read per-question counts (and the fitted profiles) into one *study*.

Input format (shared by every analysis of the release): JSON lines, one object per (benchmark, model, question,
setting),

    {"benchmark": "livecodebench" | "math500" | "hitom", "model": "<HF repo id>", "id": "<question id>",
     "temperature": float, "min_p": float or null, "n": int, "c": int}

``n`` is the number of samples and ``c`` the number correct. A file of this shape is a ``counts.jsonl``.

The temperature analyses use the fitted question profile of Appendix D (the degree-10 unimodal Bernstein fit of
Table 1). It can be supplied in either of two ways:
  - ``--profiles profiles.json``: the output of the release's profile fit (``python -m onetemp.core.fit_all``), whose
    questions carry ``P_fit`` (the fitted success probability on the allocation grid) and ``peak_fit_pct``;
  - two extra fields on every row of the counts file (a ``profiles.jsonl``):
        "p_fit":    the fitted success probability at this setting, clipped to [0, 1] (0 outside the fit's bracket);
        "peak_pct": the fitted peak of the question in PERCENT, the maximum of the fitted polynomial over a
                    1001-point grid of its bracket, rounded to 6 decimals (the same on every row of the question).
Both give the same study.

A *study* is one sweep of one sampling parameter, the *setting axis*:
  - a temperature sweep: rows with ``min_p`` null, setting = ``temperature``;
  - a min_p sweep at a fixed temperature tau: rows with ``min_p`` set, setting = ``min_p``, tau = ``temperature``
    (an optional ``"tau"`` field, if present, must equal it).
The analyses only see the setting values, so every method works on either axis (Appendix I).

Question order is the order of first appearance in the file. The random reweighting of Appendix G assigns its
Dirichlet draws to questions in hardness order, and questions of equal hardness keep the order numpy's default argsort
gives them, so its cells reproduce the paper's only with the paper's question order; any order gives the same
distribution of results.

Greedy decoding (temperature 0) is never a setting of any analysis (the paper drops it everywhere). A setting is kept
only if every question of the study has it (the parse of the paper's Appendix H data); the dropped ones are reported.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from onetemp.counts import iter_counts

BENCHMARKS = ("livecodebench", "math500", "hitom")


@dataclass
class Study:
    """One sweep: questions x settings, with counts and (optionally) the fitted profile."""

    benchmark: str
    model: str
    axis: str                      # "temperature" or "min_p"
    tau: float | None              # the fixed temperature of a min_p sweep, else None
    ids: list                      # question ids, file order
    settings: np.ndarray           # (T,) ascending
    n: np.ndarray                  # (Q, T) int64 samples
    c: np.ndarray                  # (Q, T) int64 correct
    p_fit: np.ndarray | None = None      # (Q, T) fitted success probability, or None
    peak_pct: np.ndarray | None = None   # (Q,) fitted peak in percent, or None
    dropped_settings: list = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.axis == "min_p":
            return f"{self.benchmark} | {self.model} | min_p at tau={self.tau:g}"
        return f"{self.benchmark} | {self.model} | temperature"

    def take(self, idx) -> "Study":
        """The study on the questions ``idx`` (an index array, repeats allowed, e.g. a bootstrap resample, or a
        boolean mask). Repeated questions get the ids ``<id>#<k>`` so that ids stay unique."""
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        seen: dict = {}
        ids = []
        for i in idx.tolist():
            k = seen.get(i, 0)
            seen[i] = k + 1
            ids.append(self.ids[i] if k == 0 else f"{self.ids[i]}#{k}")
        return Study(self.benchmark, self.model, self.axis, self.tau, ids, self.settings.copy(), self.n[idx],
                     self.c[idx], None if self.p_fit is None else self.p_fit[idx],
                     None if self.peak_pct is None else self.peak_pct[idx], list(self.dropped_settings))

    def subset(self, mask) -> "Study":
        """The study restricted to the questions where ``mask`` is True (order kept)."""
        return self.take(np.asarray(mask, dtype=bool))


def _read_rows(path: Path):
    """The rows of a counts.jsonl or profiles.jsonl file, through the package's one reader (validated)."""
    yield from iter_counts(path)


def _as_int(v, what):
    if isinstance(v, bool) or not float(v).is_integer():
        raise ValueError(f"{what} must be an integer, got {v!r}")
    return int(v)


def list_studies(path) -> list:
    """Every (benchmark, model, axis, tau) present in a counts/profiles file."""
    seen = {}
    for r in _read_rows(Path(path)):
        mp = r.get("min_p")
        key = (r["benchmark"], r["model"], "temperature" if mp is None else "min_p",
               None if mp is None else float(r["temperature"]))
        seen[key] = seen.get(key, 0) + 1
    return [dict(benchmark=k[0], model=k[1], axis=k[2], tau=k[3], rows=v) for k, v in seen.items()]


def load_study(path, benchmark: str | None = None, model: str | None = None, axis: str | None = None,
               tau: float | None = None, require_fit: bool = False, quiet: bool = False) -> Study:
    """Load one study from ``path``. ``benchmark``/``model``/``axis``/``tau`` select it when the file holds several.

    Raises if the selection is ambiguous, if a (question, setting) appears twice, if a count is not an integer or
    c > n, if a fitted value is not finite, or if ``require_fit`` and the rows carry no ``p_fit``/``peak_pct``.
    """
    path = Path(path)
    rows = []
    for r in _read_rows(path):
        mp = r.get("min_p")
        ax = "temperature" if mp is None else "min_p"
        if benchmark is not None and r["benchmark"] != benchmark:
            continue
        if model is not None and r["model"] != model:
            continue
        if axis is not None and ax != axis:
            continue
        if tau is not None and (mp is None or not math.isclose(float(r["temperature"]), float(tau), abs_tol=1e-12)):
            continue
        rows.append(r)
    if not rows:
        raise ValueError(f"{path}: no rows match benchmark={benchmark} model={model} axis={axis} tau={tau}")
    keys = {(r["benchmark"], r["model"], "temperature" if r.get("min_p") is None else "min_p",
             None if r.get("min_p") is None else float(r["temperature"])) for r in rows}
    if len(keys) != 1:
        raise ValueError(f"{path}: the selection holds {len(keys)} studies {sorted(map(str, keys))}; "
                         f"pass --benchmark/--model/--axis/--tau")
    (bm, md, ax, tv), = keys
    if ax == "min_p":
        for r in rows:
            if "tau" in r and r["tau"] is not None and not math.isclose(float(r["tau"]), tv, abs_tol=1e-12):
                raise ValueError(f"{path}: row {r['id']} has tau {r['tau']} but temperature {tv}")

    def setting_of(r):
        return float(r["min_p"]) if ax == "min_p" else float(r["temperature"])

    ids, index = [], {}
    cells = {}
    for r in rows:
        s = setting_of(r)
        if ax == "temperature" and s == 0.0:
            continue                                    # greedy decoding is never a setting
        q = str(r["id"])
        if q not in index:
            index[q] = len(ids)
            ids.append(q)
        key = (index[q], s)
        if key in cells:
            raise ValueError(f"{path}: question {q} has setting {s} twice")
        cells[key] = r
    if not ids:
        raise ValueError(f"{path}: no row with a sampling setting (only greedy decoding?)")
    all_settings = sorted({k[1] for k in cells})
    per_q = [set() for _ in ids]
    for (qi, s) in cells:
        per_q[qi].add(s)
    keep = [s for s in all_settings if all(s in st for st in per_q)]
    dropped = [s for s in all_settings if s not in keep]
    if not keep:
        raise ValueError(f"{path}: no setting is present for every question")
    if dropped and not quiet:
        print(f"[data] {path.name}: settings {dropped} are missing for some questions and are left out",
              file=sys.stderr)
    Q, T = len(ids), len(keep)
    n = np.zeros((Q, T), dtype=np.int64)
    c = np.zeros((Q, T), dtype=np.int64)
    has_fit = all("p_fit" in cells[(qi, s)] for qi in range(Q) for s in keep)
    has_peak = all("peak_pct" in cells[(qi, keep[0])] for qi in range(Q))
    p_fit = np.zeros((Q, T), dtype=np.float64) if has_fit else None
    peak = np.zeros(Q, dtype=np.float64) if has_peak else None
    for qi in range(Q):
        for j, s in enumerate(keep):
            r = cells[(qi, s)]
            n[qi, j] = _as_int(r["n"], f"question {ids[qi]} setting {s}: n")
            c[qi, j] = _as_int(r["c"], f"question {ids[qi]} setting {s}: c")
            if not (0 <= c[qi, j] <= n[qi, j]):
                raise ValueError(f"{path}: question {ids[qi]} setting {s}: c={c[qi, j]} n={n[qi, j]}")
            if has_fit:
                p_fit[qi, j] = float(r["p_fit"])
        if has_peak:
            peak[qi] = float(cells[(qi, keep[0])]["peak_pct"])
    st = Study(bm, md, ax, tv, ids, np.asarray(keep, dtype=np.float64), n, c, p_fit, peak, dropped)
    _check_fit(st, path)
    if require_fit and (st.p_fit is None or st.peak_pct is None):
        raise ValueError(f"{path}: no fitted profile (p_fit/peak_pct); pass --profiles from the profile fit, "
                         f"or use --rates measured")
    return st


def _check_fit(st: Study, where) -> None:
    if st.p_fit is not None and not (np.isfinite(st.p_fit).all() and st.p_fit.min() >= 0 and st.p_fit.max() <= 1):
        raise ValueError(f"{where}: p_fit must be finite and in [0, 1]")
    if st.peak_pct is not None and not np.isfinite(st.peak_pct).all():
        raise ValueError(f"{where}: peak_pct must be finite")


def attach_core_profiles(study: Study, path) -> Study:
    """Add the fitted profile from the profile fit's profiles.json (format core.fit_all/profiles-v1).

    Its questions carry ``P_fit`` on ``alloc_settings`` and ``peak_fit_pct``; every question and setting of the
    study must be there.
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not str(d.get("format", "")).startswith("core.fit_all/profiles"):
        raise ValueError(f"{path}: not a profiles.json of the profile fit (format {d.get('format')!r})")
    if d.get("axis") not in (None, study.axis):
        raise ValueError(f"{path}: axis {d.get('axis')} but the study's is {study.axis}")
    grid = [float(v) for v in d["alloc_settings"]]
    col = {}
    for j, s in enumerate(study.settings.tolist()):
        hits = [k for k, g in enumerate(grid) if math.isclose(g, s, abs_tol=1e-9)]
        if not hits:
            raise ValueError(f"{path}: setting {s} is not on the fit's allocation grid {grid}")
        col[j] = hits[0]
    by_id = {str(q["id"]): q for q in d["questions"]}
    P = np.zeros(study.n.shape, dtype=np.float64)
    peak = np.zeros(len(study.ids), dtype=np.float64)
    for i, qid in enumerate(study.ids):
        q = by_id.get(qid)
        if q is None:
            raise ValueError(f"{path}: question {qid} has no fitted profile")
        for j in range(len(study.settings)):
            P[i, j] = float(q["P_fit"][col[j]])
        peak[i] = float(q["peak_fit_pct"])
    study.p_fit, study.peak_pct = P, peak
    _check_fit(study, path)
    return study


def add_input_args(ap, measured_default: bool = False) -> None:
    """The input options every CLI shares."""
    ap.add_argument("--input", required=True, help="counts.jsonl (or a profiles.jsonl with p_fit/peak_pct)")
    ap.add_argument("--profiles", default=None, help="profiles.json of the profile fit (python -m onetemp.core.fit_all)")
    ap.add_argument("--benchmark", help="select the study when the file holds several")
    ap.add_argument("--model")
    ap.add_argument("--axis", choices=["temperature", "min_p"])
    ap.add_argument("--tau", type=float, help="the fixed temperature of a min_p study")
    ap.add_argument("--rates", choices=["fit", "measured"], default="measured" if measured_default else "fit",
                    help="fit: the fitted profiles (temperature sweeps); measured: c/n and the unbiased pass@K")


def study_from_args(a, require_fit: bool | None = None) -> Study:
    need = (a.rates == "fit") if require_fit is None else require_fit
    st = load_study(a.input, a.benchmark, a.model, a.axis, a.tau, require_fit=need and not a.profiles)
    if a.profiles:
        st = attach_core_profiles(st, a.profiles)
    return st


def guard_outputs(paths) -> None:
    """Refuse to start when an output file already exists (checked before any computation)."""
    for p in paths:
        if Path(p).exists():
            raise SystemExit(f"refusing to overwrite {p}; choose a fresh --out")


def environment() -> dict:
    """Versions and machine behind a result: the cells of the random reweighting depend on numpy's random streams
    and on the order its default argsort gives to questions of equal hardness."""
    import platform
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "machine": platform.machine(), "system": platform.system()}


def write_jsonl(path, rows) -> None:
    """Write rows as JSON lines (helper for tests and synthetic data)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
