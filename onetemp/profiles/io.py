"""Read the shared per-question count format and fitted-profile files.

``counts.jsonl`` (written by the graders and the data export, read by every analysis):

    {"benchmark": "livecodebench" | "math500" | "hitom", "model": "<HF repo id>",
     "id": "<question id>", "temperature": float, "min_p": float or null, "n": int, "c": int}

One line per (benchmark, model, question, setting): ``n`` samples, ``c`` of them correct.

* A **temperature sweep** has ``min_p`` null; the setting is ``temperature``.
* A **min_p sweep** (Appendix I) has ``min_p`` set; ``temperature`` is the fixed tau
  (an optional ``"tau"`` field, if present, must equal it).

A *study* is the set of lines that share (benchmark, model, setting axis, fixed tau).  Within a study the
questions keep the order in which they first appear in the file.  That order matters in exactly one
place: the paper sorts questions by hardness with a *stable* sort, so questions with equal hardness (many
sit at the clamp H = 0.0869, Appendix E.1) stay in file order, and that decides which of them fall into
the first sliding windows of Figures 2 and 5-7.  The data export keeps the order of the paper's data files.

Settings are rounded to 6 decimals, as the paper's loader did (``round(float(T), 6)``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from onetemp.counts import iter_counts as _iter_counts

AXES = ("temperature", "min_p")


@dataclass
class Question:
    """One question's counts on its setting grid (settings ascending)."""

    id: str
    settings: np.ndarray  # float64, ascending, unique
    n: np.ndarray  # int64, samples per setting
    c: np.ndarray  # int64, correct samples per setting

    @property
    def rates(self) -> np.ndarray:
        """Measured single-sample pass rate c / n per setting (NaN where n = 0)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n > 0, self.c / np.maximum(self.n, 1), np.nan)


@dataclass
class Study:
    """All questions of one (benchmark, model, setting axis, fixed tau)."""

    benchmark: str
    model: str
    axis: str  # "temperature" or "min_p"
    tau: float | None  # the fixed temperature of a min_p study, else None
    questions: list[Question] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.benchmark, self.model, self.axis, self.tau)

    @property
    def label(self) -> str:
        if self.axis == "min_p":
            return f"{self.benchmark} / {self.model} / min_p at tau={self.tau:g}"
        return f"{self.benchmark} / {self.model} / temperature"

    @property
    def grid(self) -> np.ndarray:
        """Sorted union of every question's settings (the paper's per-dataset measurement grid)."""
        if not self.questions:
            return np.zeros(0)
        return np.unique(np.concatenate([q.settings for q in self.questions]))

    def matrices(self, order: str = "file") -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        """(ids, grid, n[Q, S], c[Q, S]) on the union grid; a setting a question lacks gets n = c = 0.

        ``order="file"`` keeps first-appearance order, ``order="id"`` sorts ids as plain strings.
        """
        grid = self.grid
        qs = list(self.questions)
        if order == "id":
            qs.sort(key=lambda q: q.id)
        elif order != "file":
            raise ValueError(f"order must be 'file' or 'id', not {order!r}")
        pos = {float(s): j for j, s in enumerate(grid)}
        n = np.zeros((len(qs), len(grid)), dtype=np.int64)
        c = np.zeros((len(qs), len(grid)), dtype=np.int64)
        for i, q in enumerate(qs):
            for s, ni, ci in zip(q.settings, q.n, q.c):
                j = pos[float(s)]
                n[i, j] = ni
                c[i, j] = ci
        return [q.id for q in qs], grid, n, c


def _setting_of(row: dict, lineno: int, path: str) -> tuple[str, float | None, float]:
    """Return (axis, tau, setting) for one counts row."""
    t = row.get("temperature")
    mp = row.get("min_p")
    if t is None:
        raise ValueError(f"{path}:{lineno}: 'temperature' is required")
    t = round(float(t), 6)
    if mp is None:
        return "temperature", None, t
    tau = row.get("tau", t)
    if tau is not None and abs(float(tau) - t) > 1e-9:
        raise ValueError(f"{path}:{lineno}: 'tau' ({tau}) differs from 'temperature' ({t}) in a min_p row")
    return "min_p", t, round(float(mp), 6)


def iter_counts(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield (line number, row) for every row of a counts.jsonl file, through the package's one reader
    (``onetemp.counts.iter_counts``, which validates every row)."""
    yield from _iter_counts(path, with_lineno=True)


def load_counts(paths: str | Path | Iterable[str | Path]) -> list[Study]:
    """Read one or more counts.jsonl files and group them into studies.

    Rules (each violation raises ``ValueError`` naming the line):
    * ``0 <= c <= n``;
    * at most one line per (study, question, setting);
    * a study's order is the order in which its (benchmark, model, axis, tau) first appears.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    studies: dict[tuple, dict] = {}
    for path in paths:
        for lineno, row in iter_counts(path):
            axis, tau, s = _setting_of(row, lineno, str(path))
            n, c = int(row["n"]), int(row["c"])
            if n < 0 or c < 0 or c > n:
                raise ValueError(f"{path}:{lineno}: need 0 <= c <= n, got n={n}, c={c}")
            key = (str(row["benchmark"]), str(row["model"]), axis, tau)
            st = studies.setdefault(key, {"order": [], "rows": {}, "files": []})
            if str(path) not in st["files"]:
                st["files"].append(str(path))
            qid = str(row["id"])
            if qid not in st["rows"]:
                st["order"].append(qid)
                st["rows"][qid] = {}
            if s in st["rows"][qid]:
                if len(st["files"]) > 1:
                    raise ValueError(
                        f"{path}:{lineno}: these files hold the same study {key}: {st['files']}. Pass only "
                        "one of them (for example the question_n or the cell_n parse, not both).")
                raise ValueError(f"{path}:{lineno}: second line for question {qid!r} at setting {s} in {key}")
            st["rows"][qid][s] = (n, c)
    out = []
    for key, st in studies.items():
        qs = []
        for qid in st["order"]:
            cells = st["rows"][qid]
            ss = np.array(sorted(cells), dtype=np.float64)
            qs.append(Question(
                id=qid,
                settings=ss,
                n=np.array([cells[float(s)][0] for s in ss], dtype=np.int64),
                c=np.array([cells[float(s)][1] for s in ss], dtype=np.int64),
            ))
        out.append(Study(benchmark=key[0], model=key[1], axis=key[2], tau=key[3], questions=qs))
    return out


def select_study(studies: list[Study], benchmark: str | None = None, model: str | None = None,
                 axis: str = "temperature", tau: float | None = None) -> Study:
    """Pick exactly one study; ``model`` may be a case-insensitive substring of the HF repo id."""
    cand = [s for s in studies if s.axis == axis]
    if benchmark is not None:
        cand = [s for s in cand if s.benchmark == benchmark]
    if model is not None:
        exact = [s for s in cand if s.model == model]
        cand = exact or [s for s in cand if model.lower() in s.model.lower()]
    if tau is not None:
        cand = [s for s in cand if s.tau is not None and math.isclose(s.tau, tau, abs_tol=1e-9)]
    if len(cand) != 1:
        have = "\n  ".join(s.label for s in studies) or "(none)"
        raise ValueError(f"{len(cand)} studies match benchmark={benchmark!r} model={model!r} axis={axis!r} "
                         f"tau={tau!r}; available:\n  {have}")
    return cand[0]


# ----------------------------------------------------------------------------------------------------
# Fitted profiles (output of the Bernstein fit, Appendix D)
# ----------------------------------------------------------------------------------------------------

@dataclass
class FittedProfile:
    """A fitted profile: a degree-10 Legendre series on [x_lo, x_hi], in PERCENT, and 0 outside.

    This is the representation the paper's fitter returns (Appendix D: Bernstein-basis fit, converted to
    Legendre coefficients for evaluation).  ``legendre`` holds c_0..c_10 of
        P(t) = sum_k c_k L_k(u),  u = 2 (t - x_lo) / (x_hi - x_lo) - 1,
    evaluated in percent (0-100).
    """

    id: str
    x_lo: float
    x_hi: float
    legendre: np.ndarray
    extra: dict = field(default_factory=dict)

    def to_json(self, benchmark: str, model: str, axis: str = "temperature", tau: float | None = None) -> dict:
        d = {"benchmark": benchmark, "model": model, "id": self.id, "axis": axis,
             "x_lo": float(self.x_lo), "x_hi": float(self.x_hi),
             "legendre": [float(v) for v in self.legendre]}
        if tau is not None:
            d["tau"] = float(tau)
        d.update(self.extra)
        return d


def load_profiles(path: str | Path) -> dict[tuple, dict[str, FittedProfile]]:
    """Read a profiles.jsonl file into {(benchmark, model, axis, tau): {id: FittedProfile}}."""
    out: dict[tuple, dict[str, FittedProfile]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k in ("benchmark", "model", "id", "x_lo", "x_hi", "legendre"):
                if k not in row:
                    raise ValueError(f"{path}:{lineno}: missing field {k!r}")
            axis = row.get("axis", "temperature")
            tau = row.get("tau")
            key = (str(row["benchmark"]), str(row["model"]), axis, None if tau is None else round(float(tau), 6))
            extra = {k: v for k, v in row.items()
                     if k not in ("benchmark", "model", "id", "axis", "tau", "x_lo", "x_hi", "legendre")}
            out.setdefault(key, {})[str(row["id"])] = FittedProfile(
                id=str(row["id"]), x_lo=float(row["x_lo"]), x_hi=float(row["x_hi"]),
                legendre=np.asarray(row["legendre"], dtype=np.float64), extra=extra)
    return out


def write_profiles(path: str | Path, study: Study, profiles: dict[str, FittedProfile]) -> None:
    """Write fitted profiles of one study as JSON lines, in the study's question order."""
    with open(path, "w", encoding="utf-8") as fh:
        for q in study.questions:
            p = profiles[q.id]
            fh.write(json.dumps(p.to_json(study.benchmark, study.model, study.axis, study.tau)) + "\n")


def write_json(path: str | Path, obj) -> None:
    """Write ``obj`` as indented JSON (numpy scalars and arrays converted)."""

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return [conv(v) for v in o.tolist()]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(conv(obj), fh, indent=1)
        fh.write("\n")
