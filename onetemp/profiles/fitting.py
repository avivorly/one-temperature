"""Fitted profiles for the analyses: from the release's Bernstein fit (core package), or from a file.

The profile fit itself (Appendix D: degree-10 Bernstein basis, Clopper-Pearson weighted least squares,
non-negativity and unimodality projections, projected gradient descent) lives in ``core.fit_all``.  This
module only

* lays a study's counts on its union setting grid, the way the paper's code fed the fitter,
* calls the fit (``core.fit_all.fit_profiles``, or any per-question backend, e.g. in tests),
* applies the paper's rule for questions that cannot be fitted, and
* reads/writes the result as ``profiles.jsonl`` (``io.FittedProfile``).

Sample counts (``n_mode``, the names of ``core.counts_io``):

* ``"per-setting"``: each setting uses its own n; a setting a question was never run at is left out.
* ``"question-max"``: one question-level n, the largest n of the question, at every setting; a setting
  with fewer samples (or none) counts its c out of that n.  The paper's LiveCodeBench profile figures
  (Figures 1, 2, 5, 8, 11) were drawn with this rule.  On MATH-500 and Hi-ToM every setting has the same
  n and the two rules coincide.  ``paper.N_MODE`` records the choice per benchmark.

Questions the fit cannot handle (no correct sample, or fewer than 2 measured settings in the bracket) get
the zero profile on the whole grid [first setting, last setting], as in the paper's code; the 0.1% peak
filter then excludes them.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .io import FittedProfile, Study, load_profiles

N_MODES = ("question-max", "per-setting")

#: A per-question backend: (grid, n_per_setting, c_per_setting) -> (x_lo, x_hi, legendre) or None.
Backend = Callable[[np.ndarray, np.ndarray, np.ndarray], "tuple[float, float, np.ndarray] | None"]


def counts_on_grid(study: Study, n_mode: str):
    """(grid, ids, n[Q, S], c[Q, S], measured[Q, S]) as fed to the fit, in study (file) order."""
    ids, grid, n, c = study.matrices(order="file")
    measured = n > 0
    if n_mode == "question-max":
        nq = n.max(axis=1, keepdims=True)
        n = np.broadcast_to(nq, n.shape).copy()
    elif n_mode != "per-setting":
        raise ValueError(f"n_mode must be one of {N_MODES}, not {n_mode!r}")
    return grid, ids, n, c, measured


def zero_profile(qid: str, grid: np.ndarray, why: str) -> FittedProfile:
    return FittedProfile(id=qid, x_lo=float(grid[0]), x_hi=float(grid[-1]), legendre=np.zeros(11),
                         extra={"zero_profile": why})


def fit_with_core(study: Study, n_mode: str, backend: str = "auto", kernel_path: str | None = None,
                  progress: bool = False) -> dict[str, FittedProfile]:
    """Fit with the release's Bernstein fit, ``onetemp.core.fit_all.fit_profiles`` (C kernel or numpy)."""
    try:
        from onetemp.core import counts_io as core_counts  # the release's core package
        from onetemp.core import fit_all as core_fit
    except ImportError as exc:
        raise ImportError("the core package (onetemp.core.fit_all) is not importable; reinstall the package "
                          "(pip install -e .), or pass --profiles FILE with fitted profiles") from exc
    ids, grid, n, c = study.matrices(order="file")
    counts = core_counts.counts_from_arrays(list(grid), n, c, ids=ids, axis=study.axis,
                                            benchmark=study.benchmark, model=study.model, measured=n > 0)
    prof = core_fit.fit_profiles(counts, n_mode=n_mode, backend=backend, kernel_path=kernel_path,
                                 progress=progress)
    out = {}
    for i, qid in enumerate(prof.ids):
        extra = {"fit_backend": prof.backend, "n_mode": n_mode}
        if not bool(prof.fitted[i]):
            extra["zero_profile"] = "not fitted (no correct sample, or fewer than 2 measured settings)"
        out[qid] = FittedProfile(id=qid, x_lo=float(prof.x_lo[i]), x_hi=float(prof.x_hi[i]),
                                 legendre=np.asarray(prof.coeffs[i], dtype=np.float64), extra=extra)
    return out


def fit_study(study: Study, n_mode: str = "per-setting", backend: Backend | None = None,
              core_backend: str = "auto", kernel_path: str | None = None,
              progress: Callable[[int, int], None] | None = None) -> dict[str, FittedProfile]:
    """Fit every question of ``study``; returns {id: FittedProfile}.

    ``backend=None`` uses the core package (``core_backend``: "auto", "c" or "numpy"); a callable backend
    fits one question at a time (used by the tests and to plug in another fitter).
    """
    if backend is None:
        return fit_with_core(study, n_mode, backend=core_backend, kernel_path=kernel_path,
                             progress=progress is not None)
    grid, ids, n, c, measured = counts_on_grid(study, n_mode)
    if n_mode == "question-max":
        c = np.where(measured, c, 0)
    out = {}
    for i, qid in enumerate(ids):
        if n[i].sum() <= 0 or c[i].sum() == 0:
            out[qid] = zero_profile(qid, grid, "no correct sample")
        else:
            res = backend(grid, n[i], c[i])
            if res is None:
                out[qid] = zero_profile(qid, grid, "fewer than 2 measured settings in the bracket")
            else:
                x_lo, x_hi, leg = res
                out[qid] = FittedProfile(id=qid, x_lo=float(x_lo), x_hi=float(x_hi),
                                         legendre=np.asarray(leg, dtype=np.float64))
        if progress is not None:
            progress(i + 1, len(ids))
    return out


def profiles_for_study(study: Study, profiles_path: str | None = None, n_mode: str = "per-setting",
                       backend: Backend | None = None, core_backend: str = "auto",
                       kernel_path: str | None = None) -> dict[str, FittedProfile]:
    """Fitted profiles of ``study``: read from ``profiles_path`` if given (must cover every question),
    else fitted (default: the core package's fit)."""
    if profiles_path is None:
        return fit_study(study, n_mode=n_mode, backend=backend, core_backend=core_backend,
                         kernel_path=kernel_path)
    allp = load_profiles(profiles_path)
    key = (study.benchmark, study.model, study.axis, study.tau)
    if key not in allp:
        have = ", ".join(map(str, allp)) or "(none)"
        raise KeyError(f"{profiles_path} has no profiles for {key}; it has: {have}")
    got = allp[key]
    missing = [q.id for q in study.questions if q.id not in got]
    if missing:
        raise KeyError(f"{profiles_path} lacks {len(missing)} question(s) of {study.label}, e.g. {missing[:3]}")
    return got
