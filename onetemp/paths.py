"""Where the release keeps its files.

* ``data_dir()``: the repository's ``data/`` folder (the per-question counts, the manifest and the reference
  numbers of the paper's own runs). ``$ONETEMP_DATA`` overrides it; otherwise it is the ``data/`` folder next to
  the ``onetemp`` package (a checkout, or an editable install), else ``./data``.
* ``default_cache_dir()``: where the graders and the prompt builders keep the public datasets they download
  (``$XDG_CACHE_HOME/one-temperature``, by default ``~/.cache/one-temperature``). Both use the same folder, so
  each file is downloaded once.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_ENV = "ONETEMP_DATA"
CACHE_NAME = "one-temperature"


def data_dir() -> Path:
    """The release's data/ folder (see the module docstring)."""
    env = os.environ.get(DATA_ENV)
    if env:
        return Path(env)
    here = PACKAGE_DIR.parent / "data"
    if (here / "manifest.json").exists():
        return here
    return Path("data")


def counts_dir() -> Path:
    """data/counts: one counts.jsonl file per study."""
    return data_dir() / "counts"


def reference_dir() -> Path:
    """data/reference: numbers of the paper's own runs that the reproduction commands compare against."""
    return data_dir() / "reference"


def default_cache_dir() -> str:
    """The download cache shared by the graders and the prompt builders."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, CACHE_NAME)
