"""Smoke test on synthetic data: no model, no GPU, no benchmark files.

Writes a tiny counts file (12 questions x 5 temperatures, n = 200 draws per cell, correctness drawn from
unimodal synthetic temperature profiles), then runs passk.py and optimize_temperatures.py on it with
leave-one-out and with 3-fold held-out evaluation.

    python smoke_test.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # find passk.py next to this file
from passk import pass_at_k_from_rate, pass_at_k_unbiased  # noqa: E402



def main():
    rng = np.random.default_rng(0)
    temps = [0.3, 0.6, 0.9, 1.2, 1.5]
    n = 200
    # Sanity checks of the two estimators.
    assert pass_at_k_unbiased(10, 0, 3) == 0.0 and pass_at_k_unbiased(10, 8, 3) == 1.0
    assert abs(pass_at_k_unbiased(10, 1, 1) - 0.1) < 1e-12
    assert abs(float(pass_at_k_from_rate(0.5, 2)) - 0.75) < 1e-12
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "counts.jsonl")
        with open(path, "w") as f:
            for q in range(12):
                peak_t, height = rng.uniform(0.3, 1.5), rng.uniform(0.0005, 0.9)
                for t in temps:
                    p = height * np.exp(-((t - peak_t) / 0.5) ** 2)       # a unimodal synthetic profile
                    c = int(rng.binomial(n, p))
                    f.write(json.dumps({"id": f"q{q:02d}", "temperature": t, "n": n, "c": c}) + "\n")
        for args in (["passk.py", path, "--k", "1,4,16"],
                     ["passk.py", path, "--k", "1,4,16", "--estimator", "unbiased"],
                     ["optimize_temperatures.py", path, "--max-k", "64"],
                     ["optimize_temperatures.py", path, "--max-k", "64", "--folds", "3",
                      "--out", os.path.join(d, "res.json")]):
            print("$ python " + " ".join(os.path.basename(a) for a in args), flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, args[0])] + args[1:], check=True, cwd=HERE)
        res = json.load(open(os.path.join(d, "res.json")))
        for r in res["per_K"]:
            # The in-sample optimum can never lose to the best single temperature or to uniform.
            assert r["optimal_in_sample_ratio"] >= max(r["best_single_T_ratio"], 1.0) - 1e-9, r
            assert sum(r["optimal_allocation"].values()) == r["K"], r
    print("smoke test OK")


if __name__ == "__main__":
    main()
