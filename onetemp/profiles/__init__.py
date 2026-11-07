"""Question-profile analyses of "One Temperature to Rule Them All?" (Section 3, Appendices D, E and I).

A question's *profile* is its single-sample success probability P_q as a function of the sampling
setting (the temperature tau, or min_p at a fixed tau).  From fitted or measured profiles this package
computes, per question,

* the peak (ideal) setting  tau*_q = argmax P_q                         (Section 3.2.1),
* the hardness  H_q = -1 / ln(1 - max P_q)                              (Section 3.2, Eq. 3),
* the efficient range  E_q = {tau : P_q(tau) >= max P_q / 2}            (Section 3.2.2, Eq. 4),

and, per (benchmark, model), the three hardness views of Figure 2 and Figures 5-13:

* peak setting against hardness with symmetric sliding means          (Appendix E.1, "Sliding averages"),
* efficient-range size against hardness with an OLS fit on log10 H    (Appendix E.1, "OLS fit"),
* maximal coverage R_Q(H) in log-uniform hardness bins                (Eq. 5, Appendix E.1).

Modules
-------
io               read the shared ``counts.jsonl`` format and fitted-profile files
curves           evaluate fitted profiles; peak and efficient range exactly as the paper's code
hardness         Eq. 3 with the paper's clamps, and the 1e-3 peak filter (Appendix D)
stats            sliding means, the drawn-line trimming rule, OLS, log-uniform bins and coverage
analysis         per-question tables and per-dataset summaries
plots            Figure 2 and Figures 5-13
profile_examples Figure 1 (example profiles)
grid_profiles    Appendix I: the same analyses on measured pass rates over a min_p grid, with the
                 split-half and common-shape-null noise checks
fitting          obtain fitted profiles (from the Bernstein fit of the core package, or a file)
paper            the paper's settings per benchmark and its printed numbers (for ``repro``)
"""

__all__ = [
    "io",
    "curves",
    "hardness",
    "stats",
    "analysis",
    "plots",
    "profile_examples",
    "grid_profiles",
    "fitting",
    "paper",
]

__version__ = "2.0.0"
