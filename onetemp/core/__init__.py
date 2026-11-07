"""Core method of "One Temperature to Rule Them All?".

Modules (run each as ``python -m onetemp.core.<module> --help`` from the repository root):

    counts_io   read the common ``counts.jsonl`` format into per-question count matrices
    fit_all     per-question profile fit: degree-10 non-negative unimodal Bernstein polynomial,
                Clopper-Pearson weighted least squares, projected gradient descent
                (C kernel ``bernstein_kernel.c`` or an equivalent numpy fallback) - Appendix D
    passk       pass@K: the unbiased estimator and the plug-in 1 - (1 - P)^K - Section 2.2
    allocation  optimal multi-setting allocation: convex relaxation (Eq. 9, Appendix F) and the
                integer allocation (the paper's search, and an exact branch-and-bound) - Section 4.1
    heldout     best single setting, uniform baseline, ratio to uniform, and the held-out
                (leave-one-out / 10-fold) evaluation of the optimal allocation - Appendix F
    bootstrap   paired bootstrap bands on the ratio to uniform - Appendix F
    tables      pass@K tables and the summary table in the style of Appendix F
    plot_rtu    the ratio-to-uniform figure (Figure 3; Figures 14-16 of Appendix F)
    pipeline    end-to-end: counts -> fit -> comparison -> tables -> figure, one command
    paper_check checks the paper's printed numbers (Figure 3, Tables 2-6, the summary table, the
                Appendix I min_p table)

The sampling parameter is generic: every analysis works on a "setting" axis, which is the
temperature or min_p at a fixed temperature (Appendix I).
"""

__all__ = [
    "counts_io",
    "fit_all",
    "passk",
    "allocation",
    "heldout",
    "bootstrap",
    "tables",
    "plot_rtu",
    "pipeline",
    "paper_check",
]

__version__ = "2.0.0"
