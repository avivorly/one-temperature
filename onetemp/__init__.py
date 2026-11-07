"""Code for "One Temperature to Rule Them All?" (NeurIPS 2026).

Subpackages (each module has a command line, ``python -m onetemp.<sub>.<module> --help``):

* ``onetemp.counts``      the counts.jsonl format and its one reader; ``onetemp.paths`` locates data/
* ``onetemp.sampling``    vLLM sampling over a temperature (or min_p) grid, the paper's prompts and run settings,
                          the first-stage screen and the multi-GPU launcher
* ``onetemp.grading``     the MATH-500, Hi-ToM and LiveCodeBench graders (completions -> counts.jsonl)
* ``onetemp.core``        the profile fit, pass@K, the optimal allocation, held-out evaluation, bootstrap, tables
* ``onetemp.profiles``    the question-profile analyses (Section 3, Appendices D, E and I)
* ``onetemp.robustness``  the hardness-reweighting robustness tests (Section 4.3.2, Appendices G, H and I)
* ``onetemp.data``        the check of the shipped counts and the paper's fit-free numbers
* ``onetemp.reproduce``   one command per figure and table of the paper
"""
__version__ = "2.0.0"
