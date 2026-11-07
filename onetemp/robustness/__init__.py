"""Hardness-reweighting robustness analyses of "One Temperature to Rule Them All?".

Section 4.3.2 (the Goldilocks sampling effect), Figure 4, Appendix G (random Dirichlet reweighting), Appendix H
(reweighting outside a protected hardness window) and the min_p versions of both in Appendix I.

Modules
-------
data          read counts.jsonl / profiles.jsonl into one "study" (a sweep of one sampling setting)
contexts      per-K pass@K matrices, hardness and the uniform reference tau_A, in the two conventions the paper uses
dirichlet     the Dirichlet smooth-base sampler of Appendix G (linear, replayed-log and log-space draws)
election      choosing tau_B under a weight vector W (argmax, and the exact log-space election)
heatmap       KL x ratio heatmaps, top-row ("% losing under 10%") tables, CLI
plot_heatmap  drawing the heatmaps
fig4b         the 24-million-W recomputation behind Figure 4(b), in chunks, and its merge
protected     beta* against the protected half-width D (Appendix H, Figure 4(a)), CLI
certify       exact rational certificates for every beta* point, and the reachability of "> 1e6" points
worst_loss    the worst-case loss R(beta, D) of tau_A, computed exactly
tilt          hardness tilts W_q = 10^(gamma (log10 H_q - log10 K))
subsets       natural sub-datasets defined by question labels
simulate      simulated hardness distributions built from measured profiles
prot_summary  aggregate statistics of Appendix H over many studies
plot_prot     beta*-against-D panels and the two-panel Figure 4
"""

__all__ = [
    "data", "contexts", "dirichlet", "election", "heatmap", "plot_heatmap", "fig4b", "protected", "certify",
    "worst_loss", "tilt", "subsets", "simulate", "prot_summary", "plot_prot",
]
