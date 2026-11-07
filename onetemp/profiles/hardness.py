"""Hardness (Section 3.2, Eq. 3) and the question filter of Appendix D.

The hardness of question q at setting tau is the number of pass@K samples needed to reach a success
probability of 1 - 1/e, with the integrality of K relaxed:

    H_q(tau) = -1 / ln(1 - P_q(tau)),        H_q = inf_tau H_q(tau) = H_q(tau*_q).

Because H is decreasing in P, H_q is evaluated at the peak pass rate.  The paper's code clamps the peak
probability to [1e-9, 0.99999] before taking the logarithm, so a question that is (almost) always solved
sits at the floor H = -1/ln(1e-5) = 0.0869 (Appendix E.1); this is the "clamp" referred to throughout.

Filter (Appendix D, "Minimum peak success probability"): a question is included only if its fitted peak
success probability is at least 1e-3 (0.1%), and questions that no setting ever solves are excluded
everywhere.
"""
from __future__ import annotations

import numpy as np

#: Clamp applied to the peak probability before Eq. 3 (the paper's code).
P_MIN, P_MAX = 1e-9, 0.99999
#: Minimum fitted peak success probability of an included question, in percent (1e-3 = 0.1%).
MIN_PEAK_PERCENT = 0.1


def hardness_from_percent(peak_percent: float) -> float:
    """H = -1 / ln(1 - p) for a peak pass rate given in percent, with the clamp p in [1e-9, 0.99999]."""
    p = min(P_MAX, max(P_MIN, float(peak_percent) / 100.0))
    return float(-1.0 / np.log(1.0 - p))


def hardness_from_probability(p: float) -> float:
    """Eq. 3 for a probability in [0, 1] (same clamp)."""
    return hardness_from_percent(100.0 * float(p))


#: The hardness floor reached by every question with peak pass rate >= 99.999%.
H_CLAMP = hardness_from_percent(100.0)


def is_included(peak_percent: float | None, min_peak_percent: float = MIN_PEAK_PERCENT) -> bool:
    """Appendix D filter: a positive peak of at least ``min_peak_percent`` percent."""
    return peak_percent is not None and peak_percent > 0 and peak_percent >= min_peak_percent
