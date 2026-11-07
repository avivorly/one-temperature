"""Optimal multi-setting allocation (Section 4.1, Eq. 9; Appendix F "Optimal temperature allocation").

With L[i, j] = log(1 - P_i(gamma_j)) for n questions and l settings, a budget of K samples split as
K_1..K_l (sum K_j = K) leaves question i unsolved with probability exp(sum_j K_j L[i, j]), so the
average failure probability is

    F(q) / n,    F(q) = sum_i exp(L_i . q),     q in Z^l, q >= 0, sum q = K.        (Eq. 9)

Maximising pass@K is minimising F. Three solvers:

* ``solve_relaxed``: the convex relaxation (q real), solved with SLSQP and the analytic gradient,
  exactly as the paper's code calls it; ``kkt_report`` gives the optimality diagnostics the paper
  lists (equality residual, bound violations, stationarity gap, dual slack, and the first-order
  gap bound g.q - K min_j g_j).
* ``integer_paper``: the paper's integer step, ported unchanged: largest-remainder rounding, a
  greedy 1-swap descent, and a branch-and-bound over a box of radius 2 around the relaxed optimum
  (radius 1 when 12 or more settings are active; inactive settings limited to 0..2), pruned with
  the first-order bound at the relaxed point. The box is centred on the relaxed (SLSQP) point.
  Every number of the paper was computed with it, so it is the default of ``optimal_allocation`` and of
  the held-out tools (``heldout``, ``pipeline``, ``paper_check``: ``--integer paper``); this module's own
  command line defaults to ``--integer exact``.
* ``integer_exact``: the exact integer optimum over the whole integer simplex. Branch-and-bound
  on the settings' integer bounds; each node is a box lo <= q <= hi intersected with sum q = K.
  Its lower bound is valid for any feasible point x of the node, by convexity of F:

      F(q) >= F(x) + grad F(x) . (q - x) >= F(x) + min_{y in node} grad F(x) . (y - x),

  where the minimum of the linear function over the node is a greedy fill (the node polytope has
  integer vertices). x is the node's relaxed optimum from an active-set Newton method, so the bound
  is tight. A node is closed when its bound is >= the best integer value found minus ``tol``;
  the search ends with every node closed, which certifies that no integer allocation is better
  than the returned one by more than ``tol`` (an absolute tolerance on F, i.e. on the expected
  number of unsolved questions; default 1e-9). Ties within ``tol`` keep the first incumbent (the
  result of ``integer_paper``, when one is passed).

The main text calls the integer step straightforward because proximity and rounding results let
good integer solutions be recovered near the relaxed optimum (Hochbaum and Shanthikumar 1990,
Moriguchi et al. 2011, Cont and Heidari 2015); Appendix F states the paper's code searches a
radius-2 neighbourhood, which ``integer_paper`` does. ``integer_exact`` (``--integer exact``)
searches the whole integer simplex and returns a certificate of optimality to ``tol``.

Command line::

    python -m onetemp.core.allocation --profiles profiles.json --k 2 16 256 1024 --integer exact
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

INTEGER_METHODS = ("paper", "exact", "none")


# ───────────────────────── Objective ─────────────────────────

def objective(L: np.ndarray, q) -> float:
    """F(q) = sum_i exp(L_i . q): the expected number of unsolved questions under allocation q."""
    return float(np.sum(np.exp(L @ np.asarray(q, dtype=np.float64))))


def gradient(L: np.ndarray, q) -> np.ndarray:
    return L.T @ np.exp(L @ np.asarray(q, dtype=np.float64))


# ───────────────────────── Convex relaxation (the paper's SLSQP calls) ─────────────────────────

def solve_relaxed(L: np.ndarray, K: float, q_init=None, maxiter: int = 100, ftol: float = 1e-12,
                  constraint_jac: bool = True, method: str = "slsqp") -> np.ndarray:
    """Relaxed optimum of Eq. 9.

    method="slsqp" (default): SLSQP as the paper's held-out code runs it: start from
    q_init clipped at 0 and rescaled to sum K (uniform K/l when it sums to 0), bounds 0 <= q_j <= K,
    the equality sum q = K with its Jacobian, maxiter 100, ftol 1e-12. This reproduces the paper.
    ``kkt_report`` gives the returned point's optimality diagnostics; the integer step
    (``integer_paper``) works from it and returns an allocation that sums to K.

    method="newton": the active-set Newton method of ``solve_box_simplex`` on 0 <= q_j <= K, which
    is exactly feasible and converges to the optimum (check with ``kkt_report``).
    """
    n_temps = L.shape[1]
    k_f = float(K)
    if method == "newton":
        return solve_box_simplex(L, k_f, np.zeros(n_temps), np.full(n_temps, k_f), q_init)
    if method != "slsqp":
        raise ValueError("method must be 'slsqp' or 'newton'")

    def _obj(q, _L=L):
        return float(np.sum(np.exp(_L @ q)))

    def _grad(q, _L=L):
        return _L.T @ np.exp(_L @ q)

    if q_init is None:
        q_init = np.full(n_temps, k_f / n_temps)
    q0 = np.maximum(np.asarray(q_init, dtype=float), 0.0)
    s = q0.sum()
    q0 = q0 * (k_f / s) if s > 0 else np.full(n_temps, k_f / n_temps)
    cons = {"type": "eq", "fun": lambda q, _kf=k_f: np.sum(q) - _kf}
    if constraint_jac:
        cons["jac"] = lambda q: np.ones_like(q)
    res = minimize(_obj, q0, jac=_grad, method="SLSQP", bounds=[(0.0, k_f)] * n_temps,
                   constraints=[cons], options={"maxiter": maxiter, "ftol": ftol})
    return np.asarray(res.x, dtype=float)


def kkt_report(L: np.ndarray, K: float, q) -> dict:
    """Optimality diagnostics of a relaxed allocation (Appendix F, "Certificate of global
    optimality"): KKT residuals as the paper's verifier computes them, plus the first-order gap
    bound F(q) - F* <= g.q - K min_j g_j (linearise at q, minimise over the simplex)."""
    q = np.asarray(q, dtype=float)
    k_f = float(K)
    g = gradient(L, q)
    f = objective(L, q)
    tol_active = max(1e-8, 1e-10 * k_f)
    active = q > tol_active
    if active.any():
        mu = float(-np.mean(g[active]))
        stat_gap = float(np.max(np.abs(g[active] + mu)))
    else:
        mu = float(-np.mean(g))
        stat_gap = 0.0
    dual_slack_min = float(np.min(g[~active] + mu)) if (~active).any() else 0.0
    fw_gap = float(g @ q - k_f * g.min())
    return {"F": f, "eq_gap": float(abs(q.sum() - k_f)), "lb_viol": float(max(0.0, -q.min())),
            "ub_viol": float(max(0.0, q.max() - k_f)), "stat_gap": stat_gap,
            "dual_slack_min": dual_slack_min, "first_order_gap_bound": fw_gap,
            "rank_L": int(np.linalg.matrix_rank(L)), "n_settings": int(L.shape[1])}


# ───────────────────────── The paper's integer step (ported unchanged) ─────────────────────────

def _round_to_sum(q_star: np.ndarray, k_int: int) -> np.ndarray:
    """Largest-remainder rounding to enforce sum == k_int and q >= 0."""
    base = np.maximum(np.floor(q_star).astype(np.int64), 0)
    deficit = int(k_int - base.sum())
    if deficit > 0:
        frac = q_star - np.floor(q_star)
        order = np.argsort(-frac)
        for j in order[:deficit]:
            base[j] += 1
    elif deficit < 0:
        frac = q_star - np.floor(q_star)
        order = np.argsort(frac)
        removed = 0
        for j in order:
            if removed >= -deficit:
                break
            if base[j] > 0:
                base[j] -= 1
                removed += 1
    return base


def _score_alloc(q: np.ndarray, Lf: np.ndarray, n_inc: int) -> float:
    return float(n_inc - np.sum(np.exp(Lf @ q.astype(np.float64))))


def _greedy_1swap(q_init: np.ndarray, Lf: np.ndarray, n_inc: int, max_iter: int = 200):
    """Greedy 1-swap descent on the integer simplex: repeatedly move one sample from setting i to
    setting j when that most improves the score; stop at a 1-swap local optimum (or max_iter)."""
    q = np.asarray(q_init, dtype=np.int64).copy()
    Lf64 = np.asarray(Lf, dtype=np.float64)
    ev = np.exp(Lf64 @ q.astype(np.float64))
    score = float(n_inc - ev.sum())
    expL = np.exp(Lf64)
    invExpL = np.exp(-Lf64)
    for _ in range(max_iter):
        weighted = (ev[:, None] * invExpL).T
        new_f = weighted @ expL
        gain = (n_inc - score) - new_f
        mask_i = q >= 1
        gain[~mask_i, :] = -np.inf
        np.fill_diagonal(gain, -np.inf)
        i, j = np.unravel_index(int(np.argmax(gain)), gain.shape)
        best_gain = float(gain[i, j])
        if best_gain <= 1e-12:
            break
        q[i] -= 1
        q[j] += 1
        ev = ev * (expL[:, j] * invExpL[:, i])
        score = float(n_inc - ev.sum())
    return q, score


def integer_paper(q_star, k_int: int, Lf, n_inc: int, radius: int = 2):
    """The paper's integer allocation (its analysis code's integer step), unchanged.

    Maximises n_inc - F(q) over the integers in a box around the relaxed optimum q_star: active
    settings (q*_j > 0.01) range over [floor(q*_j) - r, ceil(q*_j) + r] with r = radius (r = 1
    when 12 or more are active), inactive ones over [0, radius]. Incumbents: the rounded q*, every
    single-setting allocation, and the greedy 1-swap optimum; the box is searched by
    branch-and-bound with the first-order bound F(q) >= F(q*) - R* + sum_j r_j q_j, with reduced
    costs r_j = g_j - min(g). Returns (q as int64 array, score).

    As in the paper's code, q_star is assumed to lie on the simplex: for a
    point off it by more than l, the rounding can under-fill and the result can sum to less than K.
    """
    n_dim = len(q_star)
    q_star = np.asarray(q_star, dtype=np.float64)
    Lf = np.asarray(Lf, dtype=np.float64)

    Lq_star = Lf @ q_star
    exp_Lq_star = np.exp(Lq_star)
    f_star = float(exp_Lq_star.sum())
    g = Lf.T @ exp_Lq_star
    active_mask = q_star > 0.01
    if not np.any(active_mask):
        active_mask[int(np.argmin(g))] = True
    eta = float(g.min())
    r = g - eta
    R_star = float(r @ q_star)

    q_round = _round_to_sum(q_star, k_int)
    incumbents = [q_round]
    for j in range(n_dim):
        e = np.zeros(n_dim, dtype=np.int64)
        e[j] = k_int
        incumbents.append(e)
    q_descent, _ = _greedy_1swap(q_round, Lf, n_inc)
    incumbents.append(q_descent)
    best_q = incumbents[0]
    best_score = _score_alloc(best_q, Lf, n_inc)
    for q_inc in incumbents[1:]:
        s = _score_alloc(q_inc, Lf, n_inc)
        if s > best_score:
            best_score = s
            best_q = q_inc
    U_obj = float(n_inc - best_score)

    coords_active = np.where(active_mask)[0]
    coords_inactive = np.where(~active_mask)[0]
    coords_inactive = coords_inactive[np.argsort(r[coords_inactive])]
    coord_order = np.concatenate([coords_active, coords_inactive]).astype(np.int64)

    n_active = int(active_mask.sum())
    eff_radius = 1 if n_active >= 12 else radius
    lo_band = np.zeros(n_dim, dtype=np.int64)
    hi_band = np.zeros(n_dim, dtype=np.int64)
    for j in range(n_dim):
        if active_mask[j]:
            qj = float(q_star[j])
            lo_band[j] = max(0, int(np.floor(qj)) - eff_radius)
            hi_band[j] = min(k_int, int(np.ceil(qj)) + eff_radius)
        else:
            lo_band[j] = 0
            hi_band[j] = min(k_int, radius)

    min_after = np.zeros(n_dim + 1, dtype=np.int64)
    max_after = np.zeros(n_dim + 1, dtype=np.int64)
    for i in range(n_dim - 1, -1, -1):
        j = int(coord_order[i])
        min_after[i] = min_after[i + 1] + lo_band[j]
        max_after[i] = max_after[i + 1] + hi_band[j]

    min_r_after = np.full(n_dim + 1, np.inf, dtype=np.float64)
    min_r_after[n_dim] = np.inf
    for i in range(n_dim - 1, -1, -1):
        j = int(coord_order[i])
        min_r_after[i] = min(min_r_after[i + 1], float(r[j]))

    n_rows = max(1, Lf.shape[0])
    chunk = max(1, 16_000_000 // n_rows)
    Q_buf = np.zeros((chunk, n_dim), dtype=np.float64)
    cur = np.zeros(n_dim, dtype=np.int64)
    state = {"n_in_buf": 0}

    def _flush():
        n = state["n_in_buf"]
        if n == 0:
            return
        nonlocal best_score, best_q, U_obj
        Q_view = Q_buf[:n]
        scores = n_inc - np.sum(np.exp(Lf @ Q_view.T), axis=0)
        local_best = int(np.argmax(scores))
        local_score = float(scores[local_best])
        if local_score > best_score:
            best_score = local_score
            best_q = Q_view[local_best].astype(np.int64)
            U_obj = float(n_inc - best_score)
        state["n_in_buf"] = 0

    def recurse(i, partial_sum, accumulated_r_q):
        if i == n_dim:
            if partial_sum == k_int:
                idx = state["n_in_buf"]
                Q_buf[idx] = cur
                state["n_in_buf"] = idx + 1
                if state["n_in_buf"] >= chunk:
                    _flush()
            return
        j = int(coord_order[i])
        rem_min = int(min_after[i + 1])
        rem_max = int(max_after[i + 1])
        v_lo = max(int(lo_band[j]), k_int - partial_sum - rem_max)
        v_hi = min(int(hi_band[j]), k_int - partial_sum - rem_min)
        if v_lo > v_hi:
            return
        rj = float(r[j])
        slack = U_obj - f_star + R_star
        for v in range(v_lo, v_hi + 1):
            new_acc = accumulated_r_q + rj * v
            new_sum = partial_sum + v
            remaining_mass = k_int - new_sum
            tail_lb_r = 0.0
            if remaining_mass > 0:
                m = min_r_after[i + 1]
                if np.isfinite(m):
                    tail_lb_r = m * remaining_mass
            if new_acc + tail_lb_r >= slack:
                continue
            cur[j] = v
            recurse(i + 1, new_sum, new_acc)
        cur[j] = 0

    recurse(0, 0, 0.0)
    _flush()
    return best_q, best_score


# ───────────────────────── Node relaxation: active-set Newton on a box and the simplex ─────────────────────────

def _project_box_simplex(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, K: float) -> np.ndarray:
    """Euclidean projection of v onto {lo <= x <= hi, sum x = K} (bisection on a common shift)."""
    a = float(np.min(v - hi)) - 1.0
    b = float(np.max(v - lo)) + 1.0
    for _ in range(200):
        mid = 0.5 * (a + b)
        if np.clip(v - mid, lo, hi).sum() > K:
            a = mid
        else:
            b = mid
        if b - a <= 1e-15 * (1.0 + abs(a) + abs(b)):
            break
    x = np.clip(v - 0.5 * (a + b), lo, hi)
    resid = K - x.sum()
    for _ in range(3):
        if resid == 0:
            break
        room = (hi - x) if resid > 0 else (x - lo)
        j = int(np.argmax(room))
        step = min(abs(resid), float(room[j]))
        x[j] += step if resid > 0 else -step
        resid = K - x.sum()
    return x


def solve_box_simplex(L: np.ndarray, K: float, lo: np.ndarray, hi: np.ndarray, x0=None,
                      max_iter: int = 200) -> np.ndarray:
    """min F(x) s.t. lo <= x <= hi, sum x = K, by a primal active-set Newton method.

    Working set W = settings free to move. Each step solves the Newton system of F on W with the
    equality constraint; a step that reaches a bound fixes that setting; at a stationary point the
    multipliers of the fixed settings are checked and the most violating one is released. The
    result only needs to be feasible for the branch-and-bound bound to be valid; accuracy makes the
    bound tight.
    """
    l = L.shape[1]
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    K = float(K)
    if x0 is None:
        x0 = np.full(l, K / l)
    x = _project_box_simplex(np.asarray(x0, dtype=np.float64), lo, hi, K)
    movable = hi > lo
    eps_b = 1e-12 * (1.0 + K)
    W = movable & (x > lo + eps_b) & (x < hi - eps_b)
    x = np.where(movable & ~W & (x <= lo + eps_b), lo, x)
    x = np.where(movable & ~W & (x >= hi - eps_b), hi, x)
    x = _fix_sum(x, lo, hi, K, W)
    e = np.exp(L @ x)
    F = float(e.sum())
    for _ in range(max_iter):
        g = L.T @ e
        widx = np.where(W)[0]
        if len(widx) >= 2:
            Lw = L[:, widx]
            H = Lw.T @ (e[:, None] * Lw)
            m = len(widx)
            A = np.zeros((m + 1, m + 1))
            A[:m, :m] = H + (1e-14 * (np.trace(H) / m + 1e-300)) * np.eye(m)
            A[:m, m] = 1.0
            A[m, :m] = 1.0
            rhs = np.concatenate([-g[widx], [0.0]])
            try:
                sol = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
            d = sol[:m]
            eta = -float(sol[m])
            decrement = -float(g[widx] @ d)
        elif len(widx) == 1:
            d = np.zeros(1)
            eta = float(g[widx[0]])
            decrement = 0.0
        else:
            d = np.zeros(0)
            eta = None
            decrement = 0.0
        if decrement > 1e-15 * (1.0 + F):
            step_max = 1.0
            hit = -1
            for t, j in enumerate(widx):
                if d[t] < 0:
                    s = (lo[j] - x[j]) / d[t]
                elif d[t] > 0:
                    s = (hi[j] - x[j]) / d[t]
                else:
                    continue
                if s < step_max:
                    step_max, hit = s, t
            alpha = step_max
            gd = float(g[widx] @ d)
            accepted = False
            for _ls in range(60):
                x_new = x.copy()
                x_new[widx] += alpha * d
                if alpha == step_max and hit >= 0:
                    j = widx[hit]
                    x_new[j] = lo[j] if d[hit] < 0 else hi[j]
                x_new = np.clip(x_new, lo, hi)
                e_new = np.exp(L @ x_new)
                F_new = float(e_new.sum())
                if F_new <= F + 1e-4 * alpha * gd:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                if alpha == step_max and hit >= 0:
                    W[widx[hit]] = False
                x = _fix_sum(x_new, lo, hi, K, W)
                e = np.exp(L @ x)
                F = float(e.sum())
                continue
        # Stationary on W (or no descent possible): check the fixed settings' multipliers.
        fixed = movable & ~W
        if not fixed.any():
            break
        if eta is None:
            # No free setting: release the cheapest at-lower and the dearest at-upper setting.
            at_lo = fixed & (x <= lo + eps_b)
            at_hi = fixed & (x >= hi - eps_b)
            if at_lo.any() and at_hi.any():
                jl = np.where(at_lo)[0][int(np.argmin(g[at_lo]))]
                jh = np.where(at_hi)[0][int(np.argmax(g[at_hi]))]
                if g[jl] < g[jh] - 1e-15 * (1.0 + abs(g[jl]) + abs(g[jh])):
                    W[jl] = W[jh] = True
                    continue
            break
        scale = 1e-13 * (1.0 + float(np.max(np.abs(g))))
        viol = np.zeros(l)
        at_lo = fixed & (x <= lo + eps_b)
        at_hi = fixed & (x >= hi - eps_b)
        viol[at_lo] = (eta - g[at_lo])            # > 0: setting wants to increase
        viol[at_hi] = (g[at_hi] - eta)            # > 0: setting wants to decrease
        j = int(np.argmax(viol))
        if viol[j] > scale:
            W[j] = True
            continue
        break
    return x


def _fix_sum(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, K: float, W: np.ndarray) -> np.ndarray:
    """Remove a floating-point excess of sum x - K on a free coordinate (keeps feasibility)."""
    resid = K - float(x.sum())
    if resid == 0.0:
        return x
    cand = np.where(W)[0] if W.any() else np.arange(len(x))
    room = (hi[cand] - x[cand]) if resid > 0 else (x[cand] - lo[cand])
    j = cand[int(np.argmax(room))]
    x = x.copy()
    x[j] = min(hi[j], max(lo[j], x[j] + resid))
    return x


def _linear_min(g: np.ndarray, lo: np.ndarray, hi: np.ndarray, K: int) -> np.ndarray:
    """argmin_y g.y over {lo <= y <= hi, sum y = K}, integer bounds: start at lo and fill the
    remaining mass into the settings with the smallest gradient first (integral)."""
    y = lo.astype(np.int64).copy()
    rem = int(K) - int(y.sum())
    for j in np.argsort(g, kind="stable"):
        if rem <= 0:
            break
        add = min(int(hi[j] - lo[j]), rem)
        y[j] += add
        rem -= add
    return y


def _round_in_box(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, K: int) -> Optional[np.ndarray]:
    """Largest-remainder rounding of x inside the integer box, or None if not possible."""
    base = np.clip(np.floor(x).astype(np.int64), lo, hi)
    deficit = int(K - base.sum())
    frac = x - np.floor(x)
    if deficit > 0:
        for j in np.argsort(-frac, kind="stable"):
            if deficit == 0:
                break
            if base[j] < hi[j]:
                base[j] += 1
                deficit -= 1
    elif deficit < 0:
        for j in np.argsort(frac, kind="stable"):
            if deficit == 0:
                break
            if base[j] > lo[j]:
                base[j] -= 1
                deficit += 1
    return base if deficit == 0 else None


# ───────────────────────── Exact integer allocation ─────────────────────────

def integer_exact(L: np.ndarray, K: int, x_relaxed=None, incumbent=None, tol: float = 1e-9,
                  max_nodes: int = 2_000_000):
    """Exact integer optimum of Eq. 9 over {q in Z^l : q >= 0, sum q = K} by branch-and-bound.

    incumbent: an integer allocation to start from (e.g. the paper's); it is kept unless a better
    one by more than ``tol`` is found. An incumbent that is not an allocation of K (wrong length,
    not integral, negative, or not summing to K) is ignored and reported as rejected.
    tol: absolute tolerance on F, >= 0; it must exceed the bound's floating-point error, about
    1e-12 at the paper's sizes (allocations that tie in exact arithmetic cannot be told apart).
    Returns (q, score, certificate) where score = n - F(q) and certificate = {"proven": bool,
    "nodes": int, "lower_bound": float, "F": float, "tol": tol, "improved_on_incumbent": bool,
    "incumbent_rejected": bool}. proven is True when every node was closed, i.e.
    F(q) <= F(q') + tol for every integer allocation q'.
    """
    L = np.asarray(L, dtype=np.float64)
    n, l = L.shape
    K = int(K)
    if K < 0:
        raise ValueError("K must be >= 0")
    if not tol >= 0:
        raise ValueError("tol must be >= 0 (and above the ~1e-12 rounding error of the bound)")
    incumbent_rejected = False
    if incumbent is not None:
        inc = np.asarray(incumbent)
        if (inc.shape != (l,) or not np.all(np.isfinite(inc)) or np.any(inc != np.round(inc))
                or np.any(inc < 0) or int(np.round(inc).sum()) != K):
            incumbent, incumbent_rejected = None, True      # not an allocation of K: ignore it
    if l == 1 or K == 0:
        q = np.full(l, K, dtype=np.int64) if l == 1 else np.zeros(l, dtype=np.int64)
        f = objective(L, q)
        return q, float(n - f), {"proven": True, "nodes": 0, "lower_bound": f, "F": f, "tol": tol,
                                 "improved_on_incumbent": False, "incumbent_rejected": incumbent_rejected}

    lo0 = np.zeros(l, dtype=np.int64)
    hi0 = np.full(l, K, dtype=np.int64)
    x_root = solve_box_simplex(L, K, lo0, hi0, x_relaxed)

    cands = []
    if incumbent is not None:
        cands.append(np.asarray(incumbent, dtype=np.int64))
    q_r = _round_to_sum(x_root, K)
    cands.append(q_r)
    cands.append(_greedy_1swap(q_r, L, n)[0])
    best_q = cands[0].copy()
    best_F = objective(L, best_q)
    first_F = best_F
    for q in cands[1:]:
        f = objective(L, q)
        if f < best_F - tol:
            best_q, best_F = q.copy(), f

    stack = [(lo0, hi0, x_root, -np.inf)]
    nodes = 0
    global_lb = np.inf
    while stack:
        lo, hi, x_hint, parent_lb = stack.pop()
        if parent_lb >= best_F - tol:
            global_lb = min(global_lb, parent_lb)
            continue
        nodes += 1
        if nodes > max_nodes:
            return best_q, float(n - best_F), {"proven": False, "nodes": nodes, "lower_bound": float("nan"),
                                               "F": best_F, "tol": tol,
                                               "improved_on_incumbent": bool(best_F < first_F - tol),
                                               "incumbent_rejected": incumbent_rejected,
                                               "reason": "max_nodes reached"}
        x = solve_box_simplex(L, K, lo, hi, x_hint)
        e = np.exp(L @ x)
        Fx = float(e.sum())
        g = L.T @ e
        y = _linear_min(g, lo, hi, K)
        lb = Fx + float(g @ (y - x))
        lb = max(lb, parent_lb)
        for cand in (y, _round_in_box(x, lo, hi, K)):
            if cand is None:
                continue
            f = objective(L, cand)
            if f < best_F - tol:
                best_q, best_F = cand.copy(), f
        if lb >= best_F - tol:
            global_lb = min(global_lb, lb)
            continue
        frac = x - np.floor(x)
        dist = np.minimum(frac, 1.0 - frac)
        dist = np.where(hi > lo, dist, -1.0)
        j = int(np.argmax(dist))
        if dist[j] <= 1e-9:
            # x is integral: it is the node's best integer point up to the solver's accuracy.
            xi = np.clip(np.rint(x).astype(np.int64), lo, hi)
            if int(xi.sum()) == K:
                f = objective(L, xi)
                if f < best_F - tol:
                    best_q, best_F = xi.copy(), f
            if lb >= best_F - tol:
                global_lb = min(global_lb, lb)
                continue
            widths = hi - lo
            j = int(np.argmax(widths))
            v = int((lo[j] + hi[j]) // 2)
        else:
            v = int(np.floor(x[j]))
        lo_a, hi_a = lo.copy(), hi.copy()
        hi_a[j] = v                      # child a: q_j <= v
        lo_b, hi_b = lo.copy(), hi.copy()
        lo_b[j] = v + 1                  # child b: q_j >= v + 1
        children = []
        for (cl, ch) in ((lo_a, hi_a), (lo_b, hi_b)):
            if cl[j] <= ch[j] and cl.sum() <= K <= ch.sum():
                children.append((cl, ch))
        # Depth first, nearer child (the one holding round(x_j)) explored first.
        near_first_b = (x[j] - v) >= 0.5
        if len(children) == 2 and not near_first_b:
            children = children[::-1]
        for cl, ch in children:
            stack.append((cl, ch, x, lb))
    if not np.isfinite(global_lb):
        global_lb = best_F
    return best_q, float(n - best_F), {"proven": True, "nodes": nodes, "lower_bound": float(global_lb),
                                       "F": float(best_F), "tol": tol,
                                       "improved_on_incumbent": bool(best_F < first_F - tol),
                                       "incumbent_rejected": incumbent_rejected}


# ───────────────────────── One call for a budget K ─────────────────────────

def optimal_allocation(L: np.ndarray, K: int, integer: str = "paper", q_init=None, tol: float = 1e-9,
                       relaxed: str = "slsqp") -> dict:
    """Relaxed and integer optimum at budget K. integer: "paper", "exact" (seeded with the paper's
    result) or "none" (relaxed only); relaxed: "slsqp" (the paper's) or "newton" (see
    ``solve_relaxed``). Returns a dict with the allocations, scores and checks."""
    if integer not in INTEGER_METHODS:
        raise ValueError(f"integer must be one of {INTEGER_METHODS}")
    n, l = L.shape
    k_f = float(K)
    x = solve_relaxed(L, k_f, np.full(l, k_f / l) if q_init is None else q_init, method=relaxed)
    out = {"K": int(K), "relaxed_method": relaxed, "q_relaxed": x.tolist(),
           "solved_relaxed": float(n - objective(L, x)), "kkt": kkt_report(L, k_f, x)}
    if integer == "none":
        return out
    q_p, s_p = integer_paper(x, int(K), L, n, radius=2)
    out["q_paper"] = q_p.astype(int).tolist()
    out["solved_paper"] = float(s_p)
    if integer == "exact":
        q_e, s_e, cert = integer_exact(L, int(K), x_relaxed=x, incumbent=q_p, tol=tol)
        out["q_exact"] = q_e.astype(int).tolist()
        out["solved_exact"] = float(s_e)
        out["certificate"] = cert
        out["q_int"], out["solved_int"] = out["q_exact"], out["solved_exact"]
    else:
        out["q_int"], out["solved_int"] = out["q_paper"], out["solved_paper"]
    return out


def main(argv=None) -> int:
    from onetemp.core.fit_all import MIN_PEAK_PCT, load_profiles
    from onetemp.core.passk import k_grid
    ap = argparse.ArgumentParser(description="Optimal allocation over the settings (Eq. 9) at given budgets K.")
    ap.add_argument("--profiles", required=True, help="profiles.json written by onetemp.core.fit_all")
    ap.add_argument("--mode", default="fit", choices=("fit", "meas"))
    ap.add_argument("--min-peak", type=float, default=MIN_PEAK_PCT, help="inclusion threshold, percent")
    ap.add_argument("--k", type=int, nargs="+", default=None, help="budgets (default: the paper's grid to 1024)")
    ap.add_argument("--integer", default="exact", choices=INTEGER_METHODS)
    ap.add_argument("--relaxed", default="slsqp", choices=("slsqp", "newton"),
                    help="relaxation solver: slsqp (the paper's) or newton (exactly feasible)")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", default=None, help="write the results as JSON")
    a = ap.parse_args(argv)
    prof = load_profiles(a.profiles)
    inc = prof.included(a.min_peak, a.mode)
    L = prof.L(a.mode)[inc]
    res = []
    for k in (a.k or k_grid(1024)):
        t0 = time.perf_counter()
        r = optimal_allocation(L, k, a.integer, tol=a.tol, relaxed=a.relaxed)
        r["seconds"] = round(time.perf_counter() - t0, 3)
        res.append(r)
        msg = (f"K={k:5d} pass@K relaxed {r['solved_relaxed'] / len(L):.6f} "
               f"(gap bound {r['kkt']['first_order_gap_bound']:.1e}, sum residual {r['kkt']['eq_gap']:.1e})")
        if "q_int" in r:
            msg += f"  integer {r['solved_int'] / len(L):.6f}  q={r['q_int']}"
        if "certificate" in r:
            msg += f"  proven={r['certificate']['proven']} nodes={r['certificate']['nodes']}"
            if r["q_exact"] != r["q_paper"]:
                msg += f"  (paper's search: {r['solved_paper'] / len(L):.6f}, q={r['q_paper']})"
        print(msg, flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"settings": prof.alloc_settings, "n_included": int(inc.sum()), "results": res}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
