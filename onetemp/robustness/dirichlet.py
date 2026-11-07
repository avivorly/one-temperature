"""The Dirichlet smooth-base sampler of Appendix G: random per-question weight vectors W.

Per W (Appendix G, "Experiment"):
  1. a smooth base measure m over the questions, two Gaussian bumps in the hardness coordinate u in [-1, 1]
     (centres mu ~ U(-1, 1), widths sigma = 10^U(log10 0.1, log10 2), heights h ~ U(0, 1)), normalised to sum 1;
  2. a target KL divergence kappa ~ U(0, 6) nats, and the Dirichlet concentration c whose median KL to uniform is
     kappa, read off a calibration table computed once with a uniform base measure (60 values of c from 10^-3 to
     10^3, 200 draws each);
  3. W ~ Dirichlet(c m), drawn as y_i ~ Gamma(c m_i) and W = y / sum y; KL(W || uniform) = sum_i W_i ln(n W_i).
The KL spans [0, ln n] (the entropy ceiling); targets above ln n are clipped to the table's smallest c and give
near one-hot W's.

Three ways to draw step 3, all on numpy's default generator (PCG64), each needed to reproduce one set of figures:
  "linear"  the original draw: y = Gamma(alpha) in linear space, W = y / max(sum y, 1e-300), KL in linear space.
            Some Gamma values underflow to 0, and rarely all of them (then W = 0 and its KL computes as 0).
            Used for the min_p heatmaps of Appendix I (Figures 42-44).
  "replay"  the same Gamma draws on the same random stream as "linear", but each draw's logarithm is kept, by
            replaying numpy's own Gamma algorithm (Marsaglia-Tsang, with Johnk-type boosting below shape 1) element by
            element, so that no W underflows; W = exp(log y - logsumexp(log y)) and KL from the logs. Every W that
            did not underflow is the same W as "linear" to rounding. Used for the twelve heatmaps of Appendix G
            (Figures 17-28), with the exact election.
  "log"     an exact log-space Dirichlet on a different stream: log y = log X + log(U) / alpha with X ~ Gamma(alpha +
            1) and U ~ U(0, 1] (X U^(1/alpha) ~ Gamma(alpha)). Used for the 24 million W's of Figure 4(b).

Seeds: the paper uses seed 20260606. SeedSequence(seed).spawn(2) gives the calibration stream (child 0) and the draw
stream (child 1) of a 10,000-W figure. Figure 4(b) computes the calibration table once from child 0 and draws its
chunk i of 10,000 W's from SeedSequence(seed, spawn_key=(1 + i,)).
"""
from __future__ import annotations

import math

import numpy as np

NAME = "dirichlet"
LABEL = "Dirichlet (smooth base)"

# shape draw
MU_LO, MU_HI = -1.0, 1.0
MIN_SIGMA_U = 0.1
LOG10_SIGMA_LO = float(np.log10(MIN_SIGMA_U))
LOG10_SIGMA_HI = float(np.log10(2.0))
# calibration grid for c -> KL
C_GRID_LO_LOG, C_GRID_HI_LOG, C_GRID_N = -3.0, 3.0, 60
CALIB_DRAWS = 200
# target KL
KAPPA_LO, KAPPA_HI = 0.0, 6.0
# batch size of the per-W draws (the elections batch the same way)
BATCH = 2048
# numerical floors of the original sampler
ALPHA_FLOOR = 1e-12
GAMMA_FLOOR = 1e-300
SEED = 20260606
MODES = ("linear", "replay", "log")


def base_measure(u, mu1, sigma1, h1, mu2, sigma2, h2):
    """Step 1: the normalised two-Gaussian base measure, (B, n), each row summing to 1."""
    z1 = (u[None, :] - mu1[:, None]) / sigma1[:, None]
    z2 = (u[None, :] - mu2[:, None]) / sigma2[:, None]
    g = h1[:, None] * np.exp(-0.5 * z1 * z1) + h2[:, None] * np.exp(-0.5 * z2 * z2)
    rs = np.maximum(g.sum(axis=1, keepdims=True), 1e-300)
    return g / rs


def calibrate(n_inc: int, rng: np.random.Generator):
    """Step 2's table: (c_grid, median KL) with a uniform base measure, CALIB_DRAWS draws per c (linear space).

    The KL is computed as sum W log(n W) with W = 0 terms dropped, exactly as the paper's sampler does.
    """
    c_grid = np.logspace(C_GRID_LO_LOG, C_GRID_HI_LOG, C_GRID_N)
    m_unif = np.full(n_inc, 1.0 / float(n_inc), dtype=np.float64)
    n_f = float(n_inc)
    kl_med = np.empty(C_GRID_N, dtype=np.float64)
    for i, c in enumerate(c_grid):
        alpha = np.maximum(c * m_unif, ALPHA_FLOOR)
        y = rng.gamma(shape=alpha[None, :], scale=1.0, size=(CALIB_DRAWS, n_inc))
        s = np.maximum(y.sum(axis=1, keepdims=True), GAMMA_FLOOR)
        W = y / s
        sw = np.where(W > 0, W, 1.0)
        kl = np.sum(W * np.log(n_f * sw), axis=1)
        kl_med[i] = float(np.median(kl))
    return c_grid, kl_med


def c_for_kappa(kappa, c_grid, kl_med):
    """Invert the table by interpolation in log10 c; targets outside the table are clipped to its edge values."""
    log_c = np.log10(c_grid)
    order = np.argsort(kl_med)
    return 10.0 ** np.interp(kappa, kl_med[order], log_c[order])


def shape_draws(draw_rng: np.random.Generator, n: int, c_grid, kl_med) -> dict:
    """Steps 1-2 for n W's, in the paper's order of random calls."""
    mu1 = draw_rng.uniform(MU_LO, MU_HI, size=n).astype(np.float64)
    mu2 = draw_rng.uniform(MU_LO, MU_HI, size=n).astype(np.float64)
    sigma1 = (10.0 ** draw_rng.uniform(LOG10_SIGMA_LO, LOG10_SIGMA_HI, size=n)).astype(np.float64)
    sigma2 = (10.0 ** draw_rng.uniform(LOG10_SIGMA_LO, LOG10_SIGMA_HI, size=n)).astype(np.float64)
    h1 = draw_rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    h2 = draw_rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    kappa = draw_rng.uniform(KAPPA_LO, KAPPA_HI, size=n).astype(np.float64)
    c = c_for_kappa(kappa, c_grid, kl_med).astype(np.float64)
    return {"mu1": mu1, "sigma1": sigma1, "h1": h1, "mu2": mu2, "sigma2": sigma2, "h2": h2, "kappa": kappa, "c": c}


def gamma_replay_log(rng: np.random.Generator, alpha: np.ndarray):
    """numpy's standard Gamma for every element of alpha (C order) on rng's stream, returning (x, log x).

    x is the value numpy's rng.gamma(alpha) returns on the same stream; log x is the logarithm of the same draw,
    computed before any underflow (log(U)/shape below shape 1). This replays numpy's random_standard_gamma
    (legacy-free PCG64 path) in Python; :func:`draw` checks it against numpy bit for bit.
    """
    a = np.ascontiguousarray(alpha, dtype=np.float64)
    flat = a.ravel().tolist()
    x = np.empty(len(flat))
    lx = np.empty(len(flat))
    rnd, rexp, rnorm = rng.random, rng.standard_exponential, rng.standard_normal
    log, pow_, sqrt = math.log, math.pow, math.sqrt
    ninf = -math.inf
    for i, shape in enumerate(flat):
        if shape == 1.0:
            X = rexp()
            x[i] = X
            lx[i] = log(X) if X > 0.0 else ninf
            continue
        if shape == 0.0:
            x[i] = 0.0
            lx[i] = ninf
            continue
        if shape < 1.0:
            inv = 1.0 / shape
            while True:
                U = rnd()
                V = rexp()
                if U <= 1.0 - shape:
                    X = pow_(U, inv)
                    if X <= V:
                        x[i] = X
                        lx[i] = log(U) / shape if U > 0.0 else ninf
                        break
                else:
                    Y = -log((1 - U) / shape)
                    base = 1.0 - shape + shape * Y
                    X = pow_(base, inv)
                    if X <= (V + Y):
                        x[i] = X
                        lx[i] = log(base) / shape
                        break
        else:
            b = shape - 1.0 / 3.0
            cc = 1.0 / sqrt(9 * b)
            while True:
                while True:
                    Xn = rnorm()
                    V = 1.0 + cc * Xn
                    if V > 0.0:
                        break
                V = V * V * V
                U = rnd()
                if U < 1.0 - 0.0331 * (Xn * Xn) * (Xn * Xn):
                    break
                lu = log(U) if U > 0.0 else ninf
                if lu < 0.5 * Xn * Xn + b * (1.0 - V + log(V)):
                    break
            X = b * V
            x[i] = X
            lx[i] = log(X)
    return x.reshape(a.shape), lx.reshape(a.shape)


def _normalise_log(logy: np.ndarray, ln_n: float):
    mx = logy.max(axis=1, keepdims=True)
    logW = logy - (mx + np.log(np.exp(logy - mx).sum(axis=1, keepdims=True)))
    W = np.exp(logW)
    kl = np.sum(W * (logW + ln_n), axis=1)
    return W, logW, kl


def draw(u_sorted, n: int, mode: str, draw_rng: np.random.Generator, c_grid, kl_med, check_replay: bool = True):
    """Step 3 for n W's over the questions in hardness order (u_sorted), after the shape draws on the same stream.

    Returns {"W" (n, n_inc), "logW" (n, n_inc) or None for "linear", "kl" (n,), "params" (the shape draws),
    "n_all_zero", "n_gamma_underflow"}.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    u_sorted = np.asarray(u_sorted, dtype=np.float64)
    n_inc = len(u_sorted)
    ln_n = math.log(n_inc)
    prm = shape_draws(draw_rng, n, c_grid, kl_med)
    W_all = np.empty((n, n_inc))
    logW_all = None if mode == "linear" else np.empty((n, n_inc))
    kl = np.empty(n)
    n_under = 0
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        m = base_measure(u_sorted, prm["mu1"][s:e], prm["sigma1"][s:e], prm["h1"][s:e],
                         prm["mu2"][s:e], prm["sigma2"][s:e], prm["h2"][s:e])
        alpha = np.maximum(prm["c"][s:e, None] * m, ALPHA_FLOOR)
        if mode == "linear":
            y = draw_rng.gamma(shape=alpha, scale=1.0)
            n_under += int((y < np.finfo(np.float64).tiny).sum())
            rs = np.maximum(y.sum(axis=1, keepdims=True), GAMMA_FLOOR)
            W = y / rs
            sw = np.where(W > 0, W, 1.0)
            W_all[s:e] = W
            kl[s:e] = np.sum(W * np.log(float(n_inc) * sw), axis=1)
            continue
        if mode == "replay":
            if check_replay:
                twin = np.random.Generator(type(draw_rng.bit_generator)())
                twin.bit_generator.state = draw_rng.bit_generator.state
                y_np = twin.gamma(shape=alpha, scale=1.0)
            y, logy = gamma_replay_log(draw_rng, alpha)
            if check_replay:
                if not np.array_equal(y, y_np) or twin.bit_generator.state != draw_rng.bit_generator.state:
                    raise RuntimeError("the replayed Gamma draws differ from numpy's on this numpy version")
            n_under += int((y < np.finfo(np.float64).tiny).sum())
        else:  # "log"
            xg = draw_rng.standard_gamma(alpha + 1.0)
            uu = 1.0 - draw_rng.random(size=alpha.shape)
            logy = np.log(xg) + np.log(uu) / alpha
        W, logW, k_ = _normalise_log(logy, ln_n)
        W_all[s:e], logW_all[s:e], kl[s:e] = W, logW, k_
    return {"W": W_all, "logW": logW_all, "kl": kl, "params": prm,
            "n_all_zero": int((W_all.sum(axis=1) == 0).sum()), "n_gamma_underflow": n_under}


def figure_streams(seed: int = SEED):
    """(calibration rng, draw rng) of a 10,000-W figure: SeedSequence(seed).spawn(2)."""
    calib_seed, draw_seed = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(calib_seed), np.random.default_rng(draw_seed)


def chunk_stream(i: int, seed: int = SEED) -> np.random.Generator:
    """The draw rng of Figure 4(b)'s chunk i: SeedSequence(seed, spawn_key=(1 + i,)); chunk 0 is a figure's stream."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1 + int(i),)))


def sample(u_sorted, n: int, mode: str, seed: int = SEED, check_replay: bool = True) -> dict:
    """A 10,000-W figure's draws (any n): the calibration from child 0, then n W's from child 1."""
    calib_rng, draw_rng = figure_streams(seed)
    c_grid, kl_med = calibrate(len(u_sorted), calib_rng)
    out = draw(u_sorted, n, mode, draw_rng, c_grid, kl_med, check_replay=check_replay)
    out["c_grid"], out["kl_med"] = c_grid, kl_med
    return out
