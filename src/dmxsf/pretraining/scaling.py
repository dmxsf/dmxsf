"""Chinchilla's parametric loss, the compute-optimal split, and the cost of serving.

L(N, D) = E + A / N^alpha + B / D^beta, with the coefficients of the replication by Besiroglu et al.
(2024), which are the ones the course page uses.

Course page: https://dmxsf.com/wiki/pretraining/scaling-laws/
"""

from __future__ import annotations

import numpy as np

E, A, B, ALPHA, BETA = 1.8172, 482.01, 2085.43, 0.3478, 0.3658


def loss(n_params, tokens, e=E, a=A, b=B, alpha=ALPHA, beta=BETA):
    return e + a / n_params**alpha + b / tokens**beta


def compute_optimal(flops, a=A, b=B, alpha=ALPHA, beta=BETA):
    """Minimise L(N, D) subject to 6 N D = C. Returns (N, D) in closed form."""
    n = (alpha * a / (beta * b)) ** (1 / (alpha + beta)) * (flops / 6) ** (beta / (alpha + beta))
    return n, flops / (6 * n)


def optimal_exponents(alpha=ALPHA, beta=BETA):
    """N_opt ~ C^(beta / (alpha + beta)) and D_opt ~ C^(alpha / (alpha + beta))."""
    return beta / (alpha + beta), alpha / (alpha + beta)


def tokens_needed(n_params, target, e=E, a=A, b=B, alpha=ALPHA, beta=BETA):
    """Training tokens a model of ``n_params`` needs to reach ``target`` loss (inf if it never can)."""
    gap = target - e - a / n_params**alpha
    return (b / gap) ** (1 / beta) if gap > 0 else np.inf


def best_model(target, served_tokens, sizes=None):
    """The model size that reaches ``target`` loss at the lowest total FLOPs: 6 N D to train plus
    2 N T_serve to serve ``served_tokens``. Returns (cost, N, D)."""
    best = None
    for n in np.logspace(9, 12.5, 4000) if sizes is None else sizes:
        d = tokens_needed(n, target)
        if np.isfinite(d):
            cost = 6 * n * d + 2 * n * served_tokens
            if best is None or cost < best[0]:
                best = (cost, n, d)
    return best


def fit(n_params, tokens, losses, grid=None, refine=3):
    """Fit L = E + A / N^alpha + B / D^beta to measured losses; returns a dict with a residual.

    For a fixed (alpha, beta) the loss is linear in (E, A, B), so this scans the two exponents on a
    grid, solves the three coefficients by least squares for each pair (a pair that needs a negative
    coefficient is rejected), keeps the best, and zooms in around it ``refine`` times. No optimizer
    to tune and no starting point to guess, which is what makes it safe for a tiny experiment.
    """
    n = np.asarray(n_params, dtype=float)
    d = np.asarray(tokens, dtype=float)
    y = np.asarray(losses, dtype=float)
    n_ref, d_ref = np.exp(np.log(n).mean()), np.exp(np.log(d).mean())  # keeps the columns O(1)

    def solve(alpha, beta):
        x = np.stack([np.ones_like(n), (n / n_ref) ** -alpha, (d / d_ref) ** -beta], axis=1)
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        if (coef < 0).any():
            return np.inf, coef
        return float(np.sqrt(np.mean((x @ coef - y) ** 2))), coef

    a_grid = b_grid = np.linspace(0.02, 1.5, 75) if grid is None else grid
    best = (np.inf, 0.0, 0.0, None)
    for _ in range(refine + 1):
        for alpha in a_grid:
            for beta in b_grid:
                rms, coef = solve(alpha, beta)
                if rms < best[0]:
                    best = (rms, alpha, beta, coef)
        step_a = (a_grid[1] - a_grid[0]) * 2
        step_b = (b_grid[1] - b_grid[0]) * 2
        a_grid = np.linspace(max(best[1] - step_a, 0.005), best[1] + step_a, 21)
        b_grid = np.linspace(max(best[2] - step_b, 0.005), best[2] + step_b, 21)
    rms, alpha, beta, coef = best
    if coef is None:
        raise ValueError("no pair of exponents gives non-negative coefficients")
    e, a_scaled, b_scaled = (float(c) for c in coef)
    return {
        "E": e,
        "A": a_scaled * n_ref**alpha,
        "B": b_scaled * d_ref**beta,
        "alpha": float(alpha),
        "beta": float(beta),
        "rms": rms,
    }
