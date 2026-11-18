"""Diffusion and flow matching: verify a Gaussian path.

Wiki pages
  Chinese: https://dmxsf.com/wiki/multimodal/diffusion-and-flow-matching/
  English: https://dmxsf.com/en/wiki/multimodal/diffusion-and-flow-matching/
"""

import math

import numpy as np


def forward_noise(clean: np.ndarray, noise: np.ndarray, alpha_bar: float) -> np.ndarray:
    """Sample the closed-form forward diffusion marginal."""
    return math.sqrt(alpha_bar) * clean + math.sqrt(1 - alpha_bar) * noise


def gaussian_flow(noise: np.ndarray, target_std: float, steps: int) -> np.ndarray:
    """Euler-integrate the conditional mean velocity of independent Gaussian endpoints.

    X_t = (1-t) Z + t X, with Z ~ N(0, 1) and X ~ N(0, target_std**2).
    The linear velocity coefficient is Cov(X-Z, X_t) / Var(X_t).
    """
    state = noise.copy()
    for i in range(steps):
        t = i / steps
        variance = (1 - t) ** 2 + (t * target_std) ** 2
        covariance = t * target_std**2 - (1 - t)
        state += (covariance / variance / steps) * state
    return state
