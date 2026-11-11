"""LLM-as-a-judge: a judge with biases (simulated), agreement measures, Bradley-Terry ratings.

Course page: https://dmxsf.com/wiki/evaluation/llm-as-judge/

The judge is a stand-in with known biases, so every number can be checked against an exact answer:
it scores "the first answer wins" as the quality gap, plus a bonus for being first, plus a bonus
per unit of extra length, plus Gaussian noise. Real judges are models; their biases have the same
shapes.
"""

from __future__ import annotations

from collections.abc import Callable
from math import erf, pi, sqrt

import numpy as np

# --- a judge with known biases -------------------------------------------------------------------

Answer = float | tuple[float, float]  # a quality, or (quality, length)
Judge = Callable[[Answer, Answer], str]


def _quality_length(answer: Answer) -> tuple[float, float]:
    return (float(answer), 0.0) if isinstance(answer, int | float | np.floating) else answer


def make_judge(
    position_bias: float, noise: float, rng: np.random.Generator, length_bias: float = 0.0
) -> Judge:
    """A stand-in judge. It sees the quality of both answers (and their lengths, if the answers
    are ``(quality, length)`` pairs) and prefers the answer shown first by ``position_bias`` and the
    longer answer by ``length_bias`` per unit of length, with Gaussian noise on top."""

    def judge(first: Answer, second: Answer) -> str:
        (q1, l1), (q2, l2) = _quality_length(first), _quality_length(second)
        score = q1 - q2 + position_bias + length_bias * (l1 - l2) + noise * rng.normal()
        return "first" if score > 0 else "second"

    return judge


def judge_once(judge: Judge, x: Answer, y: Answer, rng: np.random.Generator) -> str:
    """One judgment per pair, with the order chosen at random. Returns "x" or "y"."""
    if rng.random() < 0.5:
        return "x" if judge(x, y) == "first" else "y"
    return "y" if judge(y, x) == "first" else "x"


def judge_swapped(judge: Judge, x: Answer, y: Answer) -> str:
    """Judge both orders; a winner needs both judgments. Returns "x", "y" or "tie"."""
    one = judge(x, y)  # x first
    two = judge(y, x)  # y first
    if one == "first" and two == "second":
        return "x"
    if one == "second" and two == "first":
        return "y"
    return "tie"


# --- exact answers for the stand-in judge --------------------------------------------------------


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.array([erf(v / sqrt(2.0)) for v in np.ravel(x)]).reshape(np.shape(x)))


def exact_agreement(position_bias: float, noise: float, gap_sd: float = 1.0) -> dict[str, float]:
    """The exact expectations of the two procedures when the true quality gap x - y is
    N(0, gap_sd^2), by numerical integration over the gap.

    once:       accuracy of one judgment with the order drawn at random.
    tie:        share of pairs that the swapped procedure leaves undecided.
    swapped:    accuracy among the pairs the swapped procedure decides.
    """
    g = np.linspace(-8 * gap_sd, 8 * gap_sd, 4001)
    f = np.exp(-0.5 * (g / gap_sd) ** 2) / (gap_sd * sqrt(2 * pi))
    w = f * (g[1] - g[0])
    b, s = position_bias, noise
    x_first = norm_cdf((g + b) / s)  # P(judge says "first") with x shown first
    y_first = norm_cdf((-g + b) / s)  # P(judge says "first") with y shown first
    once_x = 0.5 * x_first + 0.5 * (1 - y_first)  # P(pick x) with a random order
    p_x = x_first * (1 - y_first)  # P(swapped procedure says x)
    p_y = (1 - x_first) * y_first
    correct = np.where(g > 0, once_x, 1 - once_x)
    decided_correct = np.where(g > 0, p_x, p_y)
    return {
        "once": float((w * correct).sum()),
        "tie": float(1 - (w * (p_x + p_y)).sum()),
        "swapped": float((w * decided_correct).sum() / (w * (p_x + p_y)).sum()),
    }


def agreement_no_bias(gap_sd: float, noise: float) -> float:
    """Closed form for a judge without position bias, judging once: 1/2 + arctan(s/sigma)/pi."""
    return 0.5 + np.arctan(gap_sd / noise) / pi


# --- how reliable is a judge? --------------------------------------------------------------------


def cohens_kappa(confusion: np.ndarray) -> tuple[float, float, float]:
    """(observed agreement, agreement by chance, kappa) of a square confusion matrix whose rows are
    one rater and columns the other."""
    n = confusion.sum()
    p_observed = np.trace(confusion) / n
    p_expected = (confusion.sum(axis=1) / n) @ (confusion.sum(axis=0) / n)
    return float(p_observed), float(p_expected), float((p_observed - p_expected) / (1 - p_expected))


def labels_needed(p: float, eps: float, z: float = 1.96) -> int:
    """Human labels needed to estimate an agreement rate p to within +-eps."""
    return int(np.ceil(z**2 * p * (1 - p) / eps**2))


# --- from pairwise votes to a leaderboard --------------------------------------------------------


def bt_probability(beta_i: float, beta_j: float) -> float:
    """Bradley-Terry: the chance that i beats j."""
    return 1.0 / (1.0 + np.exp(-(beta_i - beta_j)))


def fit_bt(i: np.ndarray, j: np.ndarray, win: np.ndarray, m: int, iters: int = 50) -> np.ndarray:
    """Maximum-likelihood Bradley-Terry coefficients by Newton's method (battle k: model i[k]
    against j[k], win[k] = 1 if i won). Only differences matter, so model 0 is fixed at 0."""
    beta = np.zeros(m)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(beta[i] - beta[j])))
        g = win - p
        grad = np.bincount(i, g, m) - np.bincount(j, g, m)
        w = p * (1 - p)
        hess = np.zeros((m, m))
        np.add.at(hess, (i, i), -w)
        np.add.at(hess, (j, j), -w)
        np.add.at(hess, (i, j), w)
        np.add.at(hess, (j, i), w)
        step = np.linalg.solve(-hess[1:, 1:], grad[1:])
        beta[1:] += step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def to_elo(beta: np.ndarray, anchor: float = 1000.0) -> np.ndarray:
    """Elo scale: 400 / ln 10 per unit of Bradley-Terry coefficient, model 0 at ``anchor``."""
    return 400 / np.log(10) * (beta - beta[0]) + anchor


def elo_to_beta(elo: np.ndarray) -> np.ndarray:
    return np.asarray(elo) * np.log(10) / 400


def bt_design(i: np.ndarray, j: np.ndarray, m: int, extra: np.ndarray | None = None) -> np.ndarray:
    """Design matrix of the Bradley-Terry logistic regression: +1 for model i, -1 for model j in
    every battle (model 0 is the reference and gets no column), plus optional extra columns such as
    the length difference between the two answers."""
    x = np.zeros((len(i), m - 1))
    for k in range(1, m):
        x[:, k - 1] = (i == k).astype(float) - (j == k).astype(float)
    if extra is not None:
        x = np.hstack([x, np.reshape(extra, (len(i), -1))])
    return x


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 100, ridge: float = 1e-9) -> np.ndarray:
    """Logistic regression without an intercept, fitted by Newton's method."""
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(x @ w)))
        grad = x.T @ (y - p) - ridge * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + ridge * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def online_elo(
    i: np.ndarray, j: np.ndarray, win: np.ndarray, m: int, k: float = 4.0, start: float = 1000.0
) -> np.ndarray:
    """Classic Elo, updated battle by battle. The result depends on the order of the battles."""
    rating = np.full(m, start)
    for a, b, w in zip(i, j, win, strict=True):
        expected = 1 / (1 + 10 ** ((rating[b] - rating[a]) / 400))
        rating[a] += k * (w - expected)
        rating[b] -= k * (w - expected)
    return rating
