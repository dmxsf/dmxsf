"""Preference optimization: the DPO loss, its closed-form optimum, and a Bradley-Terry toy world.

The DPO derivation says that the KL-regularised reward-maximisation problem has the solution
``pi*(y) = pi_ref(y) exp(r(y) / beta) / Z``, so a preference model ``sigmoid(r(y_w) - r(y_l))``
can be fitted by moving ``pi`` directly. The tabular helpers here make that checkable to many
digits; the model helpers turn the same loss into a few lines for a language model whose answers
can be enumerated.

Course page: https://dmxsf.com/wiki/post-training/preference-optimization/
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dmxsf.post_training.intents import LABELS, answer_bytes, render
from dmxsf.post_training.sft import finetune


def sequence_logprob(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Log-probability of each response: per-token log-probs summed over the masked positions.

    ``logits``: (B, T, V), ``labels`` and ``mask``: (B, T); ``logits[:, t]`` predicts ``labels[:, t]``.
    """
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(dim=-1)


def dpo_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The DPO loss and the share of pairs whose implicit rewards are ordered correctly.

    All four inputs are whole-response log-probabilities, shape (B,).
    """
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, accuracy


def dpo_gradient_weight(
    pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta: float = 0.1
) -> torch.Tensor:
    """``sigmoid(r_rejected - r_chosen)``: how wrongly a pair is ordered; the gradient's per-pair weight."""
    margin = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return torch.sigmoid(-margin)


def optimal_policy(ref: np.ndarray, reward: np.ndarray, beta: float) -> np.ndarray:
    """Closed-form solution of max E[r] - beta KL(pi || ref) over distributions on a finite set."""
    w = ref * np.exp((reward - reward.max()) / beta)
    return w / w.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats."""
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def bradley_terry_counts(
    reward: np.ndarray, n_pairs: int, seed: int, ref: np.ndarray | None = None
) -> np.ndarray:
    """``counts[i, j]``: how often response i was preferred to j in ``n_pairs`` simulated comparisons.

    Two responses are drawn (from ``ref`` or uniformly) and the winner is chosen with probability
    ``sigmoid(r_i - r_j)``: the Bradley-Terry model.
    """
    rng = np.random.default_rng(seed)
    k = len(reward)
    p = np.full(k, 1 / k) if ref is None else ref
    i = rng.choice(k, size=n_pairs, p=p)
    j = rng.choice(k, size=n_pairs, p=p)
    keep = i != j
    i, j = i[keep], j[keep]
    i_wins = rng.random(len(i)) < 1 / (1 + np.exp(-(reward[i] - reward[j])))
    win, lose = np.where(i_wins, i, j), np.where(i_wins, j, i)
    counts = np.zeros((k, k))
    np.add.at(counts, (win, lose), 1)
    return counts


def fit_dpo_tabular(
    counts: np.ndarray, ref: np.ndarray, beta: float, steps: int = 3000, lr: float = 0.05
) -> np.ndarray:
    """Minimise the DPO loss over the logits of a categorical policy, from preference counts only."""
    ref_log = torch.tensor(np.log(ref))
    c = torch.tensor(counts)
    theta = ref_log.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        logp = torch.log_softmax(theta, dim=0)
        implicit = beta * (logp - ref_log)  # the implicit reward of every response
        margin = implicit[:, None] - implicit[None, :]
        loss = -(c * F.logsigmoid(margin)).sum() / c.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(theta, dim=0).detach().numpy()


# --- a language model whose answers can be enumerated -------------------------------------------


def answer_batch(texts: list[str], labels: list[str]):
    """Tokens for (prompt, answer) pairs: inputs, shifted targets and a mask over the answer."""
    seqs = [list(render(t)) + list(answer_bytes(y)) for t, y in zip(texts, labels, strict=True)]
    n_prompt = len(render(texts[0]))
    longest = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), longest, dtype=torch.long)
    mask = torch.zeros(len(seqs), longest - 1)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, n_prompt - 1 : len(s) - 1] = 1.0  # positions that predict an answer byte
    return ids[:, :-1], ids[:, 1:], mask


def answer_logprobs(model, texts: list[str], labels: list[str]) -> torch.Tensor:
    """Log-probability of ``labels[i]`` as the answer to ``texts[i]``, shape (N,); keeps the graph."""
    x, y, mask = answer_batch(texts, labels)
    return sequence_logprob(model(x), y, mask)


@torch.no_grad()
def label_distribution(model, texts: list[str], chunk: int = 20) -> torch.Tensor:
    """(N, 12): log-probability of each of the twelve labels for every text."""
    out = []
    for s in range(0, len(texts), chunk):
        part = texts[s : s + chunk]
        rows = [
            answer_logprobs(model, part, [label] * len(part)) for label in LABELS
        ]  # twelve batched forward passes
        out.append(torch.stack(rows, dim=1))
    return torch.cat(out)


def normalise(logp: torch.Tensor) -> torch.Tensor:
    """Turn (N, 12) log-probabilities of whole answers into a distribution over the labels."""
    return torch.log_softmax(logp, dim=-1)


Pair = tuple[str, str, str]  # (message, chosen label, rejected label)


def make_pairs(
    texts: list[str], gold: list[str], label_probs: np.ndarray, per_prompt: int, seed: int
) -> list[Pair]:
    """Preference pairs: chosen = the correct label, rejected = a mistake the reference makes.

    The rejected label is drawn from the reference's own distribution over the eleven wrong
    labels (``label_probs``, shape (N, 12)), so the pairs are about errors this model really
    produces, as in on-policy preference data.
    """
    rng = np.random.default_rng(seed)
    out = []
    for text, label, probs in zip(texts, gold, label_probs, strict=True):
        wrong = np.array(probs, dtype=np.float64)
        wrong[LABELS.index(label)] = 0.0
        wrong /= wrong.sum()
        out += [(text, label, LABELS[rng.choice(len(LABELS), p=wrong)]) for _ in range(per_prompt)]
    return out


@torch.no_grad()
def pair_logprobs(model, pairs: list[Pair]) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-answer log-probabilities of every chosen and every rejected answer (no gradient)."""
    texts = [p[0] for p in pairs]
    chosen = answer_logprobs(model, texts, [p[1] for p in pairs])
    rejected = answer_logprobs(model, texts, [p[2] for p in pairs])
    return chosen, rejected


def train_dpo(
    policy,
    pairs: list[Pair],
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    *,
    beta: float,
    lr: float,
    steps: int,
    batch: int = 16,
    seed: int = 0,
    params=None,
) -> list[dict[str, float]]:
    """DPO fine-tuning with the reference log-probabilities computed once, up front.

    Returns one dict per step: the loss, the share of pairs whose implicit rewards are ordered correctly (``reward_acc``), the mean change of
    the chosen and rejected log-probabilities relative to the reference, and the mean gradient
    weight ``sigmoid(r_rejected - r_chosen)``.
    """
    log: list[dict[str, float]] = []

    def batch_fn(step):
        return np.random.default_rng((seed, step)).integers(0, len(pairs), size=batch)

    def loss_fn(model, idx):
        texts = [pairs[i][0] for i in idx]
        pi_c = answer_logprobs(model, texts, [pairs[i][1] for i in idx])
        pi_r = answer_logprobs(model, texts, [pairs[i][2] for i in idx])
        ref_c, ref_r = ref_chosen[idx], ref_rejected[idx]
        loss, acc = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
        log.append(
            {
                "loss": loss.item(),
                "reward_acc": acc.item(),
                "d_chosen": (pi_c - ref_c).mean().item(),
                "d_rejected": (pi_r - ref_r).mean().item(),
                "weight": dpo_gradient_weight(pi_c.detach(), pi_r.detach(), ref_c, ref_r, beta)
                .mean()
                .item(),
            }
        )
        return loss

    finetune(policy, batch_fn, steps, lr, loss_fn=loss_fn, params=params)
    return log
