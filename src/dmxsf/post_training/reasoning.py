"""Reasoning RL: verifiable rewards, GRPO advantages and loss, and a rollout-and-update loop.

The page's three pieces are here as plain functions: a rule-based verifier, the group-relative
advantage ``(r - mean) / std``, and the PPO-style clipped loss with a k3 KL penalty. ``rollout``
samples answers from a language model and ``train_grpo`` closes the loop on the intent task
(``dmxsf.post_training.intents``), where the verifier is an exact string match.

Course page: https://dmxsf.com/wiki/post-training/reasoning-rl/
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import torch

from dmxsf.inference.generate import generate
from dmxsf.post_training.intents import LABELS, STOP, render

# --- verifiers ------------------------------------------------------------------------------


def math_reward(response: str, gold: str) -> float:
    """The page's verifier: the answer sits in ``\\boxed{}`` and equals the gold string."""
    match = re.findall(r"\\boxed\{([^}]*)\}", response)
    if not match:
        return 0.0  # not in the required format
    return 1.0 if match[-1].strip() == gold.strip() else 0.0


def exact_match_reward(response: str, gold: str) -> float:
    """1 if the whole answer is the gold label, else 0 (a verifier that cannot be argued with)."""
    return 1.0 if response.strip() == gold else 0.0


def contains_reward(response: str, gold: str) -> float:
    """A leaky verifier: 1 if the gold label appears anywhere in the answer."""
    return 1.0 if gold in response else 0.0


# --- GRPO -----------------------------------------------------------------------------------


def group_advantages(
    rewards: torch.Tensor, eps: float = 1e-4, divide_by_std: bool = True, unbiased: bool = True
) -> torch.Tensor:
    """``rewards``: (num_prompts, G), one row per prompt. The page's function, with two switches.

    ``unbiased`` picks the sample (G - 1) or population (G) standard deviation; ``divide_by_std``
    False keeps only the mean subtraction.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    centred = rewards - mean
    if not divide_by_std:
        return centred
    std = rewards.std(dim=1, keepdim=True, unbiased=unbiased)
    return centred / (std + eps)


def signal_probability(p: float, group: int) -> float:
    """Chance that a group of ``group`` samples, each right with probability ``p``, is mixed."""
    return 1.0 - p**group - (1.0 - p) ** group


def k3(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """The k3 estimate of KL(pi_new || pi_ref) from tokens sampled from pi_new: non-negative."""
    d = logp_ref - logp_new
    return torch.exp(d) - d - 1


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps_clip: float = 0.2,
    beta: float = 0.04,
    token_level: bool = False,
) -> torch.Tensor:
    """The page's loss. ``logp_*``: (B, T) per-token log-probabilities, ``advantages``: (B,).

    Default: average over the tokens of each answer, then over answers (the page's version).
    ``token_level=True`` averages over all tokens of the batch instead (the DAPO choice).
    """
    ratio = torch.exp(logp_new - logp_old)
    adv = advantages.unsqueeze(1)
    surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv)
    per_token = surrogate - beta * k3(logp_new, logp_ref)
    if token_level:
        return -(per_token * mask).sum() / mask.sum()
    per_seq = (per_token * mask).sum(dim=1) / mask.sum(dim=1)
    return -per_seq.mean()


# --- sampling and the training loop ----------------------------------------------------------


def token_logprobs(model, ids: torch.Tensor, with_entropy: bool = False):
    """(N, T-1): log-probability of every token given the ones before it.

    With ``with_entropy`` it also returns the entropy (nats) of the model's distribution at each
    position, the signal that collapses when a policy stops exploring.
    """
    logp_all = torch.log_softmax(model(ids[:, :-1]), dim=-1)
    logp = logp_all.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    if not with_entropy:
        return logp
    return logp, -(logp_all.exp() * logp_all).sum(dim=-1)


@torch.no_grad()
def rollout(
    model,
    texts: list[str],
    group: int,
    *,
    max_new: int = 24,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
):
    """Sample ``group`` answers per message. Returns ``ids``, a target ``mask`` and the answers.

    ``ids`` is (len(texts) * group, P + max_new) with the samples of one message in consecutive
    rows; ``mask`` is (rows, P + max_new - 1) and is 1 on the targets that belong to the answer,
    up to and including the stop token; ``answers`` are the decoded strings.
    """
    prompts = torch.tensor([list(render(t)) for t in texts for _ in range(group)])
    out = generate(model, prompts, max_new, stop=STOP, temperature=temperature, generator=generator)
    ids = torch.cat([prompts, out], dim=1)
    n_prompt = prompts.shape[1]
    mask = torch.zeros(ids.shape[0], ids.shape[1] - 1)
    answers = []
    for i, row in enumerate(out.tolist()):
        cut = row.index(STOP) if STOP in row else len(row)
        length = min(cut + 1, len(row))  # the stop token is part of the answer
        mask[i, n_prompt - 1 : n_prompt - 1 + length] = 1.0
        answers.append(bytes(row[:cut]).decode("utf-8", errors="replace").strip())
    return ids, mask, answers


def train_grpo(
    policy,
    reference,
    texts: list[str],
    gold: list[str],
    *,
    steps: int,
    prompts_per_step: int = 16,
    group: int = 8,
    lr: float = 3e-5,
    beta: float = 0.04,
    seed: int = 0,
    reward_fn: Callable[[str, str], float] = exact_match_reward,
    divide_by_std: bool = True,
    token_level: bool = False,
    max_new: int = 24,
) -> list[dict[str, float]]:
    """GRPO with one update per rollout (so the ratio is 1 and the clip never binds).

    Per step it returns: the mean reward, the share of groups that carried a learning signal,
    the mean answer length, the share of answers that are one of the twelve labels, the mean k3 KL
    estimate on the sampled tokens, and the mean entropy of the policy at the answer positions.
    """
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    log = []
    for step in range(steps):
        rng = np.random.default_rng((seed, step))
        picks = rng.choice(len(texts), size=prompts_per_step, replace=False)
        gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
        ids, mask, answers = rollout(
            policy, [texts[i] for i in picks], group, max_new=max_new, generator=gen
        )
        golds = [gold[i] for i in picks for _ in range(group)]
        rewards = torch.tensor([reward_fn(a, g) for a, g in zip(answers, golds, strict=True)])
        rewards = rewards.view(prompts_per_step, group)
        adv = group_advantages(rewards, divide_by_std=divide_by_std).reshape(-1)
        with torch.no_grad():
            logp_ref = token_logprobs(reference, ids)
        policy.train()
        logp_new, entropy = token_logprobs(policy, ids, with_entropy=True)
        loss = grpo_loss(
            logp_new, logp_new.detach(), logp_ref, adv, mask, beta=beta, token_level=token_level
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            kl = ((k3(logp_new, logp_ref) * mask).sum() / mask.sum()).item()
            mean_entropy = ((entropy * mask).sum() / mask.sum()).item()
        log.append(
            {
                "reward": rewards.mean().item(),
                "signal": (rewards.std(dim=1) > 0).float().mean().item(),
                "length": mask.sum(dim=1).mean().item(),
                "valid": float(np.mean([a in LABELS for a in answers])),
                "kl": kl,
                "entropy": mean_entropy,
                "loss": loss.item(),
            }
        )
    policy.eval()
    return log
