"""Speculative decoding: a small model drafts, the big one verifies several tokens in one pass.

Course page: https://dmxsf.com/wiki/inference/speculative-decoding/

One round: the draft model proposes ``gamma`` tokens one after the other; the target model reads the
last accepted token and all ``gamma`` proposals in a single forward pass; proposal i is kept with
probability min(1, p_i(x_i) / q_i(x_i)); at the first rejection a replacement is drawn from the
residual distribution norm(max(0, p - q)), and if every proposal survives the target's last row
supplies one bonus token. The sequence that comes out has exactly the target's distribution.

Everything is batch size 1 and uses the KV caches of dmxsf.inference.kv_cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dmxsf.inference.kv_cache import KVCache

# ---------------------------------------------------------------------------------------------
# The rule on one position


def probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax at ``temperature``; temperature 0 is the one-hot distribution of the argmax."""
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return torch.softmax(logits / temperature, dim=-1)


def acceptance_rate(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """beta = sum_x min(p(x), q(x)) = 1 - TV(p, q): the chance that one drafted token is accepted."""
    return torch.minimum(p, q).sum(-1)


def residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(0, p - q)): where a rejected position is refilled from."""
    r = (p - q).clamp(min=0)
    total = r.sum(-1, keepdim=True)
    return torch.where(total > 0, r / total.clamp(min=1e-30), p)


def output_distribution(p: torch.Tensor, q: torch.Tensor, repair: str = "residual") -> torch.Tensor:
    """The exact distribution of the token one position yields: proposed and accepted, plus
    rejected and repaired. ``repair="target"`` refills from p itself instead of the residual: the
    tempting wrong rule, which double counts the likely tokens."""
    beta = acceptance_rate(p, q)
    fill = residual(p, q) if repair == "residual" else p
    return torch.minimum(p, q) + (1 - beta).unsqueeze(-1) * fill


def verify(p, q, drafted, generator=None):
    """One round's decision. p: (gamma + 1, V) target rows, q: (gamma, V) draft rows, drafted:
    (gamma,) proposals. Returns (k, final): k proposals kept, then the token that follows them."""
    gamma = len(drafted)
    u = torch.rand(gamma, generator=generator)
    for i in range(gamma):
        x = int(drafted[i])
        if u[i] < p[i, x] / q[i, x]:  # accepted with probability min(1, p / q)
            continue
        return i, torch.multinomial(residual(p[i], q[i]), 1, generator=generator)
    return gamma, torch.multinomial(p[gamma], 1, generator=generator)


# ---------------------------------------------------------------------------------------------
# The formulas (Leviathan et al., 2023, theorem 3.8)


def expected_tokens(alpha: float, gamma: int) -> float:
    """Tokens one round produces when every proposal is accepted independently with probability alpha."""
    return gamma + 1.0 if alpha >= 1 else (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha: float, gamma: int, c: float) -> float:
    """Against plain decoding; c is the cost of one draft step relative to one target step."""
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


def best_gamma(alpha: float, c: float, max_gamma: int = 64) -> tuple[int, float]:
    """The proposal length that maximises the speedup, and that speedup."""
    return max(((g, speedup(alpha, g, c)) for g in range(1, max_gamma + 1)), key=lambda t: t[1])


# ---------------------------------------------------------------------------------------------
# The whole loop


@dataclass
class Stats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    rejections: int = 0  # rounds that ended with a rejection (the others kept every drafted token)
    target_calls: int = 0  # forward passes of the target, the prefill included
    draft_calls: int = 0

    @property
    def alpha(self) -> float:
        """The acceptance rate of one drafted token the rule actually looked at. A round stops at its
        first rejection, so the tokens drafted after it were never tested: they belong in neither the
        numerator nor the denominator. Each tested token is either accepted or the rejection."""
        return self.accepted / max(1, self.accepted + self.rejections)

    @property
    def kept_share(self) -> float:
        """The share of all drafted tokens that ended up in the output (wasted draft work is the rest)."""
        return self.accepted / max(1, self.drafted)

    @property
    def tokens_per_round(self) -> float:
        return (self.accepted + self.rounds) / max(1, self.rounds)


@torch.no_grad()
def speculative_generate(
    target,
    draft,
    prompt: torch.Tensor,
    max_new: int,
    gamma: int = 4,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
):
    """Continue ``prompt`` (1, T) by ``max_new`` tokens. Returns ((1, max_new) tokens, Stats).

    With temperature 0 the tokens equal the target's greedy decoding; otherwise they are samples
    from the target's distribution at that temperature. The models need learned positions at least
    T + max_new + gamma long.
    """
    if prompt.shape[0] != 1:
        raise ValueError("batch size 1 only")
    t_len = prompt.shape[1]
    limit = min(target.cfg.block_size, draft.cfg.block_size)
    if target.cfg.pos == "learned" and t_len + max_new + gamma > limit:
        raise ValueError(f"prompt + max_new + gamma = {t_len + max_new + gamma} exceeds {limit}")
    t_cache, d_cache = KVCache(len(target.blocks)), KVCache(len(draft.blocks))
    stats = Stats()
    if t_len > 1:  # both caches hold everything except the newest token
        target(prompt[:, :-1], cache=t_cache)
        draft(prompt[:, :-1], cache=d_cache)
        stats.target_calls += 1
        stats.draft_calls += 1
    last, fed, new = prompt[:, -1:], t_len - 1, []
    while len(new) < max_new:
        q_rows, drafted, x = [], [], last
        for _ in range(gamma):  # the draft model proposes one token at a time
            q = probs(draft(x, cache=d_cache)[0, -1], temperature)
            token = torch.multinomial(q, 1, generator=generator)
            q_rows.append(q)
            drafted.append(token)
            x = token.view(1, 1)
        block = torch.cat([last, *[t.view(1, 1) for t in drafted]], dim=1)  # (1, gamma + 1)
        p = probs(target(block, cache=t_cache)[0], temperature)  # all gamma + 1 rows at once
        k, final = verify(p, torch.stack(q_rows), torch.cat(drafted), generator)
        new.extend(int(t) for t in drafted[:k])
        new.append(int(final))
        t_cache.crop(fed + 1 + k)  # keep the last token and the accepted proposals
        if k == gamma:  # the draft never read its own last proposal
            draft(drafted[-1].view(1, 1), cache=d_cache)
            stats.draft_calls += 1
        else:
            d_cache.crop(fed + 1 + k)
        last, fed = final.view(1, 1), fed + 1 + k
        stats.rounds += 1
        stats.drafted += gamma
        stats.accepted += k
        stats.rejections += int(k < gamma)
        stats.target_calls += 1
        stats.draft_calls += gamma
    return torch.tensor([new[:max_new]], dtype=torch.long), stats
