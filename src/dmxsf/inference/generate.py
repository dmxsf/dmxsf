"""Autoregressive generation with a KV cache: the loop every chat product runs.

Course pages: https://dmxsf.com/wiki/inference/decoding-and-sampling/
              https://dmxsf.com/wiki/inference/kv-cache/

``generate`` takes a batch of prompts *of the same length* (pad the text, not the ids: with learned
position embeddings a left-padded prompt would shift every position), runs the prompt once to fill
the cache, then feeds one new token per step. Sampling is a pure function of the logits
(``sample_logits``), so decoding strategies can be tested without a model.
"""

from __future__ import annotations

import torch

from dmxsf.inference.kv_cache import KVCache


def process_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> torch.Tensor:
    """Temperature, then top-k, then top-p, then min-p; dropped tokens get ``-inf``.

    Order matters and follows common practice: divide by the temperature, keep the ``top_k`` best,
    keep the smallest set of tokens whose probability reaches ``top_p`` (nucleus sampling), and
    drop tokens below ``min_p`` times the largest probability. ``softmax`` of the result is the
    distribution that gets sampled from.
    """
    logits = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        before = probs.cumsum(dim=-1) - probs  # mass of the tokens ranked above this one
        sorted_logits = sorted_logits.masked_fill(before >= top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    if min_p is not None:
        probs = logits.softmax(dim=-1)
        logits = logits.masked_fill(
            probs < min_p * probs.max(dim=-1, keepdim=True).values, float("-inf")
        )
    return logits


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    min_p: float | None = None,
) -> torch.Tensor:
    """Pick one token per row of ``logits`` (B, V). ``temperature=0`` is greedy decoding."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    processed = process_logits(logits, temperature, top_k, top_p, min_p)
    return torch.multinomial(processed.softmax(dim=-1), 1, generator=generator).squeeze(-1)


def apply_penalties(
    logits: torch.Tensor,
    history: torch.Tensor,
    repetition: float = 1.0,
    frequency: float = 0.0,
    presence: float = 0.0,
) -> torch.Tensor:
    """Penalise tokens that already appeared. ``logits``: (B, V); ``history``: (B, T) token ids.

    ``repetition`` (CTRL / Hugging Face style): a seen token's logit is divided by it if positive
    and multiplied by it if negative. ``frequency`` subtracts ``frequency * count``; ``presence``
    subtracts a flat ``presence`` once a token has appeared at all.
    """
    counts = torch.zeros_like(logits).scatter_add_(
        1, history, torch.ones_like(history, dtype=logits.dtype)
    )
    seen = counts > 0
    out = torch.where(
        seen, torch.where(logits > 0, logits / repetition, logits * repetition), logits
    )
    return out - frequency * counts - presence * seen.to(logits.dtype)


@torch.no_grad()
def generate(
    model,
    prompts: torch.Tensor,
    max_new: int,
    *,
    stop: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition: float = 1.0,
    frequency: float = 0.0,
    presence: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Continue each row of ``prompts`` (B, P) by up to ``max_new`` tokens; returns (B, max_new).

    The penalties (see ``apply_penalties``) look at the prompt and everything generated so far.

    Once a row has produced ``stop`` it keeps emitting ``stop``; generation ends early when every
    row is done. Tokens after the stop token are therefore padding, not text.
    """
    was_training = model.training
    model.eval()
    cache = KVCache(model.cfg.n_layer)
    logits = model(prompts, cache=cache)[:, -1]
    out = []
    done = torch.zeros(prompts.shape[0], dtype=torch.bool)
    penalised = (repetition, frequency, presence) != (1.0, 0.0, 0.0)
    history = prompts
    for _ in range(max_new):
        if penalised:
            logits = apply_penalties(logits, history, repetition, frequency, presence)
        token = sample_logits(logits, temperature, top_k, top_p, generator, min_p)
        if stop is not None:
            token = torch.where(done, torch.full_like(token, stop), token)
            done |= token == stop
        if penalised:
            history = torch.cat([history, token[:, None]], dim=1)
        out.append(token)
        if stop is not None and bool(done.all()):
            break
        logits = model(token[:, None], cache=cache)[:, -1]
    result = torch.stack(out, dim=1)
    if stop is not None and result.shape[1] < max_new:
        pad = torch.full((result.shape[0], max_new - result.shape[1]), stop, dtype=result.dtype)
        result = torch.cat([result, pad], dim=1)
    model.train(was_training)
    return result


@torch.no_grad()
def beam_search(model, prompt: torch.Tensor, beams: int, max_new: int):
    """Beam search from one prompt (1, P): keep the ``beams`` best partial sequences by total
    log-probability and extend each by every token.

    Returns ``(tokens, scores)``: tokens (beams, max_new) and the summed log-probabilities,
    best first. Every beam has the same length (there is no stop token), so no length
    normalisation is needed. With ``beams=1`` this is greedy decoding.
    """
    was_training = model.training
    model.eval()
    cache = KVCache(model.cfg.n_layer)
    logp = torch.log_softmax(model(prompt, cache=cache)[:, -1], dim=-1)
    scores, tokens = logp[0].topk(beams)
    cache.reorder(torch.zeros(beams, dtype=torch.long))  # every beam starts from the prompt
    seqs = tokens[:, None]
    for _ in range(max_new - 1):
        logp = torch.log_softmax(model(seqs[:, -1:], cache=cache)[:, -1], dim=-1)
        flat = (scores[:, None] + logp).reshape(-1)
        scores, best = flat.topk(beams)
        parent, token = best // logp.shape[1], best % logp.shape[1]
        cache.reorder(parent)
        seqs = torch.cat([seqs[parent], token[:, None]], dim=1)
    model.train(was_training)
    return seqs, scores
