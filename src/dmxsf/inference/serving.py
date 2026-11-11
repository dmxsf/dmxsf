"""Serving: batched decoding over slots, iteration-level scheduling, paged KV memory, prefix caching.

Course pages: https://dmxsf.com/wiki/inference/continuous-batching/
              https://dmxsf.com/wiki/inference/serving/

Four small pieces that together are the skeleton of vLLM and llama.cpp's server:

* ``BatchedDecoder`` decodes one token for every active sequence in a single forward pass, although the
  sequences have different lengths (every request owns a slot of a padded cache).
* ``serve`` is the scheduling loop on top of it: static batching waits for a whole batch to finish,
  continuous batching refills a slot as soon as its request is done. ``simulate`` is the same loop
  on paper: it needs only the time of a step at each batch size.
* ``BlockAllocator`` / ``PagedKV`` store keys and values in fixed-size blocks reached through a
  per-sequence block table (PagedAttention), with reference counts and copy-on-write.
* ``PrefixCache`` keeps the keys and values of full blocks under a hash of everything before them.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from dmxsf.architecture.attention import repeat_kv
from dmxsf.inference.kv_cache import KVCache

# ---------------------------------------------------------------------------------------------
# The page's workload and its scheduler on paper


def make_lengths(n: int = 48, seed: int = 7) -> list[int]:
    """Skewed (log-normal) output lengths, as on the page."""
    rng = random.Random(seed)
    return [min(512, max(8, round(100 * rng.lognormvariate(0, 0.9)))) for _ in range(n)]


def step_from_table(table: dict[int, float]) -> Callable[[int], float]:
    """Seconds per decode step at batch size b, interpolated linearly between measured sizes."""
    ks = sorted(table)

    def step(b: int) -> float:
        if b <= ks[0]:
            return table[ks[0]]
        for lo, hi in itertools.pairwise(ks):
            if b <= hi:
                return table[lo] + (table[hi] - table[lo]) * (b - lo) / (hi - lo)
        return table[ks[-1]]

    return step


def simulate(
    lengths: list[int],
    slots: int,
    continuous: bool,
    stall: float,
    step: Callable[[int], float],
    admitted: list[float] | None = None,
) -> list[float]:
    """When each request finishes. ``lengths`` are decode steps; admitting a request pauses the whole
    batch for ``stall`` seconds (its prefill); a step takes ``step(batch size)``. Static batching only
    refills when every slot is empty. If ``admitted`` is a list, the time at which each request
    starts generating is appended to it."""
    todo, run, t, done = list(lengths), [], 0.0, []
    while todo or run:
        if continuous or not run:
            while todo and len(run) < slots:
                run.append(todo.pop(0))
                t += stall
                if admitted is not None:
                    admitted.append(t)
        t += step(len(run))
        run = [n - 1 for n in run]
        done += [t] * sum(n == 0 for n in run)
        run = [n for n in run if n > 0]
    return done


# ---------------------------------------------------------------------------------------------
# One decode step for many sequences of different lengths


class BatchedDecoder:
    """Slots over one padded cache. ``prefill`` fills a slot from a prompt; ``step`` appends one token
    to each of the chosen slots in one forward pass (learned positions, pre-norm GPT only)."""

    def __init__(self, model, slots: int, max_len: int):
        cfg = model.cfg
        if cfg.pos != "learned" or not cfg.pre_norm or cfg.window:
            raise ValueError("BatchedDecoder supports the course's learned-position, pre-norm GPT")
        a = model.blocks[0].attn
        shape = (len(model.blocks), slots, a.n_kv_heads, max_len, a.head_dim)
        self.model, self.max_len = model, max_len
        self.k, self.v = torch.zeros(shape), torch.zeros(shape)
        self.length = [0] * slots

    @torch.no_grad()
    def prefill(self, slot: int, tokens: torch.Tensor) -> torch.Tensor:
        """Run a prompt (T,) through the model and keep its keys and values in ``slot``; returns the
        logits of the last position."""
        cache = KVCache(len(self.model.blocks))
        logits = self.model(tokens[None], cache=cache)[0, -1]
        n = len(tokens)
        for layer in range(len(self.model.blocks)):
            self.k[layer, slot, :, :n] = cache.k[layer][0]
            self.v[layer, slot, :, :n] = cache.v[layer][0]
        self.length[slot] = n
        return logits

    def free(self, slot: int) -> None:
        self.length[slot] = 0

    @torch.no_grad()
    def step(self, slots: list[int], tokens: torch.Tensor) -> torch.Tensor:
        """Feed ``tokens`` (B,) to ``slots`` (the next input of each sequence); returns logits (B, V)."""
        m = self.model
        idx = torch.tensor(slots)
        pos = torch.tensor([self.length[s] for s in slots])
        if int(pos.max()) >= self.max_len:
            raise ValueError("a sequence outgrew its slot")
        x = (m.tok_emb(tokens) + m.pos_emb(pos))[:, None, :]
        longest = int(pos.max()) + 1
        keep = (torch.arange(longest)[None, :] <= pos[:, None])[:, None, None, :]
        b = len(slots)
        for layer, block in enumerate(m.blocks):
            a = block.attn
            h = block.norm1(x)
            q = a.q_proj(h).view(b, 1, a.n_heads, a.head_dim).transpose(1, 2)
            k = a.k_proj(h).view(b, 1, a.n_kv_heads, a.head_dim).transpose(1, 2)
            v = a.v_proj(h).view(b, 1, a.n_kv_heads, a.head_dim).transpose(1, 2)
            self.k[layer, idx, :, pos] = k[:, :, 0]  # each sequence writes at its own position
            self.v[layer, idx, :, pos] = v[:, :, 0]
            rep = a.n_heads // a.n_kv_heads
            keys = repeat_kv(self.k[layer, idx, :, :longest], rep)
            values = repeat_kv(self.v[layer, idx, :, :longest], rep)
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=keep)
            x = x + a.o_proj(out.transpose(1, 2).reshape(b, 1, a.n_heads * a.head_dim))
            x = x + block.mlp(block.norm2(x))
        for s in slots:
            self.length[s] += 1
        return m.lm_head(m.final_norm(x))[:, 0]


@dataclass
class Served:
    """What happened to one request (seconds since the loop started; every request arrives at 0)."""

    admitted: float = 0.0
    first_token: float = 0.0
    done: float = 0.0
    tokens: list[int] = field(default_factory=list)


@torch.no_grad()
def serve(
    model,
    requests: list[tuple[torch.Tensor, int]],
    slots: int,
    continuous: bool,
    max_len: int = 128,
) -> list[Served]:
    """Greedy-decode every (prompt, n_new) request with ``slots`` slots. Admitting a request prefills
    it on its own (all other sequences wait, as in the page's model); each step then decodes one
    token for every active slot. The first token of a request comes from its prefill."""
    decoder = BatchedDecoder(model, slots, max_len)
    waiting = list(range(len(requests)))
    free, active = (
        list(range(slots)),
        {},
    )  # active: slot -> (request, tokens still to write, last token)
    out = [Served() for _ in requests]
    t0 = time.perf_counter()

    def finish(r: int, slot: int) -> None:
        out[r].done = time.perf_counter() - t0
        decoder.free(slot)
        free.append(slot)

    while waiting or active:
        if continuous or not active:
            while waiting and free:
                r, slot = waiting.pop(0), free.pop(0)
                prompt, n_new = requests[r]
                out[r].admitted = time.perf_counter() - t0
                token = int(decoder.prefill(slot, prompt).argmax())
                out[r].first_token = time.perf_counter() - t0
                out[r].tokens.append(token)
                if n_new == 1:
                    finish(r, slot)
                else:
                    active[slot] = (r, n_new - 1, token)
        if not active:
            continue
        slots_now = sorted(active)
        logits = decoder.step(slots_now, torch.tensor([active[s][2] for s in slots_now]))
        for slot, token in zip(slots_now, logits.argmax(-1).tolist(), strict=True):
            r, left, _ = active.pop(slot)
            out[r].tokens.append(token)
            if left == 1:
                finish(r, slot)
            else:
                active[slot] = (r, left - 1, token)
    return out


# ---------------------------------------------------------------------------------------------
# Paged KV memory


class BlockAllocator:
    """A pool of fixed-size blocks with reference counts, so that sequences can share a prompt."""

    def __init__(self, n_blocks: int, block_size: int = 16):
        self.block_size, self.n_blocks = block_size, n_blocks
        self.refs = [0] * n_blocks
        self.free_list = list(range(n_blocks - 1, -1, -1))

    def allocate(self) -> int:
        if not self.free_list:
            raise MemoryError("no free KV block (a server would preempt a request here)")
        block = self.free_list.pop()
        self.refs[block] = 1
        return block

    def share(self, block: int) -> None:
        self.refs[block] += 1

    def release(self, block: int) -> None:
        self.refs[block] -= 1
        if self.refs[block] == 0:
            self.free_list.append(block)

    @property
    def used(self) -> int:
        return self.n_blocks - len(self.free_list)


@dataclass
class PagedSequence:
    table: list[int] = field(
        default_factory=list
    )  # logical block i lives in physical block table[i]
    length: int = 0


class PagedKV:
    """Keys and values of one layer in a block pool, reached through block tables.

    ``append`` writes tokens at the end of a sequence, allocating blocks on demand; a block that two
    sequences share is copied before either writes into it (copy-on-write). ``gather`` rebuilds the
    contiguous (heads, tokens, head_dim) tensors that attention reads.
    """

    def __init__(self, n_blocks: int, block_size: int, n_heads: int, head_dim: int):
        self.alloc = BlockAllocator(n_blocks, block_size)
        shape = (n_blocks, n_heads, block_size, head_dim)
        self.k, self.v = torch.zeros(shape), torch.zeros(shape)
        self.copies = 0

    def new_sequence(self) -> PagedSequence:
        return PagedSequence()

    def append(self, seq: PagedSequence, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: (heads, new tokens, head_dim)."""
        bs = self.alloc.block_size
        for i in range(k.shape[1]):
            slot = seq.length % bs
            if slot == 0:
                seq.table.append(self.alloc.allocate())
            else:
                last = seq.table[-1]
                if self.alloc.refs[last] > 1:  # shared: copy before writing
                    fresh = self.alloc.allocate()
                    self.k[fresh], self.v[fresh] = self.k[last].clone(), self.v[last].clone()
                    self.alloc.release(last)
                    seq.table[-1] = fresh
                    self.copies += 1
            block = seq.table[-1]
            self.k[block, :, slot] = k[:, i]
            self.v[block, :, slot] = v[:, i]
            seq.length += 1

    def fork(self, seq: PagedSequence) -> PagedSequence:
        """A second sequence that starts as a copy of ``seq`` and shares all its blocks."""
        for block in seq.table:
            self.alloc.share(block)
        return PagedSequence(list(seq.table), seq.length)

    def free(self, seq: PagedSequence) -> None:
        for block in seq.table:
            self.alloc.release(block)
        seq.table, seq.length = [], 0

    def gather(self, seq: PagedSequence) -> tuple[torch.Tensor, torch.Tensor]:
        def pick(pool: torch.Tensor) -> torch.Tensor:
            blocks = pool[seq.table]  # (blocks, heads, block_size, head_dim)
            merged = blocks.permute(1, 0, 2, 3).reshape(pool.shape[1], -1, pool.shape[3])
            return merged[:, : seq.length]

        return pick(self.k), pick(self.v)


# ---------------------------------------------------------------------------------------------
# Prefix cache


def block_hashes(tokens: list[int], block_size: int = 16, salt: str = "") -> list[str]:
    """One hash per full block; each covers its own tokens and the hash of the block before it, so a
    block can only match if everything in front of it matches too."""
    hashes, parent = [], salt
    for i in range(len(tokens) // block_size):
        block = tuple(tokens[i * block_size : (i + 1) * block_size])
        parent = hashlib.sha256(repr((parent, block)).encode()).hexdigest()[:16]
        hashes.append(parent)
    return hashes


@dataclass
class PrefixStats:
    reused: int = 0
    computed: int = 0


class PrefixCache:
    """Keys and values of full blocks, found by block hash. ``prefill`` computes only the tokens that
    follow the longest cached prefix, and stores every full block it has now seen."""

    def __init__(self, model, block_size: int = 16):
        self.model, self.block_size = model, block_size
        self.blocks: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, salt: str = "", use_cache: bool = True):
        """Returns (logits of the last position, KVCache holding the whole prompt, PrefixStats)."""
        ids = tokens.tolist()
        hashes = block_hashes(ids, self.block_size, salt)
        n_layers = len(self.model.blocks)
        cache = KVCache(n_layers)
        hit = 0
        if use_cache:
            while hit < len(hashes) and hashes[hit] in self.blocks:
                hit += 1
        # always leave at least the last token to compute: the logits come from running it
        hit = min(hit, (len(ids) - 1) // self.block_size)
        reused = hit * self.block_size
        if hit:
            cache.k = [
                torch.cat([self.blocks[h][0][layer] for h in hashes[:hit]], 2)
                for layer in range(n_layers)
            ]
            cache.v = [
                torch.cat([self.blocks[h][1][layer] for h in hashes[:hit]], 2)
                for layer in range(n_layers)
            ]
        logits = self.model(tokens[None, reused:], cache=cache)[0, -1]
        bs = self.block_size
        for i, h in enumerate(hashes):
            if h not in self.blocks:
                lo, hi = i * bs, (i + 1) * bs
                self.blocks[h] = (
                    [cache.k[layer][:, :, lo:hi].clone() for layer in range(n_layers)],
                    [cache.v[layer][:, :, lo:hi].clone() for layer in range(n_layers)],
                )
        return logits, cache, PrefixStats(reused, len(ids) - reused)


# ---------------------------------------------------------------------------------------------
# Little's law


def queue_simulation(rate: float, mean_service: float, servers: int, n: int, seed: int = 0):
    """First-come first-served queue with Poisson arrivals and exponential service. Returns
    (arrival rate seen, mean time in the system W, time-average number in the system L), the last
    one by integrating the number of requests over time from the arrival and departure events."""
    rng = random.Random(seed)
    arrivals, t = [], 0.0
    for _ in range(n):
        t += rng.expovariate(rate)
        arrivals.append(t)
    free_at = [0.0] * servers
    heapq.heapify(free_at)
    sojourn, departures = [], []
    for a in arrivals:
        start = max(a, heapq.heappop(free_at))
        end = start + rng.expovariate(1 / mean_service)
        heapq.heappush(free_at, end)
        sojourn.append(end - a)
        departures.append(end)
    events = sorted([(a, 1) for a in arrivals] + [(d, -1) for d in departures])
    area, count, last = 0.0, 0, 0.0
    for when, delta in events:
        area += count * (when - last)
        count, last = count + delta, when
    return n / arrivals[-1], sum(sojourn) / n, area / events[-1][0]


def machines_needed(
    rate: float,
    tokens: int,
    min_speed: float,
    speed_at: dict[int, float],
    ttft_at: dict[int, float],
):
    """The page's capacity estimate. The most requests one machine can serve while each still gets
    ``min_speed`` tokens/s, the end-to-end time W at that load, the requests in flight L = rate x W,
    and the machines for it."""
    ok = [c for c, s in speed_at.items() if s >= min_speed]
    per_machine = max(ok)
    w = ttft_at[per_machine] + (tokens - 1) / speed_at[per_machine]
    in_flight = rate * w
    return per_machine, w, in_flight, math.ceil(in_flight / per_machine)
