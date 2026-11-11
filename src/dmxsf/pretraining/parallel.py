"""The arithmetic of parallel training: ring all-reduce, pipeline schedules, communication costs.

Everything here is plain Python and NumPy; the code that really runs in several processes lives
in ``distributed.py``. The formulas are the ones on the course page.

Course page: https://dmxsf.com/wiki/pretraining/distributed-training/
"""

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Ring all-reduce
# ---------------------------------------------------------------------------


def ring_all_reduce(vectors: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Ring all-reduce between len(vectors) simulated ranks.

    Returns every rank's result and the number of elements each rank sent. The vector is cut into n
    chunks. In n - 1 reduce-scatter steps each rank passes one chunk to its right neighbour, which adds
    it to its own copy; then n - 1 all-gather steps pass the finished chunks around once more.
    """
    n = len(vectors)
    chunks = [np.array_split(v.copy(), n) for v in vectors]
    sent = [0] * n
    for step in range(n - 1):  # reduce-scatter
        for r in range(n):
            c = (r - step) % n
            chunks[(r + 1) % n][c] += chunks[r][c]
            sent[r] += len(chunks[r][c])
    # rank r now holds the sum of chunk (r + 1) % n
    for step in range(n - 1):  # all-gather
        for r in range(n):
            c = (r + 1 - step) % n
            chunks[(r + 1) % n][c] = chunks[r][c].copy()
            sent[r] += len(chunks[r][c])
    return [np.concatenate(ch) for ch in chunks], sent


def ring_bytes(n: int, nbytes: float) -> float:
    """What each of n ranks sends in a ring all-reduce of nbytes: 2 (n - 1) / n of it."""
    return 2 * (n - 1) / n * nbytes


def reduce_scatter_bytes(n: int, nbytes: float) -> float:
    """Half of an all-reduce: each rank sends (n - 1) / n of the data. All-gather is the same."""
    return (n - 1) / n * nbytes


def zero3_bytes(n: int, nbytes: float) -> float:
    """ZeRO-3 / FSDP: all-gather the weights in the forward pass, again in the backward pass, and
    reduce-scatter the gradients: three half all-reduces, 1.5 times a data-parallel step."""
    return 3 * reduce_scatter_bytes(n, nbytes)


# ---------------------------------------------------------------------------
# Pipeline schedules
# ---------------------------------------------------------------------------


@dataclass
class Timeline:
    """The result of simulating one training step of a pipeline."""

    total: float  # wall time of the step
    idle: float  # fraction of the stages' time spent waiting
    peak: list[int]  # per stage: the most micro-batches whose activations are alive at once
    ops: list[tuple[int, str, int, float, float]]  # (stage, "F" or "B", micro-batch, start, end)


def _order(p: int, m: int, schedule: str) -> list[list[tuple[str, int]]]:
    """The sequence of forward (F) and backward (B) passes every stage executes."""
    orders = []
    for s in range(p):
        if schedule == "gpipe":  # all forwards, then all backwards
            orders.append([("F", i) for i in range(m)] + [("B", i) for i in range(m)])
        elif schedule == "1f1b":  # fill the pipe, then alternate one forward with one backward
            warmup = min(p - s - 1, m)
            seq = [("F", i) for i in range(warmup)]
            fwd, bwd = warmup, 0
            while fwd < m:
                seq += [("F", fwd), ("B", bwd)]
                fwd, bwd = fwd + 1, bwd + 1
            seq += [("B", i) for i in range(bwd, m)]
            orders.append(seq)
        else:
            raise ValueError("schedule must be 'gpipe' or '1f1b'")
    return orders


def pipeline(
    p: int, m: int, schedule: str = "gpipe", t_fwd: float = 1.0, t_bwd: float = 2.0
) -> Timeline:
    """Simulate p stages and m micro-batches. A stage does one thing at a time; a forward pass needs
    the previous stage's forward of the same micro-batch, a backward pass the next stage's backward."""
    orders = _order(p, m, schedule)
    free = [0.0] * p
    done: dict[tuple[str, int, int], float] = {}
    pos = [0] * p
    ops = []
    left = sum(len(o) for o in orders)
    while left:
        progressed = False
        for s in range(p):
            while pos[s] < len(orders[s]):
                kind, i = orders[s][pos[s]]
                if kind == "F":
                    ready = 0.0 if s == 0 else done.get(("F", s - 1, i))
                else:
                    ready = done[("F", s, i)] if s == p - 1 else done.get(("B", s + 1, i))
                if ready is None:
                    break
                start = max(free[s], ready)
                end = start + (t_fwd if kind == "F" else t_bwd)
                free[s] = done[(kind, s, i)] = end
                ops.append((s, kind, i, start, end))
                pos[s] += 1
                left -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("the schedule deadlocks")
    total = max(free)
    idle = 1 - m * (t_fwd + t_bwd) / total
    peak = []
    for s in range(p):  # activations live from the start of F to the end of B
        events = []
        for stage, kind, _i, start, end in ops:
            if stage == s:
                events.append((start, 1) if kind == "F" else (end, -1))
        alive = top = 0
        for _t, delta in sorted(events, key=lambda e: (e[0], e[1])):  # frees before allocations
            alive += delta
            top = max(top, alive)
        peak.append(top)
    return Timeline(total, idle, peak, ops)


def bubble_ratio(p: int, m: int, v: int = 1) -> float:
    """Megatron-LM's bubble time as a fraction of the useful compute time: (p - 1) / (v m), where v is
    the number of model chunks each device holds in the interleaved schedule."""
    return (p - 1) / (v * m)


def idle_fraction(p: int, m: int, v: int = 1) -> float:
    """The same bubble as a fraction of the whole step: (p - 1) / (v m + p - 1)."""
    r = bubble_ratio(p, m, v)
    return r / (1 + r)


def min_microbatches(p: int, idle: float) -> int:
    """The fewest micro-batches that keep the GPipe idle fraction (p - 1) / (m + p - 1) at or below idle."""
    return math.ceil((p - 1) * (1 - idle) / idle - 1e-9)  # the guard absorbs float error


# ---------------------------------------------------------------------------
# What communication costs (the page's worked example; throughput numbers are illustrative)
# ---------------------------------------------------------------------------


def tensor_parallel_bytes(t: int, layers: int, act_bytes: float) -> float:
    """Per micro-batch and rank: every layer does 2 all-reduces in the forward and 2 in the backward
    pass, each of one activation tensor (b x s x h)."""
    return layers * 4 * ring_bytes(t, act_bytes)


def pipeline_boundary_bytes(act_bytes: float) -> float:
    """One activation forward and its gradient backward cross each stage boundary, per micro-batch."""
    return 2 * act_bytes


def scaling_row(
    gpus: int, global_tokens: float, psi: float, comm_s: float, peak: float, mfu: float
) -> tuple[float, float, float]:
    """Fixed global batch, more GPUs: (compute seconds per step, exposed communication, efficiency).

    Only the backward pass (about 2/3 of the compute) can hide the gradient all-reduce, and the model
    assumes it hides it perfectly and that the step has no latency terms. So this is an upper bound.
    """
    compute = 6 * psi * (global_tokens / gpus) / (peak * mfu)
    exposed = max(0.0, comm_s - compute * 2 / 3)
    return compute, exposed, compute / (compute + exposed)


def tokens_to_hide(comm_s: float, psi: float, peak: float, mfu: float) -> float:
    """Tokens per GPU and step for which the backward pass (4 psi FLOPs per token) lasts as long as
    the gradient synchronisation."""
    return comm_s / (4 * psi / (peak * mfu))
