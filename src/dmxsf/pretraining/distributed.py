"""Parallel training that really runs: several processes on one CPU, talking through gloo.

Four things from the course page, each built small enough to read:

* how data-parallel gradients are averaged, and when that is wrong (``ddp_weighting``);
* ZeRO: the optimizer state (stage 1) or everything (stage 3) split over the ranks, next to plain
  data parallelism, with the memory and the communication measured (``train_sharded``);
* Megatron-style tensor parallelism of a Transformer block (``tensor_parallel_check``);
* the process launcher they all share (``launch``).

The numbers that matter (equal weights, bytes per rank, collectives per layer) do not depend on the
hardware; the speed does, and nothing here tries to be fast.

Course page: https://dmxsf.com/wiki/pretraining/distributed-training/
"""

import copy
import os
import socket
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from dmxsf.architecture.attention import repeat_kv
from dmxsf.architecture.gpt import GPT, Block, GPTConfig
from dmxsf.architecture.position import apply_rope, rope_cos_sin
from dmxsf.data import ByteCorpus, get_batch

# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank: int, world: int, port: int, fn, args: tuple, outdir: str) -> None:
    torch.set_num_threads(1)  # `world` processes share the cores of one machine
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=300),
    )
    try:
        torch.save(fn(rank, world, *args), os.path.join(outdir, f"{rank}.pt"))
    finally:
        dist.destroy_process_group()


def launch(fn, world: int, *args) -> list:
    """Run ``fn(rank, world, *args)`` in ``world`` processes; return their results, indexed by rank.

    ``fn`` has to live in an importable module (the processes are started with ``spawn``), and the
    script that calls this has to keep its own work under ``if __name__ == "__main__"``.
    """
    with tempfile.TemporaryDirectory() as out:
        mp.spawn(_entry, args=(world, free_port(), fn, args, out), nprocs=world, join=True)
        return [torch.load(os.path.join(out, f"{r}.pt"), weights_only=True) for r in range(world)]


class Comm:
    """The three collectives of this file, counting what each rank would send in a ring implementation.

    An all-reduce of S elements costs 2 (n - 1) / n S, a reduce-scatter or an all-gather half of that
    (the page's numbers); ``sent`` adds up the elements and ``calls`` counts the operations.
    """

    def __init__(self, world: int):
        self.world = world
        self.sent = 0.0
        self.calls = {"all_reduce": 0, "reduce_scatter": 0, "all_gather": 0}

    def all_reduce(self, x: torch.Tensor) -> None:
        dist.all_reduce(x)
        self.sent += 2 * (self.world - 1) / self.world * x.numel()
        self.calls["all_reduce"] += 1

    def reduce_scatter(self, out: torch.Tensor, x: torch.Tensor) -> None:
        dist.reduce_scatter_tensor(out, x)
        self.sent += (self.world - 1) / self.world * x.numel()
        self.calls["reduce_scatter"] += 1

    def all_gather(self, out: torch.Tensor, x: torch.Tensor) -> None:
        dist.all_gather_into_tensor(out, x)
        self.sent += (self.world - 1) / self.world * out.numel()
        self.calls["all_gather"] += 1


# ---------------------------------------------------------------------------
# 1. How gradients are averaged
# ---------------------------------------------------------------------------

VOCAB, SEQ = 50, 16  # the page's toy language model


def weighting_model(kind: str) -> nn.Module:
    """'toy' is the page's embedding plus one linear layer; 'gpt' a real two-layer decoder."""
    torch.manual_seed(0)
    if kind == "toy":
        return nn.Sequential(nn.Embedding(VOCAB, 8), nn.Linear(8, VOCAB))
    cfg = GPTConfig(
        vocab_size=VOCAB,
        block_size=SEQ,
        n_layer=2,
        n_head=2,
        d_model=16,
        norm="rmsnorm",
        pos="rope",
        mlp="swiglu",
        bias=False,
    )
    return GPT(cfg)


def token_loss(model, ids: torch.Tensor, valid: torch.Tensor):
    """Sum of the losses over the valid next-token predictions, and how many there are."""
    logits = model(ids[:, :-1])
    ce = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="none")
    mask = valid[:, 1:].flatten()
    return (ce * mask).sum(), mask.sum()


def flat_grad(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def _weighting_worker(rank: int, world: int, kind: str, ids: torch.Tensor, valid: torch.Tensor):
    per = ids.shape[0] // world
    mine = slice(rank * per, (rank + 1) * per)
    out = {}

    model = weighting_model(
        kind
    )  # A: every rank averages its own tokens, then the ranks are averaged
    total, count = token_loss(model, ids[mine], valid[mine])
    (total / count).backward()
    for p in model.parameters():
        dist.all_reduce(p.grad)
        p.grad /= world
    out["A: mean of the ranks' means"] = flat_grad(model).clone()

    model.zero_grad()  # B: sum the gradients, divide by the valid tokens of all ranks together
    total, count = token_loss(model, ids[mine], valid[mine])
    dist.all_reduce(count)
    (total / count).backward()
    for p in model.parameters():
        dist.all_reduce(p.grad)
    out["B: divide by all valid tokens"] = flat_grad(model).clone()

    ddp = DistributedDataParallel(weighting_model(kind))  # what PyTorch's DDP does by default
    total, count = token_loss(ddp, ids[mine], valid[mine])
    (total / count).backward()
    out["DDP as shipped"] = flat_grad(ddp.module).clone()

    ddp.module.zero_grad()  # ... and the recipe: DDP divides by world, so multiply the loss back
    total, count = token_loss(ddp, ids[mine], valid[mine])
    dist.all_reduce(count)
    (total / count * world).backward()
    out["DDP, loss x world / all tokens"] = flat_grad(ddp.module).clone()
    return out


def ddp_weighting(kind: str, lengths: list[int], world: int = 4, seed: int = 1) -> dict:
    """Relative gradient error of each way of averaging against 'one process, the whole batch,
    divided by the number of valid tokens in it' (the page's experiment).

    ``lengths`` are the valid lengths of the sequences, which are dealt out evenly to the ranks.
    """
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB, (len(lengths), SEQ))
    valid = (torch.arange(SEQ)[None] < torch.tensor(lengths)[:, None]).float()
    per = len(lengths) // world

    model = weighting_model(kind)
    total, count = token_loss(model, ids, valid)
    (total / count).backward()
    ref = flat_grad(model)

    model = weighting_model(kind)  # gradient accumulation: the same arithmetic, one after another
    for k in range(world):
        part = slice(k * per, (k + 1) * per)
        total, count = token_loss(model, ids[part], valid[part])
        (total / count / world).backward()
    acc = flat_grad(model)

    grads = launch(_weighting_worker, world, kind, ids, valid)[0]
    grads["gradient accumulation"] = acc
    return {
        "error": {k: ((g - ref).norm() / ref.norm()).item() for k, g in grads.items()},
        "tokens": [int(valid[k * per : (k + 1) * per, 1:].sum()) for k in range(world)],
    }


# ---------------------------------------------------------------------------
# 2. Data parallelism, ZeRO-1 and ZeRO-3, side by side
# ---------------------------------------------------------------------------

SHARD_CFG = GPTConfig(
    vocab_size=256,
    block_size=64,
    n_layer=2,
    n_head=4,
    d_model=64,
    norm="rmsnorm",
    pos="rope",
    mlp="swiglu",
    bias=False,
    tie_embeddings=False,
)
LR = 3e-3


def shard_model() -> GPT:
    """A 139,584-parameter Llama-style decoder in float64, so rounding does not hide mistakes."""
    torch.manual_seed(0)
    return GPT(SHARD_CFG).double()


def _batch(corpus: ByteCorpus, step: int, rows: int, seed: int):
    return get_batch(corpus.train, step, rows, SHARD_CFG.block_size, seed)


def _loss(model, x, y) -> torch.Tensor:
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())


def train_reference(steps: int, rows: int, seed: int = 0) -> torch.Tensor:
    """One process, the whole batch of ``rows`` sequences, AdamW: the answer all ranks must reproduce."""
    corpus, model = ByteCorpus.shakespeare(), shard_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    for step in range(steps):
        x, y = _batch(corpus, step, rows, seed)
        opt.zero_grad()
        _loss(model, x, y).backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


class _Gather(torch.autograd.Function):
    """All-gather the shards in the forward pass; reduce-scatter (and average) the gradient back."""

    @staticmethod
    def forward(ctx, shard: torch.Tensor, comm: Comm):
        ctx.comm = comm
        full = shard.new_empty(shard.numel() * comm.world)
        comm.all_gather(full, shard.detach().contiguous())
        return full

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        comm = ctx.comm
        out = grad_full.new_empty(grad_full.numel() // comm.world)
        comm.reduce_scatter(out, grad_full.contiguous())
        return out / comm.world, None


class Unit(nn.Module):
    """A group of parameters kept as one flat shard per rank and gathered only while it is used.

    This is what FSDP calls a unit. The forward pass runs under activation checkpointing, so the
    gathered weights are dropped right after use and gathered a second time when the backward pass
    recomputes the unit: one all-gather going forward, one coming back, and a reduce-scatter of the
    gradients, which is the 3 psi of the page.
    """

    def __init__(self, module: nn.Module, comm: Comm, rank: int):
        super().__init__()
        self.module, self.comm = module, comm
        self.names = [n for n, _ in module.named_parameters()]
        self.shapes = [p.shape for p in module.parameters()]
        flat = torch.cat([p.detach().flatten() for p in module.parameters()])
        self.numel = flat.numel()
        self.chunk = -(-self.numel // comm.world)
        flat = F.pad(flat, (0, self.chunk * comm.world - self.numel))
        self.shard = nn.Parameter(flat[rank * self.chunk : (rank + 1) * self.chunk].clone())
        for p in module.parameters():  # the full copy is gone; only the shard stays on this rank
            p.data = p.data.new_empty(0)

    def forward(self, *args, **kwargs):
        def run(shard, *a, **kw):
            full = _Gather.apply(shard, self.comm)[: self.numel]
            pieces = full.split_with_sizes([s.numel() for s in self.shapes])
            parts = [t.view(s) for t, s in zip(pieces, self.shapes, strict=True)]
            return functional_call(self.module, dict(zip(self.names, parts, strict=True)), a, kw)

        return checkpoint(run, self.shard, *args, use_reentrant=False, **kwargs)


def _values(opt, params, grads) -> dict[str, int]:
    """Numbers held on this rank, by kind: weights, gradients, optimizer state."""
    return {
        "weights": sum(p.numel() for p in params),
        "gradients": sum(g.numel() for g in grads),
        "optimizer": sum(
            v.numel()
            for s in opt.state.values()
            for v in s.values()
            if torch.is_tensor(v) and v.dim()
        ),
    }


def _sharded_worker(rank: int, world: int, stage: int, steps: int, rows: int, seed: int):
    """Train for ``steps`` steps as rank ``rank`` of ``world``.

    stage 0: data parallelism (all-reduce the gradients, every rank updates every weight);
    stage 1: ZeRO-1 (reduce-scatter the gradients, each rank updates its slice, all-gather the weights);
    stage 3: ZeRO-3 (the weights live as shards, gathered unit by unit).
    """
    corpus, model, comm = ByteCorpus.shakespeare(), shard_model(), Comm(world)
    n_params = sum(p.numel() for p in model.parameters())
    mine = slice(rank * rows // world, (rank + 1) * rows // world)
    if stage == 3:
        units = nn.ModuleList(
            [Unit(model.tok_emb, comm, rank)]
            + [Unit(b, comm, rank) for b in model.blocks]
            + [Unit(nn.Sequential(model.final_norm, model.lm_head), comm, rank)]
        )
        params = [u.shard for u in units]
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        positions = torch.arange(SHARD_CFG.block_size)
        cos_sin = rope_cos_sin(positions, model.inv_freq, model.rope_attention_factor)
    else:
        params = list(model.parameters())
        n, sizes = n_params, [p.numel() for p in params]
        chunk = -(-n // world)
        if stage == 1:
            vec = F.pad(torch.cat([p.detach().flatten() for p in params]), (0, chunk * world - n))
            shard = nn.Parameter(vec[rank * chunk : (rank + 1) * chunk].clone())
            opt = torch.optim.AdamW([shard], lr=LR, weight_decay=0.0)
        else:
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

    losses, held = [], {}
    for step in range(steps):
        x, y = _batch(corpus, step, rows, seed)
        x, y = x[mine], y[mine]
        opt.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)  # stage 1's optimizer only sees its own slice
        if stage == 3:
            h = units[0](x)
            rope = tuple(c[: x.shape[1]].to(h.dtype) for c in cos_sin)
            for unit in units[1:-1]:
                h = unit(h, rope=rope)
            loss = F.cross_entropy(units[-1](h).flatten(0, 1), y.flatten())
            loss.backward()  # gathers the units again, one at a time, and reduce-scatters the gradients
        else:
            loss = _loss(model, x, y)
            loss.backward()
            grad = F.pad(torch.cat([p.grad.flatten() for p in params]), (0, chunk * world - n))
            if stage == 0:
                comm.all_reduce(grad)
                grad /= world
                with torch.no_grad():
                    for p, g in zip(params, grad[:n].split_with_sizes(sizes), strict=True):
                        p.grad.copy_(g.view_as(p))
            else:
                mean = torch.empty(chunk, dtype=grad.dtype)
                comm.reduce_scatter(mean, grad)
                shard.grad = mean / world
        opt.step()
        if stage == 1:  # every rank updated its own slice; now everyone gets all of them
            full = torch.empty(chunk * world, dtype=grad.dtype)
            comm.all_gather(full, shard.detach())
            with torch.no_grad():
                for p, w in zip(params, full[:n].split_with_sizes(sizes), strict=True):
                    p.copy_(w.view_as(p))
        held = _values(opt, params, [p.grad for p in params])
        losses.append(loss.item())

    if stage == 3:  # collect the full weights on every rank, only to compare them afterwards
        pieces = []
        for u in units:
            full = torch.empty(u.chunk * world, dtype=u.shard.dtype)
            dist.all_gather_into_tensor(full, u.shard.detach().contiguous())
            pieces.append(full[: u.numel])
        flat = torch.cat(pieces)
    else:
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    return {
        "flat": flat,
        "held": held,
        "sent": comm.sent / steps,
        "calls": {k: v // steps for k, v in comm.calls.items()},
        "loss": losses,
        "n_params": n_params,
    }


def train_sharded(stage: int, world: int = 4, steps: int = 10, rows: int = 16, seed: int = 0):
    """Run ``_sharded_worker`` in ``world`` processes; returns rank 0's report (see that function)."""
    return launch(_sharded_worker, world, stage, steps, rows, seed)


# ---------------------------------------------------------------------------
# 3. Tensor parallelism
# ---------------------------------------------------------------------------

TP_CFG = GPTConfig(
    vocab_size=64,
    block_size=32,
    n_layer=2,
    n_head=8,
    n_kv_head=4,
    d_model=64,
    norm="layernorm",
    pos="rope",
    mlp="gelu",
    bias=False,
)
ALL_REDUCES = {"forward": 0, "backward": 0}  # counted in the two operators below


class _CopyToTP(torch.autograd.Function):
    """Megatron's f: the identity going forward, an all-reduce of the gradient coming back."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad)
        ALL_REDUCES["backward"] += 1
        return grad


class _ReduceFromTP(torch.autograd.Function):
    """Megatron's g: an all-reduce going forward, the identity coming back."""

    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        dist.all_reduce(x)
        ALL_REDUCES["forward"] += 1
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


def tp_slice(weight: torch.Tensor, rank: int, world: int, dim: int) -> torch.Tensor:
    """This rank's part of a weight: dim 0 is a column split of x W^T, dim 1 a row split."""
    k = weight.shape[dim] // world
    return weight.narrow(dim, rank * k, k)


class TPBlock(nn.Module):
    """A pre-norm Transformer block (attention, GELU MLP) split over the ranks the Megatron way.

    q, k, v and the first MLP matrix are split by output columns, so every rank owns whole attention
    heads and a slice of the hidden units; the output projection and the second MLP matrix are split
    by input rows, so each rank produces a partial sum and one all-reduce adds them up. Per block:
    two all-reduces forward (``_ReduceFromTP``) and two backward (``_CopyToTP``).
    """

    def __init__(self, block: Block, rank: int, world: int):
        super().__init__()
        a = block.attn
        if a.n_heads % world or a.n_kv_heads % world:
            raise ValueError("the number of heads and of KV heads must be divisible by the ranks")
        self.heads, self.kv_heads, self.head_dim = (
            a.n_heads // world,
            a.n_kv_heads // world,
            a.head_dim,
        )
        self.norm1, self.norm2 = copy.deepcopy(block.norm1), copy.deepcopy(block.norm2)
        pieces = {
            "wq": (a.q_proj.weight, 0),
            "wk": (a.k_proj.weight, 0),
            "wv": (a.v_proj.weight, 0),
            "wo": (a.o_proj.weight, 1),
            "fc": (block.mlp.fc.weight, 0),
            "proj": (block.mlp.proj.weight, 1),
        }
        for name, (w, dim) in pieces.items():
            setattr(self, name, nn.Parameter(tp_slice(w.detach(), rank, world, dim).clone()))

    def forward(self, x: torch.Tensor, rope) -> torch.Tensor:
        b, t, _ = x.shape
        h = _CopyToTP.apply(self.norm1(x))
        q = F.linear(h, self.wq).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = F.linear(h, self.wk).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = F.linear(h, self.wv).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        rep = self.heads // self.kv_heads
        out = F.scaled_dot_product_attention(
            q, repeat_kv(k, rep), repeat_kv(v, rep), is_causal=True
        )
        out = out.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        x = x + _ReduceFromTP.apply(F.linear(out, self.wo))
        h = _CopyToTP.apply(self.norm2(x))
        return x + _ReduceFromTP.apply(F.linear(F.gelu(F.linear(h, self.fc)), self.proj))


def _tp_worker(rank: int, world: int, seed: int):
    torch.manual_seed(seed)
    blocks = [Block(TP_CFG).double() for _ in range(TP_CFG.n_layer)]
    x = torch.randn(2, 16, TP_CFG.d_model, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(
        2, 16, TP_CFG.d_model, dtype=torch.float64
    )  # makes the loss a generic scalar
    d = TP_CFG.dim_per_head
    inv_freq = TP_CFG.rope_base ** (-torch.arange(0, d, 2).double() / d)
    rope = tuple(c.double() for c in rope_cos_sin(torch.arange(16), inv_freq))

    h = x  # the unsplit blocks: the answer
    for block in blocks:
        h = block(h, rope=rope)
    (h * weight).sum().backward()
    expect_out, expect_x = h.detach(), x.grad.clone()
    expect = [{n: p.grad.clone() for n, p in b.named_parameters()} for b in blocks]

    x2 = x.detach().clone().requires_grad_()
    tp = [TPBlock(b, rank, world) for b in blocks]
    ALL_REDUCES.update(forward=0, backward=0)
    h = x2
    for block in tp:
        h = block(h, rope)
    forward_count = ALL_REDUCES["forward"]
    (h * weight).sum().backward()

    dims = {"wq": 0, "wk": 0, "wv": 0, "wo": 1, "fc": 0, "proj": 1}
    names = {
        "wq": "attn.q_proj.weight",
        "wk": "attn.k_proj.weight",
        "wv": "attn.v_proj.weight",
        "wo": "attn.o_proj.weight",
        "fc": "mlp.fc.weight",
        "proj": "mlp.proj.weight",
    }
    grad_err = 0.0
    for mine, ref in zip(tp, expect, strict=True):
        for short, full in names.items():
            want = tp_slice(ref[full], rank, world, dims[short])
            got = getattr(mine, short).grad
            grad_err = max(grad_err, (got - want).abs().max().item() / want.abs().max().item())
        for norm in ("norm1", "norm2"):  # the replicated parameters get the full gradient
            want, got = ref[f"{norm}.weight"], getattr(mine, norm).weight.grad
            grad_err = max(grad_err, (got - want).abs().max().item() / want.abs().max().item())
    return {
        "out": (h.detach() - expect_out).abs().max().item(),
        "x_grad": (x2.grad - expect_x).abs().max().item() / expect_x.abs().max().item(),
        "weight_grad": grad_err,
        "forward_all_reduces": forward_count,
        "backward_all_reduces": ALL_REDUCES["backward"],
        "matrix_params": sum(getattr(b, n).numel() for b in tp for n in dims),
        "full_matrix_params": sum(
            dict(b.named_parameters())[name].numel() for b in blocks for name in names.values()
        ),
    }


def tensor_parallel_check(world: int, seed: int = 0) -> list[dict]:
    """Run ``TP_CFG.n_layer`` tensor-parallel blocks on ``world`` ranks and compare them with the
    same blocks unsplit: outputs, input gradient, every weight gradient, and the collectives used."""
    return launch(_tp_worker, world, seed)
