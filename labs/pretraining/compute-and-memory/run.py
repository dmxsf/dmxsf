"""Lab: what a training run costs in FLOPs, time and bytes.

Course page: https://dmxsf.com/wiki/pretraining/compute-and-memory/

Every formula on the page is checked against something that is measured: PyTorch's FLOP counter,
the tensors autograd saves for the backward pass, and the tensors an optimizer allocates.

    uv run python labs/pretraining/compute-and-memory/run.py [--quick]
"""

import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.flop_counter import FlopCounterMode

from dmxsf import lab
from dmxsf.architecture.configs import BY_NAME, PUBLIC_MODELS
from dmxsf.architecture.gpt import GPT, GPTConfig
from dmxsf.architecture.layers import SwiGLU
from dmxsf.pretraining import accounting as acc
from dmxsf.pretraining.train import TrainConfig, make_optimizer

args, report = lab.start(
    "Training compute and memory: 6ND, MFU, model state and activations",
    wiki=["pretraining/compute-and-memory"],
)
torch.manual_seed(args.seed)

A100, H100, TPU_V4 = 312e12, 989e12, 275e12  # dense BF16 peaks, FLOP/s
sci = lambda x: f"{x:.3g}"  # noqa: E731


def meta_model(cfg: GPTConfig) -> GPT:
    """The architecture without the memory: shapes and parameter counts of a model of any size."""
    with torch.device("meta"):
        return GPT(cfg)


def step_flops(model: GPT, ids: torch.Tensor) -> tuple[int, int]:
    """FLOPs of the forward pass and of the backward pass (recomputation, if any, is in backward)."""
    with FlopCounterMode(display=False) as counter:
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1))
        forward = counter.get_total_flops()
        loss.backward()
    return forward, counter.get_total_flops() - forward


# ---------------------------------------------------------------------------
report.section("1. A linear layer: 2 FLOPs per parameter forward, 4 backward")
d_in, d_out, tokens = 512, 1024, 256


def linear_flops(input_needs_grad: bool) -> tuple[int, int]:
    layer = nn.Linear(d_in, d_out, bias=False)
    x = torch.randn(tokens, d_in, requires_grad=input_needs_grad)
    with FlopCounterMode(display=False) as counter:
        y = layer(x)
        forward = counter.get_total_flops()
        y.sum().backward()
    return forward, counter.get_total_flops() - forward


forward, backward = linear_flops(True)
forward_const, backward_const = linear_flops(False)
report.say(
    f"  Y = X W with X of {tokens} x {d_in} and W of {d_in} x {d_out} ({d_in * d_out:,} weights)"
)
report.table(
    ["input", "forward", "backward", "backward / forward"],
    [
        ["activations (needs a gradient)", f"{forward:,}", f"{backward:,}", backward / forward],
        [
            "constant data (no dX needed)",
            f"{forward_const:,}",
            f"{backward_const:,}",
            backward_const / forward_const,
        ],
    ],
)
report.check(
    "forward is 2 * tokens * d_in * d_out", forward == 2 * tokens * d_in * d_out, f"{forward:,}"
)
report.check(
    "backward is two matrix products, dX and dW: twice the forward", backward == 2 * forward
)
report.check(
    "without dX (a constant input) backward is only dW: the 6N rule needs a trainable path below",
    backward_const == forward_const,
)

# ---------------------------------------------------------------------------
report.section("2. Counting a small Llama: the forward pass is 2NT plus attention")
cfg = GPTConfig(
    vocab_size=1000,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_kv_head=2,
    d_model=256,
    d_ff=688,
    norm="rmsnorm",
    pos="rope",
    mlp="swiglu",
    bias=False,
    tie_embeddings=False,
)
model = GPT(cfg)
batch, seq = 2, 128
ids = torch.randint(0, cfg.vocab_size, (batch, seq))
n_matmul, n_total, T = acc.matmul_params(model), model.num_params(), batch * seq
attention = cfg.n_layer * 4 * cfg.d_model * seq * seq * batch  # QK^T and AV, the full S x S matrix
linear = 2 * n_matmul * T
PAGE_FORMULA = 1_750_073_344  # the number the page prints for 2 N T + attention
report.say(
    f"  {cfg.n_layer} layers, hidden {cfg.d_model}, {n_total:,} parameters of which N = {n_matmul:,} "
    f"take part in matrix products; B x S = {batch} x {seq} = {T} tokens"
)
rows, measured = [], {}
for impl in ("manual", "sdpa"):
    model.attn_impl = impl
    model.zero_grad()
    fwd, bwd = step_flops(model, ids)
    measured[impl] = (fwd, bwd)
    rows.append(
        [impl, f"{fwd:,}", f"{linear + attention:,}", bwd / fwd, (fwd + bwd) / (6 * n_matmul * T)]
    )
report.table(["attention", "forward", "2NT + attention", "backward / forward", "total / 6NT"], rows)
report.check(
    "forward = 2NT + attention, to the FLOP, whichever way attention is computed",
    all(fwd == linear + attention for fwd, _ in measured.values()),
)
report.check(
    "it is the number on the course page (HF's version adds 8,192 for its rotary matmul)",
    measured["manual"][0] == PAGE_FORMULA,
    f"{PAGE_FORMULA:,}",
)
report.check(
    "with the attention matrix spelled out, backward is exactly twice forward",
    measured["manual"][1] == 2 * measured["manual"][0],
)
predicted = 2 * linear + 2.5 * attention  # a fused kernel recomputes scores: 2.5x on attention
report.check(
    "a fused (flash-style) kernel redoes the scores in backward: 2.5x on the attention part only",
    measured["sdpa"][1] == predicted,
    f"{measured['sdpa'][1]:,} measured, {predicted:,.0f} predicted",
)
ratio = sum(measured["manual"]) / (6 * n_matmul * T)
report.check(
    "total / 6NT = 1 + attention / 2NT: the whole excess over 6NT is the attention term",
    abs(ratio - (1 + attention / linear)) < 1e-12,
    f"{ratio:.3f} = 1 + {attention / linear:.3f}",
)
overshoot = 6 * n_total * T / sum(measured["manual"])
report.say(
    f"  6 * (all {n_total:,} parameters) * T would be {overshoot:.3f}x the measured FLOPs: "
    "the embedding table costs memory, not FLOPs"
)

# ---------------------------------------------------------------------------
report.section("3. Which parameters are N")
report.say("  built on the meta device: shapes only, so a 70B model costs nothing")
rows, n_of = [], {}
for entry in PUBLIC_MODELS:
    shape = meta_model(entry.config)
    n_of[entry.name] = acc.matmul_params(shape)
    total = shape.num_params()
    rows.append(
        [
            entry.name,
            f"{total:,}",
            f"{n_of[entry.name]:,}",
            f"{n_of[entry.name] / total:.1%}",
            f"{total / n_of[entry.name] - 1:+.1%}",
        ]
    )
report.table(["model", "parameters", "N (matmul)", "N / total", "6*total over-counts by"], rows)
l2, l3 = BY_NAME["Llama 2 7B"], BY_NAME["Llama 3 8B"]
report.check(
    "Llama 2 7B: N = 6.61B, 98.1% of the total",
    round(n_of[l2.name] / 1e9, 2) == 6.61 and f"{n_of[l2.name] / l2.params:.1%}" == "98.1%",
)
report.check(
    "Llama 3 8B: N = 7.50B, 93.5% of the total; 6 * total over-counts by about 7%",
    round(n_of[l3.name] / 1e9, 2) == 7.50
    and f"{n_of[l3.name] / l3.params:.1%}" == "93.5%"
    and round((l3.params / n_of[l3.name] - 1) * 100) == 7,
)
l3_head = meta_model(l3.config).lm_head.weight.numel()
report.check(
    "the quiz's wrong answer is a real number: the block weights alone are 6.98B",
    round((n_of[l3.name] - l3_head) / 1e9, 2) == 6.98,
    "the output matrix (0.53B) is a matrix product and does count",
)

# ---------------------------------------------------------------------------
report.section("4. The attention term: 12 L H d_h s per token")
llama_405b = GPTConfig(
    vocab_size=128256,
    block_size=8192,
    n_layer=126,
    n_head=128,
    n_kv_head=8,
    d_model=16384,
    d_ff=53248,
    norm="rmsnorm",
    pos="rope",
    mlp="swiglu",
    bias=False,
    tie_embeddings=False,
)
shapes = {
    "Llama 2 7B": (l2.config.n_layer, l2.config.n_head, l2.config.dim_per_head, n_of[l2.name]),
    "Llama 3 405B": (126, 128, 128, acc.matmul_params(meta_model(llama_405b))),
}
for name, (layers, heads, d_head, n) in shapes.items():
    report.say(f"  {name}: L={layers}, H={heads}, d_h={d_head}, N={n / 1e9:.2f}B")
PAGE_SHARES = {  # the page's table: full matrix (lower triangle only)
    "Llama 2 7B": {
        4096: "14.0 (7.5)",
        8192: "24.5 (14.0)",
        32768: "56.5 (39.4)",
        131072: "83.9 (72.2)",
    },
    "Llama 3 405B": {
        4096: "4.0 (2.1)",
        8192: "7.7 (4.0)",
        32768: "25.1 (14.4)",
        131072: "57.3 (40.1)",
    },
}
rows, same = [], True
for s in (4096, 8192, 32768, 131072):
    row = [f"{s:,}"]
    for name, (layers, heads, d_head, n) in shapes.items():
        att = acc.attention_flops_per_token(layers, heads, d_head, s)
        full, causal = att / (6 * n + att), att / 2 / (6 * n + att / 2)
        cell = f"{full:.1%} ({causal:.1%})".replace("%", "")
        # a length the page does not list is shown, not compared
        same &= cell == PAGE_SHARES[name].get(s, cell)
        row.append(cell + " %")
    rows.append(row)
report.table(["sequence length s", "Llama 2 7B shape", "Llama 3 405B shape"], rows)
report.say(
    "  (share of the training FLOPs per token that is attention; in brackets: lower triangle only)"
)
report.check("all eight shares are the ones on the page", same)
cross = {
    name: n / (2 * layers * heads * d_head) for name, (layers, heads, d_head, n) in shapes.items()
}
report.say(
    f"  attention FLOPs = linear FLOPs at s = {cross['Llama 2 7B']:,.0f} (7B shape) and "
    f"{cross['Llama 3 405B']:,.0f} (405B shape); with a causal mask, twice that"
)
report.check(
    "the crossover lengths are about 25,200 and 97,800",
    round(cross["Llama 2 7B"], -2) == 25_200 and round(cross["Llama 3 405B"], -2) == 97_800,
)
layers, heads, d_head, n = shapes["Llama 2 7B"]
before = 6 * n + acc.attention_flops_per_token(layers, heads, d_head, 4096)
after = 6 * n + acc.attention_flops_per_token(layers, heads, d_head, 8192)
report.check(
    "exercise 4: doubling the sequence to 8192 raises the per-token FLOPs by about 14%",
    f"{before:.3g} -> {after:.3g}" == "4.61e+10 -> 5.25e+10"
    and round(after / before - 1, 2) == 0.14,
    f"{before:.3g} -> {after:.3g}",
)

# ---------------------------------------------------------------------------
report.section("5. Recomputation, and the whole-network formula of the Megatron-LM paper")
B, S, L, h, V, H = 2, 128, 4, 128, 1000, 4  # a GPT-2-style network: MHA, d_ff = 4h
gpt = GPT(GPTConfig(vocab_size=V, block_size=S, n_layer=L, n_head=H, d_model=h, bias=False))
gpt.attn_impl = "manual"
tokens_ids = torch.randint(0, V, (B, S))
layer_forward = 24 * B * S * h**2 + 4 * B * S**2 * h  # per layer, forward
head = 6 * B * S * h * V  # output matrix: forward 2, backward 4, never recomputed
down_projection = L * 2 * B * S * (4 * h) * h  # the last matrix product of every block
results = {}
for mode in (None, "reentrant", "non-reentrant"):
    gpt.recompute = mode
    gpt.zero_grad()
    results[mode] = sum(step_flops(gpt, tokens_ids))
no_recompute = 3 * L * layer_forward + head
megatron = acc.megatron_flops(B, S, L, h, V)
full = 4 * L * layer_forward + head
report.say(
    f"  L={L}, h={h}, s={S}, B={B}, V={V}: forward of one layer = 24Bsh^2 + 4Bs^2h = {layer_forward:,}"
)
report.table(
    ["blocks are", "measured FLOPs", "formula", "measured / no recompute"],
    [
        ["stored", f"{results[None]:,}", f"{no_recompute:,}", 1.0],
        [
            "recomputed (reentrant)",
            f"{results['reentrant']:,}",
            f"{full:,}",
            results["reentrant"] / results[None],
        ],
        [
            "recomputed (non-reentrant)",
            f"{results['non-reentrant']:,}",
            f"{full - down_projection:,}",
            results["non-reentrant"] / results[None],
        ],
    ],
)
report.check(
    "no recomputation: 3 forward passes of the blocks (1 + 2) plus the head, exactly",
    results[None] == no_recompute,
)
report.check(
    "full recomputation is Megatron-LM's F = 96 B s l h^2 (1 + s/6h + V/16lh), exactly",
    results["reentrant"] == full and math.isclose(megatron, full, rel_tol=1e-12),
    f"{results['reentrant']:,}",
)
report.check(
    "recomputing costs one extra forward pass of the blocks: the blocks' 6 become 8, 4/3",
    results["reentrant"] - results[None] == L * layer_forward,
    f"{L * layer_forward:,} FLOPs",
)
report.check(
    "PyTorch's non-reentrant checkpoint stops early and skips each block's last matmul",
    results["non-reentrant"] == full - down_projection,
    f"{down_projection:,} fewer than the paper's formula, which assumes the whole block is redone",
)

# ---------------------------------------------------------------------------
report.section("6. Time: MFU and GPU hours")
mt = acc.mfu(65.43e3, 6 * 530e9, 2240, A100)
palm = acc.mfu(238.3e3, 6 * 540e9, 6144, TPU_V4)
report.say(
    f"  Megatron-Turing NLG 530B, 2240 A100: MFU {mt:.3f}.   PaLM 540B, 6144 TPU v4: MFU {palm:.3f}"
)
report.check(
    "the two PaLM-appendix examples give 0.298 and 0.457", f"{mt:.3f} {palm:.3f}" == "0.298 0.457"
)

report.say("\n  the training-budget widget, reproduced (H100, MFU 40%, nominal N and D):")
budget = {}
for name, n, d, gpus, peak in [
    ("Llama 3 405B", 405e9, 15.6e12, 16384, H100),
    ("DeepSeek-V3 (N = 37B)", 37e9, 14.8e12, 1, H100),
    ("Llama 2 7B (A100)", 7e9, 2.0e12, 1, A100),
]:
    flops = acc.train_flops(n, d)
    hours = acc.gpu_hours(flops, peak, 0.4)
    per_gpu = peak * 0.4 / (6 * n)
    budget[name] = (flops, hours, per_gpu)
    days = f"{hours / gpus / 24:.1f}" if gpus > 1 else "-"
    report.say(
        f"  {name:26s} FLOPs {flops:.3g}  GPU hours {hours:,.0f}  days on {gpus:,} GPUs {days}"
        f"  tokens/s per GPU {per_gpu:,.0f}"
    )
llama3, v3, l2b = (
    budget["Llama 3 405B"],
    budget["DeepSeek-V3 (N = 37B)"],
    budget["Llama 2 7B (A100)"],
)
report.check(
    "Llama 3 405B: 3.79e25 FLOPs, 26.62M GPU hours, 67.7 days on 16,384 GPUs, 163 tokens/s per GPU",
    f"{llama3[0]:.3g} {llama3[1] / 1e6:.2f} {llama3[1] / 16384 / 24:.1f} {llama3[2]:.0f}"
    == "3.79e+25 26.62 67.7 163",
)
report.check(
    "DeepSeek-V3 needs 1/11.5 of that: its N is the 37B activated, not the 671B stored",
    f"{v3[0]:.3g} {llama3[0] / v3[0]:.1f}" == "3.29e+24 11.5",
)
report.check(
    "Llama 2 7B on A100 at 40%: 187 thousand GPU hours (paper: 184,320), 2,971 tokens/s (paper implies 3,014)",
    f"{l2b[1]:,.0f} {l2b[2]:,.0f} {2.0e12 / (184_320 * 3600):,.0f}" == "186,966 2,971 3,014",
)
with_recompute = acc.gpu_hours(8 * 405e9 * 15.6e12, H100, 0.4)
report.check(
    "with full recomputation the same run takes 35.49M GPU hours and its MFU on 6ND shows 30.0%",
    f"{with_recompute / 1e6:.2f} {0.4 * 6 / 8:.1%}" == "35.49 30.0%",
)

report.say(
    "\n  what a technical report's GPU hours imply (6ND over hours and the A100's 312 TFLOP/s):"
)
rows, got = [], []
for name, n, hours in [
    ("Llama 2 7B", 6.74e9, 184_320),
    ("Llama 2 13B", 13.02e9, 368_640),
    ("Llama 2 34B", 33.74e9, 1_038_336),
    ("Llama 2 70B", 68.98e9, 1_720_320),
]:
    flops = acc.train_flops(n, 2.0e12)
    per_gpu = flops / (hours * 3600)
    mfu_ = acc.implied_mfu(flops, hours, A100)
    got.append(f"{flops:.3g} {per_gpu / 1e12:.1f} {mfu_:.1%}")
    rows.append([name, sci(flops), f"{hours:,}", f"{per_gpu / 1e12:.1f}", f"{mfu_:.1%}"])
v3_flops = acc.train_flops(37e9, 14.8e12)
v3_per_gpu = v3_flops / (2_664_000 * 3600)
rows.append(
    [
        "DeepSeek-V3 (37B active)",
        sci(v3_flops),
        "2,664,000",
        f"{v3_per_gpu / 1e12:.1f}",
        "FP8: no MFU",
    ]
)
report.table(
    ["model", "6ND (FLOPs)", "reported GPU hours", "TFLOP/s per GPU", "of A100 peak"], rows
)
report.check(
    "the four Llama 2 rows are the page's: 39.1, 37.7, 34.7 and 42.8% of the A100 peak",
    got
    == [
        "8.09e+22 121.9 39.1%",
        "1.56e+23 117.7 37.7%",
        "4.05e+23 108.3 34.7%",
        "8.28e+23 133.7 42.8%",
    ],
)
v3_days = 180e3 / 2048 / 24
report.check(
    "DeepSeek-V3: 342.6 TFLOP/s per H800; its 180K GPU hours per trillion tokens are 3.7 days on 2048 GPUs",
    f"{v3_per_gpu / 1e12:.1f} {v3_days:.1f}" == "342.6 3.7",
)
peak_total = 1e4 * 3600 * A100
need = acc.train_flops(7e9, 2e12)
report.check(
    "10,000 A100 hours for 7B on 2T tokens cannot be true: 7 times short even at MFU 100%",
    need / peak_total > 7,
    f"{need:.2g} FLOPs needed, {peak_total:.2g} available at the peak",
)


report.say("\n  MFU on this machine (fp32, CPU), against the best matrix product it can do:")


def gemm_peak(n: int, reps: int) -> float:
    a, b = torch.randn(n, n), torch.randn(n, n)
    a @ b  # warm-up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return 2 * n**3 / best


def train_step_mfu(
    layers: int, d_model: int, heads: int, batch: int, seq: int, peak: float
) -> tuple[float, float]:
    shape = GPTConfig(
        vocab_size=256, block_size=seq, n_layer=layers, n_head=heads, d_model=d_model,
        norm="rmsnorm", pos="rope", mlp="swiglu", bias=False,
    )  # fmt: skip
    net = GPT(shape)
    opt = make_optimizer(net, TrainConfig(steps=10))
    x = torch.randint(0, 256, (batch, seq))

    def step() -> None:
        loss = F.cross_entropy(net(x).reshape(-1, 256), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    step()  # warm-up
    best = float("inf")
    for _ in range(2 if args.quick else 3):
        t0 = time.perf_counter()
        step()
        best = min(best, time.perf_counter() - t0)
    per_token = 6 * acc.matmul_params(net) + acc.attention_flops_per_token(
        layers, heads, d_model // heads, seq
    )
    return best, batch * seq / best * per_token / peak


peak = gemm_peak(1024 if args.quick else 2048, 3 if args.quick else 5)
report.say(
    f"  best 2048^2 matrix product: {peak / 1e9:,.0f} GFLOP/s on {torch.get_num_threads()} threads"
)
rows, mfus = [], []
for layers, d_model, heads, batch, seq in [
    (4, 256, 4, 1, 128),
    (4, 256, 4, 8, 256),
    (6, 512, 8, 8, 256),
]:
    seconds, value = train_step_mfu(layers, d_model, heads, batch, seq, peak)
    mfus.append(value)
    rows.append([f"{layers} x {d_model}", batch * seq, f"{seconds * 1000:.0f}", f"{value:.1%}"])
report.table(["model (layers x hidden)", "tokens per step", "ms per step", "MFU"], rows)
report.check(
    "a larger batch feeds the matrix units better: MFU at 2,048 tokens is more than twice that at 128",
    mfus[1] > 2 * mfus[0],
    f"{mfus[0]:.1%} -> {mfus[1]:.1%}",
)
report.check(
    "MFU is a fraction of the peak", all(0 < m < 1.2 for m in mfus), "numbers differ by machine"
)

# ---------------------------------------------------------------------------
report.section("7. Memory: the model state is 16 bytes per parameter")
lin = nn.Linear(4096, 4096, bias=False)
opt = torch.optim.AdamW(lin.parameters())
lin(torch.randn(8, 4096)).sum().backward()
opt.step()  # the first step creates AdamW's state


def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


n_lin = sum(p.numel() for p in lin.parameters())
weights = sum(nbytes(p) for p in lin.parameters())
grads = sum(nbytes(p.grad) for p in lin.parameters())
states = sum(
    nbytes(v) for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0
)
report.say(
    f"  fp32 Linear(4096, 4096): {n_lin:,} parameters; weights {weights / n_lin:.0f}, grads {grads / n_lin:.0f}, AdamW state {states / n_lin:.0f} bytes each"
)
report.check(
    "autocast-style layout: 4 + 4 + 8 = 16 bytes per parameter",
    (weights + grads + states) / n_lin == 16.0,
)

low = nn.Linear(1024, 1024, bias=False).bfloat16()
master = [p.detach().float().requires_grad_() for p in low.parameters()]
master_opt = torch.optim.AdamW(master)
low(torch.randn(8, 1024, dtype=torch.bfloat16)).sum().backward()
for m, p in zip(master, low.parameters(), strict=True):
    m.grad = p.grad.float()
master_opt.step()
for m in master:
    m.grad = None  # the fp32 copy of the gradient is a temporary
n_low = sum(p.numel() for p in low.parameters())
layout = {
    "bf16 weights": sum(nbytes(p) for p in low.parameters()),
    "bf16 gradients": sum(nbytes(p.grad) for p in low.parameters()),
    "fp32 master weights": sum(nbytes(m) for m in master),
    "fp32 Adam m and v": sum(
        nbytes(v)
        for s in master_opt.state.values()
        for v in s.values()
        if torch.is_tensor(v) and v.dim() > 0
    ),
}
report.say(
    "  mixed precision: " + ", ".join(f"{k} {v / n_low:.0f}" for k, v in layout.items()) + " bytes"
)
report.check(
    "BF16 weights and gradients, FP32 master copy and moments: 2 + 2 + 4 + 8 = 16",
    sum(layout.values()) / n_low == 16.0,
)

N7 = l2.params
rows, cells = [], []
for zero, label in [
    (0, "ZeRO-0 (nothing sharded)"),
    (1, "ZeRO-1 (optimizer state)"),
    (2, "ZeRO-2 (+ gradients)"),
    (3, "ZeRO-3 / FSDP (+ weights)"),
]:
    eight, sixty_four = (acc.model_state_bytes(N7, n, zero) / acc.GiB for n in (8, 64))
    cells += [f"{eight:.1f}", f"{sixty_four:.1f}"]
    rows.append([label, f"{eight:.1f}", f"{sixty_four:.1f}"])
report.say(f"  Llama 2 7B, {N7:,} parameters, GiB per GPU:")
report.table(["scheme", "8 GPUs", "64 GPUs"], rows)
report.check(
    "the page's table (100.4 | 34.5 26.3 | 23.5 13.9 | 12.6 1.6)",
    cells == ["100.4", "100.4", "34.5", "26.3", "23.5", "13.9", "12.6", "1.6"],
)
zero_paper = [round(acc.model_state_bytes(7.5e9, 64, z) / 1e9, 1) for z in range(4)]
report.check(
    "the ZeRO paper's Figure 1 (7.5B, 64 GPUs): 120, 31.4, 16.6, 1.9 GB",
    zero_paper == [120.0, 31.4, 16.6, 1.9],
)
report.check(
    "exercise 3: a 13B model's state is 193.7 GiB, and 24.2 GiB per GPU under ZeRO-3 on 8",
    f"{acc.model_state_bytes(13e9) / acc.GiB:.1f} {acc.model_state_bytes(13e9, 8, 3) / acc.GiB:.1f}"
    == "193.7 24.2",
)
days_13b = acc.gpu_hours(acc.train_flops(13e9, 1e12), H100, 0.4) / 512 / 24
report.check(
    "exercise 2: 13B on 1T tokens, H100 at 40%, 512 GPUs: 54,769 GPU hours, about 4.5 days",
    f"{acc.gpu_hours(acc.train_flops(13e9, 1e12), H100, 0.4):,.0f} {days_13b:.1f}" == "54,769 4.5",
)
paper_flops = acc.train_flops(405e9, 15.6e12)
report.check(
    "exercise 1: 6ND for Llama 3 405B is 3.79e25, within 0.3% of the paper's 3.8e25",
    f"{paper_flops:.3g}" == "3.79e+25" and abs(paper_flops / 3.8e25 - 1) < 0.003,
)

# ---------------------------------------------------------------------------
report.section("8. Memory: activations grow with the number of tokens")
h_, f_, tok = 512, 1376, 4096
mlp = SwiGLU(h_, f_).to(torch.bfloat16)
x = torch.randn(tok, h_, dtype=torch.bfloat16, requires_grad=True)
ffn = acc.saved_tensor_bytes(lambda: mlp(x), mlp.parameters())
report.say(
    f"  SwiGLU FFN, h={h_}, f={f_}, bf16, {tok} tokens: {ffn.tensors} tensors saved, {ffn.total / tok:,.0f} bytes per token (2h + 8f = {2 * h_ + 8 * f_:,})"
)
report.check(
    "the FFN saves x, gate, up, silu(gate) and their product: 2h + 8f bytes per token",
    ffn.total / tok == 2 * h_ + 8 * f_ and ffn.tensors == 5,
)

heads_, kv_, t_, d_ = 8, 4, 512, 64
q = torch.randn(1, heads_, t_, d_, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, kv_, t_, d_, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, kv_, t_, d_, dtype=torch.bfloat16, requires_grad=True)
att = acc.saved_tensor_bytes(
    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
)
kernel = 2 * heads_ * d_ + 4 * kv_ * d_ + 2 * heads_ * d_  # q, k and v, the output
extra = att.total / t_ - kernel
report.say(
    f"  fused attention, {heads_} query heads, {kv_} KV heads: {att.total / t_:,.0f} bytes per token; q, k, v and the output are {kernel:,}, the rest is {extra:,.0f}"
)
report.check(
    "attention keeps q, k, v and its output (no S x S matrix) plus a small float32 logsumexp",
    0 <= extra <= 0.02 * kernel,
    f"logsumexp: 4 bytes per head and token = {4 * heads_}",
)

blk_cfg = GPTConfig(
    vocab_size=256, block_size=512, n_layer=1, n_head=8, n_kv_head=4, d_model=512, d_ff=1376,
    norm="rmsnorm", pos="rope", mlp="swiglu", bias=False,
)  # fmt: skip
net = GPT(blk_cfg).to(torch.bfloat16)
block = net.blocks[0]
_, rope, _ = net.embed(torch.zeros(1, 512, dtype=torch.long))
x = torch.randn(1, 512, 512, dtype=torch.bfloat16, requires_grad=True)
formula = acc.activation_bytes_per_token_layer(512, 4 * 64, 1376)
eager = acc.saved_tensor_bytes(lambda: block(x, rope=rope, impl="sdpa"), block.parameters())
no_rope = acc.saved_tensor_bytes(lambda: block(x, rope=None, impl="sdpa"), block.parameters())
fp32 = eager.by_dtype.get(torch.float32, 0) / 512
rope_bytes = (eager.total - no_rope.total) / 512
expanded = 2 * (8 - 4) * 64 * 2  # repeat_kv gives k and v all 8 heads instead of 4
report.say(
    f"  one block of this repo (h=512, 8 heads, 4 KV heads, f=1376, bf16): {eager.total / 512:,.0f} bytes per token in {eager.tensors} tensors, formula {formula:,}"
)
report.say(
    f"  of the measured bytes, {fp32:,.0f} are float32 (the norms' statistics in float32, the logsumexp); RoPE adds {rope_bytes:,.0f}; repeat_kv copies k and v for every query head: {expanded:,}"
)
report.check(
    "the formula is a floor for eager PyTorch, which keeps 1.0 - 1.6 times as much",
    formula <= eager.total / 512 <= 1.6 * formula,
    f"{eager.total / 512 / formula:.2f}x: the formula describes fused norm, RoPE and GQA kernels",
)

rows, per_layer = [], {}
for name in ("Llama 2 7B", "Llama 3 8B"):
    c = BY_NAME[name].config
    per_layer[name] = acc.activation_bytes_per_token_layer(
        c.d_model, c.kv_heads * c.dim_per_head, c.d_ff
    )
    whole = per_layer[name] * c.n_layer
    rows.append(
        [
            name,
            f"{per_layer[name] / 1024:.1f} KiB",
            f"{whole / 2**20:.2f} MiB",
            f"{whole * 4096 / acc.GiB:.2f} GiB",
        ]
    )
report.table(["model", "per token and layer", "per token, whole model", "4,096 tokens"], rows)
report.check(
    "the page's table: 150.0 KiB, 4.69 MiB, 18.75 GiB and 164.0 KiB, 5.12 MiB, 20.50 GiB",
    [r[1:] for r in rows]
    == [["150.0 KiB", "4.69 MiB", "18.75 GiB"], ["164.0 KiB", "5.12 MiB", "20.50 GiB"]],
)
logits = 4096 * 128256 * 4 / acc.GiB
report.check("Llama 3's float32 logits for 4,096 tokens: 1.96 GiB", f"{logits:.2f}" == "1.96")

# recomputation: what is saved with and without it, measured on a stack of 6 blocks
stack_cfg = GPTConfig(
    vocab_size=256, block_size=128, n_layer=6, n_head=4, n_kv_head=2, d_model=256, d_ff=688,
    norm="rmsnorm", pos="rope", mlp="swiglu", bias=False,
)  # fmt: skip
stack = GPT(stack_cfg).to(torch.bfloat16)
tokens_per_step = 4 * 128
x0 = torch.randn(4, 128, 256, dtype=torch.bfloat16, requires_grad=True)
_, rope, _ = stack.embed(torch.zeros(1, 128, dtype=torch.long))


def run_blocks(recompute: bool) -> torch.Tensor:
    x = x0
    for blk in stack.blocks:
        if recompute:
            x = checkpoint(
                lambda t, blk=blk: blk(t, rope=rope, impl="sdpa"), x, use_reentrant=False
            )
        else:
            x = blk(x, rope=rope, impl="sdpa")
    return x


plain = acc.saved_tensor_bytes(lambda: run_blocks(False), stack.parameters())
kept = acc.saved_tensor_bytes(lambda: run_blocks(True), stack.parameters())
one_layer = plain.total / stack_cfg.n_layer
inputs_only = stack_cfg.n_layer * 2 * stack_cfg.d_model * tokens_per_step
report.say(
    f"  6 blocks, h=256, 512 tokens (bytes per token): stored {plain.total / tokens_per_step:,.0f}; recomputed {kept.total / tokens_per_step:,.0f}; plus one block's working set {one_layer / tokens_per_step:,.0f}"
)
report.check(
    "with recomputation the only saved tensors are the blocks' inputs: layers * 2h bytes per token",
    kept.total == inputs_only,
    f"{kept.total / tokens_per_step:,.0f} = 6 * 2 * 256 per token",
)
report.say(
    f"  saved bytes shrink {plain.total / (kept.total + one_layer):.1f}x here (6 layers); the formula gives 32 layers of Llama 2 7B:"
)
c7 = l2.config
rec7 = acc.recompute_bytes_per_token(c7.d_model, c7.n_layer, per_layer["Llama 2 7B"])
report.check(
    "Llama 2 7B, 4,096 tokens: 18.75 GiB become 1.59 GiB, about 1/11.8",
    f"{per_layer['Llama 2 7B'] * 32 * 4096 / acc.GiB:.2f} {rec7 * 4096 / acc.GiB:.2f} "
    f"{per_layer['Llama 2 7B'] * 32 / rec7:.1f}" == "18.75 1.59 11.8",
)

# ---------------------------------------------------------------------------
report.section("9. All three accounts together: Llama 2 7B, 8 x 80 GiB, sequence 4096")
seq_tokens = 4096


def per_gpu_gib(micro_batch: int, zero_stage: int, recompute: bool) -> tuple[float, float, float]:
    tokens_ = micro_batch * seq_tokens
    state = acc.model_state_bytes(N7, 8, zero_stage)
    if recompute:
        activations = (
            acc.recompute_bytes_per_token(c7.d_model, c7.n_layer, per_layer["Llama 2 7B"]) * tokens_
        )
    else:
        activations = per_layer["Llama 2 7B"] * c7.n_layer * tokens_
    logit_bytes = 6 * c7.vocab_size * tokens_  # a bf16 copy and a float32 copy
    return state / acc.GiB, activations / acc.GiB, logit_bytes / acc.GiB


rows, totals = [], {}
for label, mb, z, rec in [
    ("ZeRO-0, micro-batch 1", 1, 0, False),
    ("ZeRO-3, micro-batch 1", 1, 3, False),
    ("ZeRO-3, micro-batch 3", 3, 3, False),
    ("ZeRO-3, micro-batch 4", 4, 3, False),
    ("ZeRO-3, micro-batch 1, recompute", 1, 3, True),
]:
    state, activations, logit_gib = per_gpu_gib(mb, z, rec)
    totals[label] = state + activations + logit_gib
    rows.append(
        [
            label,
            f"{state:.2f}",
            f"{activations:.2f}",
            f"{logit_gib:.2f}",
            f"{totals[label]:.2f}",
            "fits" if totals[label] <= 80 else "does not fit",
        ]
    )
report.table(["configuration", "state", "activations", "logits", "total GiB", "80 GiB"], rows)
report.check(
    "the page's totals: 119.89 (no sharding), 32.03, 71.00 and, with recomputation, 14.87 GiB",
    [f"{totals[k]:.2f}" for k in list(totals)[:3] + list(totals)[4:]]
    == ["119.89", "32.03", "71.00", "14.87"],
)
report.check(
    "a 4th sequence per micro-batch (about 90 GiB) no longer fits, a 3rd does",
    totals["ZeRO-3, micro-batch 3"] <= 80 < totals["ZeRO-3, micro-batch 4"]
    and round(totals["ZeRO-3, micro-batch 4"]) == 90,
)
state, activations, logit_gib = per_gpu_gib(1, 3, True)
fits = int((80 - state) // (activations + logit_gib))
report.check(
    "with recomputation 29 sequences would fit: 'more than twenty' on the page",
    fits == 29,
    "paid for with a third more FLOPs (section 5)",
)

report.finish()
