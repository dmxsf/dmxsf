"""Lab: the Transformer block, pre-norm vs post-norm, and the GPT-2 -> Llama recipe.

Course pages: https://dmxsf.com/wiki/architecture/transformer-block/
              https://dmxsf.com/wiki/architecture/modern-llm-architecture/

    uv run python labs/architecture/transformer-block/run.py [--quick] [--seed N]
"""

import itertools
import math
import statistics as st

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dmxsf import lab
from dmxsf.architecture.configs import (
    PUBLIC_MODELS,
    kv_bytes_per_token,
    param_breakdown,
    small_gpt,
)
from dmxsf.architecture.gpt import GPT, Block, GPTConfig
from dmxsf.data import ByteCorpus, get_batch
from dmxsf.foundations.optim import lr_factor

args, report = lab.start(
    "The Transformer block and the modern recipe",
    wiki=["architecture/transformer-block", "architecture/modern-llm-architecture"],
)

# ---------------------------------------------------------------------------
report.section("1. One GPT-2 small block: 12 d^2 + 13 d parameters")
gpt2 = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, d_model=768)
block = Block(gpt2)
parts = {"attention": 0, "feed-forward": 0, "norms": 0}
for name, p in block.named_parameters():
    key = "attention" if name.startswith("attn") else "feed-forward"
    parts[key if name.startswith(("attn", "mlp")) else "norms"] += p.numel()
total = sum(parts.values())
d = 768
report.table(
    ["part", "parameters", "share"],
    [[k, f"{v:,}", f"{v / total:.1%}"] for k, v in parts.items()],
)
report.say(f"  total {total:,}; the formula 12d^2 + 13d gives {12 * d * d + 13 * d:,}")
with torch.device("meta"):
    full = GPT(gpt2)
report.say(f"  12 blocks + token table + position table + final norm = {full.num_params():,}")
report.check(
    "the numbers on the course page (GPT-2 small is 124,439,808 parameters)",
    (parts["attention"], parts["feed-forward"], parts["norms"], total)
    == (2_362_368, 4_722_432, 3_072, 7_087_872)
    and full.num_params() == 124_439_808,
)
torch.manual_seed(args.seed)
block.eval()
x = torch.randn(1, 10, 768)
x_changed = x.clone()
x_changed[:, 6:] = torch.randn(1, 4, 768)
with torch.no_grad():
    y, y_changed = block(x), block(x_changed)
report.check(
    "output shape is unchanged and positions 0-5 ignore everything after them",
    y.shape == x.shape and torch.allclose(y[:, :6], y_changed[:, :6], atol=1e-5),
)

# ---------------------------------------------------------------------------
report.section("2. Where real models keep their parameters (shapes only, no weights)")
rows = []
for model in PUBLIC_MODELS:
    with torch.device("meta"):
        net = GPT(model.config)
    p = param_breakdown(net)
    n = sum(p.values())
    assert n == model.params
    vocab = net.tok_emb.weight.numel() + p["head"]  # input table, plus the output matrix if untied
    rows.append(
        [
            model.name,
            f"{n / 1e9:.2f}B",
            f"{p['mlp'] / n:.1%}",
            f"{p['attention'] / n:.1%}",
            f"{vocab / n:.1%}",
            f"{kv_bytes_per_token(model.config) / 1024:g} KiB",
        ]
    )
header = ["model", "params", "feed-forward", "attention", "vocab tables", "KV cache / token"]
report.table(header, rows)
by_name = {r[0]: r for r in rows}
report.check("all seven totals equal the published parameter counts", True, "asserted above")
report.check(
    "course page: Llama 3 8B is 70.2% feed-forward, 16.7% attention, 13.1% vocabulary",
    by_name["Llama 3 8B"][2:5] == ["70.2%", "16.7%", "13.1%"],
)
report.check(
    "course page: GPT-2 small spends 31.0% on its token table, Llama 2 7B 3.9%, Llama 3 70B 3.0%",
    [by_name[n][4] for n in ("GPT-2 small", "Llama 2 7B", "Llama 3 70B")]
    == ["31.0%", "3.9%", "3.0%"],
)
report.check(
    "course page: KV cache per token is 512 / 128 / 320 KiB for Llama 2 7B / Llama 3 8B / 70B",
    [by_name[n][5] for n in ("Llama 2 7B", "Llama 3 8B", "Llama 3 70B")]
    == ["512 KiB", "128 KiB", "320 KiB"],
)
report.say(
    "  Grouped-query attention is the difference: Llama 2 7B caches 32 KV heads, Llama 3 8B 8."
)

# ---------------------------------------------------------------------------
report.section("3. Pre-norm or post-norm: signal and gradient at initialisation, by depth")
seeds_depth = 1 if args.quick else 8
depths = (6, 12) if args.quick else (6, 12, 24, 48)


def depth_run(n_layers: int, pre_norm: bool, scale_residual: bool, seed: int):
    """Stack n_layers blocks at initialisation; return (RMS of the output, gradient norm)."""
    d, t, b, vocab = 256, 64, 8, 1000
    torch.manual_seed(seed)
    cfg = GPTConfig(
        vocab_size=vocab,
        block_size=t,
        n_layer=n_layers,
        n_head=4,
        d_model=d,
        d_ff=4 * d,
        pos="none",
        pre_norm=pre_norm,
    )
    blocks = nn.ModuleList(Block(cfg) for _ in range(n_layers))
    emb, head = nn.Embedding(vocab, d), nn.Linear(d, vocab, bias=False)
    final_ln = nn.LayerNorm(d) if pre_norm else nn.Identity()
    for m in [*blocks.modules(), head]:
        if isinstance(m, nn.Linear):  # variance preserving: W ~ N(0, 1/fan_in), biases 0
            nn.init.normal_(m.weight, std=1 / math.sqrt(m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    nn.init.normal_(emb.weight, std=1.0)
    if scale_residual:  # GPT-2: shrink the matrices that write into the stream by 1/sqrt(N)
        for blk in blocks:
            blk.attn.o_proj.weight.data /= math.sqrt(2 * n_layers)
            blk.mlp.proj.weight.data /= math.sqrt(2 * n_layers)
    h = emb(torch.randint(0, vocab, (b, t)))
    for blk in blocks:
        h = blk(h)
    rms = h.pow(2).mean().sqrt().item()
    logits = head(final_ln(h)).view(-1, vocab)
    F.cross_entropy(logits, torch.randint(0, vocab, (b * t,))).backward()
    last_ffn_output = blocks[-1].mlp.proj.weight
    return rms, last_ffn_output.grad.norm().item()


settings = {"Post-LN": (False, False), "Pre-LN": (True, False), "Pre-LN+1/sqrt(N)": (True, True)}
table, depth_rows = {}, []
for n_layers in depths:
    row = [n_layers]
    for name, (pre, scale) in settings.items():
        runs = [depth_run(n_layers, pre, scale, seed) for seed in range(seeds_depth)]
        table[n_layers, name] = (st.mean(r for r, _ in runs), st.mean(g for _, g in runs))
        row += list(table[n_layers, name])
    depth_rows.append(row)
report.say(
    f"  random tokens, d=256, 8 sequences of 64, mean of {seeds_depth} seed(s)\n"
    "  RMS  = size of the last layer's output\n"
    "  grad = norm of the gradient of the last feed-forward's output matrix"
)
report.table(
    ["layers", "post RMS", "post grad", "pre RMS", "pre grad", "pre+s RMS", "pre+s grad"],
    depth_rows,
)
first, last = depths[0], depths[-1]
pre_rms = [table[n, "Pre-LN"][0] for n in depths]
report.check(
    "post-LN: the gradient does not shrink with depth",
    table[last, "Post-LN"][1] > 0.9 * table[first, "Post-LN"][1],
)
report.check(
    "pre-LN: the gradient shrinks like 1/sqrt(L) (gradient x sqrt(L) stays within 15%)",
    all(
        abs(
            table[n, "Pre-LN"][1] * math.sqrt(n) / (table[first, "Pre-LN"][1] * math.sqrt(first))
            - 1
        )
        < 0.15
        for n in depths
    ),
)
report.check(
    "pre-LN: the residual stream keeps growing with depth",
    all(a < b for a, b in itertools.pairwise(pre_rms)) and pre_rms[-1] > 1.4 * pre_rms[0],
)
report.check(
    "pre-LN: gradient x RMS is nearly constant (between 0.9 and 1.1)",
    all(0.9 < table[n, "Pre-LN"][0] * table[n, "Pre-LN"][1] < 1.1 for n in depths),
)
scaled = [table[n, "Pre-LN+1/sqrt(N)"][0] for n in depths]
report.check(
    "scaling the residual branches by 1/sqrt(N) keeps the size steady (spread under 0.05)",
    max(scaled) - min(scaled) < 0.05,
    f"{min(scaled):.3f} to {max(scaled):.3f}",
)
if args.quick:
    report.skip("agreement with the course page's table", "the page averages 8 seeds at 4 depths")
else:
    page = {  # (RMS, gradient) per setting, copied from the course page
        6: [(1.00, 0.82), (2.35, 0.41), (1.14, 0.81)],
        12: [(1.00, 0.84), (3.42, 0.28), (1.14, 0.82)],
        24: [(1.00, 0.88), (5.03, 0.20), (1.14, 0.81)],
        48: [(1.00, 0.91), (7.39, 0.14), (1.14, 0.82)],
    }
    worst_rms = worst_grad = 0.0
    for n_layers, expected in page.items():
        for name, (rms, grad) in zip(settings, expected, strict=True):
            worst_rms = max(worst_rms, abs(table[n_layers, name][0] - rms))
            worst_grad = max(worst_grad, abs(table[n_layers, name][1] - grad))
    report.check(
        "all 12 cells agree with the course page's table (RMS within 0.05, gradient within 0.03)",
        worst_rms < 0.05 and worst_grad < 0.03,
        f"largest differences: RMS {worst_rms:.3f}, gradient {worst_grad:.3f}",
    )

# ---------------------------------------------------------------------------
report.section("4. From GPT-2 to Llama, one change at a time")
STEPS = 30 if args.quick else 500
LR, CLIP = 3e-3, 1.0
seeds = [args.seed] if args.quick else [args.seed + i for i in range(3)]
data = ByteCorpus.shakespeare()
val = torch.from_numpy(
    np.stack([data.val[i * 64 : (i + 1) * 64 + 1] for i in range(150)]).astype(np.int64)
)
llama = dict(norm="rmsnorm", mlp="swiglu", d_ff=170, pos="rope", bias=False)
ladder = [
    ("GPT-2 style: LayerNorm, GELU, learned positions, biases", {}),
    ("+ RMSNorm", dict(norm="rmsnorm")),
    (
        "+ SwiGLU (hidden size cut to keep the MLP's size)",
        dict(norm="rmsnorm", mlp="swiglu", d_ff=170),
    ),
    (
        "+ RoPE instead of learned positions",
        dict(norm="rmsnorm", mlp="swiglu", d_ff=170, pos="rope"),
    ),
    ("+ no biases", llama),
    ("+ grouped-query attention (2 KV heads)", {**llama, "n_kv_head": 2}),
]


def validation_loss(overrides: dict, seed: int) -> tuple[float, GPTConfig]:
    torch.manual_seed(seed)
    config = small_gpt(n_layer=2, d_model=64, n_head=4, block_size=64, **overrides)
    model = GPT(config)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    for step in range(STEPS):
        for group in opt.param_groups:
            group["lr"] = LR * lr_factor("cosine", step, STEPS, warmup=max(1, STEPS // 10))
        x, y = get_batch(data.train, step, 32, 64, seed=seed)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
    with torch.no_grad():
        val_loss = F.cross_entropy(model(val[:, :-1]).reshape(-1, 256), val[:, 1:].reshape(-1))
    return val_loss.item(), config


configs, means, spans = [], [], []
for _, overrides in ladder:
    results = [validation_loss(overrides, s) for s in seeds]
    values = [v for v, _ in results]
    configs.append(results[0][1])
    means.append(st.mean(values))
    spans.append(max(values) - min(values))
noise = max(spans)
rows = []
for i, (label, _) in enumerate(ladder):
    delta = means[i] - means[i - 1] if i else 0.0
    verdict = "better" if delta < -noise else "worse" if delta > noise else "same"
    if i == 0 or len(seeds) < 2:  # one seed gives no estimate of the noise
        verdict = ""
    rows.append(
        [
            label,
            f"{GPT(configs[i]).num_params():,}",
            f"{kv_bytes_per_token(configs[i])} B",
            means[i],
            spans[i],
            f"{delta:+.3f} {verdict}".strip() if i else "",
        ]
    )
report.say(
    f"  2 layers, d_model 64, {STEPS} steps (lr {LR:g}, gradient clipping {CLIP:g}) on Shakespeare "
    f"bytes, {len(seeds)} seed(s); validation loss in nats per byte"
)
report.table(
    ["variant", "parameters", "KV / token", "mean loss", "spread", "change vs row above"], rows
)
if args.quick:
    report.skip("which change matters", f"{STEPS} steps are too few")
else:
    report.say(
        f"  'spread' is max - min over seeds; a row is called better or worse only when it moves "
        f"the mean by more than the largest spread, {noise:.3f}"
    )
    report.check(
        "the full recipe beats GPT-2 style by more than the spread",
        means[0] - means[-1] > noise,
        f"{means[0] - means[-1]:.3f} nats",
    )
    rope_gain = means[2] - means[3]
    report.check(
        "swapping learned positions for RoPE is a clear gain",
        rope_gain > noise,
        f"{rope_gain:.3f} nats",
    )
    report.check(
        "grouped-query attention halves the KV cache without a clear loss",
        configs[5].kv_heads * 2 == configs[4].kv_heads and means[5] - means[4] < noise,
        f"{means[5] - means[4]:+.3f} nats",
    )

report.finish()
