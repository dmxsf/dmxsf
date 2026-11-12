"""Lab: shrinking the KV cache (GQA, MLA, sliding window) and tiling the score matrix (FlashAttention).

Course page: https://dmxsf.com/wiki/architecture/efficient-attention/

    uv run python labs/architecture/efficient-attention/run.py [--quick] [--seed N]
"""

import statistics as st
import time

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from dmxsf import lab
from dmxsf.architecture.attention import attention, causal_mask, tiled_attention
from dmxsf.architecture.configs import BY_NAME, kv_bytes_per_token, small_gpt
from dmxsf.architecture.gpt import GPT, to_gqa
from dmxsf.architecture.mla import MLACache, MLAttention
from dmxsf.architecture.position import rope_cos_sin, rope_inv_freq
from dmxsf.data import ByteCorpus, get_batch
from dmxsf.inference.kv_cache import KVCache, SlidingKVCache
from dmxsf.pretraining.train import TrainConfig, bits_per_byte, make_optimizer, train

args, report = lab.start("Efficient attention", wiki=["architecture/efficient-attention"])
KiB, GiB = 2**10, 2**30

# ---------------------------------------------------------------------------
report.section("1. KV cache per token: the formulas against the course page's table")
# (name, layers, KV heads, head dim): the Llama rows come from our own model configs, the 405B row
# from the Llama 3 paper (126 layers, 8 KV heads, 128 per head).
dense = {name: BY_NAME[name].config for name in ("Llama 2 7B", "Llama 3 8B", "Llama 3 70B")}
shapes = {n: (c.n_layer, c.kv_heads, c.dim_per_head) for n, c in dense.items()}
shapes["Llama 3 405B"] = (126, 8, 128)
# MLA: layers, d_c, d_R (DeepSeek-V2 and V3 paper); 128 query heads of 128 dims, if it were MHA
mla_shapes = {"DeepSeek-V2": (60, 512, 64), "DeepSeek-V3": (61, 512, 64)}
design = {"Llama 2 7B": "MHA, 32 KV heads"}
design |= dict.fromkeys(("Llama 3 8B", "Llama 3 70B", "Llama 3 405B"), "GQA, 8 KV heads")
design |= dict.fromkeys(mla_shapes, "MLA, d_c=512, d_R=64")

per_token = {}  # bytes, BF16
for name, (layers, kv, d_h) in shapes.items():
    per_token[name] = 2 * layers * kv * d_h * 2  # K and V, kv heads, d_h numbers, 2 bytes each
for name, (layers, d_c, d_r) in mla_shapes.items():
    per_token[name] = layers * (d_c + d_r) * 2
rows = [
    [n, design[n], f"{b / KiB:.1f} KiB", f"{b * 128 * KiB / GiB:.2f} GiB"]
    for n, b in per_token.items()
]
report.table(["model", "attention", "per token (BF16)", "128K tokens"], rows)
kib = [round(b / KiB, 1) for b in per_token.values()]
gib = [round(b * 128 * KiB / GiB, 2) for b in per_token.values()]
report.check(
    "per-token sizes: 512 / 128 / 320 / 504 / 67.5 / 68.6 KiB",
    kib == [512, 128, 320, 504, 67.5, 68.6],
)
report.check(
    "128K-token caches: 64 / 16 / 40 / 63 / 8.44 / 8.58 GiB",
    gib == [64, 16, 40, 63, 8.44, 8.58],
)
report.check(
    "the formula agrees with kv_bytes_per_token on the real Llama configs",
    all(kv_bytes_per_token(dense[n]) == per_token[n] for n in dense),
)
mha_v3 = 2 * 61 * 128 * 128 * 2  # what DeepSeek-V3 would cache with 128 MHA heads
report.say(
    f"  DeepSeek-V3 as MHA: {mha_v3 / 2**20:.2f} MiB per token; MLA caches {per_token['DeepSeek-V3'] / KiB:.1f} KiB, "
    f"1/{mha_v3 / per_token['DeepSeek-V3']:.1f}"
)
report.check(
    "course page: 3.81 MiB as MHA, 1/56.9 with MLA, and 576 numbers = 2.25 GQA groups",
    round(mha_v3 / 2**20, 2) == 3.81
    and round(mha_v3 / per_token["DeepSeek-V3"], 1) == 56.9
    and (512 + 64) / (2 * 128) == 2.25,
)

report.say(
    "\n  Tensor parallelism: a KV head cannot be split, so fewer KV heads than GPUs means copies."
)
rows = []
for name, kv_heads in (("MHA (64 KV heads)", 64), ("GQA (8 KV heads)", 8), ("MQA (1 KV head)", 1)):
    per_gpu = max(1, kv_heads // 8)
    rows.append([name, per_gpu, per_gpu * 8])
report.table(["8-way tensor parallel", "KV heads per GPU", "KV heads stored in total"], rows)
report.check(
    "with 8 GPUs MQA stores as many KV heads as GQA-8 (the copies cancel the saving)",
    rows[1][2] == rows[2][2] == 8,
)

# ---------------------------------------------------------------------------
report.section("2. The same numbers, measured on real caches")
torch.manual_seed(args.seed)
n_tokens, layers, heads, d_h = 100, 4, 8, 16
idx = torch.randint(0, 256, (1, n_tokens))
rows, measured = [], {}
for name, kv in (("MHA", 8), ("GQA, 2 KV heads", 2), ("MQA", 1)):
    cfg = small_gpt(n_layer=layers, n_head=heads, d_model=heads * d_h, n_kv_head=kv, pos="rope")
    cache = KVCache(layers)
    with torch.no_grad():
        GPT(cfg)(idx, cache=cache)
    measured[name] = cache.nbytes() / n_tokens
    assert measured[name] == kv_bytes_per_token(cfg, dtype_bytes=4)
    rows.append([name, 2 * kv * d_h, f"{measured[name]:.0f} B"])
d_c, d_r = 4 * d_h, d_h // 2  # DeepSeek's ratios: d_c = 4 d_h, d_R = d_h / 2
mla_layers = [MLAttention(heads * d_h, heads, d_h, d_h, d_c, d_r).eval() for _ in range(layers)]
mla_cache = MLACache(layers)
inv_freq = rope_inv_freq(d_r)
with torch.no_grad():
    h = torch.randn(1, n_tokens, heads * d_h)
    for i, layer in enumerate(mla_layers):
        layer(h, rope_cos_sin(torch.arange(n_tokens), inv_freq), cache=mla_cache, layer_idx=i)
measured["MLA"] = mla_cache.nbytes() / n_tokens
rows.append([f"MLA (d_c={d_c}, d_R={d_r})", d_c + d_r, f"{measured['MLA']:.0f} B"])
report.say(
    f"  {layers} layers, {heads} query heads of {d_h} dims, {n_tokens} tokens, float32 (4 bytes)"
)
report.table(["design", "numbers / token / layer", "measured bytes / token"], rows)
report.check(
    "measured cache sizes equal the formulas (2 g d_h L x 4 bytes; (d_c + d_R) L x 4 bytes)",
    measured["MHA"] == 2 * 8 * d_h * layers * 4
    and measured["GQA, 2 KV heads"] == 2 * 2 * d_h * layers * 4
    and measured["MLA"] == (d_c + d_r) * layers * 4,
)
report.check(
    "MLA with d_c = 4 d_h, d_R = d_h/2 caches as much as 2.25 GQA groups, whatever the number of heads",
    measured["MLA"] / measured["MQA"] == 2.25,
)

# ---------------------------------------------------------------------------
report.section("3. MLA: caching a latent vector, and never rebuilding the keys")
rng = np.random.default_rng(0)  # the course page's check, line for line
d, n_h, d_h, d_c, T = 256, 8, 32, 48, 10
H = rng.normal(size=(T, d))
W_DKV = rng.normal(size=(d_c, d)) / np.sqrt(d)
W_UK = rng.normal(size=(n_h, d_h, d_c)) / np.sqrt(d_c)
W_UV = rng.normal(size=(n_h, d_h, d_c)) / np.sqrt(d_c)
W_Q = rng.normal(size=(n_h, d_h, d)) / np.sqrt(d)
W_O = rng.normal(size=(d, n_h * d_h)) / np.sqrt(n_h * d_h)
C = H @ W_DKV.T  # the cached latents: T x d_c
t = T - 1


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


heads_out = []
for h in range(n_h):  # rebuild K and V, then ordinary multi-head attention
    q = W_Q[h] @ H[t]
    K, V = C @ W_UK[h].T, C @ W_UV[h].T
    heads_out.append(softmax(K @ q / np.sqrt(d_h)) @ V)
u_explicit = W_O @ np.concatenate(heads_out)
u_absorbed = np.zeros(d)
for h in range(n_h):  # absorb W_UK into the query and W_UV into the output projection
    q_abs = W_UK[h].T @ (W_Q[h] @ H[t])
    p = softmax(C @ q_abs / np.sqrt(d_h))
    u_absorbed += (W_O[:, h * d_h : (h + 1) * d_h] @ W_UV[h]) @ (p @ C)
err = np.abs(u_explicit - u_absorbed).max()
report.say(f"  NumPy toy (d={d}, {n_h} heads, d_c={d_c}): max |explicit - absorbed| = {err:.1e}")
report.say(
    f"  numbers cached per token and layer: MHA {2 * n_h * d_h}, MLA {d_c} (without the RoPE key)"
)
report.check(
    "the course page's absorption check (error ~1e-16)", err < 1e-12 and 2 * n_h * d_h == 512
)

torch.manual_seed(args.seed)
layer = MLAttention(256, 8, 32, 32, 64, 16).eval()
inv_freq = rope_inv_freq(16)


def rope_at(start, n):
    return rope_cos_sin(torch.arange(start, start + n), inv_freq)


x = torch.randn(2, 48, 256)
with torch.no_grad():
    full = layer(x, rope_at(0, 48), absorb=False)
    absorbed = layer(x, rope_at(0, 48), absorb=True)
    cache = MLACache(1)
    steps = [layer(x[:, :20], rope_at(0, 20), cache=cache, absorb=True)]
    steps += [
        layer(x[:, i : i + 1], rope_at(i, 1), cache=cache, absorb=True) for i in range(20, 48)
    ]
e1 = (full - absorbed).abs().max().item()
e2 = (full - torch.cat(steps, dim=1)).abs().max().item()
report.say("  full module with causal masking and decoupled RoPE, float32:")
report.say(f"    absorbed vs rebuilt keys, whole sequence : max error {e1:.1e}")
report.say(f"    absorbed decoding from the latent cache  : max error {e2:.1e}")
report.say(f"    cache after 48 tokens: {cache.numbers_per_token()} numbers per token and layer")
report.check("absorbed decoding reproduces the rebuilt-keys result", max(e1, e2) < 1e-4)
report.check(
    "the cache holds d_c + d_R = 80 numbers per token", cache.numbers_per_token() == 64 + 16
)


class FrozenCache:
    """Pretends the new token is already cached: nothing is copied, so only the attention is timed."""

    def __init__(self, c, k_rope):
        self.c, self.k_rope = c, k_rope

    def update(self, layer, c, k_rope):
        return self.c, self.k_rope


report.say("\n  Why absorb? One decode step with the cache already full (random weights, CPU):")
n_h, d_nope, d_v, d_c, d_r = 8, 32, 32, 128, 16
layer = MLAttention(256, n_h, d_nope, d_v, d_c, d_r).eval()
lengths = (512, 2048, 8192)
rows, ratio = [], 1.0
for n_cached in lengths:
    cache = FrozenCache(torch.randn(1, n_cached, d_c), torch.randn(1, 1, n_cached, d_r))
    x_new = torch.randn(1, 1, 256)
    timings = {}
    for absorb in (False, True):
        times = []
        for rep in range(16):
            t0 = time.perf_counter()
            with torch.no_grad():
                layer(x_new, rope_at(n_cached - 1, 1), cache=cache, absorb=absorb)
            if rep:  # the first call is a warm-up
                times.append(time.perf_counter() - t0)
        timings[absorb] = st.median(times) * 1000
    ratio = timings[False] / timings[True]
    # multiply-adds for the keys and values / the attention itself, per decode step
    rebuild = n_cached * d_c * n_h * (d_nope + d_v) + n_cached * n_h * (d_nope + d_r + d_v)
    absorbed = n_h * d_nope * d_c + n_cached * n_h * (2 * d_c + d_r) + n_h * d_v * d_c
    rows.append(
        [
            n_cached,
            f"{rebuild / 1e6:.1f} M",
            f"{timings[False]:.2f} ms",
            f"{absorbed / 1e6:.1f} M",
            f"{timings[True]:.2f} ms",
            f"{ratio:.1f}x",
        ]
    )
report.say(f"  8 heads, d_c={d_c}, median of 15 steps")
report.table(["cached", "rebuild MACs", "time", "absorbed MACs", "time", "faster by"], rows)
report.say(
    "  The arithmetic drops by an order of magnitude more than the time does: a step this small "
    "is no longer limited by multiply-adds."
)
report.check(
    "absorption makes a decode step at least 1.5x faster at the longest length",
    ratio > 1.5,  # 3.6x on the reference machine, 15x and 29x on Linux runners: it is a timing
    f"{ratio:.1f}x at {lengths[-1]} tokens",
)


# ---------------------------------------------------------------------------
report.section("4. FlashAttention's algorithm: tiles and an online softmax")
torch.manual_seed(args.seed)
n, d = 1000, 64
q, k, v = (torch.randn(1, 1, n, d, dtype=torch.float64) * 3 for _ in range(3))  # sharp softmax
expected, _ = attention(q, k, v, mask=causal_mask(n, n))
tiled = tiled_attention(q, k, v, block_q=128, block_k=128)
err = (expected - tiled).abs().max().item()
report.say(f"  N={n}, d={d}, tiles of 128: max |standard - tiled| = {err:.1e} (float64)")
report.say(
    f"  score matrix {n * n * 8 / 2**20:.1f} MiB, one tile {128 * 128 * 8 / KiB:.0f} KiB (float64)"
)
report.check(
    "the page's numbers: 7.6 MiB for the full matrix, 128 KiB per tile, and an exact result",
    err < 1e-10 and round(n * n * 8 / 2**20, 1) == 7.6 and 128 * KiB == 128 * 128 * 8,
)
inputs = [torch.randn(1, 2, 97, 16, dtype=torch.float64, requires_grad=True) for _ in range(3)]
ref_grads = torch.autograd.grad(attention(*inputs, causal_mask(97, 97))[0].pow(2).sum(), inputs)
tile_grads = torch.autograd.grad(tiled_attention(*inputs, 16, 16).pow(2).sum(), inputs)
worst = max((a - b).abs().max().item() for a, b in zip(ref_grads, tile_grads, strict=True))
report.check("the gradients of q, k and v agree too", worst < 1e-9, f"max difference {worst:.1e}")


class LargestTensor(TorchDispatchMode):
    """Records how many elements the biggest tensor produced by any operation has."""

    def __init__(self):
        super().__init__()
        self.numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, tuple | list) else [out]:
            if isinstance(t, torch.Tensor):
                self.numel = max(self.numel, t.numel())
        return out


def biggest_tensor(fn, *tensors) -> int:
    """Number of elements of the largest tensor any operation inside fn(*tensors) produces."""
    with torch.no_grad(), LargestTensor() as watch:
        fn(*tensors)
    return watch.numel


def standard(q, k, v):
    return attention(q, k, v, causal_mask(q.shape[-2], k.shape[-2]))[0]


def tiled(q, k, v):
    return tiled_attention(q, k, v, 128, 128)


rows, standard_ok, tiled_ok = [], True, True
for n in (512, 2048) if args.quick else (512, 2048, 8192):
    q, k, v = (torch.randn(1, 1, n, 64) for _ in range(3))
    big_standard, big_tiled = biggest_tensor(standard, q, k, v), biggest_tensor(tiled, q, k, v)
    rows.append([n, f"{big_standard * 4 / 2**20:.2f} MiB", f"{big_tiled * 4 / 2**20:.2f} MiB"])
    standard_ok &= big_standard == n * n  # the score matrix
    tiled_ok &= big_tiled == n * 64  # the output itself
report.say("\n  The largest tensor any operation creates (float32, d=64, tiles of 128):")
report.table(["N", "standard", "tiled"], rows)
report.check(
    "standard attention builds an N x N matrix; the tiled version's biggest tensor is its N x d output",
    standard_ok and tiled_ok,
)

n, d, n_heads = 1024, 64, 4
q, k, v = (torch.randn(1, n_heads, n, d) for _ in range(3))
flops = {}
for name, fn in (
    ("standard", lambda: attention(q, k, v)[0]),
    ("tiled", lambda: tiled_attention(q, k, v, 128, 128, causal=False)),
    ("tiled, causal", lambda: tiled_attention(q, k, v, 128, 128, causal=True)),
):
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    flops[name] = counter.get_total_flops()
report.say(f"\n  Matrix-multiply FLOPs, N={n}, d={d}, {n_heads} heads (forward):")
report.table(
    ["version", "GFLOPs", "relative"],
    [[name, f"{f / 1e9:.3f}", f"{f / flops['standard']:.4f}"] for name, f in flops.items()],
)
report.check(
    "tiling alone does not reduce the arithmetic: the same 4 N^2 d per head",
    flops["standard"] == flops["tiled"] == 4 * n * n * d * n_heads,
)
blocks = n // 128
report.check(
    f"skipping the tiles above the diagonal (causal) does: {blocks * (blocks + 1) // 2}/{blocks**2} of the work",
    flops["tiled, causal"] * blocks**2 == flops["standard"] * (blocks * (blocks + 1) // 2),
)

# ---------------------------------------------------------------------------
report.section("5. Sliding-window attention: bounded cache, and information still travels")
window, receptive = 4, []
for n_layers in (1, 2, 3, 4):
    torch.manual_seed(args.seed)
    cfg = small_gpt(n_layer=n_layers, pos="rope", norm="rmsnorm", mlp="swiglu", window=window)
    model = GPT(cfg).eval()
    tokens = torch.randint(0, 256, (1, 24))
    other = tokens.clone()
    other[0, 0] = (other[0, 0] + 1) % 256  # change the very first token
    with torch.no_grad():
        moved = (model(tokens) - model(other)).abs().amax(dim=-1)[0] > 1e-7
    receptive.append((n_layers, moved.nonzero().max().item()))
report.say(f"  window {window}: the last position that still 'hears' a change to token 0")
report.table(
    ["layers", "last position", "layers x (window - 1)"],
    [[n_layers, last, n_layers * (window - 1)] for n_layers, last in receptive],
)
report.check(
    "information travels window - 1 positions per layer (a window of 4 over 4 layers reaches 12)",
    all(last == n_layers * (window - 1) for n_layers, last in receptive),
)
report.say(
    f"  Mistral 7B: 32 layers x 4096 = {32 * 4096:,} tokens of theoretical reach; a 32K sequence "
    f"needs {32768 // 4096}x less cache"
)

window = 16
torch.manual_seed(args.seed)
cfg = small_gpt(n_layer=2, pos="rope", norm="rmsnorm", mlp="swiglu", window=window)
model = GPT(cfg).eval()
tokens = torch.randint(0, 256, (1, 128))
with torch.no_grad():
    reference = model(tokens)
    full_cache, ring = KVCache(2), SlidingKVCache(2, window)
    steps = {"full": [], "sliding": []}
    rows = []
    for i in range(128):
        steps["full"].append(model(tokens[:, i : i + 1], cache=full_cache))
        steps["sliding"].append(model(tokens[:, i : i + 1], cache=ring))
        if i + 1 in (8, 16, 32, 64, 128):
            rows.append([i + 1, f"{full_cache.nbytes():,} B", f"{ring.nbytes():,} B"])
report.say(f"\n  Decoding 128 tokens one by one, window {window} (float32, 2 layers):")
report.table(["tokens so far", "full cache", "sliding cache"], rows)
gap = max((torch.cat(steps[k], dim=1) - reference).abs().max().item() for k in ("full", "sliding"))
report.check(
    "the sliding cache holds window - 1 tokens and gives the same logits as recomputing everything",
    ring.k[0].shape[2] == window - 1 and gap < 1e-4,
    f"max logit difference {gap:.1e}",
)

# ---------------------------------------------------------------------------
report.section("6. From an MHA model to GQA: mean-pool the heads, then train a little")
steps_total, uptrain = (30, 5) if args.quick else (500, 25)
seeds = [args.seed] if args.quick else [args.seed + i for i in range(3)]
corpus = ByteCorpus.shakespeare()
cfg_train = TrainConfig(steps=steps_total, peak_lr=3e-3, warmup=max(1, steps_total // 10))
cfg_up = TrainConfig(steps=uptrain, peak_lr=1e-3, warmup=max(1, uptrain // 8))
cfg_up4 = TrainConfig(steps=4 * uptrain, peak_lr=1e-3, warmup=max(1, uptrain // 2))


def score(model):
    return bits_per_byte(model, corpus.val, block_size=64, windows=150) * np.log(2)  # nats


def fit(model, first_step, config, seed):
    opt = make_optimizer(model, config)
    batch = lambda step: get_batch(corpus.train, first_step + step, 32, 64, seed=seed)  # noqa: E731
    train(model, opt, batch, config)
    return model


def first_head_of_each_group(model, n_kv):
    """The control: keep head 0 of every group instead of averaging the group."""
    new = to_gqa(model, n_kv)
    cfg = model.cfg
    for old_block, new_block in zip(model.blocks, new.blocks, strict=True):
        for name in ("k_proj", "v_proj"):
            heads = getattr(old_block.attn, name).weight.view(cfg.kv_heads, cfg.dim_per_head, -1)
            kept = heads[:: cfg.kv_heads // n_kv]
            getattr(new_block.attn, name).weight.data = kept.reshape(-1, kept.shape[-1]).clone()
    return new


MHA, EXTRA = "MHA, trained", f"MHA, {uptrain} more steps (reference)"
SCRATCH = "GQA-2, trained from scratch"
POOLED, POOLED_UP = "MHA -> GQA-2, mean-pooled", f"  ... then {uptrain} steps"
FIRST, FIRST_UP = "MHA -> GQA-2, first head kept", f"  ... then {uptrain} steps (first head)"
POOLED_UP4 = f"  ... or {4 * uptrain} steps ({4 * uptrain / steps_total:.0%})"
keys = (MHA, EXTRA, SCRATCH, POOLED, POOLED_UP, POOLED_UP4, FIRST, FIRST_UP)
results = {key: [] for key in keys}
for seed in seeds:
    shape = dict(n_layer=2, n_head=4, d_model=64, block_size=64, pos="rope", norm="rmsnorm")
    shape |= dict(mlp="swiglu", d_ff=170, bias=False)
    torch.manual_seed(seed)
    mha = fit(GPT(small_gpt(**shape)), 0, cfg_train, seed)
    results[MHA].append(score(mha))
    torch.manual_seed(seed)
    results[SCRATCH].append(score(fit(GPT(small_gpt(**shape, n_kv_head=2)), 0, cfg_train, seed)))
    for convert, before, after in (
        (to_gqa, POOLED, POOLED_UP),
        (first_head_of_each_group, FIRST, FIRST_UP),
    ):
        converted = convert(mha, 2)
        results[before].append(score(converted))
        results[after].append(score(fit(converted, steps_total, cfg_up, seed)))
    results[POOLED_UP4].append(score(fit(to_gqa(mha, 2), steps_total, cfg_up4, seed)))
    longer = GPT(small_gpt(**shape))
    longer.load_state_dict(mha.state_dict())
    results[EXTRA].append(score(fit(longer, steps_total, cfg_up, seed)))
report.say(
    f"  2 layers, 4 query heads, {steps_total} steps, then {uptrain} more ({uptrain / steps_total:.0%}); "
    f"validation loss in nats per byte over {len(seeds)} seed(s)"
)
report.table(
    ["model", "mean loss", "range over seeds"],
    [[k, st.mean(v), f"{min(v):.3f} - {max(v):.3f}"] for k, v in results.items()],
)
mean = {k: st.mean(v) for k, v in results.items()}
if args.quick:
    report.skip("conversion results", f"{steps_total} training steps are too few to compare")
else:
    gap = mean[POOLED] - mean[FIRST]
    noise = max(max(results[k]) - min(results[k]) for k in (POOLED, FIRST))
    if abs(gap) > noise:
        better = "mean-pooling" if gap < 0 else "keeping the first head"
        report.say(
            f"  {better} starts {abs(gap):.3f} nats ahead, more than the seed range {noise:.3f}"
        )
    else:
        report.say(
            f"  The two starting points differ by {abs(gap):.3f} nats, less than the seed-to-seed range "
            f"{noise:.3f}:\n  this experiment cannot say which is better. (The GQA paper reports "
            "mean-pooling ahead on T5-XXL; nothing here contradicts that.)"
        )
    for before, after in ((POOLED, POOLED_UP), (FIRST, FIRST_UP)):
        report.check(
            f"{uptrain} steps repair most of the damage of the conversion ({before.strip()})",
            mean[before] - mean[after] > 0.5 * (mean[before] - mean[MHA]),
            f"loss {mean[before]:.3f} -> {mean[after]:.3f}, MHA {mean[MHA]:.3f}",
        )
    report.check(
        "the more steps after the conversion, the lower the loss",
        mean[POOLED] > mean[POOLED_UP] > mean[POOLED_UP4],
        f"{mean[POOLED]:.3f} > {mean[POOLED_UP]:.3f} > {mean[POOLED_UP4]:.3f}",
    )
    short = mean[POOLED_UP4] - mean[SCRATCH]
    report.say(
        f"  After {4 * uptrain} steps the converted model is {short:+.3f} nats from the GQA-2 model "
        f"trained from scratch (range of the latter {min(results[SCRATCH]):.3f} - "
        f"{max(results[SCRATCH]):.3f}); the checkpoint it started from cost {steps_total} steps."
    )

report.finish()
