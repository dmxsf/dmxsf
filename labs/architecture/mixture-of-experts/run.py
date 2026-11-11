"""Lab: mixture of experts, the routing, the balance and the parameter accounts.

Course page: https://dmxsf.com/wiki/architecture/mixture-of-experts/

    uv run python labs/architecture/mixture-of-experts/run.py [--quick]
"""

import math
import statistics as st
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from dmxsf import lab
from dmxsf.architecture.configs import BY_NAME, small_gpt
from dmxsf.architecture.deepseek import DEEPSEEK_V3, DeepSeekConfig, DeepSeekLM
from dmxsf.architecture.gpt import GPT
from dmxsf.architecture.layers import LayerNorm, RMSNorm
from dmxsf.architecture.moe import (
    MoELayer,
    active_params,
    aux_loss,
    balance_loss,
    capacity,
    moe_layers,
    moeify,
    route,
)
from dmxsf.data import ByteCorpus, get_batch
from dmxsf.pretraining.train import TrainConfig, bits_per_byte, lm_loss, make_optimizer, train

args, report = lab.start(
    "Mixture of experts: routing, load balance and the parameter accounts",
    wiki=["architecture/mixture-of-experts"],
)

# ---------------------------------------------------------------------------
report.section("1. A MoE layer computes only the tokens an expert was chosen for")
torch.manual_seed(args.seed)
T, d, f, E, k = (
    16,
    8,
    32,
    4,
    2,
)  # the course page's toy: tokens, width, expert width, experts, top-k
layer = MoELayer(d, f, E, k).double().eval()
x = torch.randn(1, T, d, dtype=torch.double)
with torch.no_grad():
    sparse, dense = layer(x), layer.dense_reference(x)
error = (sparse - dense).abs().max().item()
pairs = int(layer.load.sum())
report.say(f"  T={T} tokens, E={E} experts, k={k}: max |sparse - dense| = {error:.1e} (float64)")
report.say(f"  token-expert pairs: sparse {pairs}, dense {T * E}, ratio {pairs / (T * E):.2f}")
report.check("computing only the chosen tokens gives the dense result", error < 1e-12)
report.check("32 pairs against 64: the sparse form does k/E of the expert work", pairs == 32)


def flops(fn, *inputs):
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        fn(*inputs)
    return counter.get_total_flops()


router_flops = 2 * T * d * E
per_pair = 3 * 2 * d * f  # a SwiGLU expert: three matrix products of 2*d*f FLOPs per token
flops_sparse, flops_dense = flops(layer, x), flops(layer.dense_reference, x)
report.say(
    f"  measured FLOPs: sparse {flops_sparse:,}, dense {flops_dense:,}"
    f" (router {router_flops:,} + pairs x {per_pair:,})"
)
report.check(
    "FLOPs = router + (token-expert pairs) x 6 d f, counted by torch, not assumed",
    flops_sparse == router_flops + pairs * per_pair
    and flops_dense == router_flops + T * E * per_pair,
)

logits = torch.tensor([[2.0, 0.5, 1.2, -0.3, 1.7, 0.1]])
rows = []
for name, gate, top_k, scale in (
    ("Switch", "switch", 1, 1.0),
    ("Mixtral", "mixtral", 2, 1.0),
    ("DeepSeek-V3", "deepseek", 2, 2.5),
):
    r = route(logits, top_k, gate, route_scale=scale)
    gates = [round(g, 3) for g in r.gates[0].tolist()]
    rows.append([f"{name} (k={top_k})", r.experts[0].tolist(), gates, round(sum(gates), 3)])
report.say("\n  one token, six experts, logits [2.0, 0.5, 1.2, -0.3, 1.7, 0.1]:")
report.table(["gate", "experts", "weights", "sum"], rows)
report.check(
    "Mixtral's two weights sum to 1; DeepSeek-V3's sum to its route scale 2.5",
    rows[1][3] == 1.0 and rows[2][3] == 2.5,
)
plain = route(logits, 2, "deepseek", route_scale=2.5)
bias = torch.tensor([-1.0, 0, 0, 0, 0, 0])
biased = route(logits, 2, "deepseek", bias=bias, route_scale=2.5)
s = logits.sigmoid()[0, biased.experts[0]]
report.say(
    f"  a bias of -1 on expert 0 changes the choice {plain.experts[0].tolist()} -> "
    f"{biased.experts[0].tolist()}; the weights {[round(g, 3) for g in biased.gates[0].tolist()]}"
    " are still the raw scores, normalised"
)
report.check(
    "the bias decides who is chosen and never enters a weight",
    plain.experts[0].tolist() != biased.experts[0].tolist()
    and torch.allclose(biased.gates[0], 2.5 * s / s.sum()),
)
one = MoELayer(d, f, n_experts=1, top_k=1).double()
y = torch.randn(1, 5, d, dtype=torch.double)
report.check(
    "one expert and k=1 is an ordinary feed-forward network", torch.equal(one(y), one.experts[0](y))
)

# ---------------------------------------------------------------------------
report.section("2. Total and active parameters, counted on real modules")
GROUPS = ("routed experts", "shared experts", "attention", "dense FFN", "routers", "vocab matrices")


def breakdown(model):
    parts = dict.fromkeys((*GROUPS, "norm gains"), 0)
    norm_gains = {
        id(p) for m in model.modules() if isinstance(m, RMSNorm | LayerNorm) for p in m.parameters()
    }
    for name, p in model.named_parameters():
        if id(p) in norm_gains:
            key = "norm gains"
        elif name.startswith(("tok_emb", "lm_head")):
            key = "vocab matrices"
        elif ".experts." in name:
            key = "routed experts"
        elif ".shared." in name:
            key = "shared experts"
        elif ".router." in name:
            key = "routers"
        elif ".attn." in name:
            key = "attention"
        else:
            key = "dense FFN"
        parts[key] += p.numel()
    return parts


mistral_cfg = BY_NAME["Mistral 7B"].config
t0 = time.perf_counter()
with torch.device("meta"):  # shapes only: nothing is allocated
    mistral = GPT(mistral_cfg)
    mixtral = moeify(GPT(mistral_cfg), n_experts=8, top_k=2)
    mixtral_k1 = moeify(GPT(mistral_cfg), n_experts=8, top_k=1)
    v3 = DeepSeekLM(DEEPSEEK_V3)
report.say(
    f"  built Mixtral 8x7B and DeepSeek-V3 on the meta device in {time.perf_counter() - t0:.1f}s"
)
cells = {}
for name, model, embed in (
    ("Mixtral 8x7B", mixtral, 32000 * 4096),
    ("DeepSeek-V3", v3, 129280 * 7168),
):
    total, active = model.num_params(), active_params(model)
    parts = breakdown(model)
    cells[name] = (total, active, active - embed, parts)
    report.say(
        f"  {name}: {total:,} parameters; {active:,} active per token "
        f"({active / 1e9:.2f}B; {(active - embed) / 1e9:.2f}B without the input table)"
    )
    report.say(
        f"      the page's closed form has no norm gains ({parts['norm gains']:,} here): "
        f"without them {(active - embed - parts['norm gains']) / 1e9:.2f}B, "
        f"{(active - embed - parts['norm gains']) / (total - parts['norm gains']):.1%} of the total"
    )
report.check(
    "totals and active parameters of the course page: 46.70B / 12.88B and 671.03B / 37.55B",
    [round(cells[n][i] / 1e9, 2) for n in cells for i in (0, 1)] == [46.70, 12.88, 671.03, 37.55],
)
report.check(
    "without the input table and the norm gains: 12.75B and 36.62B, 27.3% and 5.5% of the total",
    [round((cells[n][2] - cells[n][3]["norm gains"]) / 1e9, 2) for n in cells] == [12.75, 36.62]
    and [
        round(
            100
            * (cells[n][2] - cells[n][3]["norm gains"])
            / (cells[n][0] - cells[n][3]["norm gains"]),
            1,
        )
        for n in cells
    ]
    == [27.3, 5.5],
    "counting the gains moves DeepSeek-V3 from 36.62B to 36.63B: the page's figure leaves them out",
)


def show(value, part):
    return f"{value / 1e6:,.0f}M" if part == "routers" else f"{value / 1e9:.2f}B"


rows = []
for part in GROUPS:
    row = [part]
    for name in cells:
        value = cells[name][3][part]
        row.append(show(value, part) + (f" ({value / cells[name][0]:.1%})" if value > 3e10 else ""))
    rows.append(row)
rows.append(["total", *(f"{cells[n][0] / 1e9:.2f}B" for n in cells)])
report.table(["part", "Mixtral 8x7B", "DeepSeek-V3"], rows)
mix, ds = cells["Mixtral 8x7B"][3], cells["DeepSeek-V3"][3]
report.check(
    "the breakdown of the course page (experts 45.10B / 653.91B, attention 1.34B / 11.41B, ...)",
    [round(mix["routed experts"] / 1e9, 2), round(ds["routed experts"] / 1e9, 2)] == [45.10, 653.91]
    and [round(mix["attention"] / 1e9, 2), round(ds["attention"] / 1e9, 2)] == [1.34, 11.41]
    and round(ds["shared experts"] / 1e9, 2) == 2.55
    and round(ds["dense FFN"] / 1e9, 2) == 1.19
    and [round(mix["vocab matrices"] / 1e9, 2), round(ds["vocab matrices"] / 1e9, 2)]
    == [0.26, 1.85]
    and [mix["routers"], round(ds["routers"] / 1e6)] == [1_048_576, 106],
)
report.say(
    f"  Mixtral's routers: {mix['routers']:,} parameters, {mix['routers'] / cells['Mixtral 8x7B'][0]:.4%}"
    " of the total"
)

expert = 3 * 4096 * 14336
shared_part = cells["Mixtral 8x7B"][0] - 32 * 8 * expert
report.say(
    f"  Mixtral = {shared_part / 1e9:.2f}B shared + 8 x {32 * expert / 1e9:.2f}B of experts; "
    f"eight full Mistral-7B models would be {8 * mistral.num_params() / 1e9:.1f}B"
)
k1_gap = active_params(mixtral_k1) - mistral.num_params()
report.say(
    f"  with k=1 the active parameters are {active_params(mixtral_k1):,}; Mistral 7B has {mistral.num_params():,}"
)
report.check(
    "'8x7B' is not 8 x 7B, and k=1 gives the dense Mistral 7B plus the routers",
    round(shared_part / 1e9, 2) == 1.61
    and round(8 * mistral.num_params() / 1e9, 1) == 57.9
    and k1_gap == mix["routers"],
    f"difference = {k1_gap:,} = the routers",
)
v3_expert = 3 * 7168 * 2048
tokens = 14.8e12
n_active = cells["DeepSeek-V3"][2]
dense_671 = 6 * cells["DeepSeek-V3"][0] * tokens
moe_flops = 6 * n_active * tokens
report.say(
    f"  6ND with 14.8T tokens: {moe_flops:.1e} FLOPs for the MoE, {dense_671:.1e} for a dense 671B "
    f"({dense_671 / moe_flops:.0f}x)"
)
report.check(
    "one routed expert 44.04M; the activated FFN width (8+1) x 2048 = the dense layers' 18432; 3.3e24 vs 6.0e25",
    round(v3_expert / 1e6, 2) == 44.04
    and (DEEPSEEK_V3.top_k + DEEPSEEK_V3.n_shared) * DEEPSEEK_V3.d_expert == DEEPSEEK_V3.d_dense
    and f"{moe_flops:.1e} {dense_671:.1e}" == "3.3e+24 6.0e+25",
)
report.check(
    "possible expert combinations: C(8,2) = 28 and C(256,8) = 4.1e14",
    math.comb(8, 2) == 28 and f"{math.comb(256, 8):.1e}" == "4.1e+14",
)

# ---------------------------------------------------------------------------
report.section("3. The same classes at toy size: a DeepSeek-V3-style model that runs")
toy = DeepSeekConfig(n_layer=4, first_dense=1, n_experts=16, top_k=4, n_group=4, topk_group=2)
torch.manual_seed(args.seed)
toy_model = DeepSeekLM(toy).eval()
ids = torch.randint(0, 256, (2, 24))
with torch.no_grad():
    full = toy_model(ids)
    cache = toy_model.new_cache()
    pieces = [toy_model(ids[:, :10], cache, absorb=True)]
    pieces += [toy_model(ids[:, t : t + 1], cache, absorb=True) for t in range(10, 24)]
gap = (torch.cat(pieces, dim=1) - full).abs().max().item()
report.say(
    f"  toy: {toy_model.num_params():,} parameters, {active_params(toy_model):,} active per token, "
    f"{len(moe_layers(toy_model))} MoE layers after {toy.first_dense} dense layer"
)
report.say(
    f"  decoding token by token from the latent cache: max logit error {gap:.1e}; the cache holds "
    f"{cache.numbers_per_token()} numbers per token = {toy.n_layer} x ({toy.d_c} + {toy.d_rope})"
)
report.check(
    "MLA attention + shared and routed experts: decoding from the cache equals the full pass",
    gap < 1e-4 and cache.numbers_per_token() == toy.n_layer * (toy.d_c + toy.d_rope),
)

gen = torch.Generator().manual_seed(args.seed)
raw = torch.randn(1000 if args.quick else 4000, 256, generator=gen)
free, limited = route(raw, 8, "deepseek"), route(raw, 8, "deepseek", n_group=8, topk_group=4)


def nodes_per_token(r):  # 8 groups of 32 experts, one group per node
    return torch.zeros(len(r.experts), 8).scatter_(1, r.experts // 32, 1.0).sum(dim=1)


exact = 8 * (1 - math.comb(224, 8) / math.comb(256, 8))
n_free, n_limited = nodes_per_token(free), nodes_per_token(limited)
report.say(
    f"  nodes a token is sent to (256 experts on 8 nodes, k=8): free routing {n_free.mean():.2f} on "
    f"average (exact {exact:.2f}), at most {int(n_free.max())}; node-limited {n_limited.mean():.2f}, "
    f"at most {int(n_limited.max())}"
)
report.check(
    "node-limited routing sends every token to at most 4 nodes",
    n_limited.max() <= 4 and n_free.max() > 4 and abs(n_free.mean() - exact) < 0.1,
)

# ---------------------------------------------------------------------------
report.section("4. Capacity: how many assignments an expert may take")
rows = []
for label, tokens, top_k, n_experts, factor in (
    ("course page", 256, 2, 8, 1.25),
    ("quiz", 512, 2, 16, 1.25),
    ("exercise 2", 1024, 8, 64, 1.25),
):
    rows.append(
        [
            label,
            tokens,
            top_k,
            n_experts,
            factor,
            tokens * top_k / n_experts,
            capacity(tokens, top_k, n_experts, factor),
        ]
    )
report.table(["case", "T", "k", "E", "factor", "average load", "capacity"], rows)
crowd = torch.zeros(285, 64)
crowd[:, 7] = 9.0  # all 285 tokens want expert 7
r = route(crowd, 1, "switch", cap=capacity(1024, 8, 64, 1.25))
dropped = int((~r.keep).sum())
report.say(
    f"  285 assignments for one expert with capacity 160: {int(r.keep.sum())} kept, {dropped} dropped"
)
report.check(
    "capacities 80 / 80 / 160, and 285 - 160 = 125 assignments dropped",
    [row[-1] for row in rows] == [80, 80, 160] and dropped == 125 and r.keep[:160].all(),
    "the first 160 arrivals are the ones kept",
)

# ---------------------------------------------------------------------------
report.section("5. The balance loss N * sum(f_i P_i)")
torch.manual_seed(args.seed)
n_experts, n_tokens = 8, 64
values = []
for _ in range(5):  # perfectly even assignments, arbitrary probabilities
    scores = torch.randn(n_tokens, n_experts) * 0.3
    scores[torch.arange(n_tokens), torch.arange(n_tokens) % n_experts] += 4.0
    values.append(balance_loss(route(scores, 1, "switch")).item())
collapsed = torch.zeros(n_tokens, n_experts)
collapsed[:, 0] = 20.0
w = torch.randn(n_tokens, n_experts, requires_grad=True)
r = route(w, 2, "switch")
r.probs.retain_grad()
loss = balance_loss(r)
loss.backward()
f_share = r.load.float() / r.load.sum()
expected_grad = n_experts * f_share / n_tokens
report.say(f"  perfectly even f, five random P: loss = {', '.join(f'{v:.6f}' for v in values)}")
report.say(
    f"  every token on one expert: loss = {balance_loss(route(collapsed, 1, 'switch')).item():.3f} (N = {n_experts})"
)
report.say(
    f"  random router, k=2: loss = {loss.item():.3f}; d loss / d p[t, i] = N f_i / T, f = {[round(v, 3) for v in f_share.tolist()]}"
)
report.check(
    "f uniform gives exactly 1 whatever P is; a collapsed router gives N",
    max(abs(v - 1) for v in values) < 1e-5
    and abs(balance_loss(route(collapsed, 1, "switch")).item() - n_experts) < 1e-3,
)
report.check(
    "the gradient reaches the router through P only: d loss / d p[t, i] = N f_i / T",
    torch.allclose(r.probs.grad, expected_grad.expand(n_tokens, -1), atol=1e-7)
    and not r.load.requires_grad,
)

# ---------------------------------------------------------------------------
report.section("6. Decoding: how many experts does a batch touch?")
trials = 200 if args.quick else 1000
rows, read_ok = [], []
for name, model, n_e, top_k, n_moe in (
    ("Mixtral 8x7B", mixtral, 8, 2, 32),
    ("DeepSeek-V3", v3, 256, 8, 58),
):
    per_expert = sum(p.numel() for p in model.blocks[-1].mlp.experts[0].parameters())
    embed = model.tok_emb.weight.numel()
    resident = model.num_params() - n_moe * n_e * per_expert - embed  # lookups read only a few rows
    for batch in (1, 4, 16, 64):
        formula = n_e * (1 - (1 - top_k / n_e) ** batch)
        gen = torch.Generator().manual_seed(args.seed)
        picks = route(torch.randn(trials * batch, n_e, generator=gen), top_k).experts
        hit = torch.zeros(trials, n_e).scatter_(1, picks.view(trials, -1), 1.0)
        simulated = hit.sum(dim=1).mean().item()
        fraction = (resident + n_moe * formula * per_expert) / model.num_params()
        rows.append([name, batch, f"{formula:.2f}", f"{simulated:.2f}", f"{fraction:.1%}"])
        read_ok.append(abs(simulated - formula) < 0.03 * n_e + 0.1)
report.table(["model", "batch", "experts used / layer", "simulated", "weights read"], rows)
report.check(
    "the course page's table: 2.00 / 5.47 / 7.92 / 8.00 and 8.00 / 30.53 / 101.96 / 222.44 experts",
    [r[2] for r in rows] == ["2.00", "5.47", "7.92", "8.00", "8.00", "30.53", "101.96", "222.44"],
)
report.check(
    "weights read per decoding step: 27.3% / 69.2% / 98.8% / 99.7% and 5.5% / 14.0% / 41.2% / 87.1%",
    [r[4] for r in rows] == ["27.3%", "69.2%", "98.8%", "99.7%", "5.5%", "14.0%", "41.2%", "87.1%"],
)
report.check("a Monte Carlo over random routers agrees with the formula", all(read_ok))
exact_half = math.log(0.5) / math.log(31 / 32)
need = math.ceil(exact_half)
at_21, at_22 = (256 * (1 - (31 / 32) ** b) for b in (21, 22))
report.say(
    f"  exercise 3: more than half of the 256 experts need B >= {exact_half:.1f}: B=21 -> {at_21:.1f}, B=22 -> {at_22:.1f}"
)
report.check("B = 22 (B = 21 gives 124.6)", need == 22 and round(at_21, 1) == 124.6 and at_22 > 128)
gen = torch.Generator().manual_seed(args.seed)
tilt_logits = 0.8 * torch.randn(256, generator=gen) + torch.randn(trials * 64, 256, generator=gen)
skewed = route(tilt_logits, 8).experts
used = torch.zeros(trials, 256).scatter_(1, skewed.view(trials, -1), 1.0).sum(dim=1).mean().item()
report.say(
    f"  with a popularity-skewed router, a batch of 64 touches {used:.1f} experts, not 222.4: skew only helps"
)
report.check(
    "non-uniform routing touches fewer experts than the uniform estimate", used < 222.44 - 5
)

# ---------------------------------------------------------------------------
report.section("7. Imbalance, capacity and bias balancing on synthetic router scores")


def router_layer(n_experts, top_k, factor):
    """A MoE layer whose router is the identity: the scores you feed are the router logits."""
    moe = MoELayer(n_experts, 4, n_experts, top_k, gate="deepseek", capacity_factor=factor)
    with torch.no_grad():
        moe.router.weight.copy_(torch.eye(n_experts))
    return moe.eval()


def popularity(n_experts, seed):
    g = torch.Generator().manual_seed(seed + 7)
    p = torch.randn(n_experts, generator=g)
    return (p - p.mean()) / p.std()


def simulate(tilt, gamma, steps, n_experts=8, top_k=2, tokens=256, factor=1.25):
    """Route fresh batches of scores = tilt * popularity + noise; with gamma, balance the bias."""
    moe = router_layer(n_experts, top_k, factor)
    pop = popularity(n_experts, args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    stats = []
    for _ in range(steps):
        scores = (tilt * pop + torch.randn(tokens, n_experts, generator=gen))[None]
        with torch.no_grad():
            moe(scores)
        load = moe.load.float()
        stats.append(
            (
                (load.max() / load.mean()).item(),
                moe.dropped / (tokens * top_k),
                moe.aux_loss.item(),
            )
        )
        if gamma:
            moe.update_bias(gamma)
    return moe, pop, stats


def mean_of(stats, column, window=None):
    return st.mean(s[column] for s in (stats if window is None else stats[window]))


reps = 8 if args.quick else 40
report.say(
    f"  8 experts, k=2, 256 tokens per batch, capacity factor 1.25; mean over {reps} batches\n"
)
rows, plain = [], {}
for tilt in (0.0, 0.4, 1.0):
    plain[tilt] = simulate(tilt, 0.0, reps)[2]
    rows.append(
        [
            f"{tilt:g}",
            mean_of(plain[tilt], 0),
            f"{mean_of(plain[tilt], 1):.1%}",
            mean_of(plain[tilt], 2),
        ]
    )
report.table(["tilt", "busiest / average", "dropped", "balance loss"], rows)
report.check(
    "more preference, more imbalance: busiest expert, drops and balance loss all rise with the tilt",
    all(rows[i][1] < rows[i + 1][1] and rows[i][3] < rows[i + 1][3] for i in range(2)),
)
floor = {}
for tokens in (64, 256, 1024):
    floor[tokens] = mean_of(simulate(0.0, 0.0, reps, tokens=tokens)[2], 0)
report.say(
    "  with no preference at all the busiest expert is still above average, from sampling alone: "
    + ", ".join(f"{floor[t]:.2f}x at {t} tokens" for t in floor)
    + "  (the course page's single runs: 1.31x, 1.16x, 1.09x)"
)
report.check(
    "that sampling floor shrinks with the batch and matches the page within 0.1",
    floor[64] > floor[256] > floor[1024]
    and all(abs(floor[t] - p) < 0.1 for t, p in zip(floor, (1.31, 1.16, 1.09), strict=True)),
)

ladder = {c: mean_of(simulate(0.4, 0.0, reps, factor=c)[2], 1) for c in (1.0, 1.25, 1.5, 2.0)}
report.say(
    "  tilt 0.4, dropped assignments by capacity factor: "
    + ", ".join(f"{c}: {v:.1%}" for c, v in ladder.items())
)
top1 = mean_of(simulate(0.4, 0.0, reps, top_k=1)[2], 1)
report.say(
    f"  the same router with k=1 (Switch) at factor 1.25: {top1:.1%} dropped, against {ladder[1.25]:.1%} for k=2"
)
report.check(
    "a larger capacity factor drops fewer assignments; k=1 drops more than k=2",
    list(ladder.values()) == sorted(ladder.values(), reverse=True) and top1 > ladder[1.25],
)

steps = 20 if args.quick else 200
report.say(
    f"\n  bias balancing (DeepSeek-V3 rule, scores are sigmoids), {steps} steps, fresh batch each step:"
)
rows, runs = [], {}
for tilt in (0.4, 1.5):
    for gamma in (0.001, 0.01, 0.05, 0.2):
        moe, pop, stats = simulate(tilt, gamma, steps)
        runs[tilt, gamma] = (moe, pop, stats)
        tail = stats[-steps // 5 :]
        rows.append(
            [
                f"{tilt:g}",
                f"{gamma:g}",
                mean_of(stats, 0, slice(max(0, steps // 8 - 5), steps // 8)),
                mean_of(stats, 0, slice(max(0, steps // 2 - 5), steps // 2)),
                mean_of(tail, 0),
                st.pstdev(s[0] for s in tail),
                f"{mean_of(tail, 1):.1%}",
            ]
        )
report.table(
    ["tilt", "gamma", f"@{steps // 8}", f"@{steps // 2}", "last 20%", "jitter", "dropped"],
    rows,
)
report.say(
    "  (columns: busiest / average expert load; jitter = its standard deviation in the last 20%)"
)
if args.quick:
    report.skip("convergence of the bias", f"{steps} steps are too few")
else:
    moe, pop, stats = runs[0.4, 0.01]
    rank = lambda v: v.argsort().argsort().float()  # noqa: E731
    rho = torch.corrcoef(torch.stack([rank(moe.route_bias), rank(pop)]))[0, 1].item()
    report.say(
        f"  tilt 0.4, gamma 0.01: Spearman correlation between the learned bias and the popularity "
        f"{rho:.2f}; most popular expert {int(pop.argmax())}, lowest bias {int(moe.route_bias.argmin())}"
    )
    report.check(
        "a reasonable step reaches the sampling floor without any loss term",
        mean_of(runs[0.4, 0.01][2], 0, slice(-40, None)) < 1.3
        and mean_of(runs[0.4, 0.01][2], 1, slice(-40, None)) < 0.01
        and mean_of(plain[0.4], 1) > 0.05,
        f"{mean_of(runs[0.4, 0.01][2], 0, slice(-40, None)):.2f}x against {floor[256]:.2f}x with no preference",
    )
    report.check("the learned bias opposes the popularity", rho < -0.9)
    report.check(
        "a step that is too small is slow (tilt 1.5, gamma 0.001: still above 2.5x after 100 steps)",
        mean_of(runs[1.5, 0.001][2], 0, slice(95, 100)) > 2.5,
        f"{mean_of(runs[1.5, 0.001][2], 0, slice(95, 100)):.2f}x",
    )
    report.check(
        "a step that is too large oscillates and is worse than no bias at all (tilt 0.4, gamma 0.2)",
        mean_of(runs[0.4, 0.2][2], 0, slice(-40, None)) > mean_of(plain[0.4], 0),
        f"{mean_of(runs[0.4, 0.2][2], 0, slice(-40, None)):.2f}x against {mean_of(plain[0.4], 0):.2f}x",
    )
    big = [
        simulate(tilt, g, 100, n_experts=64, top_k=8, tokens=1024)
        for tilt, g in ((0.4, 0.0), (0.4, 0.005))
    ]
    report.say(
        f"  64 experts, k=8, 1024 tokens, tilt 0.4: busiest expert {mean_of(big[0][2], 0):.2f}x and "
        f"{mean_of(big[0][2], 1):.1%} dropped with no bias; after 100 steps of gamma 0.005: "
        f"{mean_of(big[1][2], 0, slice(-20, None)):.2f}x and {mean_of(big[1][2], 1, slice(-20, None)):.1%}"
    )
    report.check(
        "the same mechanism balances a DeepSeek-V3-shaped layer",
        mean_of(big[1][2], 0, slice(-20, None)) < 0.6 * mean_of(big[0][2], 0)
        and mean_of(big[1][2], 1, slice(-20, None)) < 0.01,
    )

# ---------------------------------------------------------------------------
report.section("8. Trained on bytes: a MoE against dense models of the same cost and the same size")
steps_total = 30 if args.quick else 500
seeds = [args.seed] if args.quick else [args.seed + i for i in range(3)]
corpus = ByteCorpus.shakespeare()
cfg_train = TrainConfig(steps=steps_total, peak_lr=3e-3, warmup=max(1, steps_total // 10))
shape = dict(n_layer=2, n_head=4, d_model=64, block_size=64, pos="rope", norm="rmsnorm")
shape |= dict(mlp="swiglu", bias=False)
VARIANTS = {  # name: (dense FFN width, (experts, expert width, top-k) or None, aux-loss weight)
    "dense, FFN 128": (128, None, 0.0),
    "MoE 8x64, top-2, aux 0.01": (128, (8, 64, 2), 0.01),
    "MoE 8x64, top-2, no aux": (128, (8, 64, 2), 0.0),
    "dense, FFN 512": (512, None, 0.0),
}
holdout = torch.from_numpy(
    np.stack([corpus.val[j * 700 : j * 700 + 65] for j in range(64)]).astype(np.int64)
)


def build(name, seed):
    d_ff, moe_spec, _ = VARIANTS[name]
    torch.manual_seed(seed)
    model = GPT(small_gpt(**shape, d_ff=d_ff))
    if moe_spec:
        n_e, d_e, top_k = moe_spec
        moeify(model, n_experts=n_e, top_k=top_k, d_expert=d_e)
    return model


def objective(alpha):
    def loss_fn(model, batch):
        return lm_loss(model, batch) + (alpha * aux_loss(model) if alpha else 0.0)

    return loss_fn


results = {name: [] for name in VARIANTS}
for seed in seeds:
    for name, (_, moe_spec, alpha) in VARIANTS.items():
        model = build(name, seed)
        opt = make_optimizer(model, cfg_train)
        t0 = time.perf_counter()
        train(
            model,
            opt,
            lambda step, seed=seed: get_batch(corpus.train, step, 32, 64, seed=seed),
            cfg_train,
            loss_fn=objective(alpha),
        )
        seconds = time.perf_counter() - t0
        loss = bits_per_byte(model, corpus.val, block_size=64, windows=150) * math.log(2)
        sample = holdout[:, :64]
        row = dict(
            loss=loss,
            ms=1000 * seconds / steps_total,
            flops=flops(model, sample[:1]) / 64,
            params=model.num_params(),
            active=active_params(model),
        )
        if moe_spec:
            with torch.no_grad():
                model.eval()
                model(sample)
            layers = moe_layers(model)
            cap = capacity(sample.numel(), moe_spec[2], moe_spec[0], 1.25)
            row["busiest"] = st.mean((m.load.max() / m.load.float().mean()).item() for m in layers)
            row["dropped"] = st.mean(
                (m.load - cap).clamp(min=0).sum().item() / m.load.sum().item() for m in layers
            )
        results[name].append(row)


def column(name, key):
    return [r[key] for r in results[name]]


rows = []
for name in VARIANTS:
    losses = column(name, "loss")
    rows.append(
        [
            name,
            f"{results[name][0]['params']:,}",
            f"{results[name][0]['active']:,}",
            f"{results[name][0]['flops'] / 1e3:,.0f}k",
            f"{st.mean(column(name, 'ms')):.1f}",
            st.mean(losses),
            f"{min(losses):.3f} - {max(losses):.3f}",
        ]
    )
report.say(
    f"  2 layers, d_model 64, {steps_total} steps, validation loss in nats per byte, {len(seeds)} seed(s)"
)
report.table(
    ["model", "params", "active", "FLOPs/token", "ms/step", "val loss", "range over seeds"], rows
)
moe_names = [n for n in VARIANTS if VARIANTS[n][1]]
report.table(
    ["MoE", "busiest / average load", "dropped at factor 1.25"],
    [[n, st.mean(column(n, "busiest")), f"{st.mean(column(n, 'dropped')):.1%}"] for n in moe_names],
)
a, b, c, dd = (st.mean(column(n, "loss")) for n in VARIANTS)
noise = max(max(column(n, "loss")) - min(column(n, "loss")) for n in VARIANTS)


def verdict(label, gap):
    if len(seeds) < 2:
        return f"  {label}: {gap:+.3f} nats (one seed: no way to tell noise from difference)"
    if abs(gap) > noise:
        return f"  {label}: {gap:+.3f} nats, larger than the seed range {noise:.3f}"
    return (
        f"  {label}: {gap:+.3f} nats, within the seed range {noise:.3f}: no difference we can claim"
    )


report.say(verdict("MoE (aux 0.01) minus dense FFN 128", b - a))
report.say(verdict("MoE (aux 0.01) minus dense FFN 512", b - dd))
report.say(verdict("MoE without aux minus MoE with aux", c - b))
dense_small = results["dense, FFN 128"][0]
moe_row = results["MoE 8x64, top-2, aux 0.01"][0]
router_params = 2 * 64 * 8
report.check(
    "same active parameters as the dense model, plus the routers; the total is that of the wide dense FFN",
    moe_row["active"] == dense_small["params"] + router_params
    and moe_row["params"] - results["dense, FFN 512"][0]["params"] == router_params,
    f"{moe_row['active']:,} = {dense_small['params']:,} + {router_params}",
)
report.check(
    "measured FLOPs per token: the MoE costs what the small dense model costs, the wide one more",
    moe_row["flops"] < 1.05 * dense_small["flops"]
    and results["dense, FFN 512"][0]["flops"] > 1.5 * dense_small["flops"],
    f"{moe_row['flops'] / dense_small['flops']:.3f}x and "
    f"{results['dense, FFN 512'][0]['flops'] / dense_small['flops']:.2f}x of the small dense model",
)
if args.quick:
    report.skip("what the trained models score", f"{steps_total} steps are too few")
else:
    report.check(
        "every variant learned (well below the 5.55 nats of a uniform guess)",
        max(a, b, c, dd) < 2.4,
        f"worst {max(a, b, c, dd):.3f}",
    )

report.finish()
