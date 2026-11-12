"""Lab: quantization. The ledger, the grid, and four experiments on a real (small) model.

Course page: https://dmxsf.com/wiki/inference/quantization/

The page runs its experiments on Qwen2.5-0.5B and measures perplexity. This lab runs the same
algorithms (round to nearest, SmoothQuant, an AWQ-style scale search, GPTQ, a quantized KV cache) on
the course's 0.86M-parameter byte-level GPT and measures bits per byte on held-out Shakespeare. The
ledger (section 1) and the grid (section 2) reproduce the page's numbers exactly. A model this small
does not grow the activation and key outliers that make the big models hard to quantize, so sections
4 to 6 also build outliers on demand: copies of the model that compute the same function with one
channel rescaled. Those results are about a mechanism, and are labelled as constructed.

All quantization here is fake quantization: weights are rounded to an integer grid and multiplied back,
which measures accuracy loss, not speed.

    uv run python labs/inference/quantization/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import torch

from dmxsf import lab, models
from dmxsf.architecture.configs import BY_NAME, kv_bytes_per_token
from dmxsf.data import ByteCorpus
from dmxsf.inference.quantization import (
    QuantKVCache,
    Target,
    act_quant,
    awq,
    bits_per_byte_cached,
    collect_stats,
    effective_bits,
    fake_quant,
    gptq,
    output_error,
    rtn,
    smooth_scales,
    sqnr_db,
    widget_weights,
    with_activation_outlier,
    with_key_outlier,
)

BLOCK = models.BLOCK


def windows(data: np.ndarray, n: int) -> torch.Tensor:
    """n evenly spaced windows of BLOCK bytes: the calibration batches."""
    starts = np.linspace(0, len(data) - BLOCK - 2, n).astype(int)
    return torch.from_numpy(np.stack([data[s : s + BLOCK] for s in starts]).astype(np.int64))


def main() -> None:
    args, report = lab.start(
        "Quantization: the ledger, the grid, weights, activations, calibration, the KV cache",
        wiki=["inference/quantization"],
    )
    QUICK = args.quick
    torch.manual_seed(args.seed)
    EVAL_WINDOWS = 16 if QUICK else 128

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.check(name, ok, detail)

    def full_observe(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.observe(name, ok, detail)

    # -----------------------------------------------------------------------------------------
    report.section("1. The ledger: Llama 3 8B on one 80 GB card (exact)")
    llama = BY_NAME["Llama 3 8B"]
    cfg = llama.config
    d, ff, layers = cfg.d_model, cfg.d_ff, cfg.n_layer
    kv_dim = cfg.kv_heads * cfg.dim_per_head
    n_lin = layers * (2 * d * d + 2 * d * kv_dim + 3 * d * ff)  # q, o | k, v | gate, up, down
    n_emb = n_head = cfg.vocab_size * d
    n_norm = (2 * layers + 1) * d
    report.say(
        f"  parameters: {llama.params / 1e9:.2f}B in all; decoder linear layers {n_lin / 1e9:.2f}B,"
        f" input embedding {n_emb / 1e9:.2f}B, output layer {n_head / 1e9:.2f}B."
    )
    report.check(
        "the three groups and the norm gains add up to the model's total",
        n_lin + n_emb + n_head + n_norm == llama.params,
        f"{llama.params:,}",
    )
    bandwidth, memory = 2e12, 0.9 * 80e9
    kv_tok = 2 * layers * cfg.kv_heads * cfg.dim_per_head  # elements per token
    report.say(
        f"  KV cache: {kv_tok * 2} bytes per token (BF16), {kv_tok * 2 * 8192 / 1e9:.2f} GB for one"
        " sequence of 8,192 tokens."
    )
    rows, got = [], {}
    for tag, w_bits, kv_bits in (
        ("BF16 + BF16", 16, 16),
        ("INT4 + BF16", 4.125, 16),
        ("INT4 + INT8", 4.125, 8.125),
    ):
        lin = n_lin * w_bits / 8  # only the decoder's linear layers are quantized
        weights = lin + (n_emb + n_head) * 2
        read = lin + n_head * 2  # per decoded token: linear layers and the output layer
        seqs = (memory - weights) // (kv_tok * kv_bits / 8 * 8192)
        got[tag] = (
            round(weights / 1e9, 1),
            round(read / 1e9, 1),
            round(bandwidth / read),
            int(seqs),
        )
        rows.append(
            [
                tag,
                f"{weights / 1e9:.1f}",
                f"{read / 1e9:.1f}",
                f"{bandwidth / read:.0f}",
                f"{seqs:.0f}",
            ]
        )
    report.table(
        ["weights + KV", "weights GB", "read per step GB", "tokens/s bound", "8K sequences"], rows
    )
    report.check(
        "the page's table: 16.1 / 5.7 / 5.7 GB, 15.0 / 4.6 / 4.6 GB per step, 133 / 430 / 430 tokens/s,"
        " 52 / 61 / 121 sequences",
        got
        == {
            "BF16 + BF16": (16.1, 15.0, 133, 52),
            "INT4 + BF16": (5.7, 4.6, 430, 61),
            "INT4 + INT8": (5.7, 4.6, 430, 121),
        },
        str(got),
    )
    report.say(
        "  weights shrink 2.8x, not 4x (the embeddings stay BF16); the decode bound rises 3.2x; the number\n"
        "  of sequences barely moves until the cache itself is quantized."
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. Rounding to a grid: the page's widget, number for number")
    x, x_out = widget_weights(), widget_weights(outlier=True)
    sym = {b: sqnr_db(x, fake_quant(x, b, "tensor")) for b in range(2, 9)}
    report.say("  1,024 fixed-seed normal weights (the widget's own sample), symmetric, one scale:")
    report.table(
        ["bits", "SQNR dB", "gain over previous bit"],
        [[b, f"{sym[b]:.1f}", "" if b == 2 else f"{sym[b] - sym[b - 1]:+.1f}"] for b in sym],
    )
    report.check(
        "SQNR for 2 to 8 bits is 1.1, 9.6, 16.9, 23.5, 29.8, 35.6, 42.1 dB",
        [round(sym[b], 1) for b in range(2, 9)] == [1.1, 9.6, 16.9, 23.5, 29.8, 35.6, 42.1],
    )
    groups = [
        ("whole tensor", "tensor"),
        ("g256", 256),
        ("g128", 128),
        ("g64", 64),
        ("g32", 32),
        ("g16", 16),
    ]
    rows, with_outlier = [], {}
    for label, g in groups:
        plain = sqnr_db(x, fake_quant(x, 4, g))
        with_outlier[label] = sqnr_db(x_out, fake_quant(x_out, 4, g))
        bits = effective_bits(4, [(1024,)], g)
        rows.append([label, f"{bits:.3f}", f"{plain:.1f}", f"{with_outlier[label]:.1f}"])
    report.say(
        "\n  INT4, symmetric, by group size; the last column replaces one weight by 20 (20 sigma):"
    )
    report.table(["scale shared by", "effective bits", "SQNR dB", "SQNR dB with an outlier"], rows)
    report.check(
        "one outlier takes INT4 from 16.9 dB to 3.7 dB; groups of 128 / 64 / 16 win back 11.6 / 14.7 / 19.1 dB",
        [round(with_outlier[k], 1) for k in ("whole tensor", "g128", "g64", "g16")]
        == [3.7, 11.6, 14.7, 19.1],
    )
    asym = fake_quant(x, 4, 128, symmetric=False)
    sym128 = fake_quant(x, 4, 128)
    report.say(
        f"\n  asymmetric against symmetric at INT4, g128: {sqnr_db(x, asym):.1f} dB"
        f" ({effective_bits(4, [(8, 128)], 128, symmetric=False):.3f} bits)"
        f" against {sqnr_db(x, sym128):.1f} dB ({effective_bits(4, [(8, 128)], 128):.3f} bits)."
    )
    report.check(
        "asymmetric: 19.6 dB at 4.156 bits; symmetric: 18.1 dB at 4.125 bits",
        round(sqnr_db(x, asym), 1) == 19.6
        and round(sqnr_db(x, sym128), 1) == 18.1
        and round(effective_bits(4, [(8, 128)], 128, symmetric=False), 3) == 4.156,
    )
    seven_b = 7e9
    report.check(
        "a 7B model: 14 GB in BF16, about 3.6 GB at 4.125 bits per weight",
        seven_b * 2 / 1e9 == 14.0 and round(seven_b * 4.125 / 8 / 1e9, 1) == 3.6,
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. Round to nearest on a real model: bits, granularity, sensitivity")
    corpus = ByteCorpus.shakespeare()
    model = models.pretrained("base", quick=QUICK, seed=args.seed).eval()
    t = Target(model)
    shapes = [tuple(w.shape) for w in t.orig]

    def bpb(m) -> float:
        return models.shakespeare_bpb(m, windows=EVAL_WINDOWS)

    base = bpb(model)
    n_all = model.num_params()
    report.say(
        f"  the course's byte-level GPT: {n_all:,} parameters, of which {t.numel:,} are in the"
        f" {len(t.linears)} decoder linear layers; the tied embedding has {model.tok_emb.weight.numel():,}"
        f" ({model.tok_emb.weight.numel() / n_all:.1%}).\n"
        f"  held-out Shakespeare: {base:.4f} bits/byte in float32. The page's metric is perplexity on"
        " WikiText-2;\n  bits per byte is its logarithm per character, so 'delta' below is a log-ratio."
    )

    grid = [(8, "tensor"), (8, "row"), (4, "tensor"), (4, "row"), (4, 64), (4, 32), (4, 16), (4, 8)]
    grid += [(3, 32), (3, 16), (2, 16)]
    rows, delta, eff = [], {}, {}
    for bits, g in grid:
        with t.quantized(lambda i, w, bits=bits, g=g: fake_quant(w, bits, g)):
            value = bpb(model)
        zero = float(np.mean([(fake_quant(w, bits, g) == 0).float().mean().item() for w in t.orig]))
        delta[bits, g] = value - base
        eff[bits, g] = effective_bits(bits, shapes, g)
        rows.append(
            [
                f"INT{bits} {g}",
                f"{eff[bits, g]:.3f}",
                f"{t.numel * eff[bits, g] / 8 / 1024:.0f}",
                f"{value:.4f}",
                f"{value - base:+.4f}",
                f"{zero:.1%}",
            ]
        )
    report.table(
        ["scheme", "effective bits", "linear weights KiB", "bits/byte", "delta", "rounded to 0"],
        rows,
    )
    ratio = torch.stack([w.abs().max() / w.abs().median() for w in t.orig])
    report.say(
        f"  max|w| / median|w| per layer: median {ratio.median():.1f}, largest {ratio.max():.1f}"
        " (the page's 0.5B model: 27 and 117)."
    )
    full_check(
        "INT8 with one scale per row is invisible",
        abs(delta[8, "row"]) < 0.005,
        f"{delta[8, 'row']:+.4f} bits/byte",
    )
    full_check(
        "INT4 with one scale for the whole tensor is more than twice as bad as one per row",
        delta[4, "tensor"] > 2 * delta[4, "row"],
        f"{delta[4, 'tensor']:+.4f} against {delta[4, 'row']:+.4f}",
    )
    chain = [(4, "row"), (4, 64), (4, 32), (4, 16), (4, 8)]
    slopes = [
        (delta[a] - delta[b]) / (eff[b] - eff[a]) for a, b in itertools.pairwise(chain)
    ]  # bits/byte saved per extra stored bit
    full_check(
        "smaller groups help, and the first extra bits buy more than the last ones",
        # neighbouring sizes (64 against 32) can swap by half a thousandth of a bit: rounding noise
        delta[4, "row"] > delta[4, 32] > delta[4, 8] and slopes[0] > slopes[-1],
        "saved per extra bit: " + ", ".join(f"{s:.3f}" for s in slopes),
    )
    # seen over eight trained models (six on an M2, two on Linux): INT3 costs 4.9x to 10x INT4 (g32),
    # INT2 loses 1.6 to 1.7 bits
    full_check(
        "round to nearest breaks below 4 bits: INT3 costs 3x INT4 (g32), INT2 loses half a bit per byte",
        delta[3, 32] > 3 * delta[4, 32] and delta[2, 16] > 0.5,
        f"INT3 {delta[3, 32]:+.3f}, INT4 {delta[4, 32]:+.3f}, INT2 {delta[2, 16]:+.3f}",
    )

    kinds = ["q_proj", "k_proj", "v_proj", "o_proj", "fc", "proj"]
    by_kind = {}
    for kind in kinds:
        with t.quantized(
            lambda i, w, kind=kind: (
                fake_quant(w, 4, 32) if t.names[i].endswith("." + kind) else w.clone()
            )
        ):
            by_kind[kind] = bpb(model) - base
    by_layer = {}
    for layer in range(model.cfg.n_layer):
        with t.quantized(
            lambda i, w, layer=layer: fake_quant(w, 4, 32) if t.layer_of(i) == layer else w.clone()
        ):
            by_layer[layer] = bpb(model) - base
    report.say("\n  INT4 g32 on one kind of layer at a time (delta bits/byte):")
    report.table(["layer kind", *kinds], [["delta", *[f"{by_kind[k]:+.4f}" for k in kinds]]])
    report.say("  ...and on one block at a time:")
    report.table(
        ["block", *[str(i) for i in by_layer]],
        [["delta", *[f"{v:+.4f}" for v in by_layer.values()]]],
    )
    emb = model.tok_emb.weight
    emb_orig = emb.data.clone()
    with t.quantized(lambda i, w: fake_quant(w, 4, 32)):
        linears_only = bpb(model) - base
        emb.data = fake_quant(emb_orig, 4, 32)  # the output layer shares this matrix
        both = bpb(model) - base
        emb.data = emb_orig
    report.say(
        f"\n  all 24 linear layers at INT4 g32: {linears_only:+.4f}; adding the tied embedding / output"
        f" matrix: {both:+.4f}\n  (the single-block rows above sum to {sum(by_layer.values()):+.4f}:"
        " roughly additive)."
    )
    full_check(
        "rounding the embedding alone costs more than rounding all 24 linear layers, with a fraction of the parameters",
        both - linears_only > linears_only,
        f"{both - linears_only:+.4f} against {linears_only:+.4f}; {model.tok_emb.weight.numel() / n_all:.1%} of the parameters",
    )
    size = t.numel * eff[4, 32] / 8 + (n_all - t.numel) * 2
    report.say(
        f"  INT4 g32 linear layers, the rest in 16 bits: {size / 1024:.0f} KiB, {size / (n_all * 2):.0%} of the"
        " 16-bit model (the page's 0.5B model: 46%)."
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. Activations: why they are harder, and an outlier you can dial")
    mid = model.cfg.n_layer // 2
    which = {
        "q/k/v input": f"blocks.{mid}.attn.q_proj",
        "o input": f"blocks.{mid}.attn.o_proj",
        "fc input": f"blocks.{mid}.mlp.fc",
        "proj input": f"blocks.{mid}.mlp.proj",
    }
    seen, handles = {}, []
    for key, name in which.items():
        handles.append(
            model.get_submodule(name).register_forward_pre_hook(
                lambda mod, a, key=key: seen.__setitem__(key, a[0][0].abs())
            )
        )
    with torch.no_grad():
        model(torch.from_numpy(corpus.val[:BLOCK].astype(np.int64))[None])
    for h in handles:
        h.remove()
    rows = []
    for key, a in seen.items():
        med = a.median(0).values
        top = int(med.argmax())
        rows.append(
            [
                key,
                f"{a.median():.3f}",
                f"{a.amax(1).median():.2f}",
                f"{a.max():.1f}",
                f"{med[top] / a.median():.1f}",
                f"{(a.argmax(1) == top).float().mean():.0%}",
            ]
        )
    report.say(f"  what enters four linear layers of block {mid} (one {BLOCK}-byte window):")
    report.table(
        [
            "input",
            "median |x|",
            "median of per-token max",
            "max",
            "top channel / median",
            "top channel is the max in",
        ],
        rows,
    )
    strongest = max(float(r[4]) for r in rows)
    report.say(
        "  the page's Qwen2.5-0.5B has a channel 21 times the median that is the largest entry in 44-58% of the\n"
        f"  tokens; here the strongest channel is {strongest:.1f} times the median."
    )

    def w8a8(m, t_m, per_token: bool) -> float:
        with t_m.quantized(lambda i, w: fake_quant(w, 8, "row")):
            handles = t_m.hook_inputs(lambda i, h: act_quant(h, per_token))
            try:
                return bpb(m) - base
            finally:
                for hd in handles:
                    hd.remove()

    with t.quantized(lambda i, w: fake_quant(w, 8, "row")):
        w8a16 = bpb(model) - base
    natural = {False: w8a8(model, t, False), True: w8a8(model, t, True)}
    report.say(
        f"\n  natural model, delta bits/byte: W8A16 {w8a16:+.4f}; W8A8 one scale per tensor"
        f" {natural[False]:+.4f}; one per token {natural[True]:+.4f}."
    )
    full_check(
        "on this small model activation INT8 is nearly free even with one scale per tensor: there is no outlier to pay for",
        natural[False] < 0.01,
        f"{natural[False]:+.4f} bits/byte",
    )

    cs = (1.0, 30.0) if QUICK else (1.0, 10.0, 30.0, 100.0)
    probe = torch.from_numpy(corpus.val[: 4 * BLOCK].astype(np.int64)).view(4, BLOCK)
    calib = windows(corpus.train, 16 if QUICK else 64)
    loud, stats = {}, {}
    for c in cs:
        loud[c] = with_activation_outlier(model, c).eval()
        t_c = Target(loud[c])
        stats[c] = (t_c, collect_stats(t_c, calib.split(16)))
    rows, equal_ok, drift = [], True, 0.0
    cost = {}
    for c in cs:
        m, (t_c, _) = loud[c], stats[c]
        with torch.no_grad():
            equal_ok &= (m(probe) - model(probe)).abs().max().item() < 1e-3
        drift = max(drift, abs(bpb(m) - base))
        with t_c.quantized(lambda i, w: fake_quant(w, 8, "row")):
            w16 = bpb(m) - base
        cost[c, "tensor"], cost[c, "token"] = w8a8(m, t_c, False), w8a8(m, t_c, True)
        rows.append(
            [f"{c:g}x", f"{w16:+.4f}", f"{cost[c, 'tensor']:+.4f}", f"{cost[c, 'token']:+.4f}"]
        )
    report.say(
        "\n  constructed outlier: one residual channel made c times larger (the model computes the same function):"
    )
    report.table(["channel scale", "W8A16", "W8A8 per tensor", "W8A8 per token"], rows)
    report.check(
        "the constructed models compute what the original does (max logit difference under 1e-3)",
        equal_ok and drift < 1e-3,
        f"bits/byte moved by at most {drift:.1e}",
    )
    series = [cost[c, "tensor"] for c in cs]
    full_check(
        "an outlier channel wrecks per-tensor activation INT8: the damage grows with its size, past a bit per byte at 100x",
        all(a < b for a, b in itertools.pairwise(series)) and series[-1] > 1.0,
        "delta " + ", ".join(f"{v:+.3f}" for v in series),
    )
    full_check(
        "one scale per token softens the damage but does not remove it",
        all(cost[c, "token"] < cost[c, "tensor"] for c in cs[1:]) and cost[cs[-1], "token"] > 0.05,
        f"at {cs[-1]:g}x: {cost[cs[-1], 'token']:+.3f} against {cost[cs[-1], 'tensor']:+.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("5. Calibration: SmoothQuant, an AWQ-style search, GPTQ")
    alphas = (0.5, 0.7, 0.9) if QUICK else (0.3, 0.5, 0.7, 0.8, 0.9, 1.0)

    def smoothed(m, t_m, st, alpha: float) -> float:
        scales = []

        def one(i, w):
            scales.append(smooth_scales(st[2][i], w, alpha))
            return fake_quant(w * scales[-1], 8, "row")

        with t_m.quantized(one):
            handles = t_m.hook_inputs(lambda i, h: act_quant(h / scales[i], False))
            try:
                return bpb(m) - base
            finally:
                for hd in handles:
                    hd.remove()

    c_demo = 30.0
    t_c, st_c = stats[c_demo]
    sweep = {a: smoothed(loud[c_demo], t_c, st_c, a) for a in alphas}
    report.say(
        f"  SmoothQuant on the {c_demo:g}x model, W8A8 with one scale per tensor (alpha is the knob):"
    )
    report.table(
        ["alpha", "delta bits/byte"],
        [
            ["none", f"{cost[c_demo, 'tensor']:+.4f}"],
            *[[f"{a}", f"{v:+.4f}"] for a, v in sweep.items()],
        ],
    )
    smooth_alpha = 0.7
    fixed = {c: smoothed(loud[c], stats[c][0], stats[c][1], smooth_alpha) for c in cs}
    report.say(
        f"  alpha fixed in advance at {smooth_alpha}, every outlier size: "
        + ", ".join(f"{c:g}x {fixed[c]:+.4f}" for c in cs)
    )
    full_check(
        "smoothing moves the outlier into the weights and recovers the loss: under 0.01 bits/byte at every size",
        all(v < 0.01 for v in fixed.values()),
        f"largest {max(fixed.values()):+.4f}; unsmoothed at {cs[-1]:g}x it was {cost[cs[-1], 'tensor']:+.3f}",
    )

    def run_method(m, t_m, st, fn, bits, group=32):
        rel, tic = [], time.perf_counter()

        def one(i, w):
            wq = fn(w, st[0][i], st[1][i], bits, group)
            rel.append(output_error(w, wq, st[0][i]) / ((w @ st[0][i]) * w).sum().item())
            return wq

        with t_m.quantized(one):
            value = bpb(m) - base
        return float(np.mean(rel)), value, time.perf_counter() - tic

    methods = (("RTN", rtn), ("AWQ-style", awq), ("GPTQ", gptq))
    t_nat = t
    st_nat = collect_stats(t_nat, calib.split(16))
    results = {}
    rows = []
    for label, (m, t_m, st) in (
        ("natural model", (model, t_nat, st_nat)),
        (f"{c_demo:g}x outlier", (loud[c_demo], *stats[c_demo])),
    ):
        for bits in (4, 3):
            for name, fn in methods:
                rel, value, secs = run_method(m, t_m, st, fn, bits)
                results[label, bits, name] = (rel, value)
                rows.append(
                    [label, f"INT{bits} g32", name, f"{rel:.4f}", f"{value:+.4f}", f"{secs:.1f}"]
                )
    report.say(
        "\n  weight quantization with calibration data (64 windows of the training set, groups of 32):"
    )
    report.table(
        ["model", "weights", "method", "relative output error", "delta bits/byte", "seconds"], rows
    )
    nat = "natural model"
    out = f"{c_demo:g}x outlier"
    full_check(
        "output error: GPTQ is the lowest and the AWQ-style search is never above round to nearest",
        all(
            results[lab, b, "GPTQ"][0]
            < results[lab, b, "AWQ-style"][0]
            <= results[lab, b, "RTN"][0]
            for lab in (nat, out)
            for b in (4, 3)
        ),
    )
    full_check(
        "GPTQ loses fewer bits per byte than round to nearest, at INT4 and INT3",
        all(results[nat, b, "GPTQ"][1] < results[nat, b, "RTN"][1] for b in (4, 3)),
        ", ".join(
            f"INT{b}: {results[nat, b, 'GPTQ'][1]:+.4f} against {results[nat, b, 'RTN'][1]:+.4f}"
            for b in (4, 3)
        ),
    )
    ratio = {b: results[nat, b, "RTN"][1] / max(results[nat, b, "GPTQ"][1], 1e-9) for b in (4, 3)}
    full_observe(  # seen: 1.9x to 8.3x at INT4, 2.1x to 3.2x at INT3, over eight trained models
        "...and by more than 2x at both",
        all(r > 2 for r in ratio.values()),
        f"RTN loses {ratio[4]:.1f}x as much at INT4 and {ratio[3]:.1f}x as much at INT3",
    )
    gain_nat = results[nat, 4, "AWQ-style"][1] / results[nat, 4, "RTN"][1]
    gain_out = results[out, 4, "AWQ-style"][1] / results[out, 4, "RTN"][1]
    verdict = {True: "helps", False: "does not help"}
    report.say(
        f"  the AWQ-style search at INT4 leaves {gain_nat:.2f}x of RTN's loss on the natural model and"
        f" {gain_out:.2f}x with the constructed outlier:\n  it {verdict[gain_nat < 0.8]} on the first and"
        f" {verdict[gain_out < 0.8]} on the second (below 0.8x counts as help)."
    )
    full_observe(  # seen: 0.34x to 0.60x on six trained models, 0.84x and 0.94x on two more
        "with an outlier channel the AWQ-style search cuts round-to-nearest's INT4 loss by 20% or more",
        gain_out < 0.8,
        f"{gain_out:.2f}x",
    )

    n_calib = (4, 16) if QUICK else (1, 4, 16, 64, 256)
    py = None if QUICK else ByteCorpus.python_code()
    variants = [(f"Shakespeare, {n} windows", windows(corpus.train, n)) for n in n_calib]
    if py is not None:
        variants.append(("Python source, 64 windows", windows(py.train, 64)))
        g = torch.Generator().manual_seed(args.seed + 1)
        variants.append(
            ("random bytes, 64 windows", torch.randint(0, 256, (64, BLOCK), generator=g))
        )
    rows, spread = [], {3: {}, 4: {}}
    for label, data in variants:
        st = collect_stats(t_nat, data.split(16))
        row = [label]
        for bits in (4, 3):
            _, value, _ = run_method(model, t_nat, st, gptq, bits)
            spread[bits][label] = value
            row.append(f"{value:+.4f}")
        rows.append(row)
    report.say(
        "\n  GPTQ, groups of 32, as the calibration data changes (delta bits/byte; RTN: "
        f"INT4 {results[nat, 4, 'RTN'][1]:+.4f}, INT3 {results[nat, 3, 'RTN'][1]:+.4f}):"
    )
    report.table(["calibration data", "INT4", "INT3"], rows)
    if not QUICK:
        three = spread[3]
        full_check(
            "INT3: more calibration data helps (256 windows against 1), and random bytes lose much of the gain",
            three["Shakespeare, 256 windows"] < three["Shakespeare, 1 windows"] - 0.02
            and three["random bytes, 64 windows"] > three["Shakespeare, 64 windows"] + 0.02,
            f"{three['Shakespeare, 1 windows']:+.3f} -> {three['Shakespeare, 256 windows']:+.3f}; random"
            f" {three['random bytes, 64 windows']:+.3f}",
        )
        four = list(spread[4].values())
        rtn4 = results[nat, 4, "RTN"][1]
        below = "all below" if max(four) < rtn4 else "not all below"
        report.say(
            f"  INT4 GPTQ ranges over {max(four) - min(four):.4f} bits/byte across the {len(four)} calibration"
            f" sets, {below} RTN's loss: differences of a few thousandths between two\n"
            "  quantized models of this size are not worth interpreting. Python source calibrates about as well"
            f" as Shakespeare here (INT3 {three['Python source, 64 windows']:+.4f} against"
            f" {three['Shakespeare, 64 windows']:+.4f}):\n  the page's warning about domain mismatch is not"
            " visible in this model, which is no evidence against it for a large one."
        )

    # -----------------------------------------------------------------------------------------
    report.section("6. The KV cache: keys and values")
    data = corpus.val
    n_layer = model.cfg.n_layer

    def cache_bpb(m, k_bits, k_chan, v_bits) -> float:
        return bits_per_byte_cached(
            m, data, lambda: QuantKVCache(n_layer, k_bits, k_chan, v_bits), BLOCK, EVAL_WINDOWS
        )

    cases = [
        ("K 16, V 16", 16, False, 16),
        ("K INT8, V INT8", 8, False, 8),
        ("K INT4, V INT4", 4, False, 4),
        ("K INT4, V 16", 4, False, 16),
        ("K 16, V INT4", 16, False, 4),
        ("K INT8 per channel, V INT8", 8, True, 8),
        ("K INT4 per channel, V INT4", 4, True, 4),
        ("K INT2, V INT2", 2, False, 2),
    ]
    head_dim = model.cfg.dim_per_head
    elems = n_layer * model.cfg.kv_heads * head_dim  # per token, for K (and again for V)
    rows, kv = [], {}
    for label, kb, kc, vb in cases:
        kv[label] = cache_bpb(model, kb, kc, vb) - base
        bits_k = kb if kb == 16 else kb + 16 / head_dim  # one fp16 scale per head_dim numbers
        bits_v = vb if vb == 16 else vb + 16 / head_dim
        rows.append([label, f"{(bits_k + bits_v) * elems / 8:.0f}", f"{kv[label]:+.4f}"])
    report.say(
        f"  natural model; the cache holds {2 * elems * 2:,} bytes per token in 16 bits"
        f" (formula: {kv_bytes_per_token(model.cfg, 2):,}):"
    )
    report.table(["cache format", "bytes per token, scales included", "delta bits/byte"], rows)
    full_check(
        "INT8 keys and values per token are free here",
        abs(kv["K INT8, V INT8"]) < 0.002,
        f"{kv['K INT8, V INT8']:+.4f}",
    )
    full_check(
        "at INT4 the keys are the more fragile half: rounding K alone costs more than 1.5x rounding V alone",
        kv["K INT4, V 16"] > 1.5 * kv["K 16, V INT4"],
        f"K {kv['K INT4, V 16']:+.4f}, V {kv['K 16, V INT4']:+.4f}",
    )
    full_check("INT2 is not usable", kv["K INT2, V INT2"] > 0.5, f"{kv['K INT2, V INT2']:+.3f}")
    report.say(
        f"  per-channel scales for K change INT4 from {kv['K INT4, V INT4']:+.4f} to"
        f" {kv['K INT4 per channel, V INT4']:+.4f} here (the page's model: perplexity 62.46 to 18.00)."
        "\n  What a per-channel scale is for are key channels that are far larger than the rest, so build them:"
    )
    ck = (1.0, 30.0) if QUICK else (1.0, 10.0, 30.0)
    rows, series, same_function = [], {}, True
    for c in ck:
        m = with_key_outlier(model, c).eval()
        with torch.no_grad():
            same_function &= (m(probe) - model(probe)).abs().max().item() < 1e-3
        series[c] = {
            "tok8": cache_bpb(m, 8, False, 16) - base,
            "chan8": cache_bpb(m, 8, True, 16) - base,
            "tok4": cache_bpb(m, 4, False, 16) - base,
            "chan4": cache_bpb(m, 4, True, 16) - base,
        }
        rows.append(
            [f"{c:g}x"] + [f"{series[c][k]:+.4f}" for k in ("tok8", "chan8", "tok4", "chan4")]
        )
    report.check(
        "every key-outlier copy computes the same function (max logit difference under 1e-3)",
        same_function,
    )
    report.say(
        "\n  constructed outlier: one key channel per head made c times larger, only K rounded (delta bits/byte):"
    )
    report.table(
        [
            "key channel scale",
            "INT8 per token",
            "INT8 per channel",
            "INT4 per token",
            "INT4 per channel",
        ],
        rows,
    )
    full_check(
        "a per-channel scale does not care how big the channel is (identical at every size); a per-token scale pays for it",
        max(abs(series[c]["chan4"] - series[1.0]["chan4"]) for c in ck) < 1e-3
        and series[ck[-1]]["tok4"] > 1.0
        and series[ck[-1]]["tok4"] > 50 * series[ck[-1]]["chan4"],
        f"INT4 at {ck[-1]:g}x: {series[ck[-1]]['tok4']:+.3f} per token, {series[ck[-1]]['chan4']:+.4f} per channel",
    )
    report.say(
        "  the per-channel numbers need a whole group of 32 tokens before they can be rounded: a real cache\n"
        "  keeps the newest tokens in 16 bits (KIVI); this evaluation rounds whole windows, so it is optimistic."
    )

    report.finish()


if __name__ == "__main__":
    main()
