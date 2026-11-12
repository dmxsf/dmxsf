"""Lab: LoRA from scratch, compared with full fine-tuning, merged, and stored in 4 bits.

Course page: https://dmxsf.com/wiki/post-training/lora-finetuning/

The page fine-tunes Qwen2.5-0.5B on banking77. This lab implements the same pieces (LoRALinear,
merging, NF4) in dmxsf.post_training.lora and runs them on the byte-level GPT of dmxsf.models with
the generated intent task of dmxsf.post_training.intents. The parameter arithmetic is exact and
needs no model; the experiments are a different model and task, so only the mechanisms carry over.

    uv run python labs/post-training/lora-finetuning/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import copy
import time

import torch
from torch import nn

from dmxsf import lab, models
from dmxsf.post_training import intents
from dmxsf.post_training.lora import (
    LoRALinear,
    adapter_params,
    add_lora,
    merge_lora,
    nf4_bits_per_weight,
    nf4_roundtrip,
    trainable_params,
)
from dmxsf.post_training.sft import build_example, finetune, sample_batch

GIB = 2**30


def kept_share(change: torch.Tensor, dw: torch.Tensor) -> float:
    """Projection of the change that storage made to a weight onto the update dW: 1 keeps all of it."""
    return ((change * dw).sum() / (dw * dw).sum()).item()


def main() -> None:
    args, report = lab.start(
        "LoRA: the parameter bill, the mechanics, LoRA against full fine-tuning, merging, NF4",
        wiki=["post-training/lora-finetuning"],
    )
    QUICK = args.quick
    STEPS = 40 if QUICK else 300
    DEFAULT_K = 5 if QUICK else 20
    BATCH = 16

    # -----------------------------------------------------------------------------------------
    report.section("1. The parameter bill (exact, no model needed)")
    qwen = {
        "q_proj": (896, 896),
        "k_proj": (896, 128),
        "v_proj": (896, 128),
        "o_proj": (896, 896),
        "gate_proj": (896, 4864),
        "up_proj": (896, 4864),
        "down_proj": (4864, 896),
    }
    rows = [[n, f"{i} -> {o}", f"{16 * (i + o):,}"] for n, (i, o) in qwen.items()]
    per_layer = adapter_params(qwen, 16)
    report.say("  Qwen2.5-0.5B (hidden 896, MLP 4864, 24 layers, 2 KV heads of 64), r = 16:")
    report.table(["matrix", "d_in -> d_out", "LoRA parameters"], rows)
    report.check(
        "per layer 366,592; times 24 layers = 8,798,208 (the number peft reports)",
        per_layer == 366_592 and per_layer * 24 == 8_798_208,
        f"{per_layer:,} x 24 = {per_layer * 24:,}",
    )
    d, ff = 4096, 11008
    qv = adapter_params({"q": (d, d), "v": (d, d)}, 8) * 32
    llama = adapter_params(
        dict.fromkeys("qkvo", (d, d)) | {"gate": (d, ff), "up": (d, ff), "down": (ff, d)}, 16
    )
    report.check(
        "Llama 2 7B, r = 8 on q and v: 4,194,304 parameters, 0.06% of the model",
        qv == 4_194_304,
        f"{qv:,} = {qv / 6_738_415_616:.2%} of 6,738,415,616",
    )
    report.check(
        "Llama 2 7B, r = 16 on every linear layer: 39,976,960 parameters, 76.25 MiB in bf16",
        llama * 32 == 39_976_960 and abs(llama * 32 * 2 / 2**20 - 76.25) < 0.01,
        f"{llama * 32:,} parameters, {llama * 32 * 2 / 2**20:.2f} MiB",
    )

    # QLoRA's static memory for Llama 2 7B, the page's accounting
    linear = 32 * (4 * d * d + 3 * d * ff)
    other = 2 * 32000 * d + (2 * 32 + 1) * d
    plain, dq = nf4_bits_per_weight(), nf4_bits_per_weight(double_quant=True)
    bf16 = (linear + other) * 2 / GIB
    qlora = (linear * dq / 8 + other * 2) / GIB
    state = llama * 32 * 16 / GIB
    full = (linear + other) * 16 / GIB
    report.table(
        ["quantity", "value"],
        [
            ["bits per weight: NF4 / NF4 + double quantization", f"{plain:.3f} / {dq:.3f}"],
            ["base model, bf16 -> NF4 + double quant (GiB)", f"{bf16:.2f} -> {qlora:.2f}"],
            ["adapter + gradient + Adam states, r=16 (GiB)", f"{state:.2f}"],
            ["full fine-tuning, 16 bytes per parameter (GiB)", f"{full:.1f}"],
        ],
    )
    report.check(
        "QLoRA accounting: 12.55 -> 3.60 GiB base, 0.60 GiB state, against 100.4 GiB for full fine-tuning",
        (round(bf16, 2), round(qlora, 2), round(state, 2), round(full, 1))
        == (12.55, 3.60, 0.60, 100.4),
        f"{bf16:.2f} -> {qlora:.2f}, {state:.2f}, {full:.1f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. Mechanics on the pretrained byte model")
    base = models.pretrained("base", quick=QUICK, seed=args.seed)
    base_bpb = models.shakespeare_bpb(base)
    torch.manual_seed(args.seed)
    layer = base.blocks[0]
    rows = []
    for name, lin in (("attn.q_proj", layer.attn.q_proj), ("mlp.fc", layer.mlp.fc)):
        x = torch.randn(4, 10, lin.in_features)
        lora = LoRALinear(copy.deepcopy(lin), r=8, alpha=16)
        same = (lora(x) - lin(x)).abs().max().item()
        lora(x).pow(2).sum().backward()
        a_grad, b_grad = lora.A.grad.abs().max().item(), lora.B.grad.norm().item()
        zero = LoRALinear(copy.deepcopy(lin), r=8, alpha=16, zero_a=True)
        zero(x).pow(2).sum().backward()
        dead = max(zero.A.grad.abs().max().item(), zero.B.grad.abs().max().item())
        with torch.no_grad():
            lora.B.normal_(0, 0.02)
        y = lora(x)
        merged = nn.functional.linear(x, lora.merged(), lin.bias)
        rel = ((y - merged).abs().max() / y.abs().max()).item()
        rows.append(
            [
                name,
                f"{lin.in_features} -> {lin.out_features}",
                f"{same:.1f}",
                f"{a_grad:.1f}",
                f"{b_grad:.1f}",
                f"{dead:.1f}",
                f"{rel:.1e}",
                f"{8 * (lin.in_features + lin.out_features):,}",
            ]
        )
        ok = same == 0 and a_grad == 0 and b_grad > 0 and dead == 0 and rel < 1e-5
        report.check(
            f"{name}: zero start, dL/dA = 0 on step one, A = B = 0 never learns, merge exact",
            ok,
            f"|lora - base| {same}, max |dA| {a_grad}, |dB| {b_grad:.2f}, merged error {rel:.1e}",
        )
    report.table(
        [
            "layer",
            "shape",
            "max|lora-base| at init",
            "max|dA| step 1",
            "|dB| step 1",
            "max grad if A=B=0",
            "merge error (rel.)",
            "r(d_in+d_out)",
        ],
        rows,
    )

    # -----------------------------------------------------------------------------------------
    train_pairs = intents.make_dataset(DEFAULT_K, seed=1 + args.seed)
    test = intents.make_dataset(
        25 if not QUICK else 5, seed=99, exclude={t for t, _ in train_pairs}
    )
    exclude = {t for t, _ in test}

    def examples(pairs):
        return [
            build_example(list(intents.render(t)), list(intents.answer_bytes(y))) for t, y in pairs
        ]

    def quantize_base(model):
        """NF4 round trip of every linear layer inside the blocks (embeddings stay as they are)."""
        for block in model.blocks:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = nf4_roundtrip(m.weight.data)

    def run(
        mode="lora", *, lr=3e-3, r=8, alpha=16.0, targets="all", seed=0, data_seed=1, quant=False
    ):
        """One fine-tuning run from the pretrained model; returns scores and the trained model.

        ``seed`` and ``data_seed`` are relative to ``--seed``: every run shifts by the same amount.
        """
        seed, data_seed = seed + args.seed, data_seed + args.seed
        pairs = intents.make_dataset(DEFAULT_K, seed=data_seed, exclude=exclude)
        ex = examples(pairs)
        model = copy.deepcopy(base)
        if quant:
            quantize_base(model)
        torch.manual_seed(seed)
        if mode == "lora":
            add_lora(model, targets, r, alpha)
        t0 = time.perf_counter()
        losses = finetune(model, lambda s: sample_batch(ex, s, BATCH, seed), STEPS, lr)
        seconds = time.perf_counter() - t0
        out = intents.score(model, test)
        out |= {
            "bpb": models.shakespeare_bpb(model),
            "trainable": trainable_params(model)
            if mode == "lora"
            else sum(p.numel() for p in model.parameters()),
            "seconds": seconds,
            "loss": losses[-1],
            "model": model,
        }
        return out

    def fmt(res):
        return [f"{res['accuracy']:.1%}", f"{res['bpb']:.2f}"]

    # -----------------------------------------------------------------------------------------
    report.section("3. Learning rate: LoRA and full fine-tuning each have a window")
    lora_lrs = (3e-3,) if QUICK else (3e-4, 1e-3, 3e-3, 1e-2)
    full_lrs = (3e-4,) if QUICK else (3e-5, 1e-4, 3e-4, 1e-3)
    lora_runs = {lr: run("lora", lr=lr) for lr in lora_lrs}
    full_runs = {lr: run("full", lr=lr) for lr in full_lrs}
    rows = [
        ["LoRA r=8 (all linear)", f"{lr:g}", f"{res['trainable']:,}", *fmt(res)]
        for lr, res in lora_runs.items()
    ] + [
        ["full fine-tuning", f"{lr:g}", f"{res['trainable']:,}", *fmt(res)]
        for lr, res in full_runs.items()
    ]
    report.table(
        ["method", "learning rate", "trained parameters", "accuracy", "shakespeare bits/byte"], rows
    )
    report.say(f"  (the pretrained model: {base_bpb:.2f} bits/byte on held-out Shakespeare)")
    default = lora_runs[3e-3]
    extra = [run("lora", lr=3e-3, seed=sd) for sd in ((1,) if QUICK else (1, 2))]
    seeds_acc = [default["accuracy"]] + [e["accuracy"] for e in extra]
    spread = max(seeds_acc) - min(seeds_acc)
    report.say(
        "  the default LoRA (r=8, alpha=16, lr 3e-3) with other training seeds: "
        + ", ".join(f"{a:.1%}" for a in seeds_acc)
        + f" (range {spread * 100:.1f} points; smaller differences below are not findings)"
    )
    if QUICK:
        report.skip(
            "accuracy and forgetting both depend strongly on the learning rate",
            "needs the full run",
        )
    else:
        for name, runs in (("LoRA", lora_runs), ("full fine-tuning", full_runs)):
            accs = [r["accuracy"] for r in runs.values()]
            bpbs = [r["bpb"] for r in runs.values()]
            report.check(
                f"{name}: the learning rate moves accuracy and forgetting a lot (not a flat curve)",
                max(accs) - min(accs) > 0.1 and max(bpbs) - min(bpbs) > 0.5,
                f"accuracy {min(accs):.0%}-{max(accs):.0%}, bits/byte {min(bpbs):.2f}-{max(bpbs):.2f}",
            )
        best_lora = max(lora_runs.values(), key=lambda r: (r["accuracy"], -r["bpb"]))
        best_full = max(full_runs.values(), key=lambda r: (r["accuracy"], -r["bpb"]))
        report.say(
            f"  best LoRA: {best_lora['accuracy']:.1%} / {best_lora['bpb']:.2f} bits;"
            f" best full fine-tuning: {best_full['accuracy']:.1%} / {best_full['bpb']:.2f} bits"
        )

    # -----------------------------------------------------------------------------------------
    report.section("4. Rank and alpha: capacity, or just a bigger learning rate?")
    ranks = (1, 8) if QUICK else (1, 8, 64)
    rows, fixed, scaled = [], {}, {}
    for r in ranks:
        fixed[r] = default if r == 8 else run("lora", r=r, alpha=16.0)
        scaled[r] = default if r == 8 else run("lora", r=r, alpha=2.0 * r)
        rows.append(
            [r, f"{fixed[r]['trainable']:,}", *fmt(fixed[r]), f"{2 * r:g}", *fmt(scaled[r])]
        )
    report.table(
        [
            "rank",
            "adapter parameters",
            "alpha=16: accuracy",
            "bits/byte",
            "alpha=2r",
            "accuracy",
            "bits/byte",
        ],
        rows,
    )
    report.say(
        "  alpha / r is the multiplier on the update: with alpha fixed it shrinks as r grows, with\n"
        "  alpha = 2r it stays 2. Which column moves tells you whether rank matters, or the step size."
    )

    # -----------------------------------------------------------------------------------------
    report.section("5. Which matrices: position against parameter count")
    names = ("qv", "mlp", "all") if not QUICK else ("qv", "all")
    target_runs = {n: default if n == "all" else run("lora", targets=n) for n in names}
    report.table(
        ["adapted matrices", "adapter parameters", "accuracy", "shakespeare bits/byte"],
        [[n, f"{res['trainable']:,}", *fmt(res)] for n, res in target_runs.items()],
    )
    if QUICK:
        report.skip(
            "the adapter parameter counts follow r(d_in + d_out) per matrix", "needs the full run"
        )
    else:
        d, ffd = 128, 512
        expect = {
            "qv": 4 * 2 * 8 * (d + d),
            "mlp": 4 * 2 * 8 * (d + ffd),
            "all": 4 * (4 * 8 * (d + d) + 2 * 8 * (d + ffd)),
        }
        report.check(
            "adapter parameters equal r(d_in + d_out) summed over the adapted matrices",
            all(target_runs[n]["trainable"] == expect[n] for n in names),
            ", ".join(f"{n} {target_runs[n]['trainable']:,}" for n in names),
        )

    # -----------------------------------------------------------------------------------------
    report.section("6. How low-rank is what fine-tuning learns?")
    full_model = full_runs[3e-4]["model"]
    lora_model = default["model"]
    ranks_e = (1, 2, 4, 8, 16)

    def energy(delta):
        s2 = torch.linalg.svdvals(delta.double()) ** 2
        cum = s2.cumsum(0) / s2.sum()
        return [cum[r - 1].item() for r in ranks_e]

    kinds = {"q": "attn.q_proj", "v": "attn.v_proj", "fc": "mlp.fc", "proj": "mlp.proj"}

    def at(model, path):
        return [block.get_submodule(path) for block in model.blocks]

    rows, full_e, rand_e = [], {}, {}
    gen = torch.Generator().manual_seed(args.seed)
    for key, path in kinds.items():
        base_w = [m.weight for m in at(base, path)]
        full_w = [m.weight for m in at(full_model, path)]
        lora_d = [m.delta() for m in at(lora_model, path)]
        for label, deltas in (
            ("full fine-tuning", [f - b for f, b in zip(full_w, base_w, strict=True)]),
            ("LoRA r=8", lora_d),
            ("random matrix", [torch.randn(w.shape, generator=gen) for w in base_w]),
        ):
            es = [energy(d_.detach()) for d_ in deltas]
            mean = [sum(col) / len(es) for col in zip(*es, strict=True)]
            rows.append([label, key, *[f"{v:.3f}" for v in mean]])
            if label == "full fine-tuning":
                full_e[key] = mean
            if label == "random matrix":
                rand_e[key] = mean
    report.say(
        "  share of the squared Frobenius norm of the update in its top-r singular directions,"
    )
    report.say("  averaged over the four blocks:")
    report.table(["update", "matrix", *[f"r={r}" for r in ranks_e]], rows)
    if QUICK:
        report.skip(
            "the full fine-tuning update is more concentrated than noise, but not low rank",
            "needs the full run",
        )
    else:
        i8 = ranks_e.index(8)
        report.check(
            "a full fine-tuning update is far more concentrated than a random matrix (top 8 directions)",
            all(full_e[k][i8] > 2 * rand_e[k][i8] for k in kinds),
            ", ".join(f"{k} {full_e[k][i8]:.2f} vs {rand_e[k][i8]:.2f}" for k in kinds),
        )
        report.check(
            "...but it is not low rank: the top 8 directions hold well under all of it",
            all(full_e[k][i8] < 0.9 for k in kinds),
            ", ".join(f"{k} {full_e[k][i8]:.2f}" for k in kinds),
        )
        report.check(
            "a rank-8 LoRA update has exactly 8 directions (energy 1.000 at r=8)",
            all(float(row[ranks_e.index(8) + 2]) == 1.0 for row in rows if row[0] == "LoRA r=8"),
        )

    # -----------------------------------------------------------------------------------------
    report.section("7. Merging, and merging into 4-bit weights")
    w = base.blocks[0].attn.q_proj.weight.data
    rel_err = ((w - nf4_roundtrip(w)).norm() / w.norm()).item()
    report.say(f"  NF4 round-trip error of block 0's q_proj: {rel_err:.1%} of the weight norm")
    report.check(
        "NF4 costs roughly a tenth of the weight norm (page: 9.3% to 9.7% on Qwen)",
        0.05 < rel_err < 0.15,
        f"{rel_err:.1%}",
    )

    # how much of dW survives re-quantization, as a function of its size
    paths = ("attn.q_proj", "mlp.fc", "mlp.proj")
    rows = []
    kept_small = kept_big = kept_fp32_small = None
    for scale in (0.01, 0.1, 1.0, 10.0):
        kept_nf4, kept_fp32 = [], []
        for path in paths:
            ad = lora_model.blocks[0].get_submodule(path)
            w0 = base.blocks[0].get_submodule(path).weight.data
            dw = scale * ad.delta()
            grid = nf4_roundtrip(w0)
            into_nf4 = nf4_roundtrip(grid + dw) - grid
            into_fp32 = nf4_roundtrip(w0 + dw) - grid

            kept_nf4.append(kept_share(into_nf4, dw))
            kept_fp32.append(kept_share(into_fp32, dw))
        mean_dw = (
            sum(lora_model.blocks[0].get_submodule(p).delta().abs().mean().item() for p in paths)
            / 3
            * scale
        )
        mean_err = (
            sum(
                (
                    base.blocks[0].get_submodule(p).weight.data
                    - nf4_roundtrip(base.blocks[0].get_submodule(p).weight.data)
                )
                .abs()
                .mean()
                .item()
                for p in paths
            )
            / 3
        )
        rows.append(
            [
                f"{scale:g} x trained",
                f"{mean_dw:.1e}",
                f"{mean_err:.1e}",
                f"{sum(kept_nf4) / 3:.3f}",
                f"{sum(kept_fp32) / 3:.3f}",
            ]
        )
        if scale == 0.01:
            kept_small, kept_fp32_small = sum(kept_nf4) / 3, sum(kept_fp32) / 3
        if scale == 10.0:
            kept_big = sum(kept_nf4) / 3
    report.say(
        "\n  share of the update dW that survives, for the LoRA adapter trained above (block 0, three matrices):"
    )
    report.table(
        [
            "update size",
            "mean |dW|",
            "mean NF4 error",
            "merged into NF4, re-quantized",
            "merged into fp32, then quantized",
        ],
        rows,
    )
    report.check(
        "an update far below the 4-bit grid spacing is rounded away when merged into an already quantized weight, but survives from fp32",
        kept_small < 0.2 and kept_fp32_small > 0.8,
        f"at 0.01 x the trained size: {kept_small:.3f} against {kept_fp32_small:.3f}",
    )
    report.check(
        "an update well above the spacing mostly survives either way",
        kept_big > 0.7,
        f"at 10 x the trained size: {kept_big:.3f}",
    )

    # deployment variants of the default adapter, and QLoRA
    def attached_to_nf4(model):
        m = copy.deepcopy(model)
        quantize_base(m)  # reaches the frozen weight inside every adapter too
        return m

    def merged_then_quantized(model):
        m = copy.deepcopy(model)
        merge_lora(m)
        quantize_base(m)
        return m

    def merged_into_nf4(model):
        m = copy.deepcopy(model)
        quantize_base(m)  # the quantized base the adapter is merged into...
        merge_lora(m)
        quantize_base(m)  # ...and the merged weights quantized again, the same way
        return m

    rows = []
    variants = {
        "fp32 base, no adapter": copy.deepcopy(base),
        "adapter attached to the fp32 base": copy.deepcopy(lora_model),
        "adapter attached to an NF4 base": attached_to_nf4(lora_model),
        "merged into fp32, then quantized": merged_then_quantized(lora_model),
        "merged into NF4, quantized again": merged_into_nf4(lora_model),
    }
    nf4_only = copy.deepcopy(base)
    quantize_base(nf4_only)
    variants["NF4 base, no adapter"] = nf4_only
    results = {}
    for name, model in variants.items():
        sc = intents.score(model, test)
        results[name] = (sc["accuracy"], models.shakespeare_bpb(model))
        rows.append([name, f"{sc['accuracy']:.1%}", f"{results[name][1]:.2f}"])
    qlora_run = run("lora", quant=True)
    rows.append(["QLoRA: trained on the NF4 base, attached to it", *fmt(qlora_run)])
    report.say(
        "\n  one adapter, four ways to deploy it (and QLoRA: trained on the quantized base):"
    )
    report.table(["deployment", "accuracy", "shakespeare bits/byte"], rows)
    nf4_cost = results["NF4 base, no adapter"][1] - base_bpb
    report.say(
        f"  4-bit storage alone moves held-out Shakespeare by {nf4_cost:+.3f} bits/byte"
        f" ({nf4_cost / base_bpb:+.1%}); the page's 0.5B model lost 13% in perplexity."
    )

    # -----------------------------------------------------------------------------------------
    report.section("8. Memory and speed")
    full_n = sum(p.numel() for p in base.parameters())
    lora_n = default["trainable"]
    rows = [
        [
            "full fine-tuning",
            f"{full_n:,}",
            f"{full_n * 4 * 4 / 2**20:.2f}",
            f"{full_runs[3e-4]['seconds'] / STEPS * 1000:.0f}",
        ],
        [
            "LoRA r=8, all linear",
            f"{lora_n:,}",
            f"{(full_n + 3 * lora_n + lora_n) * 4 / 2**20:.2f}",
            f"{default['seconds'] / STEPS * 1000:.0f}",
        ],
    ]
    report.table(
        [
            "method",
            "trained parameters",
            "weights+grads+Adam (MiB, fp32)",
            "ms per step (this machine)",
        ],
        rows,
    )
    report.say(
        "  static memory = one frozen copy of the weights, plus four copies (weights, gradient, two Adam\n"
        "  moments) of whatever is trained. Activations come on top and shrink the gap, as on the page."
    )
    report.check(
        "LoRA trains a small fraction of the parameters and needs a fraction of the static memory",
        lora_n < 0.2 * full_n,
        f"{lora_n / full_n:.1%} of the parameters",
    )

    report.finish()


if __name__ == "__main__":
    main()
