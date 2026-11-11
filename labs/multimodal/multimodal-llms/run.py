"""Lab: Multimodal LLMs. An image as tokens, what the tokens cost, and CLIP's contrastive loss.

Course page: https://dmxsf.com/wiki/multimodal/multimodal-llms/

Runs offline. The patch merge, projector and splice of the page's code; the token counts, KV cache and
FLOP tables of the page, recomputed (they are arithmetic on the models' configuration files); LLaVA's
projector bill and its two training stages; CLIP's symmetric loss checked against the formula; and a small
CLIP trained from scratch on rendered coloured shapes, to see what the temperature and the number of
negatives do. With `--hf` (extra `hf`) the page's CLIP experiment is rerun on the real
openai/clip-vit-base-patch32 and CIFAR-10 (zero-shot accuracy for six prompt templates, the loss of eight
pairs). The page's Qwen3-VL experiments (latency, reading small digits) need MLX and an Apple GPU and are
not rerun; their numbers appear here only where the page's own table can be checked against the arithmetic.

    uv run python labs/multimodal/multimodal-llms/run.py [--hf] [--seed N] [--quick]
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from dmxsf import lab
from dmxsf.multimodal import clip, vision

# What the page prints. Tokens: (width, height) -> (resized width, resized height, tokens, KV MiB).
PAGE_TOKENS = {
    ("Qwen2-VL-7B", (224, 224)): (224, 224, 64, 4),
    ("Qwen2-VL-7B", (1024, 768)): (1036, 756, 999, 55),
    ("Qwen2-VL-7B", (1920, 1080)): (1932, 1092, 2691, 147),
    ("Qwen2-VL-7B", (4032, 3024)): (4032, 3024, 15552, 850),
    ("Qwen3-VL-4B", (224, 224)): (256, 256, 64, 9),
    ("Qwen3-VL-4B", (1024, 768)): (1024, 768, 768, 108),
    ("Qwen3-VL-4B", (1920, 1080)): (1920, 1088, 2040, 287),
    ("Qwen3-VL-4B", (4032, 3024)): (4032, 3008, 11844, 1666),
}
# Visual tokens -> (ViT TFLOPs, LLM TFLOPs, ViT share of the compute) and the share of the *time* the
# vision part took on an M2 Max (measured with MLX on the page; not rerun here).
PAGE_FLOPS = {
    64: (0.16, 0.47, 26, 26),
    256: (0.72, 1.88, 28, 25),
    576: (1.91, 4.28, 31, 30),
    1024: (4.12, 7.75, 35, 33),
    2304: (13.92, 18.31, 43, 39),
    4096: (36.28, 34.71, 51, 48),
    9216: (155.86, 92.02, 63, 60),
    16384: (461.79, 198.22, 70, 70),
}
# Reading six-digit numbers with Qwen3-VL-4B (4-bit): font size -> accuracy ignoring stray punctuation at
# scale x0.5, x1, x2, x3 (the page's table, measured; not rerun here).
PAGE_OCR = {
    8: (0.04, 0.90, 0.99, 0.97),
    12: (0.75, 1.00, 1.00, 1.00),
    16: (0.92, 1.00, 1.00, 1.00),
    24: (1.00, 1.00, 1.00, 1.00),
}
# openai/clip-vit-base-patch32 on the 10,000 CIFAR-10 test images, and the loss of 8 pairs.
PAGE_CLIP = {
    "scale": 100,
    "accuracy": {
        "bare class name": 0.8735,
        "a photo of a {}.": 0.8830,
        "a drawing of a {}.": 0.8967,
        "a low resolution photo of a {}.": 0.8903,
        "unrelated sentence": 0.8733,
        "mean of 7 templates": 0.8917,
    },
    "loss": {"correct": 0.307, "shifted": 6.587, "scale 1": 2.027, "chance": 2.079},
}


def add_args(parser) -> None:
    parser.add_argument(
        "--hf",
        action="store_true",
        help="also rerun the page's CLIP experiment on openai/clip-vit-base-patch32 and CIFAR-10",
    )


def main() -> None:
    args, report = lab.start(
        "Multimodal LLMs: images as tokens", wiki=["multimodal/multimodal-llms"], add_args=add_args
    )
    quick = args.quick

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if quick:
            report.skip(name, "--quick trains too little")
        else:
            report.check(name, ok, detail)

    # ------------------------------------------------------------------------------------------------
    report.section("1. From patches to visual tokens (the page's code, with random numbers)")
    torch.manual_seed(0)
    grid_h, grid_w, merge = 8, 12, 2  # an image cut into 8 x 12 patches
    d_vit, d_llm = 1024, 2560  # Qwen3-VL-4B: ViT width, LLM width
    vit_out = torch.randn(grid_h * grid_w, d_vit)  # one vector per patch
    blocks = vision.merge_patches(vit_out, grid_h, grid_w, merge)
    merger = vision.projector(merge * merge * d_vit, d_llm)
    visual = merger(blocks)
    n_params = sum(p.numel() for p in merger.parameters())
    report.say(
        f"  patches {tuple(vit_out.shape)} -> blocks {tuple(blocks.shape)} -> visual tokens "
        f"{tuple(visual.shape)}; merger parameters: {n_params:,}"
    )
    placeholder = -1
    ids = torch.tensor([5, 9] + [placeholder] * len(visual) + [7, 3, 4])
    embed = nn.Embedding(100, d_llm)
    inputs = vision.splice(ids, embed, visual, placeholder)
    report.say(
        f"  LLM input: {tuple(inputs.shape)} = 2 text + {len(visual)} visual + 3 text tokens"
    )
    report.check(
        "shapes: 96 patches -> 24 blocks of 4 x 1024 -> 24 visual tokens of 2560; 29 rows reach the LLM",
        tuple(blocks.shape) == (24, 4096)
        and tuple(visual.shape) == (24, 2560)
        and tuple(inputs.shape) == (29, 2560),
    )
    report.check(
        "the merger has (4096 x 4096 + 4096) + (4096 x 2560 + 2560) = 27,269,632 parameters",
        n_params
        == vision.linear_params(4096, 4096) + vision.linear_params(4096, 2560)
        == 27_269_632,
    )
    grid = vit_out.view(grid_h, grid_w, d_vit)
    first_block = torch.cat([grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]])
    last_block = torch.cat([grid[6, 10], grid[6, 11], grid[7, 10], grid[7, 11]])
    report.check(
        "each visual token is built from one 2 x 2 block of neighbouring patches, in row-major order",
        torch.equal(blocks[0], first_block) and torch.equal(blocks[-1], last_block),
    )
    is_image = ids == placeholder
    report.check(
        "after the splice the placeholder rows are the visual tokens and the others are table rows: the LLM "
        "cannot tell which is which",
        torch.equal(inputs[is_image], visual)
        and torch.equal(inputs[~is_image], embed(ids[~is_image])),
    )

    rows = []
    for name, side, patch, merge_ in (
        ("CLIP ViT-B/32 (224x224)", 224, 32, 1),
        ("LLaVA-1.5-7B (336x336)", 336, 14, 1),
    ):
        n = (side // patch) ** 2
        rows.append([name, patch, n, n // merge_**2])
    for cfg, (w, h) in ((vision.QWEN2_VL, (1024, 768)), (vision.QWEN3_VL, (1024, 768))):
        t = vision.visual_tokens(cfg, h, w)
        rows.append(
            [f"{cfg.name} ({w}x{h} -> {t.width}x{t.height})", cfg.patch, t.patches, t.tokens]
        )
    report.say("\n  Patches in the vision encoder, tokens into the language model:")
    report.table(["model (input)", "patch side", "patches", "visual tokens"], rows)
    report.check(
        "the page's table: 49, 576, 3,996 -> 999 and 3,072 -> 768",
        [(r[2], r[3]) for r in rows] == [(49, 49), (576, 576), (3996, 999), (3072, 768)],
    )
    small = vision.visual_tokens(vision.QWEN3_VL, 512, 512)
    double = vision.visual_tokens(vision.QWEN3_VL, 1024, 1024)
    report.check(
        "doubling the side of the image quadruples the patches (and the visual tokens), so the "
        "vision encoder's attention pairs, which grow with patches squared, grow sixteen-fold",
        double.patches == 4 * small.patches
        and double.tokens == 4 * small.tokens
        and double.patches**2 == 16 * small.patches**2,
        f"{small.patches} -> {double.patches} patches",
    )

    # ------------------------------------------------------------------------------------------------
    report.section("2. How many tokens does an image take, and what do they cost in KV cache")
    rows, mismatch = [], []
    for cfg in (vision.QWEN2_VL, vision.QWEN3_VL):
        per_token = cfg.kv_bytes_per_token
        for w, h in ((224, 224), (1024, 768), (1920, 1080), (4032, 3024)):
            t = vision.visual_tokens(cfg, h, w)
            mib = f"{t.tokens * per_token / 2**20:.0f}"
            rows.append([cfg.name, f"{w}x{h}", f"{t.width}x{t.height}", t.tokens, mib])
            if (t.width, t.height, t.tokens, int(mib)) != PAGE_TOKENS[cfg.name, (w, h)]:
                mismatch.append((cfg.name, (w, h)))
    report.table(["model", "image", "after resize", "tokens", "KV cache (MiB, FP16)"], rows)
    report.check("all eight rows of the page's table", not mismatch, str(mismatch))
    kv = {
        c.name: c.kv_bytes_per_token // 1024
        for c in (vision.LLAVA_15, vision.QWEN2_VL, vision.QWEN3_VL)
    }
    report.check(
        "KV cache per token: 512 KiB (LLaVA-1.5, no GQA), 56 KiB (Qwen2-VL-7B), 144 KiB (Qwen3-VL-4B)",
        kv == {"LLaVA-1.5-7B": 512, "Qwen2-VL-7B": 56, "Qwen3-VL-4B": 144},
    )
    llava = vision.visual_tokens(vision.LLAVA_15, 1080, 1920).tokens
    qwen2 = vision.visual_tokens(vision.QWEN2_VL, 768, 1024).tokens
    llava_mib = llava * vision.LLAVA_15.kv_bytes_per_token / 2**20
    qwen2_mib = qwen2 * vision.QWEN2_VL.kv_bytes_per_token / 2**20
    report.check(
        "LLaVA-1.5 always gives 576 tokens and 288 MiB; Qwen2-VL-7B on 1024x768 has 1.7x the tokens and "
        "less than a fifth of the cache",
        llava == 576
        and round(llava_mib) == 288
        and round(qwen2 / llava, 1) == 1.7
        and qwen2_mib / llava_mib < 0.2,
        f"{qwen2 / llava:.2f}x the tokens, {qwen2_mib / llava_mib:.2f} of the cache",
    )
    big = [vision.visual_tokens(c, 3024, 4032).tokens for c in (vision.QWEN2_VL, vision.QWEN3_VL)]
    report.check(
        "a 12-megapixel photo takes 47% and 36% of a 32K context window",
        [round(100 * n / 32768) for n in big] == [47, 36],
        f"{big}",
    )
    capped = [
        vision.visual_tokens(vision.QWEN3_VL, 3024, 4032, cap).tokens for cap in (256, 1024, 4096)
    ]
    report.check(
        "a token cap shrinks the image so that it fits: 256, 1024 and 4096 tokens at most",
        all(n <= cap for n, cap in zip(capped, (256, 1024, 4096), strict=True)),
        f"{capped}",
    )
    sizes = [(h, w) for h in (150, 480, 1000, 2500) for w in (200, 640, 1333, 3000)]
    ok = True
    for cfg in (vision.QWEN2_VL, vision.QWEN3_VL):
        for h, w in sizes:
            t = vision.visual_tokens(cfg, h, w)
            ok &= t.height % cfg.factor == 0 and t.width % cfg.factor == 0
            ok &= cfg.min_pixels <= t.height * t.width <= cfg.max_pixels
    report.check(
        "for 16 image sizes and both models: both sides are multiples of patch x merge, and the pixel count "
        "stays inside [min_pixels, max_pixels]",
        ok,
    )

    # ------------------------------------------------------------------------------------------------
    report.section("3. LLaVA: the projector is small, and each stage trains different parts")
    vicuna = vision.llama_params(layers=32, d=4096, ff=11008, vocab=32000)
    linear = vision.linear_params(1024, 4096)
    mlp = linear + vision.linear_params(4096, 4096)
    report.table(
        ["part", "parameters", "share of the 6.74B language model"],
        [
            ["Vicuna-7B (LLaMA-7B shapes)", f"{vicuna:,}", "100%"],
            ["LLaVA projector, one linear layer", f"{linear:,}", f"{linear / vicuna:.2%}"],
            ["LLaVA-1.5 projector, two-layer MLP", f"{mlp:,}", f"{mlp / vicuna:.2%}"],
        ],
    )
    report.check(
        "the language model has 6.74 billion parameters; the projector 4,198,400 or 20,979,712, "
        "which is 0.06% and 0.31%",
        (round(vicuna / 1e9, 2), linear, mlp) == (6.74, 4_198_400, 20_979_712)
        and (f"{linear / vicuna:.2%}", f"{mlp / vicuna:.2%}") == ("0.06%", "0.31%"),
    )

    toy_vision, toy_proj, toy_lm = nn.Linear(8, 8), nn.Linear(8, 6), nn.Linear(6, 6)
    counts, grads = {}, {}
    for stage in (1, 2):
        counts[stage] = vision.set_llava_stage(toy_vision, toy_proj, toy_lm, stage)
        for m in (toy_vision, toy_proj, toy_lm):
            m.zero_grad(set_to_none=True)
        toy_lm(toy_proj(toy_vision(torch.randn(3, 8)))).sum().backward()
        grads[stage] = tuple(
            all(p.grad is not None for p in m.parameters()) for m in (toy_vision, toy_proj, toy_lm)
        )
    report.say(
        "  A toy vision encoder -> projector -> language model, one backward pass per stage:"
    )
    report.table(
        ["stage", "trainable parameters", "vision gets a gradient", "projector", "language model"],
        [[s, counts[s], *("yes" if g else "no" for g in grads[s])] for s in (1, 2)],
    )
    n_vision = sum(p.numel() for p in toy_vision.parameters())
    n_proj = sum(p.numel() for p in toy_proj.parameters())
    n_lm = sum(p.numel() for p in toy_lm.parameters())
    report.check(
        "stage 1 trains only the projector; stage 2 the projector and the language model; the vision "
        "encoder never",
        counts == {1: n_proj, 2: n_proj + n_lm}
        and grads[1] == (False, True, False)
        and grads[2] == (False, True, True)
        and n_vision > 0,
    )

    # ------------------------------------------------------------------------------------------------
    report.section("4. CLIP's loss")
    rng = np.random.default_rng(args.seed)
    n, d = 6, 16
    a = torch.nn.functional.normalize(torch.from_numpy(rng.normal(size=(n, d))).float(), dim=-1)
    b = torch.nn.functional.normalize(torch.from_numpy(rng.normal(size=(n, d))).float(), dim=-1)
    ours = clip.clip_loss(a, b, 14.3).item()
    by_hand = clip.clip_loss_by_hand(a, b, 14.3)
    report.check(
        "the loss equals the formula of the paper, summed term by term",
        abs(ours - by_hand) < 1e-5,
        f"{ours:.6f} vs {by_hand:.6f}",
    )
    report.check(
        "it is symmetric in images and texts, and at temperature infinity (scale 0) it is exactly ln n",
        abs(clip.clip_loss(a, b, 5.0).item() - clip.clip_loss(b, a, 5.0).item()) < 1e-6
        and abs(clip.clip_loss(a, b, 0.0).item() - math.log(n)) < 1e-6,
    )

    n, d = 8, 512
    shared, matched, noise = (
        0.19,
        0.08,
        0.73,
    )  # cosine of mismatched pairs, extra for matched, the rest

    def unit(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    common = unit(rng.normal(size=d))
    own = unit(rng.normal(size=(n, d)))

    def side() -> torch.Tensor:
        v = (
            math.sqrt(shared) * common
            + math.sqrt(matched) * own
            + math.sqrt(noise) * unit(rng.normal(size=(n, d)))
        )
        return torch.from_numpy(unit(v)).float()

    a, b = side(), side()
    cos = (a @ b.T).numpy()
    diag = np.diag(cos)
    off = cos[~np.eye(n, dtype=bool)]
    loss = {
        "correct pairs, 1/tau = 100": clip.clip_loss(a, b, 100.0).item(),
        "shifted pairs, 1/tau = 100": clip.clip_loss(a, b.roll(1, dims=0), 100.0).item(),
        "correct pairs, 1/tau = 1": clip.clip_loss(a, b, 1.0).item(),
        "chance level, ln 8": math.log(8),
    }
    report.say(
        f"  Eight made-up image/text pairs whose similarities look like the page's real CLIP ones: matched\n"
        f"  pairs {diag.min():.2f} to {diag.max():.2f}, the others {off.min():.2f} to {off.max():.2f}."
    )
    report.table(["", "loss"], [[k, v] for k, v in loss.items()])
    report.check(
        "the cosines are close together, as on the page: a gap of about 0.08 between matched and "
        "mismatched pairs",
        0.04 < diag.mean() - off.mean() < 0.12 and 0.15 < off.mean() < 0.23,
        f"matched {diag.mean():.3f}, mismatched {off.mean():.3f}",
    )
    report.check(
        "without the temperature the softmax is nearly uniform: the loss sits within 0.1 of ln 8",
        abs(loss["correct pairs, 1/tau = 1"] - math.log(8)) < 0.1,
    )
    report.check(
        "multiplied by 100 the same gaps give a loss far below chance, and a shifted pairing is "
        "punished hard",
        loss["correct pairs, 1/tau = 100"] < 0.6 * math.log(8)
        and loss["shifted pairs, 1/tau = 100"] > 3 * math.log(8),
    )

    # ------------------------------------------------------------------------------------------------
    report.section("5. A small CLIP, trained from scratch on rendered shapes")
    sample = clip.render(np.array([clip.N_CLASSES - 1]), np.random.default_rng(0))
    report.say(
        f"  {clip.N_CLASSES} captions ('{clip.caption(0)}', ..., '{clip.caption(clip.N_CLASSES - 1)}'), "
        f"24x24 pictures {tuple(sample.shape[1:])}, chance = {1 / clip.N_CLASSES:.1%}. Zero-shot accuracy: "
        "the nearest of the\n  96 caption embeddings to a fresh picture, no classifier trained."
    )
    steps = 40 if quick else 300
    seeds = [args.seed] if quick else [args.seed, args.seed + 1, args.seed + 2]
    runs: dict[tuple[str, int], list[clip.Result]] = {}

    def run(batch: int, scale: float | None) -> list[clip.Result]:
        key = ("learned" if scale is None else f"{scale:g}", batch)
        if key not in runs:
            runs[key] = [clip.train(steps, batch, scale, s)[1] for s in seeds]
        return runs[key]

    def row(label: str, rs: list[clip.Result]) -> list:
        acc = [100 * r.accuracy for r in rs]
        return [
            label,
            f"{np.mean(acc):.1f}%",
            f"{min(acc):.1f}-{max(acc):.1f}",
            f"{np.mean([r.loss for r in rs]):.3f}",
            f"{np.mean([r.scale for r in rs]):.1f}",
        ]

    headers = ["", "zero-shot accuracy", "range over seeds", "final loss", "final 1/tau"]
    report.say(
        f"\n  The temperature: batch of 32, {steps} steps, {len(seeds)} seed(s). 'learned' starts at 1/0.07."
    )
    by_scale = {
        name: run(32, s)
        for name, s in (("1", 1.0), ("10", 10.0), ("100", 100.0), ("learned", None))
    }
    report.table(
        headers,
        [
            row(f"1/tau = {k}" if k != "learned" else "1/tau learned", v)
            for k, v in by_scale.items()
        ],
    )
    report.say(
        f"\n  The number of negatives: learned temperature, {steps} steps, {len(seeds)} seed(s)."
    )
    by_batch = {b: run(b, None) for b in (2, 8, 32, 96)}
    report.table(headers, [row(f"batch {b}", v) for b, v in by_batch.items()])
    report.say(
        "  (A bigger batch also shows more pictures per step, and the same number of steps are taken, so\n"
        "  'more negatives' and 'more data per step' are not separated here.)"
    )

    def mean(rs: list[clip.Result]) -> float:
        return float(np.mean([r.accuracy for r in rs]))

    full_check(
        "every setting beats chance: the temperature runs by a wide margin, a single negative by a few times",
        all(mean(rs) > 0.2 for rs in by_scale.values())
        and all(mean(rs) > 3 / clip.N_CLASSES for rs in by_batch.values()),
        f"worst temperature run {min(mean(rs) for rs in by_scale.values()):.1%}, "
        f"worst batch {min(mean(rs) for rs in by_batch.values()):.1%}, chance {1 / clip.N_CLASSES:.1%}",
    )
    full_check(
        "without a temperature (1/tau = 1) the loss cannot get small and the embeddings separate worse "
        "than with 1/tau = 100 or a learned one",
        mean(by_scale["1"]) < min(mean(by_scale["100"]), mean(by_scale["learned"]))
        and np.mean([r.loss for r in by_scale["1"]]) > 1.0,
        f"{mean(by_scale['1']):.1%} vs {mean(by_scale['100']):.1%} and {mean(by_scale['learned']):.1%}",
    )
    full_check(
        "the learned temperature moves away from its initial 1/0.07 = 14.3 towards a sharper softmax",
        float(np.mean([r.scale for r in by_scale["learned"]])) > 1 / 0.07,
        f"{np.mean([r.scale for r in by_scale['learned']]):.1f}",
    )
    full_check(
        "with only one negative per image (batch 2) the model learns less than with 31",
        mean(by_batch[2]) < mean(by_batch[32]),
        f"{mean(by_batch[2]):.1%} vs {mean(by_batch[32]):.1%}",
    )

    # ------------------------------------------------------------------------------------------------
    report.section("6. The bill: how much of the compute is the vision encoder")
    rows, gaps = [], []
    for tokens, (vit_t, llm_t, share, measured) in PAGE_FLOPS.items():
        v, lm = vision.vit_flops(tokens), vision.llm_flops(tokens)
        mine = round(100 * v / (v + lm))
        rows.append(
            [f"{tokens:,}", f"{v / 1e12:.2f}", f"{lm / 1e12:.2f}", f"{mine}%", f"{measured}%"]
        )
        gaps.append(
            (
                mine - measured,
                (f"{v / 1e12:.2f}", f"{lm / 1e12:.2f}", mine)
                == (f"{vit_t:.2f}", f"{llm_t:.2f}", share),
            )
        )
    report.table(
        [
            "visual tokens",
            "ViT TFLOPs",
            "LLM TFLOPs",
            "ViT share (FLOPs)",
            "ViT share of time (page)",
        ],
        rows,
    )
    report.check("the page's FLOP table, all eight rows", all(ok for _, ok in gaps))
    report.check(
        "the FLOP share is within 4 points of the measured time share, and never below it: a reconciliation, "
        "not a predictor",
        all(0 <= g <= 4 for g, _ in gaps),
        f"gaps {[g for g, _ in gaps]}",
    )
    vit_attn = 4 * (4 * 1) ** 2 * vision.VIT["d"] * vision.VIT["depth"]  # per n^2, m = 4 n
    llm_attn = 2 * vision.LLM["heads"] * vision.LLM["hd"] * vision.LLM["depth"]  # per n^2
    report.check(
        "the vision encoder's attention term has a coefficient about 5.3 times the language model's",
        round(vit_attn / llm_attn, 1) == 5.3,
        f"{vit_attn / llm_attn:.2f}",
    )
    shares = [vision.vit_share(t) for t in PAGE_FLOPS]
    report.check(
        "the share rises with the number of tokens (26% to 70%) because the encoder's attention grows with "
        "the square of 4 x tokens",
        shares == sorted(shares) and round(100 * shares[0]) == 26 and round(100 * shares[-1]) == 70,
    )

    # ------------------------------------------------------------------------------------------------
    report.section("7. Resolution and the size of the characters")
    rows = []
    for font, accuracies in PAGE_OCR.items():
        row_ = [f"{font} px"]
        for scale, acc in zip((0.5, 1, 2, 3), accuracies, strict=True):
            tokens = vision.visual_tokens(
                vision.QWEN3_VL, int(512 * scale), int(512 * scale)
            ).tokens
            row_.append(f"{font * scale:g} px, {tokens:,} tok: {acc:.2f}")
        rows.append(row_)
    report.say(
        "  The page's measured accuracy (Qwen3-VL-4B, 4-bit, MLX, ignoring stray punctuation; not rerun) "
        "next to what\n  each cell means in the model's input: the effective font size and the visual tokens of a "
        "512x512 canvas."
    )
    report.table(["font", "x0.5", "x1", "x2", "x3"], rows)
    report.check(
        "the token columns of the page's table are 64, 256, 1,024 and 2,304",
        [
            vision.visual_tokens(vision.QWEN3_VL, int(512 * s), int(512 * s)).tokens
            for s in (0.5, 1, 2, 3)
        ]
        == [64, 256, 1024, 2304],
    )
    by_effective: dict[float, list[tuple[float, int]]] = {}
    for font, accuracies in PAGE_OCR.items():
        for scale, acc in zip((0.5, 1, 2, 3), accuracies, strict=True):
            tokens = vision.visual_tokens(
                vision.QWEN3_VL, int(512 * scale), int(512 * scale)
            ).tokens
            by_effective.setdefault(font * scale, []).append((acc, tokens))
    spread = max(max(a for a, _ in v) - min(a for a, _ in v) for v in by_effective.values())
    fold = max(max(t for _, t in v) / min(t for _, t in v) for v in by_effective.values())
    report.check(
        "in the page's table, cells with the same effective font size (font x scale) agree to within 3 "
        f"points although their token counts differ up to {fold:g}-fold: the size of the characters, not "
        "the token count, sets the accuracy",
        spread <= 0.03 + 1e-9 and fold >= 4,
        f"largest spread {spread:.2f}",
    )

    # ------------------------------------------------------------------------------------------------
    if args.hf:
        run_clip(args, report)
    else:
        report.say(
            "\n(8. Rerun the page's CLIP experiment on the real model: add --hf. It needs the extra `hf`, "
            "the model and CIFAR-10.)"
        )
    report.finish()


def run_clip(args, report) -> None:
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import CLIPModel, CLIPProcessor

    report.section("8. CLIP ViT-B/32 on CIFAR-10, zero-shot (float32, CPU)")
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    proc = CLIPProcessor.from_pretrained(name)
    data = load_dataset("uoft-cs/cifar10", split="test")
    classes = data.features["label"].names
    count = 300 if args.quick else len(data)
    report.say(f"  {count} test images; temperature scale {model.logit_scale.exp().item():.0f}")
    report.check(
        "the learned 1/tau sits at the cap of 100, as the paper clips it",
        round(model.logit_scale.exp().item()) == PAGE_CLIP["scale"],
    )

    def tensor_of(out):  # transformers 5 returns an output object
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.no_grad()
    def embed_images(images):
        px = proc(images=images, return_tensors="pt")["pixel_values"]
        return F.normalize(tensor_of(model.get_image_features(pixel_values=px)), dim=-1)

    @torch.no_grad()
    def embed_classes(templates):
        rows = []
        for c in classes:
            tok = proc(text=[t.format(c) for t in templates], return_tensors="pt", padding=True)
            e = F.normalize(tensor_of(model.get_text_features(**tok)), dim=-1)
            rows.append(F.normalize(e.mean(0), dim=-1))
        return torch.stack(rows)

    images = [data[i]["img"].convert("RGB") for i in range(count)]
    labels = torch.tensor(data["label"][:count])
    img = torch.cat([embed_images(images[i : i + 250]) for i in range(0, count, 250)])
    templates = {
        "bare class name": ["{}"],
        "a photo of a {}.": ["a photo of a {}."],
        "a drawing of a {}.": ["a drawing of a {}."],
        "a low resolution photo of a {}.": ["a low resolution photo of a {}."],
        "unrelated sentence": ["the weather is nice today, {}"],
        "mean of 7 templates": [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a low resolution photo of a {}.",
            "a bad photo of a {}.",
            "a good photo of a {}.",
            "a photo of the {}.",
            "a close-up photo of a {}.",
        ],
    }
    got, rows = {}, []
    for tname, tpl in templates.items():
        acc = ((img @ embed_classes(tpl).T).argmax(-1) == labels).float().mean().item()
        got[tname] = acc
        rows.append([tname, f"{acc:.4f}", f"{PAGE_CLIP['accuracy'][tname]:.4f}"])
    report.table(["template", "accuracy", "page"], rows)
    if args.quick:
        report.skip("the page's six accuracies", "--quick uses 300 images")
    else:
        off = {
            k: round(got[k] - v, 4)
            for k, v in PAGE_CLIP["accuracy"].items()
            if f"{got[k]:.4f}" != f"{v:.4f}"
        }
        report.check(
            "the page's six accuracies, to within two images in 10,000 (the page ran on MPS, this on the CPU: "
            "an image that is a near-tie can flip)",
            all(abs(d) <= 0.0002 + 1e-9 for d in off.values()),
            f"differences {off}" if off else "all six equal to four digits",
        )
        report.check(
            "a template is a hyperparameter: 'a drawing of a {}.' beats 'a photo of a {}.' by more than a point, "
            "and an unrelated sentence is no better than the bare class name",
            got["a drawing of a {}."] - got["a photo of a {}."] > 0.01
            and abs(got["unrelated sentence"] - got["bare class name"]) < 0.005,
        )

    scale = model.logit_scale.exp().item()
    labels_all = torch.tensor(data["label"])
    one_each = [(labels_all == c).nonzero()[0].item() for c in range(8)]
    if args.quick:
        report.skip(
            "the loss of eight pairs", "needs the images the page uses (computed from the full set)"
        )
        return
    a, b = img[one_each], embed_classes(["a photo of a {}."])[:8]
    cos = (a @ b.T).numpy()
    loss = {
        "correct": clip.clip_loss(a, b, scale).item(),
        "shifted": clip.clip_loss(a, b.roll(1, dims=0), scale).item(),
        "scale 1": clip.clip_loss(a, b, 1.0).item(),
        "chance": math.log(8),
    }
    report.say(
        "  loss of one image per class against the 8 class texts: "
        + ", ".join(f"{k} {v:.3f}" for k, v in loss.items())
    )
    off = {
        k: round(loss[k], 3) for k, v in PAGE_CLIP["loss"].items() if f"{loss[k]:.3f}" != f"{v:.3f}"
    }
    report.check("the page's four loss values", not off, f"got {off}")
    report.check(
        "in every row of the 8 x 8 similarity matrix the maximum is on the diagonal",
        bool((cos.argmax(1) == np.arange(8)).all()),
    )


if __name__ == "__main__":
    main()
