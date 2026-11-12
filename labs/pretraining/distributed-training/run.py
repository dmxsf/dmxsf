"""Lab: distributed training, built and measured in a few processes on one machine.

Course page: https://dmxsf.com/wiki/pretraining/distributed-training/

1. how data-parallel gradients are averaged, and when that is wrong (real processes, gloo);
2. the ring all-reduce, simulated, and why its cost hardly grows with the number of GPUs;
3. data parallelism, ZeRO-1 and ZeRO-3 written out in ~150 lines, trained side by side: same weights,
   measured memory per rank, measured communication;
4. Megatron-style tensor parallelism: the MLP split, then a whole Transformer block on 2 and 4 ranks;
5. pipeline schedules (GPipe, 1F1B) simulated: bubbles and activation memory;
6. the page's communication arithmetic for a 7B model.

    uv run python labs/pretraining/distributed-training/run.py [--quick] [--seed N]

The workers are started with ``spawn``, which imports this file again in every process: that is why
everything below sits under the main guard.
"""

import numpy as np
import torch
import torch.nn.functional as F

from dmxsf import lab
from dmxsf.pretraining import accounting as acc
from dmxsf.pretraining import distributed as dst
from dmxsf.pretraining import parallel as pl

EQUAL = [16] * 8
UNEQUAL = [16, 14, 12, 12, 10, 8, 8, 4]  # the page's lengths
# the keys of ddp_weighting()["error"]
MEAN_OF_MEANS, ALL_TOKENS = "A: mean of the ranks' means", "B: divide by all valid tokens"
DDP, DDP_RECIPE = "DDP as shipped", "DDP, loss x world / all tokens"
ACCUMULATION = "gradient accumulation"


def sci(x: float) -> str:
    return f"{x:.2e}"


def main() -> None:
    args, report = lab.start(
        "Distributed training: gradients, ZeRO, tensor and pipeline parallelism",
        wiki=["pretraining/distributed-training"],
    )
    quick = args.quick

    # -----------------------------------------------------------------------
    report.section("1. How data-parallel gradients are averaged (4 processes, gloo)")
    report.say(
        "  8 sequences, 2 per process; error = |g - g_ref| / |g_ref| against one process, the whole batch,"
    )
    report.say("  loss divided by the number of valid next-token predictions in it")
    results = {}
    for kind in ("toy", "gpt"):
        results[kind] = {
            name: dst.ddp_weighting(kind, lengths)
            for name, lengths in (("equal", EQUAL), ("unequal", UNEQUAL))
        }
        label = (
            "the page's toy model (embedding + linear)"
            if kind == "toy"
            else "a real 2-layer decoder"
        )
        report.say(f"\n  {label}:")
        methods = list(results[kind]["equal"]["error"])
        report.table(
            ["way of averaging", "equal lengths", f"lengths {UNEQUAL}"],
            [
                [
                    m,
                    sci(results[kind]["equal"]["error"][m]),
                    sci(results[kind]["unequal"]["error"][m]),
                ]
                for m in methods
            ],
        )
    tokens = results["toy"]["unequal"]["tokens"]
    share = [t / sum(tokens) for t in tokens]
    report.say(
        f"\n  valid predictions per rank {tokens}: they should weigh "
        f"{', '.join(f'{s:.1%}' for s in share)}; averaging the means gives each 25%"
    )
    toy_eq, toy_un = results["toy"]["equal"]["error"], results["toy"]["unequal"]["error"]
    gpt_eq, gpt_un = results["gpt"]["equal"]["error"], results["gpt"]["unequal"]["error"]
    report.check(
        "equal lengths: every way of averaging agrees with the reference to float32 rounding",
        max(toy_eq.values()) < 1e-6 and max(gpt_eq.values()) < 1e-6,
        f"largest error {max(*toy_eq.values(), *gpt_eq.values()):.1e}; the page reports 2.45e-07",
    )
    report.check(
        "the page's numbers: unequal lengths bias the average of the means by 36.5%",
        0.36 < toy_un[MEAN_OF_MEANS] < 0.37,
        f"{toy_un[MEAN_OF_MEANS]:.3f}",
    )
    report.check(
        "dividing by all valid tokens brings the error back to rounding",
        toy_un[ALL_TOKENS] < 1e-6 and gpt_un[ALL_TOKENS] < 1e-6,
        f"{toy_un[ALL_TOKENS]:.1e} on the toy model, {gpt_un[ALL_TOKENS]:.1e} on the decoder; "
        "the page reports 1.60e-07",
    )
    report.check(
        "PyTorch's DDP is method A, and 'loss x world / all tokens' is method B",
        abs(toy_un[DDP] - toy_un[MEAN_OF_MEANS]) < 1e-3 * toy_un[MEAN_OF_MEANS]
        and toy_un[DDP_RECIPE] < 1e-6,
        f"DDP {toy_un[DDP]:.3f} against {toy_un[MEAN_OF_MEANS]:.3f}; the recipe {toy_un[DDP_RECIPE]:.1e}",
    )
    report.check(
        "gradient accumulation has the same bias: time or space, the means are averaged the same way",
        abs(toy_un[ACCUMULATION] - toy_un[MEAN_OF_MEANS]) < 1e-3 * toy_un[MEAN_OF_MEANS]
        and gpt_un[ACCUMULATION] > 0.1,
        f"{toy_un[ACCUMULATION]:.3f} on the toy model, {gpt_un[ACCUMULATION]:.3f} on the decoder",
    )
    report.check(
        "a real decoder is biased too",
        gpt_un[MEAN_OF_MEANS] > 0.1,
        f"{gpt_un[MEAN_OF_MEANS]:.3f}",
    )
    report.check(
        "the rank with 10 valid tokens counts 25% instead of 13.2%, 1.9 times too much",
        round(share[3], 3) == 0.132 and round(0.25 / share[3], 1) == 1.9,
        f"{share[3]:.1%}, x{0.25 / share[3]:.2f}",
    )

    # -----------------------------------------------------------------------
    report.section("2. Ring all-reduce, simulated")
    rng = np.random.default_rng(args.seed)
    size = 100_000 if quick else 1_000_000
    rows, ok = [], True
    for n in (2, 4, 8, 16):
        vectors = [rng.standard_normal(size) for _ in range(n)]
        out, sent = pl.ring_all_reduce(vectors)
        correct = all(np.allclose(o, sum(vectors)) for o in out)
        ok &= correct and abs(sent[0] / size - 2 * (n - 1) / n) < 1e-9
        rows.append([n, correct, f"{sent[0] / size:.4f}", f"{2 * (n - 1) / n:.4f}", 2 * (n - 1)])
    report.table(["n", "all ranks hold the sum", "sent per rank / size", "2(n-1)/n", "steps"], rows)
    report.check(
        "each rank sends 2(n-1)/n of the gradient: about 2S however many ranks, but 2(n-1) steps of latency",
        ok,
    )

    # -----------------------------------------------------------------------
    n_ranks, steps, batch = 4, (3 if quick else 10), 16
    report.section(
        f"3. Data parallelism, ZeRO-1 and ZeRO-3 side by side ({n_ranks} processes, {steps} steps)"
    )
    ref = dst.train_reference(steps, batch)
    runs = {stage: dst.train_sharded(stage, n_ranks, steps, batch) for stage in (0, 1, 3)}
    n = runs[0][0]["n_params"]
    names = {0: "data parallel", 1: "ZeRO-1", 3: "ZeRO-3 / FSDP"}
    report.say(
        f"  a {n:,}-parameter Llama-style decoder in float64, AdamW without weight decay, batch of {batch} "
        f"sequences x {dst.SHARD_CFG.block_size} bytes of TinyShakespeare, {n_ranks} ranks"
    )
    report.say(
        "  values held per rank, as a multiple of the number of parameters (what each rank keeps between steps):"
    )
    table, diffs, ratios = [], {}, {}
    for stage, ranks in runs.items():
        held, first = ranks[0]["held"], ranks[0]
        diffs[stage] = max((r["flat"] - ref).abs().max().item() for r in ranks)
        ratios[stage] = first["sent"] / n
        calls = ", ".join(f"{v} {k.replace('_', '-')}" for k, v in first["calls"].items() if v)
        table.append(
            [
                names[stage],
                f"{held['weights'] / n:.3f}",
                f"{held['gradients'] / n:.3f}",
                f"{held['optimizer'] / n:.3f}",
                f"{sum(held.values()) / n:.3f}",
                f"{ratios[stage]:.3f}",
                calls,
                sci(diffs[stage]),
            ]
        )
    report.table(
        [
            "",
            "weights",
            "gradients",
            "Adam state",
            "total",
            "sent / step",
            "collectives per step",
            "max |w - ref|",
        ],
        table,
    )
    report.say(
        f"  (sent / step: elements a ring implementation sends per rank, in units of the {n:,} parameters)"
    )
    held = {s: r[0]["held"] for s, r in runs.items()}
    report.check(
        "all three end with the weights of one process on the whole batch, on every rank",
        max(diffs.values()) < 1e-9,
        f"largest difference {max(diffs.values()):.1e} (float64 rounding)",
    )
    report.check(
        "ZeRO-1 keeps full weights and gradients but 1/n of the optimizer state",
        held[1]["weights"] == n
        and held[1]["gradients"] == n
        and held[1]["optimizer"] * n_ranks == held[0]["optimizer"],
        f"Adam state {held[1]['optimizer']:,} against {held[0]['optimizer']:,}",
    )
    report.check(
        "ZeRO-3 keeps 1/n of all three",
        all(abs(held[3][k] * n_ranks - held[0][k]) <= 0.01 * held[0][k] for k in held[0]),
        f"weights {held[3]['weights']:,} of {n:,}",
    )
    report.check(
        "ZeRO-1 moves exactly as much as data parallelism (reduce-scatter + all-gather = all-reduce)",
        abs(ratios[1] - ratios[0]) < 1e-9,
        f"{ratios[1]:.3f} x psi against {ratios[0]:.3f} x psi, 2(n-1)/n = {2 * (n_ranks - 1) / n_ranks:.3f}",
    )
    report.check(
        "ZeRO-3 moves 1.5 times as much: all-gather going forward and coming back, reduce-scatter",
        abs(ratios[3] / ratios[0] - 1.5) < 0.01,
        f"{ratios[3] / ratios[0]:.3f}x; each of the 4 units is gathered twice and its gradient scattered once",
    )
    report.check(
        "the same loss curve in all three",
        max(abs(x - y) for x, y in zip(runs[0][0]["loss"], runs[3][0]["loss"], strict=True)) < 1e-9,
        f"loss {runs[0][0]['loss'][0]:.4f} -> {runs[0][0]['loss'][-1]:.4f}",
    )
    report.say(
        "\n  the same sharding in mixed precision (2 + 2 + 12 bytes per parameter), the page's example:"
    )
    psi, gpus = 7.5e9, 64
    gib = [acc.model_state_bytes(psi, gpus, z) / 2**30 for z in range(4)]
    paper = [acc.model_state_bytes(psi, gpus, z) / 1e9 for z in range(4)]
    report.table(
        ["7.5B on 64 GPUs", "GiB per GPU", "GB (the ZeRO paper's unit)"],
        [
            [nm, f"{g:.2f}", f"{p:.2f}"]
            for nm, g, p in zip(
                ["data parallel", "ZeRO-1", "ZeRO-2", "ZeRO-3"], gib, paper, strict=True
            )
        ],
    )
    report.check(
        "7.5B on 64 GPUs: 111.8, 29.2, 15.5, 1.75 GiB (the paper: 120, 31.4, 16.6, 1.88 GB)",
        [f"{g:.1f}" for g in gib[:3]] + [f"{gib[3]:.2f}"] == ["111.8", "29.2", "15.5", "1.75"]
        and [round(p, 1) for p in paper[1:3]] == [31.4, 16.6],
        ", ".join(f"{g:.2f}" for g in gib),
    )
    gpu = 80 * 2**30
    fits = [acc.model_state_bytes(70e9, 10**6, z) <= gpu for z in range(4)]  # a million GPUs
    report.check(
        "70B on 80 GiB cards: ZeRO-1 and ZeRO-2 never fit; ZeRO-3 needs 14 cards; 8 cards would need 130 GiB each",
        fits == [False, False, False, True]
        and acc.min_gpus_for_zero3(70e9, gpu) == 14
        and round(acc.model_state_bytes(70e9, 8, 3) / 2**30) == 130,
        f"{acc.model_state_bytes(70e9, 8, 3) / 2**30:.1f} GiB on 8 cards",
    )

    # -----------------------------------------------------------------------
    report.section("4. Tensor parallelism: first column, then row")
    torch.manual_seed(0)
    x = torch.randn(4, 16, 64, dtype=torch.float64)
    w1 = torch.randn(64, 256, dtype=torch.float64) / 8
    w2 = torch.randn(256, 64, dtype=torch.float64) / 16
    expect = F.gelu(x @ w1) @ w2
    errors = {}
    for t in (2, 4, 8):
        parts = [
            F.gelu(x @ a) @ b for a, b in zip(w1.chunk(t, dim=1), w2.chunk(t, dim=0), strict=True)
        ]
        errors[t] = (sum(parts) - expect).abs().max().item()
    xs, ws = x.chunk(4, dim=-1), w1.chunk(4, dim=0)
    wrong = sum(F.gelu(xi @ wi) for xi, wi in zip(xs, ws, strict=True)) @ w2
    right = F.gelu(sum(xi @ wi for xi, wi in zip(xs, ws, strict=True))) @ w2
    wrong_err, right_err = (wrong - expect).abs().max().item(), (right - expect).abs().max().item()
    report.table(
        ["split of the MLP  Y = GeLU(XA) B", "max error"],
        [[f"A by columns, B by rows, t={t}", sci(e)] for t, e in errors.items()]
        + [["A by rows, activate each part, then add", sci(wrong_err)]]
        + [["A by rows, add, then activate (one more all-reduce)", sci(right_err)]],
    )
    report.check(
        "column then row is exact: one all-reduce at the end of the MLP",
        max(errors.values()) < 1e-13,
        f"largest error {max(errors.values()):.1e}; the page reports 2.2e-15 ... 3.1e-15",
    )
    report.check(
        "activating partial sums is wrong (error of the order of the output)",
        wrong_err > 0.1 and right_err < 1e-13,
        f"{wrong_err:.1e} against {right_err:.1e} when the sum comes first; the page: 1.2e+00 and 1.8e-15",
    )
    layers = dst.TP_CFG.n_layer
    report.say(
        f"\n  a whole pre-norm block (GQA attention with 8 query / 4 KV heads, RoPE, GELU MLP), {layers} layers, "
        "float64, real processes:"
    )
    table, ok_counts, ok_equal, ok_params = [], True, True, True
    for t in (2,) if quick else (2, 4):
        res = dst.tensor_parallel_check(t)
        r = res[0]
        ok_counts &= all(
            q["forward_all_reduces"] == 2 * layers and q["backward_all_reduces"] == 2 * layers
            for q in res
        )
        ok_equal &= all(max(q["out"], q["x_grad"], q["weight_grad"]) < 1e-12 for q in res)
        ok_params &= r["matrix_params"] * t == r["full_matrix_params"]
        table.append(
            [
                t,
                sci(max(q["out"] for q in res)),
                sci(max(q["x_grad"] for q in res)),
                sci(max(q["weight_grad"] for q in res)),
                f"{r['forward_all_reduces'] // layers} + {r['backward_all_reduces'] // layers}",
                f"{r['matrix_params']:,} of {r['full_matrix_params']:,}",
            ]
        )
    report.table(
        [
            "t",
            "output",
            "input grad",
            "weight grads",
            "all-reduces per layer (fwd + bwd)",
            "matrix params per rank",
        ],
        table,
    )
    report.check(
        "split over t ranks, the block gives the same output, input gradient and every weight gradient",
        ok_equal,
        "relative errors at rounding level, over every rank",
    )
    report.check(
        "the page's count: 2 all-reduces in the forward pass and 2 in the backward pass, per layer",
        ok_counts,
        f"{2 * layers} + {2 * layers} for {layers} layers",
    )
    report.check("every rank holds 1/t of the matrix weights", ok_params)

    # -----------------------------------------------------------------------
    report.section(
        "5. Pipeline schedules: bubbles and activation memory (1 time unit forward, 2 backward)"
    )
    cases = [(4, 4), (4, 8), (4, 16), (4, 32), (8, 32)]
    page_total = {(4, 4): 21, (4, 8): 33, (4, 16): 57, (4, 32): 105, (8, 32): 117}
    rows, ok_sim, ok_same = [], True, True
    for p, m in cases:
        g, f1 = pl.pipeline(p, m, "gpipe"), pl.pipeline(p, m, "1f1b")
        formula = pl.idle_fraction(p, m)
        ok_sim &= g.total == page_total[(p, m)] and abs(g.idle - formula) < 1e-12
        ok_same &= f1.total == g.total and abs(f1.idle - g.idle) < 1e-12
        rows.append(
            [
                p,
                m,
                f"{g.total:.0f}",
                f"{g.idle:.1%}",
                f"{formula:.1%}",
                f"{pl.idle_fraction(p, m, v=2):.1%}",
                g.peak[0],
                f1.peak[0],
            ]
        )
    report.table(
        [
            "p",
            "m",
            "step time",
            "idle (simulated)",
            "(p-1)/(m+p-1)",
            "idle, 2 chunks per GPU",
            "GPipe: live on stage 1",
            "1F1B",
        ],
        rows,
    )
    report.check(
        "the simulated time is (m+p-1)(t_fwd+t_bwd) and the idle share (p-1)/(m+p-1): the page's 21, 33, 57, 105, 117",
        ok_sim,
    )
    report.check("1F1B has exactly the same bubble as GPipe", ok_same)
    g, f1 = pl.pipeline(4, 8, "gpipe"), pl.pipeline(4, 8, "1f1b")
    report.check(
        "what 1F1B changes is memory: p=4, m=8 keeps 8 micro-batches alive on the first stage under GPipe, 4 under 1F1B",
        (g.peak[0], f1.peak[0]) == (8, 4),
        f"live activations per stage: GPipe {g.peak}, 1F1B {f1.peak}",
    )
    report.check(
        "cutting the pipeline finer needs more micro-batches: m=32 gives 8.6% with 4 stages and 17.9% with 8",
        round(pl.idle_fraction(4, 32), 3) == 0.086 and round(pl.idle_fraction(8, 32), 3) == 0.179,
    )
    report.check(
        "exercise 2: 8 stages and at most 10% idle need at least 63 micro-batches",
        pl.min_microbatches(8, 0.10) == 63
        and pl.idle_fraction(8, 63) <= 0.1 < pl.idle_fraction(8, 62),
        f"{pl.idle_fraction(8, 63):.2%} at 63, {pl.idle_fraction(8, 62):.2%} at 62",
    )

    # -----------------------------------------------------------------------
    report.section("6. What the communication costs for a 7B model (illustrative hardware numbers)")
    psi, h, layers_, seq, bf16, gb = 6.738e9, 4096, 32, 4096, 2, 1e9
    peak, mfu, nvlink, ib = 989e12, 0.40, 160 * gb, 50 * gb
    report.say(
        "  Llama 2 7B shapes (6.738B parameters, 32 layers, hidden 4096, sequence 4096), BF16 gradients;\n"
        "  bandwidths from the DeepSeek-V3 paper (NVLink 160 GB/s, InfiniBand 50 GB/s), an H100 at 989 TFLOPs and\n"
        "  MFU 40%. These do not come from one machine: they give the orders of magnitude, nothing measured."
    )
    grads = psi * bf16
    ddp_gb, z3_gb = pl.ring_bytes(64, grads) / gb, pl.zero3_bytes(64, grads) / gb
    report.table(
        ["data parallel, 64 GPUs", "GB per GPU per step"],
        [["DDP / ZeRO-1 / ZeRO-2", f"{ddp_gb:.1f}"], ["ZeRO-3 / FSDP", f"{z3_gb:.1f}"]],
    )
    act = 1 * seq * h * bf16
    tp_rows, tp_ok = [], []
    for t in (2, 4, 8):
        tp = pl.tensor_parallel_bytes(t, layers_, act)
        compute = 6 * psi * seq / t / (peak * mfu)
        tp_rows.append(
            [
                t,
                f"{tp / gb:.1f}",
                f"{tp / nvlink * 1e3:.0f}",
                f"{tp / ib * 1e3:.0f}",
                f"{compute * 1e3:.0f}",
            ]
        )
        tp_ok.append(
            (
                round(tp / gb, 1),
                round(tp / nvlink * 1e3),
                round(tp / ib * 1e3),
                round(compute * 1e3),
            )
        )
    report.say(
        "\n  tensor parallelism, one micro-batch of one sequence (ms; NVLink / InfiniBand are the transfer times):"
    )
    report.table(["t", "GB per GPU", "NVLink ms", "InfiniBand ms", "compute ms"], tp_rows)
    boundary = pl.pipeline_boundary_bytes(act) / 1e6
    report.say(
        f"\n  pipeline parallelism: {boundary:.1f} MB per micro-batch across each stage boundary"
    )
    comm = pl.ring_bytes(64, grads) / ib
    glob = 4 * 2**20
    report.say(
        f"\n  fixed global batch of {glob // 2**20}M tokens, gradient sync {comm:.2f} s (an upper bound on scaling):"
    )
    rows, eff = [], {}
    for g_ in (64, 128, 256, 512, 1024):
        compute, exposed, e = pl.scaling_row(g_, glob, psi, comm, peak, mfu)
        eff[g_] = (round(compute, 2), round(exposed, 2), round(e, 3))
        rows.append([g_, glob // g_, f"{compute:.2f}", f"{exposed:.2f}", f"{e:.1%}"])
    report.table(["GPUs", "tokens per GPU", "compute s", "exposed comm s", "efficiency"], rows)
    need = pl.tokens_to_hide(comm, psi, peak, mfu)
    report.say(f"  the backward pass hides the sync from about {need:,.0f} tokens per GPU and step")
    report.check(
        "the page: 26.5 GB (DDP) and 39.8 GB (ZeRO-3), a factor 1.5",
        (round(ddp_gb, 1), round(z3_gb, 1)) == (26.5, 39.8) and abs(z3_gb / ddp_gb - 1.5) < 1e-9,
    )
    report.check(
        "the page: tensor parallelism at t=8 needs 7.5 GB, 47 ms on NVLink against 52 ms of compute, 150 ms on InfiniBand",
        tp_ok == [(4.3, 27, 86, 209), (6.4, 40, 129, 105), (7.5, 47, 150, 52)],
    )
    report.check(
        "the page: a pipeline boundary moves 67.1 MB per micro-batch, two orders of magnitude less than t=8",
        round(boundary, 1) == 67.1
        and pl.tensor_parallel_bytes(8, layers_, act) / 1e6 / boundary > 100,
        f"{pl.tensor_parallel_bytes(8, layers_, act) / 1e6 / boundary:.0f}x",
    )
    report.check(
        "the page: efficiency stays at 100% to 512 GPUs and falls to 62.5% at 1024; about 7,788 tokens per GPU hide the sync",
        eff[512] == (0.84, 0.0, 1.0) and eff[1024] == (0.42, 0.25, 0.625) and round(need) == 7788,
    )
    ex1 = [pl.ring_bytes(k, grads) / gb for k in (8, 1024)]
    report.check(
        "exercise 1: ring all-reduce of the 13.476 GB of gradients sends 23.6 GB per GPU on 8 GPUs and 26.9 GB on 1024",
        [round(v, 1) for v in ex1] == [23.6, 26.9],
        f"{ex1[1] / ex1[0] - 1:.0%} more for 128 times the GPUs",
    )
    ratio = {hh: pl.tensor_parallel_bytes(8, 1, hh * bf16) / (72 * hh * hh) for hh in (4096, 8192)}
    report.check(
        "exercise 4: doubling the hidden size halves tensor parallelism's communication per FLOP",
        abs(ratio[8192] / ratio[4096] - 0.5) < 1e-9,
        "16 (t-1)/t h bytes against 72 h^2 FLOPs per token and layer",
    )
    report.finish()


if __name__ == "__main__":
    main()
