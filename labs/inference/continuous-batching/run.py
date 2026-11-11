"""Lab: serving. Continuous batching, paged KV memory, the prefix cache, TTFT and capacity planning.

Course pages: https://dmxsf.com/wiki/inference/continuous-batching/
              https://dmxsf.com/wiki/inference/serving/

The pages measure llama.cpp on an Apple M2 Max. This lab does two things with them. It recomputes
every number that is arithmetic or a simulation (the KV-cache ledger, the scheduler on paper, the
capacity example): those come out identical on any machine. And it runs the same ideas on a real
model, dmxsf's 0.84M-parameter byte-level GPT on the CPU, through a small engine written for this
course (dmxsf.inference.serving): ragged batched decoding over slots, static and continuous
scheduling, a paged KV pool with copy-on-write, a hash-based prefix cache. Read the ratios there,
not the milliseconds: a tiny model on a CPU is limited by per-operation overhead, not by memory
bandwidth, so its curves have the pages' shapes but not their numbers.

    uv run python labs/inference/continuous-batching/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import math
import statistics as st
import time

import numpy as np
import torch

from dmxsf import lab, models
from dmxsf.architecture.configs import BY_NAME, kv_bytes_per_token
from dmxsf.data import ByteCorpus
from dmxsf.inference.generate import generate
from dmxsf.inference.kv_cache import KVCache
from dmxsf.inference.serving import (
    BatchedDecoder,
    PagedKV,
    PrefixCache,
    block_hashes,
    machines_needed,
    make_lengths,
    queue_simulation,
    serve,
    simulate,
    step_from_table,
)

GIB, MIB = 2**30, 2**20

# The pages' measured tables (llama.cpp, Qwen2.5-7B Q4_K_M, Apple M2 Max): seconds per decode step
# by batch size; the time one 128-token prompt stalls the batch; per-request speed and mean time to
# first token by concurrency.
PAGE_STEP = {1: 0.0249, 2: 0.0404, 3: 0.0562, 4: 0.0579, 6: 0.0822}
PAGE_STEP |= {8: 0.0984, 12: 0.0965, 16: 0.1003, 24: 0.1016, 32: 0.1035}
PAGE_STALL = 0.343
PAGE_SPEED = {1: 40.1, 2: 24.8, 3: 17.8, 4: 17.3, 6: 12.2, 8: 10.2, 12: 10.4, 16: 10.0}
PAGE_TTFT = {1: 0.19, 2: 0.31, 3: 0.39, 4: 0.47, 6: 0.76, 8: 0.94, 12: 1.50, 16: 1.99}


def summary(done: list[float], total_tokens: int) -> list[float]:
    """Makespan, tokens per second, mean and 90th percentile of the completion times."""
    d = sorted(done)
    return [d[-1], total_tokens / d[-1], st.mean(d), d[int(0.9 * len(d))]]


def main() -> None:
    args, report = lab.start(
        "Serving: continuous batching, paged KV memory, prefix cache, TTFT, capacity",
        wiki=["inference/continuous-batching", "inference/serving"],
    )
    QUICK = args.quick
    torch.manual_seed(args.seed)

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.check(name, ok, detail)

    model = models.pretrained("base", quick=QUICK, seed=args.seed).eval()
    val = ByteCorpus.shakespeare().val

    def tokens(start: int, n: int) -> torch.Tensor:
        return torch.from_numpy(val[start : start + n].astype(np.int64))

    # -----------------------------------------------------------------------------------------
    report.section("1. Why batching pays: the roofline, and a batched decode step that is exact")
    ratio = 989e12 / 3.35e12
    q7 = BY_NAME["Qwen2.5 7B"]
    per_token = kv_bytes_per_token(q7.config, 2)
    weights = 2 * q7.params
    breakeven = weights / (4096 * per_token)
    report.say(
        f"  H100 SXM: 989 TFLOPS / 3.35 TB/s = {ratio:.0f} FLOP per byte. A linear layer applied to B tokens\n"
        f"  does B FLOP for every byte of BF16 weight it reads, so below B of about {ratio:.0f} a step is paid for"
        " by\n  reading the weights and extra sequences ride along almost free. The KV cache cannot be shared"
        f" that way:\n  Qwen2.5-7B reads {4096 * per_token / MIB:.0f} MiB of it per 4,096-token sequence against"
        f" {weights / GIB:.1f} GiB of weights, so {breakeven:.0f}\n  sequences read as much KV as weights."
    )
    report.check(
        "295 FLOP/B; Qwen2.5-7B: 224 MiB of KV per 4,096-token sequence, 14.2 GiB of weights, 65 sequences",
        round(ratio) == 295
        and 4096 * per_token == 224 * MIB
        and round(weights / GIB, 1) == 14.2
        and round(breakeven) == 65,
        f"{ratio:.1f}, {4096 * per_token / MIB:.0f} MiB, {weights / GIB:.2f} GiB, {breakeven:.1f}",
    )

    # one decode step for B sequences, by B
    sizes = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    context, reps = 64, 8 if QUICK else 30
    decoder = BatchedDecoder(model, 32, 128)
    for slot in range(32):
        decoder.prefill(slot, tokens(0, context))
    step_ms: dict[int, float] = {}
    for b in sizes:
        times = []
        for _ in range(reps):
            decoder.length = [context] * 32  # the same context every time
            ids = torch.randint(0, 256, (b,))
            t0 = time.perf_counter()
            decoder.step(list(range(b)), ids)
            times.append(time.perf_counter() - t0)
        step_ms[b] = st.median(times[2:]) * 1e3
    rows = [
        [
            b,
            f"{step_ms[b]:.2f}",
            f"{b / step_ms[b] * 1e3:,.0f}",
            f"{1e3 / step_ms[b]:,.0f}",
            f"{b * step_ms[1] / step_ms[b]:.1f}x",
        ]
        for b in sizes
    ]
    report.say(
        f"\n  one decode step on the CPU, {context} tokens of context per sequence (median of {reps - 2}):"
    )
    report.table(["batch", "step ms", "total tok/s", "per-request tok/s", "total vs batch 1"], rows)
    gain16 = 16 * step_ms[1] / step_ms[16]
    report.check(
        "batching pays on this machine too: 16 sequences give more than 3x the tokens per second of one",
        gain16 > 3,
        f"{gain16:.1f}x the total throughput; each request runs at {step_ms[1] / step_ms[16]:.2f} of its solo speed",
    )

    # ragged: sequences of different lengths decoded together equal each one decoded alone
    prompts = [tokens(50 * i, n) for i, n in enumerate((5, 13, 29, 41, 60))]
    dec = BatchedDecoder(model, 5, 128)
    solo = [KVCache(len(model.blocks)) for _ in prompts]
    last = []
    for i, p in enumerate(prompts):
        dec.prefill(i, p)
        last.append(int(model(p[None], cache=solo[i])[0, -1].argmax()))
    worst, same, count = 0.0, 0, 0
    for _ in range(10):
        batched = dec.step(list(range(5)), torch.tensor(last))
        new = []
        for i in range(5):
            ref = model(torch.tensor([[last[i]]]), cache=solo[i])[0, -1]
            worst = max(worst, (batched[i] - ref).abs().max().item())
            same += int(batched[i].argmax() == ref.argmax())
            count += 1
            new.append(int(ref.argmax()))
        last = new
    report.check(
        "five sequences of 5, 13, 29, 41 and 60 tokens decoded in one batch match each decoded alone",
        worst < 1e-4 and same == count,
        f"largest logit difference {worst:.1e}; same next byte in {same} of {count} steps",
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. The page's scheduler on paper: static against continuous batching (exact)")
    lengths = make_lengths()
    total = sum(lengths)
    step = step_from_table(PAGE_STEP)
    shape = (min(lengths), st.median(lengths), round(st.mean(lengths)), max(lengths), total)
    report.say(
        f"  {len(lengths)} requests, output tokens: min {shape[0]}, median {shape[1]:.0f}, mean {shape[2]},"
        f" max {shape[3]}, total {total:,}"
    )
    report.check(
        "the page's workload: min 15, median 61, mean 97, max 453, total 4,656",
        shape == (15, 61, 97, 453, 4656),
        str(shape),
    )
    result, rows = {}, []
    for name, continuous in (("continuous", True), ("static", False)):
        result[name] = summary(simulate(lengths, 16, continuous, PAGE_STALL, step), total)
        rows.append([name, *[f"{x:.1f}" for x in result[name]]])
    report.say(
        f"\n  16 slots; a 128-token prompt stalls the whole batch for {PAGE_STALL} s; step times from the page's table:"
    )
    report.table(["mode", "makespan s", "tok/s", "mean done s", "p90 done s"], rows)
    report.check(
        "continuous 52.5 s, 88.6 tok/s, 29.9, 47.0 and static 68.6, 67.9, 37.7, 64.0, as printed on the page",
        [round(x, 1) for x in result["continuous"]] == [52.5, 88.6, 29.9, 47.0]
        and [round(x, 1) for x in result["static"]] == [68.6, 67.9, 37.7, 64.0],
        f"continuous {[round(x, 1) for x in result['continuous']]}, static {[round(x, 1) for x in result['static']]}",
    )

    ideal = total / 16 * PAGE_STEP[16]
    prefill_all = len(lengths) * PAGE_STALL
    admitted: list[float] = []
    simulate(lengths, 16, True, PAGE_STALL, step, admitted)
    drain = result["continuous"][0] - admitted[-1]
    longest = [max(lengths[i : i + 16]) for i in (0, 16, 32)]
    decode_c = result["continuous"][0] - prefill_all
    decode_s = result["static"][0] - prefill_all
    report.say(
        f"\n  where the time goes: {total:,} tokens at 16 sequences of {PAGE_STEP[16] * 1e3:.1f} ms would take {ideal:.1f} s;"
        f" prefill takes {prefill_all:.1f} s in both schedules.\n  decoding takes {decode_c:.1f} s"
        f" ({ideal / decode_c:.0%} of ideal) when continuous and {decode_s:.1f} s ({ideal / decode_s:.0%}) when static."
        f" Static batches\n  of 16 wait for outputs of {longest[0]}, {longest[1]} and {longest[2]} tokens; in the"
        f" continuous schedule the last request starts at {admitted[-1]:.1f} s,\n  then {drain:.1f} s of drain"
        " while the batch shrinks."
    )
    report.check(
        "ideal 29 s, prefill 16.5 s, decoding 36 s against 52 s (81% against 56% of ideal); batches wait for"
        " 218, 453, 192; last start 38 s, 14.6 s drain",
        round(ideal) == 29
        and round(prefill_all, 1) == 16.5
        and (round(decode_c), round(decode_s)) == (36, 52)
        and (round(ideal / decode_c * 100), round(ideal / decode_s * 100)) == (81, 56)
        and longest == [218, 453, 192]
        and round(admitted[-1]) == 38
        and round(drain, 1) == 14.6,
        f"{ideal:.1f}, {prefill_all:.1f}, {decode_c:.1f}, {decode_s:.1f}, {longest}, {admitted[-1]:.1f}, {drain:.1f}",
    )

    scenarios = [
        ("the page's setting", lengths, 16, PAGE_STALL, step),
        ("every output 100 tokens", [100] * 48, 16, PAGE_STALL, step),
        ("step time independent of batch", lengths, 16, PAGE_STALL, lambda b: PAGE_STEP[1]),
        ("step time proportional to batch", lengths, 16, PAGE_STALL, lambda b: b * PAGE_STEP[1]),
        ("1,024-token prompts (2.18 s)", lengths, 16, 2.18, step),
        ("2 slots", lengths, 2, PAGE_STALL, step),
    ]
    sc, rows = {}, []
    for name, lens, slots, stall, fn in scenarios:
        c = simulate(lens, slots, True, stall, fn)[-1]
        s = simulate(lens, slots, False, stall, fn)[-1]
        sc[name] = (c, s, sum(lens))
        rows.append(
            [
                name,
                f"{c:.1f}",
                f"{sum(lens) / c:.0f}",
                f"{s:.1f}",
                f"{sum(lens) / s:.0f}",
                f"{1 - c / s:.0%}",
            ]
        )
    report.say("\n  the bullets under the page's widget, recomputed (same simulator):")
    report.table(
        ["scenario", "continuous s", "tok/s", "static s", "tok/s", "continuous saves"], rows
    )
    c_id, s_id, _ = sc["every output 100 tokens"]
    c_ideal, s_ideal, _ = sc["step time independent of batch"]
    c_prop, s_prop, _ = sc["step time proportional to batch"]
    c_long, s_long, _ = sc["1,024-token prompts (2.18 s)"]
    c_two, s_two, _ = sc["2 slots"]
    report.check(
        "equal outputs: both 46.6 s. Batch-independent steps: 164 and 123 tok/s, 25% saved. Proportional steps:"
        " no difference. 1,024-token prompts: 105 s of prefill, 141 s, 10%. Two slots: 7%",
        (round(c_id, 1), round(s_id, 1)) == (46.6, 46.6)
        and (round(total / c_ideal), round(total / s_ideal), round(1 - c_ideal / s_ideal, 2))
        == (164, 123, 0.25)
        and abs(c_prop - s_prop) < 1e-9
        and (round(48 * 2.18), round(c_long), round(1 - c_long / s_long, 2)) == (105, 141, 0.10)
        and round(1 - c_two / s_two, 2) == 0.07,
        f"{c_id:.1f}/{s_id:.1f}, {total / c_ideal:.0f}/{total / s_ideal:.0f}, {c_prop:.1f}/{s_prop:.1f},"
        f" {c_long:.1f} ({1 - c_long / s_long:.1%}), {1 - c_two / s_two:.1%}",
    )
    one_long = [100] + [10] * 9
    steps = [simulate(one_long, 4, c, 0.0, lambda b: 1.0)[-1] for c in (False, True)]
    report.check(
        "the page's exercise: 4 slots, one request of 100 steps and nine of 10 take 120 steps static, 100 continuous",
        steps == [120.0, 100.0],
        f"{steps}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. The same loop on a real model")
    n_out = [
        min(100, max(3, round(0.2 * n))) for n in lengths
    ]  # the page's workload, scaled to 128 positions
    requests = [(tokens(2000 + 37 * i, 16), n) for i, n in enumerate(n_out)]
    reference = [generate(model, p[None], n, temperature=0.0)[0].tolist() for p, n in requests]
    n_rep = 2 if QUICK else 3
    runs: dict[str, list] = {"continuous": [], "static": []}
    exact = {"continuous": True, "static": True}
    for _ in range(n_rep):  # alternate, so a drift in the machine hits both the same way
        for name, continuous in (("continuous", True), ("static", False)):
            out = serve(model, requests, 16, continuous, 128)
            runs[name].append(out)
            exact[name] &= all(o.tokens == r for o, r in zip(out, reference, strict=True))
    stalls = []
    for _ in range(20):
        t0 = time.perf_counter()
        decoder.prefill(0, requests[0][0])
        stalls.append(time.perf_counter() - t0)
    stall = st.median(stalls)
    table = {b: ms / 1e3 for b, ms in step_ms.items()}
    sim_steps = [n - 1 for n in n_out]  # the first byte of a request comes from its prefill

    def cells(vals: list[float]) -> list[str]:  # makespan, tok/s, mean, p90 -> ms, tok/s, ms, ms
        return [
            f"{vals[0] * 1e3:.0f}",
            f"{vals[1]:,.0f}",
            f"{vals[2] * 1e3:.0f}",
            f"{vals[3] * 1e3:.0f}",
        ]

    rows, measured, predicted = [], {}, {}
    for name, continuous in (("continuous", True), ("static", False)):
        per_run = [summary([o.done for o in out], sum(n_out)) for out in runs[name]]
        measured[name] = [st.median(col) for col in zip(*per_run, strict=True)]
        ttft = st.median(st.mean(o.first_token for o in out) for out in runs[name])
        simulated = simulate(sim_steps, 16, continuous, stall, step_from_table(table))
        predicted[name] = summary(simulated, sum(n_out))
        rows.append([name, "real", *cells(measured[name]), f"{ttft * 1e3:.0f}"])
        rows.append([name, "simulated", *cells(predicted[name]), ""])
    report.say(
        f"  the page's workload scaled to this model's 128 positions: {len(requests)} requests, 16-byte prompts,"
        f" outputs of {min(n_out)} to {max(n_out)} bytes ({sum(n_out):,} in all),\n  16 slots, greedy decoding,"
        f" median of {n_rep} alternating runs. The simulation uses this machine's own step table (section 1)"
        f" and\n  its own prefill time ({stall * 1e3:.2f} ms for 16 bytes):"
    )
    report.table(
        ["mode", "", "makespan ms", "bytes/s", "mean done ms", "p90 done ms", "mean TTFT ms"], rows
    )
    report.check(
        "every request gets exactly the bytes it would get decoded alone, in both schedules and every run",
        all(exact.values()),
        f"{len(requests)} requests, {n_rep} runs each",
    )
    gap_c = measured["continuous"][0] / measured["static"][0]
    report.check(
        "continuous batching finishes the batch of requests sooner than static batching, and their average sooner",
        gap_c < 1 and measured["continuous"][2] < measured["static"][2],
        f"makespan {gap_c:.2f} of static, mean completion {measured['continuous'][2] / measured['static'][2]:.2f} of static",
    )
    errs = {m: predicted[m][0] / measured[m][0] - 1 for m in measured}
    full_check(
        "the simulator, fed only the measured step table and prefill time, predicts both makespans within 15%",
        all(abs(e) < 0.15 for e in errs.values()),
        ", ".join(f"{m}: {e:+.1%}" for m, e in errs.items()),
    )
    counted = {c: simulate(sim_steps, 16, c, 0.0, lambda b: 1.0)[-1] for c in (True, False)}
    report.say(
        f"  counted in steps alone, with no clock involved: continuous {counted[True]:.0f} decode steps, static"
        f" {counted[False]:.0f}, so {1 - counted[True] / counted[False]:.0%} fewer;"
        " that saving comes from the schedule, not from the hardware."
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. KV memory: the ledger, reserving against paging, the block table")
    pool = 0.9 * 80 * GIB - weights - 4 * GIB
    pool_tokens = pool / per_token
    reserve_fit = int(pool_tokens // 8192)
    paged_fit = int(pool_tokens // (math.ceil(1000 / 16) * 16))
    opt = 2 * 40 * 5120 * 2
    report.table(
        ["quantity (Qwen2.5-7B, 80 GiB card)", "value"],
        [
            ["KV cache per token", f"{per_token:,} bytes"],
            [
                "a 4,096-token slot / a 65,536-token pool",
                f"{4096 * per_token / MIB:.0f} MiB / {65536 * per_token / GIB:.2f} GiB",
            ],
            ["parameters / FP16 weights", f"{q7.params / 1e9:.3f} B / {weights / GIB:.1f} GiB"],
            [
                "KV pool (90% of memory, 4 GiB for activations)",
                f"{pool / GIB:.1f} GiB = {pool_tokens / 1e6:.2f} M tokens",
            ],
            ["sequences when every one reserves 8,192 tokens", f"{reserve_fit}"],
            ["sequences in 16-token blocks, 1,000 tokens each", f"{paged_fit}"],
            [
                "OPT-13B per token (2 x 40 x 5120 x 2 bytes)",
                f"{opt:,} bytes, {opt / per_token:.1f}x more",
            ],
        ],
    )
    report.check(
        "57,344 bytes per token, 224 MiB, 3.50 GiB, 7.616 B and 14.2 GiB, 53.8 GiB = 1.01 M tokens, 123 against 999 sequences, OPT-13B 14x",
        per_token == 57_344
        and round(65536 * per_token / GIB, 2) == 3.5
        and round(q7.params / 1e9, 3) == 7.616
        and round(pool / GIB, 1) == 53.8
        and round(pool_tokens / 1e6, 2) == 1.01
        and (reserve_fit, paged_fit) == (123, 999)
        and round(opt / per_token) == 14,
        f"{per_token}, {pool / GIB:.2f} GiB, {reserve_fit}, {paged_fit}, {opt / per_token:.2f}",
    )
    llama3 = BY_NAME["Llama 3 8B"]
    l3_token = kv_bytes_per_token(llama3.config, 2)
    room = 24 * GIB - 2 * llama3.params
    report.check(
        "exercise: Llama 3 8B needs 131,072 bytes per token, 1 GiB per 8,192-token request, and a 24 GiB card holds 9",
        l3_token == 131_072 and l3_token * 8192 == GIB and int(room // (l3_token * 8192)) == 9,
        f"{l3_token:,} bytes, {room / GIB:.2f} GiB left, {room / (l3_token * 8192):.2f} requests",
    )

    total_len = [128 + n for n in lengths]  # the page's workload with its 128-token prompts
    held = sum(total_len)
    blocks = sum(math.ceil(n / 16) for n in total_len) * 16
    waste = (blocks - held) / len(total_len)
    report.say(
        f"\n  the page's 48 requests hold {held:,} tokens in all. Reserving 8,192 for each puts {held / (48 * 8192):.1%} of the"
        f" reserved\n  memory to use; 16-token blocks allocated as needed put {held / blocks:.1%} to use, a waste of"
        f" {waste:.1f} tokens per request (never more than 15)."
    )
    report.check(
        "paged allocation wastes under 16 tokens per request and keeps over 90% of memory in use; reserving the maximum keeps under 5%",
        0 <= waste < 16 and held / blocks > 0.9 and held / (48 * 8192) < 0.05,
        f"{held / blocks:.1%} against {held / (48 * 8192):.1%}",
    )

    g = torch.Generator().manual_seed(args.seed)
    kv = PagedKV(n_blocks=16, block_size=4, n_heads=2, head_dim=8)
    seqs, dense = [kv.new_sequence() for _ in range(3)], [([], []) for _ in range(3)]
    goal = (10, 19, 23)
    for t in range(max(goal)):  # the sequences grow one token at a time, interleaved
        for i, seq in enumerate(seqs):
            if t < goal[i]:
                k, v = torch.randn(2, 1, 8, generator=g), torch.randn(2, 1, 8, generator=g)
                kv.append(seq, k, v)
                dense[i][0].append(k)
                dense[i][1].append(v)
    query = torch.randn(1, 2, 1, 8, generator=g)
    exact_kv = exact_attn = True
    for seq, (ks, vs) in zip(seqs, dense, strict=True):
        k, v = kv.gather(seq)
        kd, vd = torch.cat(ks, 1), torch.cat(vs, 1)
        exact_kv &= torch.equal(k, kd) and torch.equal(v, vd)
        paged_out = torch.nn.functional.scaled_dot_product_attention(query, k[None], v[None])
        dense_out = torch.nn.functional.scaled_dot_product_attention(query, kd[None], vd[None])
        exact_attn &= torch.equal(paged_out, dense_out)
    report.say(
        f"\n  three sequences of {goal[0]}, {goal[1]} and {goal[2]} tokens grow interleaved in a pool of 16 blocks of 4 tokens;"
        f" their block tables:"
    )
    report.table(
        ["sequence", "tokens", "block table (physical blocks)"],
        [[i, goal[i], str(seqs[i].table)] for i in range(3)],
    )
    needed = sum(math.ceil(n / 4) for n in goal)
    report.check(
        "the blocks of a sequence are not next to each other, yet gathering them gives back the dense keys and values, and attention is bit-for-bit the same",
        any(b - a != 1 for s in seqs for a, b in zip(s.table, s.table[1:], strict=False))
        and exact_kv
        and exact_attn
        and kv.alloc.used == needed,
        f"{kv.alloc.used} blocks in use for {sum(goal)} tokens; a dense layout reserving 32 tokens each would take 24",
    )
    # parallel sampling: four samples continue one 12-token prompt
    pool2 = PagedKV(n_blocks=24, block_size=4, n_heads=1, head_dim=2)
    first = pool2.new_sequence()
    pool2.append(first, torch.randn(1, 12, 2, generator=g), torch.randn(1, 12, 2, generator=g))
    samples = [first, *[pool2.fork(first) for _ in range(3)]]
    for sample in samples:
        pool2.append(sample, torch.randn(1, 5, 2, generator=g), torch.randn(1, 5, 2, generator=g))
    shared_used = pool2.alloc.used
    unshared = 4 * (math.ceil(12 / 4) + math.ceil(5 / 4))
    # a fork of a sequence whose last block is half full
    half = pool2.new_sequence()
    pool2.append(half, torch.randn(1, 6, 2, generator=g), torch.randn(1, 6, 2, generator=g))
    before = pool2.gather(half)[0].clone()
    twin = pool2.fork(half)
    pool2.append(twin, torch.randn(1, 1, 2, generator=g), torch.randn(1, 1, 2, generator=g))
    report.say(
        "\n  parallel sampling: four samples of 5 tokens continue one 12-token prompt. Sharing the prompt's blocks they"
        f" use {shared_used} blocks,\n  {unshared} if each owned a copy. A fork of a sequence whose last block is half"
        f" full copies that block the first time either side writes\n  into it ({pool2.copies} copy, the original is untouched)."
    )
    report.check(
        "sharing the prompt: 3 prompt blocks plus 2 per sample, no copies; writing into a shared half-full block copies it once and leaves the original unchanged",
        shared_used == 3 + 4 * 2
        and unshared == 20
        and pool2.copies == 1
        and torch.equal(pool2.gather(half)[0], before),
        f"{shared_used} blocks against {unshared}",
    )

    # llama.cpp slots: the context is split between them unless the cache is unified
    def per_slot(ctx: int, parallel: int, unified: bool) -> int:
        return ctx if unified else ctx // parallel

    report.check(
        "llama.cpp: -c 65536 -np 16 gives 4,096 tokens per slot, so a 6,000-token request is refused, unless the cache is unified (65,536)",
        per_slot(65536, 16, False) == 4096 < 6000 <= per_slot(65536, 16, True) == 65536,
        "4096 against 65536",
    )
    report.check(
        "exercise: -c 32768 -np 8 gives 4,096 per slot, so 20,000 tokens are refused; unified or -np 1 gives 32,768; -c 160000 is needed to keep eight slots",
        per_slot(32768, 8, False) == 4096
        and per_slot(32768, 8, True) == 32768
        and per_slot(32768, 1, False) == 32768
        and 20000 * 8 == 160000
        and per_slot(160000, 8, False) == 20000,
        "4096, 32768, 32768, 160000",
    )

    # -----------------------------------------------------------------------------------------
    report.section("5. The prefix cache: the same beginning is computed once")
    bs = 16
    doc = tokens(4000, 96)  # six full blocks
    question = lambda i: tokens(6000 + 50 * i, 16)  # noqa: E731
    header = lambda i: tokens(9000 + 10 * i, 8)  # noqa: E731
    cases = [
        ("A doc first, cache on", True, lambda i: torch.cat([doc, question(i)])),
        ("B header first, cache on", True, lambda i: torch.cat([header(i), doc, question(i)])),
        ("C doc first, cache off", False, lambda i: torch.cat([doc, question(i)])),
    ]
    n_rep, rows, counts, ms, worst = (3 if QUICK else 9), [], {}, {}, 0.0
    for name, on, build in cases:
        per_request = [[] for _ in range(4)]
        for rep in range(n_rep):
            cache = PrefixCache(model, bs)  # empty at the start of every repetition
            for i in range(4):
                prompt = build(i)
                t0 = time.perf_counter()
                logits, _, stats = cache.prefill(prompt, use_cache=on)
                per_request[i].append(time.perf_counter() - t0)
                if rep == 0:
                    counts.setdefault(name, []).append((stats.reused, stats.computed))
                    ref = model(prompt[None])[0, -1]
                    worst = max(worst, (logits - ref).abs().max().item())
        ms[name] = [st.median(t) * 1e3 for t in per_request]
        for i in range(4):
            rows.append([name, i + 1, counts[name][i][0], counts[name][i][1], f"{ms[name][i]:.2f}"])
    report.say(
        "  a 96-byte document (six 16-byte blocks) and a 16-byte question, four requests per case, each with a"
        f" different question;\n  B puts 8 bytes that change every time in front of the document. The"
        f" cache starts empty in each case; times are the median of {n_rep} repetitions:"
    )
    report.table(["case", "request", "reused", "computed", "prefill ms"], rows)
    report.check(
        "A: the first request computes all 112 bytes, the next three reuse the 96 of the document and compute 16; B: nothing is ever reused; C: nothing is reused",
        counts["A doc first, cache on"] == [(0, 112), (96, 16), (96, 16), (96, 16)]
        and counts["B header first, cache on"] == [(0, 120)] * 4
        and counts["C doc first, cache off"] == [(0, 112)] * 4,
        f"A {counts['A doc first, cache on']}, B {counts['B header first, cache on'][1]}",
    )
    report.check(
        "reusing cached keys and values does not change the logits",
        worst < 1e-4,
        f"largest difference from a full prefill {worst:.1e} over {sum(len(v) for v in counts.values())} requests",
    )
    hit_ms = st.mean(ms["A doc first, cache on"][1:])
    cold_ms = st.mean(ms["C doc first, cache off"])
    full_check(
        "a request that hits the cache prefills faster than one that does not",
        hit_ms < cold_ms / 1.3,
        f"{hit_ms:.2f} ms against {cold_ms:.2f} ms ({cold_ms / hit_ms:.1f}x); 16 of 112 bytes computed, not 1/7 of the time: fixed costs per forward pass",
    )

    system = list(range(1000))
    first = block_hashes([*system, *range(1000, 1050)], 16)
    second = block_hashes([*system, *range(2000, 2050)], 16)
    stamped = block_hashes([7777, *system, *range(2000, 2050)], 16)
    other = block_hashes([*system, *range(1000, 1050)], 16, salt="tenant-b")

    def shared_tokens(a: list[str], b: list[str]) -> int:
        n = next(
            (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
        )
        return 16 * n

    report.check(
        "the page's exercise: with 16-token blocks, a second request with the same 1,000-token system prompt reuses 992 tokens; one with a timestamp in front reuses 0; another tenant's salt reuses 0",
        (shared_tokens(first, second), shared_tokens(first, stamped), shared_tokens(first, other))
        == (992, 0, 0),
        f"{shared_tokens(first, second)}, {shared_tokens(first, stamped)}, {shared_tokens(first, other)}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("6. What a request costs: TTFT, TPOT, the two regimes, capacity")
    n_new, rounds = 40, 3 if QUICK else 7
    one = (tokens(3000, 16), n_new)
    alone, queued = [], []
    for _ in range(rounds):  # alone and queued alternate, so a drift in the machine hits both
        alone.append(serve(model, [one], 1, True)[0])
        queued.append(serve(model, [one] * 4, 1, True))  # one slot, four requests arriving together
    prefill_s = st.median(x.first_token for x in alone)
    tpot_s = st.median((x.done - x.first_token) / (n_new - 1) for x in alone)
    rows, errors = [], []
    for i in range(4):
        queue = st.median(q[i].admitted for q in queued)
        first = st.median(q[i].first_token for q in queued)
        done = st.median(q[i].done for q in queued)
        tpot = st.median((q[i].done - q[i].first_token) / (n_new - 1) for q in queued)
        formula = queue + prefill_s + (n_new - 1) * tpot_s  # queue + prefill + (N - 1) x TPOT
        errors.append(formula / done - 1)
        rows.append(
            [
                i + 1,
                f"{queue * 1e3:.1f}",
                f"{(first - queue) * 1e3:.2f}",
                f"{first * 1e3:.1f}",
                f"{tpot * 1e3:.2f}",
                f"{done * 1e3:.1f}",
                f"{formula * 1e3:.1f}",
            ]
        )
    report.say(
        f"  one slot, four identical requests at t = 0 (16-byte prompt, {n_new} new bytes). Alone, one request has"
        f" a prefill of {prefill_s * 1e3:.2f} ms and a TPOT of {tpot_s * 1e3:.2f} ms:"
    )
    report.table(
        [
            "request",
            "queue ms",
            "prefill ms",
            "TTFT ms",
            "TPOT ms",
            "end-to-end ms",
            "queue + prefill + (N-1) TPOT, ms",
        ],
        rows,
    )
    full_check(
        "end-to-end = queue + prefill + (N - 1) x TPOT: with prefill and TPOT measured alone, the formula is within 10% for all four requests",
        all(abs(e) < 0.10 for e in errors),
        ", ".join(f"{e:+.1%}" for e in errors),
    )

    rows, prefill_rate = [], {}
    for length in (1, 4, 16, 64, 120):
        times = []
        for _ in range(8 if QUICK else 20):
            t0 = time.perf_counter()
            decoder.prefill(0, tokens(100, length))
            times.append(time.perf_counter() - t0)
        prefill_rate[length] = length / st.median(times)
        rows.append([length, f"{st.median(times) * 1e3:.2f}", f"{prefill_rate[length]:,.0f}"])
    decode_rate = 1e3 / step_ms[1]
    report.say(
        f"\n  prefill speed against prompt length (decoding one sequence runs at {decode_rate:,.0f} bytes/s):"
    )
    report.table(["prompt bytes", "prefill ms", "bytes/s"], rows)
    report.check(
        "prefill, which handles every prompt position in one pass, is more than 10x faster per byte than decoding one sequence, and gets faster per byte as the prompt grows",
        prefill_rate[120] > 10 * decode_rate
        and prefill_rate[120] > prefill_rate[4] > prefill_rate[1],
        f"{prefill_rate[120]:,.0f} against {decode_rate:,.0f} bytes/s",
    )
    bound = 400 / 4.68
    report.table(
        ["the page's arithmetic", "value"],
        [
            ["M2 Max decode ceiling, 400 GB/s / 4.68 GB", f"{bound:.0f} tokens/s"],
            [
                "42.3 tokens/s sustained reads",
                f"{42.3 * 4.68:.0f} GB/s = {42.3 * 4.68 / 400:.0%} of the bandwidth",
            ],
            ["64.8 tokens/s when cool reads", f"{64.8 * 4.68:.0f} GB/s = {64.8 * 4.68 / 400:.0%}"],
            [
                "exercise: 3,000-token prompt, 200 tokens out",
                f"TTFT {3000 / 485:.2f} s, end to end {3000 / 485 + 199 / 42:.1f} s",
            ],
            [
                "exercise: 3.35 TB/s, FP16 weights 15.23 GB / Q4_K_M 4.68 GB",
                f"{3350 / (weights / 1e9):.0f} / {3350 / 4.68:.1f} tokens/s, {weights / 1e9 / 4.68:.2f}x",
            ],
        ],
    )
    report.check(
        "85 tokens/s ceiling, 303 GB/s = 76% at 64.8 tokens/s, 6.19 s and 10.9 s, 220 and 715 tokens/s (715.8, which the page truncates), 3.3x",
        round(bound) == 85
        and round(64.8 * 4.68) == 303
        and round(64.8 * 4.68 / 400, 2) == 0.76
        and (round(3000 / 485, 2), round(3000 / 485 + 199 / 42, 1)) == (6.19, 10.9)
        and (round(3350 / (weights / 1e9)), int(3350 / 4.68), round(weights / 1e9 / 4.68, 1))
        == (220, 715, 3.3),
        "",
    )

    # benchmark hygiene on this machine: the same work as "A" and "B"
    ids = torch.randint(0, 256, (8,))

    def unit() -> float:
        t0 = time.perf_counter()
        for _ in range(10):
            decoder.length = [context] * 32
            decoder.step(list(range(8)), ids)
        return time.perf_counter() - t0

    unit()
    n, trials = (6, 3) if QUICK else (20, 5)
    ratios: dict[str, list[float]] = {"A first, then B": [], "A, B, A, B, ...": []}
    for _ in range(trials):
        a = [unit() for _ in range(n)]
        b = [unit() for _ in range(n)]
        ratios["A first, then B"].append(st.mean(b) / st.mean(a))
        a, b = [], []
        for _ in range(n):
            a.append(unit())
            b.append(unit())
        ratios["A, B, A, B, ..."].append(st.mean(b) / st.mean(a))
    report.say(
        f"\n  A and B are the same code ({n} runs of 10 batched steps each, {trials} trials). What does a benchmark say"
        " about their ratio B / A,\n  depending on the order they run in?"
    )
    report.table(
        ["order", "B / A in each trial", "range"],
        [
            [k, " ".join(f"{x:.3f}" for x in v), f"{min(v):.3f} to {max(v):.3f}"]
            for k, v in ratios.items()
        ],
    )
    spread = {k: max(v) - min(v) for k, v in ratios.items()}
    narrower = spread["A, B, A, B, ..."] < spread["A first, then B"]
    report.say(
        "  alternating gave the narrower range here, as the page advises."
        if narrower
        else "  alternating did not give the narrower range on this run: the machine drifted little, so order mattered little."
    )
    floor = max(abs(x - 1) for v in ratios.values() for x in v)
    report.say(
        f"  across all trials B / A strayed up to {floor:.1%} from 1 with identical code: a difference between two"
        " real configurations\n  smaller than that is not a difference this benchmark can see."
    )

    rows, worst_gap = [], 0.0
    for rate in (
        1.5,
        2.5,
        3.5,
    ):  # four servers of mean service time 1: utilization 0.375, 0.625, 0.875
        lam, w, ell = queue_simulation(rate, 1.0, 4, 20_000 if QUICK else 200_000, seed=args.seed)
        worst_gap = max(worst_gap, abs(ell / (lam * w) - 1))
        rows.append([f"{rate / 4:.3f}", f"{lam:.3f}", f"{w:.3f}", f"{ell:.3f}", f"{lam * w:.3f}"])
    report.say(
        "\n  Little's law in a simulated queue (Poisson arrivals, exponential service, 4 servers):"
    )
    report.table(
        [
            "utilization",
            "arrivals/s",
            "W: mean time in system",
            "L: time-average number",
            "lambda x W",
        ],
        rows,
    )
    report.check(
        "L = lambda x W: the time-average number of requests in the system equals the arrival rate times the mean time in it",
        worst_gap < 0.005,
        f"largest relative gap {worst_gap:.2%}",
    )
    per_machine, w, in_flight, machines = machines_needed(0.8, 200, 17.0, PAGE_SPEED, PAGE_TTFT)
    report.say(
        f"\n  the page's capacity exercise: 17 tokens/s per user, 200 tokens per answer, 0.8 requests/s at the peak. One machine"
        f" serves {per_machine} at once\n  (17.3 tokens/s each; at 6 it falls to 12.2), a request stays W = 0.47 + 199 / 17.3 = {w:.1f} s, so"
        f" L = 0.8 x {w:.1f} = {in_flight:.1f}\n  requests are in flight and {machines} machines are needed; one machine"
        f" sustains {per_machine / w:.2f} requests/s."
    )
    report.check(
        "4 requests per machine, W = 12.0 s, L = 9.6, 3 machines, 0.33 requests/s per machine",
        (per_machine, round(w, 1), round(in_flight, 1), machines, round(per_machine / w, 2))
        == (4, 12.0, 9.6, 3, 0.33),
        f"{per_machine}, {w:.2f}, {in_flight:.2f}, {machines}",
    )
    report.finish()


if __name__ == "__main__":
    main()
