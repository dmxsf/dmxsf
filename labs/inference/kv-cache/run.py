"""Lab: the KV cache. The memory formula, what a cache saves, and the prefill / decode split.

Course page: https://dmxsf.com/wiki/inference/kv-cache/

The page's numbers come from one formula, bytes per token = 2 x layers x kv_heads x head_dim x
bytes, which is exact arithmetic and is checked against the real tensors of dmxsf's cache. The
speed claims are checked two ways: by counting how many token positions a forward pass has to
process (exact, hardware independent) and by timing a CPU run (machine dependent: read the ratios,
not the milliseconds).

    uv run python labs/inference/kv-cache/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import dataclasses
import statistics
import time

import torch

from dmxsf import lab
from dmxsf.architecture.configs import BY_NAME, kv_bytes_per_token, small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.inference.kv_cache import KVCache, StaticKVCache
from dmxsf.pretraining.accounting import matmul_params

KIB, GIB = 2**10, 2**30


def main() -> None:
    args, report = lab.start(
        "KV cache: the memory formula, exactness, the saving, prefill against decode",
        wiki=["inference/kv-cache"],
    )
    QUICK = args.quick
    torch.manual_seed(args.seed)

    # -----------------------------------------------------------------------------------------
    report.section("1. The memory formula and the page's numbers (exact)")
    names = ("Llama 2 7B", "Llama 3 8B", "Llama 3 70B", "Qwen2.5-0.5B")
    rows = []
    for name in names:
        m = BY_NAME[name]
        per_token = kv_bytes_per_token(m.config, 2)
        rows.append(
            [
                name,
                f"{m.config.n_layer} x {m.config.kv_heads} x {m.config.dim_per_head}",
                f"{per_token / KIB:,.0f}",
                f"{per_token * 8192 / GIB:.2f}",
                f"{per_token * 8192 * 32 / GIB:.1f}",
                f"{m.params * 2 / 1e9:.1f}",
            ]
        )
    report.say("  fp16 cache; the shape is layers x kv_heads x head_dim:")
    report.table(
        [
            "model",
            "shape",
            "KiB per token",
            "GiB per 8,192 tokens",
            "GiB for 32 sequences",
            "fp16 weights (GB)",
        ],
        rows,
    )
    llama3, llama2 = BY_NAME["Llama 3 8B"], BY_NAME["Llama 2 7B"]
    report.check(
        "Llama 3 8B: 131,072 bytes = 128 KiB per token, 1 GiB per 8,192 tokens, 32 GiB for 32 sequences",
        kv_bytes_per_token(llama3.config, 2) == 131_072
        and kv_bytes_per_token(llama3.config, 2) * 8192 == GIB
        and kv_bytes_per_token(llama3.config, 2) * 8192 * 32 == 32 * GIB,
    )
    report.check(
        "...which is more than the weights (16.1 GB); Llama 2 7B (32 KV heads) needs 512 KiB per token, 4x as much",
        llama3.params * 2 < 32 * GIB
        and kv_bytes_per_token(llama2.config, 2)
        == 4 * kv_bytes_per_token(llama3.config, 2)
        == 524_288,
        f"weights {llama3.params * 2 / 1e9:.2f} GB",
    )

    shape = llama3.config
    rows = []
    for label, kv_heads in (
        ("MHA (32 KV heads)", 32),
        ("GQA (8, as released)", 8),
        ("GQA (4)", 4),
        ("MQA (1)", 1),
    ):
        cfg = dataclasses.replace(shape, n_kv_head=kv_heads)
        row = [label]
        for bytes_ in (2, 1, 0.5):
            row.append(f"{kv_bytes_per_token(cfg, 2) * bytes_ / 2 / KIB:,.1f}")
        rows.append(row)
    report.say(
        "\n  KiB per token for the Llama 3 8B shape, by number of KV heads and cache precision:"
    )
    report.table(["attention", "fp16", "int8", "int4"], rows)
    report.check(
        "fewer KV heads and fewer bits multiply: MHA fp16 512 KiB -> GQA-8 int8 64 KiB -> MQA int4 4 KiB",
        rows[0][1] == "512.0" and rows[1][2] == "64.0" and rows[3][3] == "4.0",
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. The formula against real tensors")
    prefill = 24 if QUICK else 100
    rows, ok = [], True
    for kv in (4, 2, 1):
        model = GPT(small_gpt(block_size=128, n_kv_head=kv)).eval()
        cache = KVCache(model.cfg.n_layer)
        with torch.no_grad():
            model(torch.randint(0, 256, (1, prefill)), cache=cache)
        want = kv_bytes_per_token(model.cfg, 4) * prefill
        ok &= cache.nbytes() == want
        rows.append([f"{kv} KV heads", cache.nbytes(), want])
    report.say(f"  the 0.86M byte-level GPT, fp32, after a {prefill}-token prefill:")
    report.table(["model", "bytes in the cache", "formula"], rows)
    report.check(
        "the cache holds exactly what the formula says (and GQA shrinks it in proportion)", ok
    )
    static = StaticKVCache(4, 1, 4, 32, 128)
    report.check(
        "a preallocated cache costs its full length from the first token (reserve for the longest request)",
        static.nbytes() == kv_bytes_per_token(small_gpt(block_size=128), 4) * 128,
        f"{static.nbytes():,} bytes before any token is stored",
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. Cached and uncached decoding agree")
    model = GPT(small_gpt(block_size=128)).eval()
    p_len, n_new = (16, 16) if QUICK else (64, 64)
    prompt = torch.randint(0, 256, (1, p_len))

    class Counter:
        """Counts how many token positions the model is asked to process."""

        def __init__(self, m):
            self.m, self.positions = m, 0

        def __getattr__(self, name):
            return getattr(self.m, name)

        def __call__(self, idx, **kw):
            self.positions += idx.shape[1]
            return self.m(idx, **kw)

    def generate_cached(m, steps):
        cache, out = KVCache(m.cfg.n_layer), []
        logits = m(prompt, cache=cache)[:, -1]
        for _ in range(steps):
            token = logits.argmax(-1, keepdim=True)
            out.append(token)
            logits = m(token, cache=cache)[:, -1]
        return torch.cat(out, 1)

    def generate_naive(m, steps):
        ids = prompt
        for _ in range(steps):
            ids = torch.cat([ids, m(ids)[:, -1].argmax(-1, keepdim=True)], 1)
        return ids[:, p_len:]

    with torch.no_grad():
        counter = Counter(model)
        a = generate_cached(counter, n_new)
        cached_positions = counter.positions
        counter = Counter(model)
        b = generate_naive(counter, n_new)
        naive_positions = counter.positions
        full = model(torch.cat([prompt, a], 1))[:, p_len - 1 : -1]
        cache = KVCache(model.cfg.n_layer)
        step = [model(prompt, cache=cache)[:, -1]]
        for t in range(n_new - 1):
            step.append(model(a[:, t : t + 1], cache=cache)[:, -1])
        incremental = torch.stack(step, 1)
        diff = (incremental - full).abs().max().item()
    report.check(
        f"the same {n_new} greedy tokens with and without the cache",
        torch.equal(a, b),
    )
    report.check(
        "logits from the incremental path equal a full recompute up to float rounding",
        diff < 1e-4,
        f"max difference {diff:.1e}",
    )
    expect_naive = sum(p_len + t for t in range(n_new))
    expect_cached = p_len + n_new
    report.say(
        f"\n  token positions processed to write {n_new} tokens after a {p_len}-token prompt:"
        f" without a cache {naive_positions:,}, with one {cached_positions:,}"
        f" ({naive_positions / cached_positions:.0f}x)."
    )
    report.check(
        "counted positions match the page's arithmetic: sum of (P + t) without a cache, P + n with one",
        naive_positions == expect_naive and cached_positions == expect_cached,
        f"{expect_naive:,} and {expect_cached:,}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. Time: the saving, and prefill against decode (CPU; read the ratios)")
    big = GPT(small_gpt(d_model=256, n_layer=6, n_head=8, block_size=128 if QUICK else 512)).eval()
    n_params = matmul_params(big)
    p_big, n_big = (32, 32) if QUICK else (128, 256)
    prompt_big = torch.randint(0, 256, (1, p_big))

    def timed_generate(use_cache: bool):
        times = []
        with torch.no_grad():
            if use_cache:
                cache = KVCache(big.cfg.n_layer)
                t0 = time.perf_counter()
                logits = big(prompt_big, cache=cache)[:, -1]
                prefill_s = time.perf_counter() - t0
                for _ in range(n_big):
                    t0 = time.perf_counter()
                    token = logits.argmax(-1, keepdim=True)
                    logits = big(token, cache=cache)[:, -1]
                    times.append(time.perf_counter() - t0)
            else:
                ids, prefill_s = prompt_big, 0.0
                for _ in range(n_big):
                    t0 = time.perf_counter()
                    ids = torch.cat([ids, big(ids)[:, -1].argmax(-1, keepdim=True)], 1)
                    times.append(time.perf_counter() - t0)
        return times, prefill_s

    with_cache, prefill_s = timed_generate(True)
    without, _ = timed_generate(False)
    thirds = [slice(0, n_big // 3), slice(n_big // 3, 2 * n_big // 3), slice(2 * n_big // 3, n_big)]
    # the median per third: one stall of the machine must not move a whole third
    ms = lambda xs: 1000 * statistics.median(xs)  # noqa: E731
    report.say(
        f"  a {n_params / 1e6:.1f}M-parameter GPT (6 layers, d = 256) writing {n_big} tokens after a"
        f" {p_big}-token prompt; median ms per new token, by third of the run:"
    )
    report.table(
        ["", "first third", "middle third", "last third", "whole run (s)"],
        [
            ["no cache", *[f"{ms(without[s]):.1f}" for s in thirds], f"{sum(without):.2f}"],
            ["KV cache", *[f"{ms(with_cache[s]):.1f}" for s in thirds], f"{sum(with_cache):.2f}"],
        ],
    )
    speedup = sum(without) / (sum(with_cache) + prefill_s)
    grow_no_cache = ms(without[thirds[2]]) / ms(without[thirds[0]])
    grow_cache = ms(with_cache[thirds[2]]) / ms(with_cache[thirds[0]])
    if QUICK:
        report.skip("cost per token against length, with and without a cache", "needs the full run")
    else:
        # seen on an M2: x1.26 to x1.49 without a cache, x1.00 to x1.04 with one
        report.check(
            "the cost per token rises through the run without a cache, and rises more slowly with one",
            grow_no_cache > 1.1 and grow_cache < grow_no_cache,
            f"last third over first: no cache x{grow_no_cache:.2f}, cache x{grow_cache:.2f}",
        )
        report.check(
            "the whole run is at least 3x faster with the cache",
            sum(without) > 3 * (sum(with_cache) + prefill_s),
            f"{sum(without) / (sum(with_cache) + prefill_s):.1f}x",
        )
    report.say(f"  the whole run is {speedup:.1f}x faster with the cache (prefill included).")

    per_token_prefill = 1000 * prefill_s / p_big
    per_token_decode = ms(with_cache[thirds[0]])
    flops = 2 * n_params  # per token, matmuls only; attention adds a little
    report.say(
        f"\n  prefill: {1000 * prefill_s:.1f} ms for {p_big} tokens = {per_token_prefill:.2f} ms per token,"
        f" about {flops / per_token_prefill / 1e6:.0f} GFLOP/s;"
    )
    report.say(
        f"  decode:  {per_token_decode:.2f} ms per token, about {flops / per_token_decode / 1e6:.1f} GFLOP/s."
    )
    if QUICK:
        report.skip("prefill is cheaper per token than decode", "needs the full run")
    else:
        report.check(
            "prefill handles a token far cheaper than decode does (parallel over the prompt)",
            per_token_prefill < per_token_decode / 2,
            f"{per_token_decode / per_token_prefill:.0f}x cheaper per token",
        )
    report.say(
        "  decode reads the weights and the whole cache to produce one token; at batch 1 that is\n"
        "  memory and overhead bound, which is why serving systems batch many sequences per step."
    )

    report.finish()


if __name__ == "__main__":
    main()
