"""Lab: speculative decoding. The rule, the formulas, and a draft / target pair of real models.

Course page: https://dmxsf.com/wiki/inference/speculative-decoding/

A small draft model guesses gamma bytes; the target model checks all of them in one forward pass;
each guess is kept with probability min(1, p/q), the first rejected one is replaced by a draw from
norm(max(0, p - q)), and if every guess survives the target's last row adds one more byte. The
output has exactly the target's distribution. This lab checks that exactness (on the page's toy
distribution, by exact algebra, and on the real models), reproduces the page's formulas, and runs the
whole loop with KV caches on the course's two byte-level models: the 0.86M target and a 2-layer,
64-wide draft, both trained on TinyShakespeare.

    uv run python labs/inference/speculative-decoding/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from dmxsf import lab, models
from dmxsf.data import ByteCorpus
from dmxsf.inference.generate import generate
from dmxsf.inference.kv_cache import KVCache
from dmxsf.inference.speculative import (
    acceptance_rate,
    best_gamma,
    expected_tokens,
    output_distribution,
    probs,
    residual,
    speculative_generate,
    speedup,
)
from dmxsf.utils import chi_square_gof


def main() -> None:
    args, report = lab.start(
        "Speculative decoding: the rule, the formulas, a real draft and target",
        wiki=["inference/speculative-decoding"],
    )
    QUICK = args.quick
    torch.manual_seed(args.seed)

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.check(name, ok, detail)

    # -----------------------------------------------------------------------------------------
    report.section("1. Why the extra positions are almost free (the page's A100 ledger, exact)")
    params, bandwidth, peak = 7e9, 2039e9, 312e12
    t_read = params * 2 / bandwidth
    t_token = 2 * params / peak
    report.table(
        ["quantity", "value", "how"],
        [
            ["BF16 weights", "14 GB", "7e9 parameters x 2 bytes"],
            [
                "time to read them once",
                f"{t_read * 1e3:.2f} ms ({1 / t_read:.0f} tokens/s at most)",
                "14 GB / 2,039 GB/s",
            ],
            ["compute for one token", f"{t_token * 1e3:.3f} ms", "2 x 7e9 FLOP / 312 TFLOPS"],
            ["compute for 5 positions", f"{5 * t_token * 1e3:.2f} ms", "5 x 0.045 ms"],
            ["FLOP per byte to saturate the chip", f"{peak / bandwidth:.0f}", "312e12 / 2,039e9"],
        ],
    )
    report.check(
        "6.87 ms to read the weights (146 tokens/s), 0.045 ms to compute a token, 0.22 ms for five, 153 FLOP/byte",
        round(t_read * 1e3, 2) == 6.87
        and round(1 / t_read) == 146
        and round(t_token * 1e3, 3) == 0.045
        and round(5 * t_token * 1e3, 2) == 0.22
        and round(peak / bandwidth) == 153,
    )
    report.say(
        "  reading the weights costs 153x one token's arithmetic, so checking gamma + 1 positions in the same\n"
        "  pass adds little. It stops being free around a batch of 150, where the arithmetic catches up."
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. The rule on the page's four-token distribution")
    p = np.array([0.50, 0.25, 0.15, 0.10])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    pt, qt = torch.tensor(p), torch.tensor(q)
    right = output_distribution(pt, qt)
    wrong = output_distribution(pt, qt, repair="target")
    report.say("  exact, no sampling: what one position yields under each repair rule")
    report.table(
        [
            "token",
            "p (target)",
            "q (draft)",
            "min(p, q)",
            "residual",
            "output, residual",
            "output, refill from p",
        ],
        [
            [
                i,
                f"{p[i]:.2f}",
                f"{q[i]:.2f}",
                f"{min(p[i], q[i]):.2f}",
                f"{residual(pt, qt)[i]:.3f}",
                f"{right[i]:.4f}",
                f"{wrong[i]:.4f}",
            ]
            for i in range(4)
        ],
    )
    report.check(
        "refilling a rejected position from the residual gives back p exactly; refilling from p itself does not",
        torch.allclose(right, pt, rtol=0, atol=1e-12) and bool((wrong - pt).abs().max() > 0.1),
        f"token 0 would come out at {wrong[0]:.3f} instead of {p[0]:.2f}, tokens 1-3 at"
        f" {wrong[1]:.3f}, {wrong[2]:.3f}, {wrong[3]:.3f}",
    )
    report.say(
        "  the draft already covers the target's mass on tokens where q >= p; refilling from p counts that mass"
        " a second\n  time, and token 0, which the draft under-proposes, ends up short."
    )
    report.check(
        "the acceptance rate is sum(min(p, q)) = 0.75 and the residual is (1, 0, 0, 0)",
        abs(acceptance_rate(pt, qt).item() - 0.75) < 1e-12
        and torch.allclose(
            residual(pt, qt), torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        ),
    )

    # the page's own code, character for character
    rng = np.random.default_rng(0)
    residual_np = np.maximum(p - q, 0)
    residual_np /= residual_np.sum()
    alpha = np.minimum(p, q).sum()

    def one_position(n):
        x = rng.choice(4, size=n, p=q)
        accept = rng.random(n) < np.minimum(1, p[x] / q[x])
        out = np.where(accept, x, rng.choice(4, size=n, p=residual_np))
        return out, accept.mean()

    out, rate = one_position(1_000_000)
    freq = np.bincount(out, minlength=4) / len(out)
    gamma_toy, rounds = 4, 200_000
    x = rng.choice(4, size=(rounds, gamma_toy), p=q)
    accept = rng.random((rounds, gamma_toy)) < np.minimum(1, p[x] / q[x])
    n_accepted = np.where(accept.all(axis=1), gamma_toy, accept.argmin(axis=1))
    sim = float(np.mean(n_accepted + 1))
    formula = (1 - alpha ** (gamma_toy + 1)) / (1 - alpha)
    report.say(
        f"\n  the page's simulation: output frequencies {np.round(freq, 3)} against p {p};"
        f" acceptance {rate:.3f} (theory {alpha:.2f});\n  {sim:.3f} tokens per round against the formula's {formula:.3f}."
    )
    report.check(
        "the page's printed numbers: [0.5, 0.25, 0.149, 0.1], 0.750, 3.054 and 3.051",
        np.round(freq, 3).tolist() == [0.5, 0.25, 0.149, 0.1]
        and round(rate, 3) == 0.75
        and round(sim, 3) == 3.054
        and round(formula, 3) == 3.051,
        f"{np.round(freq, 3).tolist()}, {rate:.3f}, {sim:.3f}, {formula:.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. The formulas (Leviathan et al., theorem 3.8) and the widget's numbers")
    rows = []
    for a in (0.6, 0.8, 0.9):
        row = [f"alpha = {a}"]
        for g in (1, 2, 4, 8, 16):
            row.append(f"{speedup(a, g, 0.05):.2f}")
        g_best, s_best = best_gamma(a, 0.05)
        row.append(f"{g_best}: {s_best:.2f}")
        rows.append(row)
    report.say("  speedup over plain decoding, c = 0.05 (a draft step costs 5% of a target step):")
    report.table(
        ["", "gamma 1", "gamma 2", "gamma 4", "gamma 8", "gamma 16", "best gamma: speedup"], rows
    )
    g8, s8 = best_gamma(0.8, 0.05)
    g3, s3 = best_gamma(0.8, 0.3)
    report.check(
        "alpha 0.9, gamma 10, free draft: 6.86x; alpha 0.8, c 0.05: gamma 4 gives 2.80x, the best gamma is 8 (3.09x), gamma 16 falls to 2.72x",
        round(speedup(0.9, 10, 0.0), 2) == 6.86
        and round(speedup(0.8, 4, 0.05), 2) == 2.80
        and (g8, round(s8, 2)) == (8, 3.09)
        and round(speedup(0.8, 16, 0.05), 2) == 2.72,
    )
    report.check(
        "with a free draft alpha 0.8 caps at 1/(1 - 0.8) = 5x: gamma 4 reaches 3.36x, gamma 32 about 5.0x",
        round(speedup(0.8, 4, 0.0), 2) == 3.36 and round(speedup(0.8, 32, 0.0), 2) == 5.0,
        f"gamma 32: {speedup(0.8, 32, 0.0):.3f}",
    )
    report.check(
        "alpha 0.6 at gamma 4: 1.92x, and the target does 1.49x (alpha 0.8) or 2.17x (alpha 0.6) positions per kept token",
        round(speedup(0.6, 4, 0.05), 2) == 1.92
        and round(5 / expected_tokens(0.8, 4), 2) == 1.49
        and round(5 / expected_tokens(0.6, 4), 2) == 2.17,
    )
    report.check(
        "a draft six times as dear (c = 0.3) shrinks the best gamma to 3 and the speedup to 1.55x; alpha 0.3 with gamma 4 is 0.65x, slower than not using it",
        (g3, round(s3, 2)) == (3, 1.55) and round(speedup(0.3, 4, 0.3), 2) == 0.65,
    )
    grid = [(a, c) for a in (0.3, 0.7, 0.9) for c in (0.05, 0.3)]
    report.check(
        "with gamma = 1 the speedup is (1 + alpha) / (1 + c), above 1 exactly when alpha > c",
        all(
            abs(speedup(a, 1, c) - (1 + a) / (1 + c)) < 1e-12 and (speedup(a, 1, c) > 1) == (a > c)
            for a, c in [*grid, (0.2, 0.3), (0.3, 0.3)]
        ),
    )
    gen = np.random.default_rng(args.seed)
    ok, worst = True, 0.0
    for a, g in ((0.8, 4), (0.5, 6), (0.95, 3)):
        acc = gen.random((200_000, g)) < a
        n_acc = np.where(acc.all(1), g, acc.argmin(1))
        worst = max(worst, abs((n_acc + 1).mean() - expected_tokens(a, g)))
        ok &= worst < 0.03
    report.check(
        "simulated rounds with independent acceptance average the formula's number of tokens",
        ok,
        f"largest gap {worst:.3f} tokens over three (alpha, gamma) pairs",
    )
    c_low, c_high = 1.85 / 1.8 - 1, 1.90 / 1.8 - 1
    report.check(
        "DeepSeek-V3's 1.8x with one extra token (gamma = 1) at 85-90% acceptance means c between 0.03 and 0.06",
        round(c_low, 2) == 0.03 and round(c_high, 2) == 0.06,
        f"{c_low:.3f} to {c_high:.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. A real draft and target")
    corpus = ByteCorpus.shakespeare()
    target = models.pretrained("base", quick=QUICK, seed=args.seed).eval()
    draft = models.pretrained("draft", quick=QUICK, seed=args.seed).eval()
    bpb_t = models.shakespeare_bpb(target, windows=16 if QUICK else 128)
    bpb_d = models.shakespeare_bpb(draft, windows=16 if QUICK else 128)
    report.say(
        f"  target: {target.num_params():,} parameters, 4 layers, d = 128, {bpb_t:.3f} bits/byte on held-out text.\n"
        f"  draft:  {draft.num_params():,} parameters, 2 layers, d = 64,  {bpb_d:.3f} bits/byte."
    )

    # the cost of a step, and what extra positions cost, on this machine
    def step_time(model, k: int, context: int = 64, repeats: int = 40) -> float:
        cache = KVCache(len(model.blocks))
        with torch.no_grad():
            model(torch.randint(0, 256, (1, context)), cache=cache)
            times = []
            for _ in range(repeats + 5):
                new = torch.randint(0, 256, (1, k))
                t0 = time.perf_counter()
                model(new, cache=cache)
                times.append(time.perf_counter() - t0)
                cache.crop(context)
        return statistics.median(times[5:])

    ks = (1, 2, 3, 5, 9)
    t_target = {k: step_time(target, k) for k in ks}
    t_draft = {k: step_time(draft, k) for k in ks}
    report.say(
        "\n  time of one forward pass on the CPU with 64 bytes already cached, by number of new positions:"
    )
    report.table(
        ["new positions", *[str(k) for k in ks]],
        [
            ["target, ms", *[f"{t_target[k] * 1e3:.2f}" for k in ks]],
            ["  vs one position", *[f"{t_target[k] / t_target[1]:.2f}x" for k in ks]],
            ["draft, ms", *[f"{t_draft[k] * 1e3:.2f}" for k in ks]],
        ],
    )
    c = t_draft[1] / t_target[1]
    report.say(
        f"  one draft step costs {c:.2f} of a target step here (c = {c:.2f}); a 0.86M model on a CPU is limited\n"
        "  by per-layer overhead, not by memory bandwidth, so this is a different regime from the A100 ledger."
    )
    full_check(
        "five positions in one target pass cost well under five single steps",
        t_target[5] < 2.5 * t_target[1],
        f"{t_target[5] / t_target[1]:.2f}x one step",
    )

    # how often would a guess be kept? the draft and target distributions on real text
    val = corpus.val
    n_win = 16 if QUICK else 128
    starts = np.linspace(0, len(val) - 130, n_win).astype(int)
    windows = torch.from_numpy(np.stack([val[s : s + 128] for s in starts]).astype(np.int64))
    rows = []
    betas = {}
    with torch.no_grad():
        lt, ld = target(windows), draft(windows)
        agree = (lt.argmax(-1) == ld.argmax(-1)).float().mean().item()
        for temp in (1.0, 0.7, 0.4):
            betas[temp] = acceptance_rate(probs(lt, temp), probs(ld, temp)).mean().item()
            rows.append([f"{temp}", f"{betas[temp]:.3f}"])
    report.say(
        f"\n  on {n_win} held-out windows (every position, teacher-forced), the chance that a drafted byte is kept,\n"
        "  beta = sum(min(p, q)), by sampling temperature:"
    )
    report.table(["temperature", "beta"], rows)
    report.say(
        f"  the two models pick the same most likely byte at {agree:.1%} of positions (greedy decoding)."
    )

    # the loop itself
    n_prompts, n_new, prompt_len = (3, 16, 32) if QUICK else (24, 64, 32)
    p_starts = np.linspace(0, len(val) - prompt_len - 1, n_prompts).astype(int)
    prompts = [torch.from_numpy(val[s : s + prompt_len].astype(np.int64))[None] for s in p_starts]
    gammas = (2, 4) if QUICK else (1, 2, 4, 6, 8)

    greedy_same, greedy_total = 0, 0
    for g in gammas:
        for prompt in prompts:
            want = generate(target, prompt, n_new, temperature=0.0)
            got, _ = speculative_generate(target, draft, prompt, n_new, g, temperature=0.0)
            greedy_same += int(torch.equal(got, want))
            greedy_total += 1
    report.check(
        f"greedy speculative decoding writes the same {n_new} bytes as the target alone, for every gamma and prompt",
        greedy_same == greedy_total,
        f"{greedy_same} of {greedy_total} runs",
    )

    def run_plain(seed: int) -> float:
        g = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        for prompt in prompts:
            generate(target, prompt, n_new, temperature=1.0, generator=g)
        return time.perf_counter() - t0

    def run_spec(gamma: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        total = None
        t0 = time.perf_counter()
        for prompt in prompts:
            _, st = speculative_generate(target, draft, prompt, n_new, gamma, 1.0, g)
            if total is None:
                total = st
            else:
                for f in (
                    "rounds",
                    "drafted",
                    "accepted",
                    "rejections",
                    "target_calls",
                    "draft_calls",
                ):
                    setattr(total, f, getattr(total, f) + getattr(st, f))
        return time.perf_counter() - t0, total

    run_plain(99)  # warm up the allocator and the thread pool
    plain = min(run_plain(args.seed + 1), run_plain(args.seed + 2))
    tokens = n_prompts * n_new
    rows, measured = [], {}
    for g in gammas:
        t1, stats = run_spec(g, args.seed + 1)
        t2, _ = run_spec(g, args.seed + 2)
        secs = min(t1, t2)
        a_hat = stats.alpha
        measured[g] = dict(
            alpha=a_hat,
            kept=stats.kept_share,
            per_round=stats.tokens_per_round,
            iid=expected_tokens(a_hat, g),
            calls=stats.target_calls / tokens,
            model=speedup(a_hat, g, c),
            wall=plain / secs,
        )
        m = measured[g]
        rows.append(
            [
                g,
                f"{m['alpha']:.3f}",
                f"{m['kept']:.2f}",
                f"{m['per_round']:.2f}",
                f"{m['iid']:.2f}",
                f"{m['calls']:.3f}",
                f"{1 / m['calls']:.2f}x",
                f"{m['model']:.2f}x",
                f"{m['wall']:.2f}x",
            ]
        )
    report.say(
        f"\n  sampling at temperature 1 from {n_prompts} prompts, {n_new} new bytes each. Plain decoding:"
        f" {plain / tokens * 1e3:.2f} ms per byte."
    )
    report.table(
        [
            "gamma",
            "alpha measured",
            "drafts kept",
            "bytes per round",
            "formula at that alpha",
            "target passes per byte",
            "fewer target passes",
            f"formula speedup (c = {c:.2f})",
            "wall-clock speedup",
        ],
        rows,
    )
    full_check(
        "every gamma needs fewer target passes per byte than plain decoding (which needs one)",
        all(m["calls"] < 1 for m in measured.values()),
        ", ".join(f"gamma {g}: {m['calls']:.2f}" for g, m in measured.items()),
    )
    best_wall = max(measured, key=lambda g: measured[g]["wall"])
    best_model = max(measured, key=lambda g: measured[g]["model"])
    report.say(
        f"  fastest on the clock: gamma {best_wall} ({measured[best_wall]['wall']:.2f}x); the formula, fed the"
        f" measured alpha and c, picks gamma {best_model} ({measured[best_model]['model']:.2f}x)."
    )
    best_possible = best_gamma(float(np.mean([m["alpha"] for m in measured.values()])), c)
    if measured[best_wall]["wall"] < 1:
        report.say(
            f"  no setting beats plain decoding on this machine. With c = {c:.2f} the formula's best possible"
            f" speedup at this alpha is {best_possible[1]:.2f}x (gamma {best_possible[0]}), before the loop's own"
            " work\n  (sampling, cache cropping, bookkeeping), which the formula leaves out and this lab did not"
            " time separately."
        )
    spread = [
        (measured[g]["per_round"] - measured[g]["iid"]) / measured[g]["iid"]
        for g in gammas
        if g > 1
    ]
    report.say(
        f"  bytes per round against the formula at the measured alpha: {min(spread):+.1%} to {max(spread):+.1%}"
        " (the formula assumes every\n  position accepts with the same probability; a gap means contexts differ in how easy they are to guess)."
    )
    full_check(
        "the formula, fed the measured alpha, predicts the bytes per round within 10%",
        max(abs(x) for x in spread) < 0.10,
        ", ".join(
            f"gamma {g}: {measured[g]['per_round']:.2f} against {measured[g]['iid']:.2f}"
            for g in gammas
            if g > 1
        ),
    )
    one = measured.get(1)  # gamma = 1 is only run in the full run
    full_check(
        "the acceptance rate of the real loop at gamma 1 agrees with the offline beta at temperature 1",
        one is not None and abs(one["alpha"] - betas[1.0]) < 0.05,
        f"alpha {one['alpha']:.3f}, beta {betas[1.0]:.3f}" if one else "",
    )

    # is it really the target's distribution? the first sampled byte, many times
    n_runs = 60 if QUICK else 2000
    prompt = prompts[0]
    with torch.no_grad():
        p_first = torch.softmax(target(prompt)[0, -1], -1).numpy()
    g = torch.Generator().manual_seed(args.seed + 7)
    counts = np.zeros(256)
    for _ in range(n_runs):
        out, _ = speculative_generate(target, draft, prompt, 1, 4, 1.0, g)
        counts[int(out[0, 0])] += 1
    stat, df, p_value = chi_square_gof(counts, p_first)
    top = np.argsort(-p_first)[:3]
    report.say(
        f"\n  the first sampled byte over {n_runs:,} runs (gamma 4), against the target's own next-byte distribution:\n"
        + "\n".join(
            f"    byte {b:3d} ({chr(b)!r:>6}): target {p_first[b]:.3f}, speculative {counts[b] / n_runs:.3f}"
            for b in top
        )
    )
    report.say(
        f"  chi-square {stat:.1f} on {df} degrees of freedom, p = {p_value:.3f} (cells expected under 5 pooled)."
    )
    full_check(
        "the first byte's frequencies match the target's distribution (p above 0.001)",
        p_value > 0.001,
    )

    # -----------------------------------------------------------------------------------------
    report.section("5. When it does not pay")
    cs = (0.02, 0.1, 0.3, 0.6)
    rows = []
    for a in (0.3, 0.5, 0.7, 0.9):
        row = [f"alpha = {a}"]
        for cc in cs:
            g_b, s_b = best_gamma(a, cc)
            row.append(f"{s_b:.2f}x (gamma {g_b})" if s_b > 1.005 else "no gain")
        rows.append(row)
    report.say("  best achievable speedup by acceptance rate (rows) and draft cost (columns):")
    report.table(["", *[f"c = {cc}" for cc in cs]], rows)
    report.check(
        "a low acceptance rate and a dear draft leave nothing to gain: with alpha 0.3 and c 0.5 even gamma = 1 loses",
        best_gamma(0.3, 0.5)[0] == 1 and best_gamma(0.3, 0.5)[1] < 1,
        f"best: gamma {best_gamma(0.3, 0.5)[0]}, {best_gamma(0.3, 0.5)[1]:.2f}x",
    )
    report.say(
        "  and the page's third condition, a large batch, is the ledger of section 1: once the batch is near 150\n"
        "  tokens per weight read the chip is busy, and checking gamma extra positions per sequence costs real time."
    )

    report.finish()


if __name__ == "__main__":
    main()
