"""Lab: the statistics of evaluation. pass@k, the error of a score, paired comparisons, sample sizes.

Course page: https://dmxsf.com/wiki/evaluation/eval-statistics/

Every formula on the page is run, and then tested against what it claims: the unbiased pass@k
estimator against brute-force enumeration and a Monte Carlo of the naive one, the 95% interval
against how often it really covers, and the sample-size table against the power of an actual test.
The last section compares two real models (the base and draft byte-level models) on held-out text.

    uv run python labs/evaluation/eval-statistics/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import torch
import torch.nn.functional as F

from dmxsf import lab, models
from dmxsf.data import ByteCorpus
from dmxsf.evaluation.stats import (
    bootstrap_diff,
    correlated_bernoulli,
    detects,
    diff_ci,
    half_width,
    pass_at_k,
    pass_at_k_naive,
    questions_needed,
)


def main() -> None:
    args, report = lab.start(
        "Evaluation statistics: pass@k, the error of a score, paired tests, sample sizes",
        wiki=["evaluation/eval-statistics"],
    )
    QUICK = args.quick

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.check(name, ok, detail)

    # -----------------------------------------------------------------------------------------
    report.section("1. pass@k: the page's example, then the estimator against brute force")
    correct, n = [10, 7, 3, 1, 0], 10  # five problems, ten samples each
    rows, table = [], {}
    for k in (1, 5, 10):
        exact = sum(pass_at_k(n, c, k) for c in correct) / len(correct)
        naive = sum(pass_at_k_naive(n, c, k) for c in correct) / len(correct)
        table[k] = (exact, naive)
        rows.append([f"pass@{k}", f"{exact:.4f}", f"{naive:.4f}"])
    report.table(["", "unbiased", "naive"], rows)
    report.check(
        "the page's table: unbiased 0.4200 / 0.6833 / 0.8000, naive 0.4200 / 0.6478 / 0.7246",
        [f"{table[k][0]:.4f}" for k in (1, 5, 10)] == ["0.4200", "0.6833", "0.8000"]
        and [f"{table[k][1]:.4f}" for k in (1, 5, 10)] == ["0.4200", "0.6478", "0.7246"],
    )
    # Brute force: draw every possible set of k samples out of n and ask whether any passed.
    worst = 0.0
    for nn, c, k in [(6, 2, 3), (7, 3, 2), (8, 1, 4), (8, 6, 5), (9, 0, 3)]:
        samples = [1] * c + [0] * (nn - c)
        sets = list(itertools.combinations(range(nn), k))
        brute = sum(any(samples[i] for i in s) for s in sets) / len(sets)
        assert len(sets) == comb(nn, k)
        worst = max(worst, abs(brute - pass_at_k(nn, c, k)))
    report.check(
        "pass@k equals the average over every possible k-subset of the n samples",
        worst < 1e-12,
        f"largest gap {worst:.1e} over 5 cases",
    )

    # Monte Carlo: problems with known success rates, n samples each, many repetitions.
    rng = np.random.default_rng(args.seed)
    p_true = rng.beta(0.6, 0.9, size=40)  # 40 problems of very different difficulty
    trials = 300 if QUICK else 4000
    n = 10
    counts = rng.binomial(n, p_true, size=(trials, len(p_true)))
    rows, unbiased_ok, naive_low = [], True, True
    for k in (1, 2, 5, 10):
        truth = float(np.mean(1 - (1 - p_true) ** k))
        # Both estimators, per problem, for every possible count (0..n), then looked up.
        good = np.array([pass_at_k(n, c, k) for c in range(n + 1)])
        bad = np.array([pass_at_k_naive(n, c, k) for c in range(n + 1)])
        est_good = good[counts].mean(axis=1)  # one benchmark score per trial
        est_bad = bad[counts].mean(axis=1)
        se = est_good.std(ddof=1) / np.sqrt(trials)
        unbiased_ok &= abs(est_good.mean() - truth) < 3.5 * se + 1e-9
        naive_low &= k == 1 or est_bad.mean() < truth - 3.5 * se
        rows.append(
            [
                f"pass@{k}",
                f"{truth:.4f}",
                f"{est_good.mean():.4f}",
                f"{est_bad.mean():.4f}",
                f"{est_bad.mean() - truth:+.4f}",
            ]
        )
    report.say(f"  40 problems with known success rates, n = {n} samples each, {trials} repeats:")
    report.table(["", "true value", "unbiased", "naive", "naive - true"], rows)
    report.check(
        "the unbiased estimator averages to the true value (within 3.5 standard errors)",
        unbiased_ok,
    )
    report.check("the naive estimate is below the truth for every k > 1", naive_low)

    # -----------------------------------------------------------------------------------------
    report.section("2. The error of one score")
    rows = []
    for name, size in [("HumanEval", 164), ("GSM8K", 1319), ("MMLU", 14079)]:
        rows.append([name, size, f"80% +- {100 * half_width(0.8, size):.1f} points"])
    report.table(["benchmark", "questions", "95% interval at 80%"], rows)
    report.check(
        "+-6.1, +-2.2 and +-0.7 points, as on the page",
        [f"{100 * half_width(0.8, s):.1f}" for s in (164, 1319, 14079)] == ["6.1", "2.2", "0.7"],
    )
    reps = 2000 if QUICK else 40_000
    rows, coverage = [], {}
    for size in (164, 1319):
        got = rng.binomial(size, 0.8, size=reps) / size
        hw = 1.96 * np.sqrt(got * (1 - got) / size)
        coverage[size] = float(np.mean(np.abs(got - 0.8) <= hw))
        rows.append([size, f"{coverage[size]:.4f}"])
    report.say(f"\n  How often the interval contains the true accuracy ({reps:,} simulated runs):")
    report.table(["questions", "coverage of the 95% interval"], rows)
    full_check(
        "the 95% interval covers about 95% of the time (0.93 to 0.96)",
        all(0.93 < c < 0.96 for c in coverage.values()),
        f"{coverage[164]:.4f} and {coverage[1319]:.4f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. Comparing two models: the same questions, so pair them")
    page = np.random.default_rng(2)  # the page's seed, independent of --seed
    n_q = 164
    difficulty = page.normal(scale=3.0, size=n_q)  # shared by both models

    def answers(skill: float) -> np.ndarray:
        p = 1 / (1 + np.exp(-(skill - difficulty)))
        return (page.random(n_q) < p).astype(float)

    a, b = answers(3.0), answers(2.2)
    rho = float(np.corrcoef(a, b)[0, 1])
    mean_d, lo_u, hi_u = diff_ci(a, b, paired=False)
    _, lo_p, hi_p = diff_ci(a, b, paired=True)
    boot = bootstrap_diff(a - b, page)
    report.say(f"  A {a.mean():.3f}   B {b.mean():.3f}   correlation {rho:.2f}   gap {mean_d:+.3f}")
    report.table(
        ["interval", "low", "high", "half-width"],
        [
            ["unpaired", f"{lo_u:+.3f}", f"{hi_u:+.3f}", f"{(hi_u - lo_u) / 2:.3f}"],
            ["paired", f"{lo_p:+.3f}", f"{hi_p:+.3f}", f"{(hi_p - lo_p) / 2:.3f}"],
            [
                "paired bootstrap",
                f"{boot[0]:+.3f}",
                f"{boot[1]:+.3f}",
                f"{(boot[1] - boot[0]) / 2:.3f}",
            ],
        ],
    )
    report.check(
        "the page's simulation: A 0.829, B 0.756, correlation 0.50, gap +0.073",
        (f"{a.mean():.3f}", f"{b.mean():.3f}", f"{rho:.2f}", f"{mean_d:+.3f}")
        == ("0.829", "0.756", "0.50", "+0.073"),
    )
    report.check(
        "unpaired interval [-0.014, +0.161] contains 0, paired [+0.011, +0.136] does not",
        (f"{lo_u:+.3f}", f"{hi_u:+.3f}", f"{lo_p:+.3f}", f"{hi_p:+.3f}")
        == ("-0.014", "+0.161", "+0.011", "+0.136")
        and lo_u < 0 < lo_p,
    )
    report.check(
        "the bootstrap over questions agrees with the paired formula: [0.012, 0.140]",
        f"{boot[0]:.3f} {boot[1]:.3f}" == "0.012 0.140",
    )

    # One run proves nothing. Repeat the whole experiment with fresh questions and fresh answers.
    reps = 300 if QUICK else 3000
    gen = np.random.default_rng(args.seed + 1000)
    hits = {"paired": 0, "unpaired": 0}
    for _ in range(reps):
        diff_q = gen.normal(scale=3.0, size=n_q)

        def draw(skill: float, d=diff_q) -> np.ndarray:
            return (gen.random(n_q) < 1 / (1 + np.exp(-(skill - d)))).astype(float)

        x, y = draw(3.0), draw(2.2)
        hits["paired"] += detects(x, y, paired=True)
        hits["unpaired"] += detects(x, y, paired=False)
    report.say(f"\n  The same comparison repeated {reps:,} times with new questions each time:")
    report.table(
        ["analysis", "runs in which A is found better than B"],
        [[k, f"{v / reps:.3f}"] for k, v in hits.items()],
    )
    report.check(
        "pairing finds the real difference more often than treating the runs as independent",
        hits["paired"] > hits["unpaired"],
        f"{hits['paired'] / reps:.3f} against {hits['unpaired'] / reps:.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. How many questions does it take?")
    rows = []
    needed = {}
    for gap in (5, 3, 2, 1):
        p_b = 0.80 - gap / 100
        needed[gap] = (questions_needed(0.80, p_b, 0.5), questions_needed(0.80, p_b, 0.0))
        rows.append(
            [f"{gap} point{'s' if gap > 1 else ''}", f"{needed[gap][0]:,}", f"{needed[gap][1]:,}"]
        )
    report.table(["gap", "paired (correlation 0.5)", "unpaired (correlation 0)"], rows)
    report.check(
        "the page's table: 548 / 1,091, 1,472 / 2,940, 3,256 / 6,507, 12,792 / 25,580",
        [needed[g] for g in (5, 3, 2, 1)]
        == [(548, 1091), (1472, 2940), (3256, 6507), (12792, 25580)],
    )
    report.check(
        "halving the gap takes about four times the questions (2 points -> 1 point)",
        3.9 < needed[1][0] / needed[2][0] < 4.0,
        f"x{needed[1][0] / needed[2][0]:.2f}",
    )

    # The table promises 80% power. Run a real test at exactly that size.
    reps = 600 if QUICK else 4000
    gen = np.random.default_rng(args.seed + 2000)
    rows, power = [], {}
    for label, gap, rho_, paired in [
        ("5 points, correlation 0.5, paired", 5, 0.5, True),
        ("5 points, correlation 0.0, unpaired", 5, 0.0, False),
        ("3 points, correlation 0.5, paired", 3, 0.5, True),
    ]:
        size = questions_needed(0.80, 0.80 - gap / 100, rho_)
        found = 0
        for _ in range(reps):
            x, y = correlated_bernoulli(gen, size, 0.80, 0.80 - gap / 100, rho_)
            found += detects(x, y, paired)
        power[label] = found / reps
        rows.append([label, f"{size:,}", f"{power[label]:.3f}"])
    report.say(
        f"\n  Power of a real test at the table's size ({reps:,} simulated benchmarks each):"
    )
    report.table(["case", "questions", "found the gap"], rows)
    full_check(
        "at the size the table gives, the test finds the gap about 80% of the time (0.75 to 0.85)",
        all(0.75 < v < 0.85 for v in power.values()),
        ", ".join(f"{v:.3f}" for v in power.values()),
    )

    # -----------------------------------------------------------------------------------------
    report.section("5. Two real models on held-out text (base against draft)")
    base, draft = models.pretrained("base", quick=QUICK), models.pretrained("draft", quick=QUICK)
    val = ByteCorpus.shakespeare().val
    block = models.BLOCK
    count = len(val) // block
    chunks = torch.from_numpy(
        np.stack([val[i * block : i * block + block + 1] for i in range(count - 1)]).astype(
            np.int64
        )
    )

    @torch.no_grad()
    def window_bits(model) -> np.ndarray:
        """Bits per byte of every non-overlapping 128-byte window: one score per 'question'."""
        model.eval()
        out = []
        for part in chunks.split(64):
            logits = model(part[:, :-1])
            nats = F.cross_entropy(logits.transpose(1, 2), part[:, 1:], reduction="none").mean(1)
            out.append(nats / np.log(2))
        return torch.cat(out).numpy().astype(np.float64)

    sa, sb = window_bits(draft), window_bits(base)
    size = len(sa)
    rho = float(np.corrcoef(sa, sb)[0, 1])
    mean_d, lo_u, hi_u = diff_ci(sa, sb, paired=False)
    _, lo_p, hi_p = diff_ci(sa, sb, paired=True)
    boot = bootstrap_diff(sa - sb, np.random.default_rng(args.seed))
    report.say(f"  {size} windows of {block} bytes; score = bits per byte (lower is better)")
    report.say(
        f"  draft {sa.mean():.3f}   base {sb.mean():.3f}   correlation between the two {rho:.2f}"
    )
    report.table(
        ["interval for draft - base", "low", "high", "half-width"],
        [
            ["unpaired", f"{lo_u:+.3f}", f"{hi_u:+.3f}", f"{(hi_u - lo_u) / 2:.4f}"],
            ["paired", f"{lo_p:+.3f}", f"{hi_p:+.3f}", f"{(hi_p - lo_p) / 2:.4f}"],
            [
                "paired bootstrap",
                f"{boot[0]:+.3f}",
                f"{boot[1]:+.3f}",
                f"{(boot[1] - boot[0]) / 2:.4f}",
            ],
        ],
    )
    report.check(
        "the two models' window scores are positively correlated, so pairing narrows the interval",
        rho > 0 and (hi_p - lo_p) < (hi_u - lo_u),
        f"correlation {rho:.2f}; half-width {(hi_p - lo_p) / 2:.4f} against {(hi_u - lo_u) / 2:.4f}",
    )
    report.check(
        "the paired normal interval and the bootstrap interval agree to within 0.01 bits",
        abs(lo_p - boot[0]) < 0.01 and abs(hi_p - boot[1]) < 0.01,
    )
    full_check(
        "the paired interval for draft - base excludes zero: the draft really is worse",
        lo_p > 0,
        f"draft - base = {mean_d:+.3f} bits per byte, interval [{lo_p:+.3f}, {hi_p:+.3f}]",
    )
    report.say(
        "\n  Windows are the 'questions' here; they all come from one text, so they are not independent\n"
        "  in the way the formulas assume (neighbouring speeches share characters and style). The\n"
        "  intervals are best read as lower bounds on the real uncertainty."
    )
    report.finish()


if __name__ == "__main__":
    main()
