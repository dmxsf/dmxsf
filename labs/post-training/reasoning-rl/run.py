"""Lab: reasoning RL. Verifiable rewards, GRPO advantages and loss, and a real RLVR run.

Course page: https://dmxsf.com/wiki/post-training/reasoning-rl/

The page's pieces are checked first, exactly: the verifier, the group advantages (including the
numbers it prints), the chance that a group carries a signal, the gradient weight each token gets
from the loss, the clipping rule, and the k3 estimator. Then the loop is closed on the intent task:
the model writes an answer, an exact-match verifier gives 0 or 1, and GRPO updates the policy from
nothing but those bits. The model is tiny and the task is generated, so the run shows the
mechanics (signal, entropy, KL, the starting point), not what reasoning RL does to a large model.

    uv run python labs/post-training/reasoning-rl/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from dmxsf import lab, models
from dmxsf.post_training import intents
from dmxsf.post_training import reasoning as R
from dmxsf.post_training.sft import build_example, finetune, sample_batch


def grad_weights(adv, lengths, token_level=False):
    """Gradient of the loss with respect to each token's log-probability (ratio 1, no KL)."""
    t = max(lengths)
    mask = torch.tensor([[1.0] * n + [0.0] * (t - n) for n in lengths])
    new = torch.zeros(len(lengths), t, requires_grad=True)
    loss = R.grpo_loss(
        new, torch.zeros_like(new), torch.zeros_like(new), torch.tensor(adv), mask,
        beta=0.0, token_level=token_level,
    )  # fmt: skip
    loss.backward()
    return new.grad, mask


def main() -> None:
    args, report = lab.start(
        "Reasoning RL: verifiers, GRPO advantages and loss, and a real RLVR run",
        wiki=["post-training/reasoning-rl"],
    )
    QUICK = args.quick

    # -----------------------------------------------------------------------------------------
    report.section("1. The verifier")
    cases = [
        (r"so \boxed{3}", "3", "correct"),
        (r"so \boxed{4}", "3", "wrong"),
        ("the answer is 3", "3", "right answer, wrong format"),
        (r"\boxed{4} no wait \boxed{3}", "3", "the last box counts"),
        (r"\boxed{0.5}", "1/2", "string equality is not mathematical equivalence"),
    ]
    rewards = [R.math_reward(r, g) for r, g, _ in cases]
    report.table(
        ["response", "gold", "reward", "why"],
        [[r, g, f"{x:.0f}", why] for (r, g, why), x in zip(cases, rewards, strict=True)],
    )
    report.check(
        "the page's verifier: 1, 0, 0, 1, 0 (the last row is a false negative)",
        rewards == [1.0, 0.0, 0.0, 1.0, 0.0],
    )
    dump = "card_fee card_limit transfer_declined"
    answers = [("card_fee", "the label"), (dump, "three labels, one of them right")]
    report.say("\n  the intent task, gold label card_fee:")
    report.table(
        ["answer", "exact match", "'contains the label'"],
        [
            [
                a,
                f"{R.exact_match_reward(a, 'card_fee'):.0f}",
                f"{R.contains_reward(a, 'card_fee'):.0f}",
            ]
            for a, _ in answers
        ],
    )
    report.check(
        "a leaky verifier pays for listing every label; exact match does not",
        R.contains_reward(dump, "card_fee") == 1.0
        and R.exact_match_reward(dump, "card_fee") == 0.0,
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. Group advantages")
    group = torch.tensor([[1.0, 0, 0, 1], [1.0, 0, 0, 0], [1.0, 1, 1, 1]])
    adv = R.group_advantages(group)
    report.say("  rewards of three groups of four samples and their advantages (the page's code):")
    report.table(
        ["rewards", "A1", "A2", "A3", "A4"],
        [
            [str([int(x) for x in r]), *[f"{a:+.4f}" for a in a_]]
            for r, a_ in zip(group, adv, strict=True)
        ],
    )
    page = torch.tensor(
        [
            [0.8659, -0.8659, -0.8659, 0.8659],
            [1.4997, -0.4999, -0.4999, -0.4999],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    report.check(
        "these are the numbers the page prints (4 decimals), and a unanimous group has no signal",
        torch.allclose(adv, page, atol=5e-5),
    )
    scaled = R.group_advantages(10 * group.double())
    report.check(
        "multiplying every reward by 10 changes nothing (only the comparison inside the group counts)",
        torch.allclose(scaled, adv.double(), atol=1e-3),
    )
    first = group[:1]
    variants = {
        "sample std (G - 1), the page": R.group_advantages(first)[0, 0].item(),
        "population std (G)": R.group_advantages(first, unbiased=False)[0, 0].item(),
        "mean only, no division": R.group_advantages(first, divide_by_std=False)[0, 0].item(),
    }
    report.say("\n  the advantage of a correct answer in the group [1, 0, 0, 1]:")
    report.table(["normalisation", "A"], [[k, f"{v:+.4f}"] for k, v in variants.items()])
    report.check(
        "sample and population std differ by exactly sqrt(G / (G - 1)) = 1.1547 for G = 4",
        abs(
            variants["population std (G)"] / variants["sample std (G - 1), the page"]
            - math.sqrt(4 / 3)
        )
        < 1e-3,
    )

    ps, gs = (0.05, 0.2, 0.5, 0.8), (4, 8, 16, 64)
    report.say(
        "\n  chance that a group of G samples is mixed (some right, some wrong) at pass rate p:"
    )
    report.table(
        ["pass rate p", *[f"G = {g}" for g in gs]],
        [[f"{p:.0%}", *[f"{R.signal_probability(p, g):.1%}" for g in gs]] for p in ps],
    )
    rng = np.random.default_rng(args.seed)
    draws = (rng.random((100_000, 8)) < 0.2).sum(axis=1)
    simulated = float(((draws > 0) & (draws < 8)).mean())
    report.check(
        "the page's numbers: 87.5% at p = 50%, G = 4; about 44% of groups all wrong at p = 5%, G = 16",
        abs(R.signal_probability(0.5, 4) - 0.875) < 1e-12 and round(0.95**16, 2) == 0.44,
        f"all wrong at 5%: {0.95**16:.1%}",
    )
    report.check(
        "the formula agrees with 100,000 simulated groups (p = 20%, G = 8)",
        abs(simulated - R.signal_probability(0.2, 8)) < 0.005,
        f"{simulated:.3f} against {R.signal_probability(0.2, 8):.3f}",
    )

    g = 16
    rows, ratios = [], []
    for k in (1, 2, 4, 8, 12, 15):
        r = torch.tensor([[1.0] * k + [0.0] * (g - k)])
        plain = R.group_advantages(r, divide_by_std=False).abs().sum().item()
        normed = R.group_advantages(r).abs().sum().item()
        ratios.append(normed / plain)
        rows.append([f"{k} of {g}", plain, normed, normed / plain])
    report.say("\n  how much total push a group gets (sum of |A| over the group, G = 16):")
    report.table(["right", "mean only", "divided by std", "ratio"], rows)
    report.check(
        "dividing by the std boosts nearly-solved and nearly-unsolved prompts (the bias Dr. GRPO points out)",
        ratios[0] > 1.5 * ratios[3] and ratios[-1] > 1.5 * ratios[3],
        f"ratio {ratios[0]:.2f} at 1 of 16 and {ratios[3]:.2f} at 8 of 16",
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. The loss")
    w_seq, mask = grad_weights([-0.87, -0.87], [10, 100])
    w_tok, _ = grad_weights([-0.87, -0.87], [10, 100], token_level=True)
    seq_w = (w_seq[mask > 0] * 2).tolist()
    tok_w = (w_tok[mask > 0] * 110).tolist()
    report.say(
        "  two wrong answers (A = -0.87), 10 and 100 tokens long: push-down weight per token"
    )
    report.table(
        ["loss averaged", "short answer", "long answer"],
        [
            ["per answer, then over answers (the page)", seq_w[0], seq_w[-1]],
            ["over all tokens (DAPO)", tok_w[0], tok_w[-1]],
        ],
    )
    report.check(
        "per-answer averaging punishes a long wrong answer 10x less per token; token-level does not",
        abs(seq_w[0] / seq_w[-1] - 10) < 1e-3 and abs(tok_w[0] - tok_w[-1]) < 1e-5,
        f"{seq_w[0]:.3f} against {seq_w[-1]:.4f} per token",
    )

    rows, ok = [], True
    for a, ratio, expect in (
        (1.0, 1.5, "none"),
        (1.0, 0.5, "up"),
        (-1.0, 0.5, "none"),
        (-1.0, 1.5, "down"),
    ):
        new = torch.zeros(2, 3, requires_grad=True)
        old = torch.full((2, 3), -math.log(ratio))
        R.grpo_loss(
            new, old, torch.zeros_like(new), torch.tensor([a, 0.0]), torch.ones(2, 3), beta=0.0
        ).backward()
        grad = new.grad[0, 0].item()
        got = "none" if grad == 0 else ("up" if grad < 0 else "down")
        ok &= got == expect
        rows.append([f"{a:+.0f}", f"{ratio:g}", got])
    report.say("\n  PPO clipping (eps = 0.2): does the token's probability still get pushed?")
    report.table(["advantage", "ratio new/old", "push"], rows)
    report.check(
        "clipping stops only the pushes that already went far enough in the intended direction",
        ok,
    )

    rng = np.random.default_rng(1)
    pi = torch.tensor(rng.dirichlet(np.ones(6)))
    ref = torch.tensor(rng.dirichlet(np.ones(6)))
    k3 = R.k3(pi.log(), ref.log())
    k1 = pi.log() - ref.log()
    exact = (pi * k1).sum().item()
    report.say(
        "\n  KL(pi || ref) of two distributions over 6 outcomes, and what each estimator averages to:"
    )
    report.table(
        ["estimator", "expectation over samples from pi", "smallest value"],
        [
            ["exact KL", exact, "-"],
            ["k1 = log(pi / ref)", (pi * k1).sum().item(), k1.min().item()],
            ["k3 = ref/pi - log(ref/pi) - 1", (pi * k3).sum().item(), k3.min().item()],
        ],
    )
    report.check(
        "k3 is unbiased and never negative; k1 is unbiased but goes negative",
        abs((pi * k3).sum().item() - exact) < 1e-12 and k3.min() >= 0 and k1.min() < 0,
    )
    ratios = []
    for _ in range(200):
        base_p = rng.dirichlet(np.ones(6) * 3)
        near = base_p * np.exp(rng.normal(scale=0.2, size=6))
        near /= near.sum()
        p, q = torch.tensor(near), torch.tensor(base_p)
        a, b = p.log() - q.log(), R.k3(p.log(), q.log())
        var1 = (p * a**2).sum() - (p * a).sum() ** 2
        var3 = (p * b**2).sum() - (p * b).sum() ** 2
        ratios.append((var3 / var1).item())
    report.check(
        "for policies close to the reference, k3 has far lower variance than k1 (200 random pairs)",
        float(np.median(ratios)) < 0.2,
        f"median Var(k3) / Var(k1) = {np.median(ratios):.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. A real RLVR run on the intent task")
    STEPS = 4 if QUICK else 80
    PPS, GROUP, LR, BETA = (4, 4, 1e-4, 0.04) if QUICK else (16, 8, 1e-4, 0.04)
    SFT_STEPS = 40 if QUICK else 300
    base = models.pretrained("base", quick=QUICK, seed=args.seed)
    train_pairs = intents.make_dataset(2 if QUICK else 5, seed=1 + args.seed)
    test = intents.make_dataset(5 if QUICK else 25, seed=99, exclude={x for x, _ in train_pairs})
    seen = {x for x, _ in train_pairs} | {x for x, _ in test}
    prompts = intents.make_dataset(3 if QUICK else 30, seed=7 + args.seed, exclude=seen)
    texts, gold = [x for x, _ in prompts], [y for _, y in prompts]

    def examples(pairs):
        return [
            build_example(list(intents.render(x)), list(intents.answer_bytes(y))) for x, y in pairs
        ]

    reference = copy.deepcopy(base)
    torch.manual_seed(args.seed)
    ex = examples(train_pairs)
    finetune(reference, lambda s: sample_batch(ex, s, 16, args.seed), SFT_STEPS, 3e-4)
    reference.eval()
    ref_score = intents.score(reference, test)
    ref_bpb = models.shakespeare_bpb(reference)
    report.say(
        f"  reference: pretrained model + SFT on {len(train_pairs)} messages ({SFT_STEPS} steps):"
        f" accuracy {ref_score['accuracy']:.1%} on {len(test)} test messages, {ref_bpb:.2f} bits/byte."
    )
    report.say(
        f"  RLVR: {PPS} messages x {GROUP} sampled answers per step, reward 1 for an exact match,"
        f" lr {LR:g}, beta {BETA},\n  {STEPS} steps = {STEPS * PPS * GROUP:,} sampled answers,"
        " one update per batch (so the ratio is 1 and the clip never binds)."
    )

    def run(pool_texts, pool_gold, seed, start=reference):
        """``seed`` is relative to ``--seed``."""
        policy = copy.deepcopy(start)
        log = R.train_grpo(
            policy, reference, pool_texts, pool_gold, steps=STEPS, prompts_per_step=PPS,
            group=GROUP, lr=LR, beta=BETA, seed=seed + args.seed,
        )  # fmt: skip
        out = intents.score(policy, test)
        return {"log": log, "model": policy, "bpb": models.shakespeare_bpb(policy), **out}

    def window(log, key, lo, hi):
        return float(np.mean([r[key] for r in log[lo:hi]]))

    main = run(texts, gold, 0)
    log = main["log"]
    width = max(1, STEPS // 4)
    rows = []
    for s in range(0, STEPS, width):
        hi = min(s + width, STEPS)
        rows.append(
            [f"{s + 1}-{hi}"]
            + [f"{window(log, k, s, hi):.2f}" for k in ("reward", "signal", "entropy", "length")]
            + [f"{window(log, 'kl', s, hi):.3f}"]
        )
    report.say("\n  training curve (sampled at temperature 1, averages over windows of steps):")
    report.table(
        ["steps", "mean reward", "groups with signal", "entropy (nats)", "answer tokens", "k3 KL"],
        rows,
    )
    kls = [r["kl"] for r in log]
    tail = (
        "a heavy right tail (a sampled token that the policy now makes rarer than the reference does)"
        if max(kls) > 5 * max(np.median(kls), 1e-9)
        else "no heavy tail in this run"
    )
    report.say(
        f"  after training: greedy accuracy {main['accuracy']:.1%} (reference {ref_score['accuracy']:.1%}),"
        f" {main['bpb']:.2f} bits/byte (reference {ref_bpb:.2f});\n  k3 per step: median"
        f" {np.median(kls):.3f}, largest {max(kls):.3f}: {tail}."
    )

    seed1 = run(texts, gold, 1)
    accs = [main["accuracy"], seed1["accuracy"]]
    spread = max(accs) - min(accs)
    report.say(
        f"  another seed: accuracy {seed1['accuracy']:.1%} (range {spread * 100:.1f} points over"
        f" 2 runs; one test message is {100 / len(test):.2f} points)"
    )
    if QUICK:
        for name in (
            "the mean reward rises over training",
            "the share of groups with a signal falls as the pass rate rises",
            "RLVR beats the reference by more than the seed range",
        ):
            report.skip(name, "needs the full run")
    else:
        report.check(
            "the mean reward rises over training (last window against first, by more than 0.15)",
            window(log, "reward", STEPS - width, STEPS) > window(log, "reward", 0, width) + 0.15,
            f"{window(log, 'reward', 0, width):.2f} -> {window(log, 'reward', STEPS - width, STEPS):.2f}",
        )
        signal_first = window(log, "signal", 0, width)
        signal_last = window(log, "signal", STEPS - width, STEPS)
        fell = signal_last < signal_first - 0.03  # down by 0.13 on one machine, 0.03 on another
        report.observe(
            "the share of groups with a signal falls as the pass rate rises",
            fell,
            f"{signal_first:.2f} -> {signal_last:.2f}",
        )
        if not fell:
            report.say(
                "  A group has a signal only when its answers are not all right or all wrong, so the"
                " share falls once most prompts are solved; at a mean reward of"
                f" {window(log, 'reward', STEPS - width, STEPS):.2f} many are still mixed."
            )
        report.check(
            "RLVR beats the reference by more than the seed range (mean of 2 seeds)",
            float(np.mean(accs)) - ref_score["accuracy"] > spread,
            f"{ref_score['accuracy']:.1%} -> {np.mean(accs):.1%}",
        )

    # the starting point
    policy = copy.deepcopy(base)
    before = [p.detach().clone() for p in policy.parameters()]
    cold = R.train_grpo(
        policy,
        base,
        texts,
        gold,
        steps=2 if QUICK else 10,
        prompts_per_step=PPS,
        group=GROUP,
        lr=LR,
        beta=BETA,
    )
    changed = max(
        (p - b).abs().max().item() for p, b in zip(policy.parameters(), before, strict=True)
    )
    hits = sum(r["reward"] * PPS * GROUP for r in cold)
    report.say(
        f"\n  the same loop from the pretrained model without SFT ({len(cold)} steps):"
        f" {hits:.0f} of {len(cold) * PPS * GROUP:,} sampled answers earned a reward,"
        f" groups with a signal {np.mean([r['signal'] for r in cold]):.0%},"
        f" largest weight change {changed:.1e}."
    )
    report.check(
        "no answer is ever right, so no group has a signal and the weights do not move at all",
        hits == 0 and changed == 0.0,
        "RL cannot start from a policy that never succeeds",
    )

    # difficulty filter
    def pass_rates(model, samples=8):
        rates = []
        for s in range(0, len(texts), 16):
            gen = torch.Generator().manual_seed(1000 + s + args.seed)
            _, _, answers = R.rollout(model, texts[s : s + 16], samples, generator=gen)
            for j, label in enumerate(gold[s : s + 16]):
                got = answers[j * samples : (j + 1) * samples]
                rates.append(np.mean([R.exact_match_reward(a, label) for a in got]))
        return np.array(rates)

    rates = pass_rates(reference, 4 if QUICK else 8)
    mixed = [i for i, r in enumerate(rates) if 0 < r < 1]
    report.say(
        f"\n  pass rates of the reference on the {len(texts)} training messages (8 samples each):"
        f" all wrong {np.mean(rates == 0):.0%}, all right {np.mean(rates == 1):.0%},"
        f" mixed {len(mixed) / len(texts):.0%}."
    )
    if len(mixed) < PPS:
        report.say("  too few mixed messages for the filtered run in this tiny setting.")
        filtered = []
    else:
        filtered = [
            run([texts[i] for i in mixed], [gold[i] for i in mixed], seed) for seed in (0, 1)
        ]
    if filtered:
        full_runs = [main, seed1]
        report.table(
            [
                "training pool",
                "mean reward, last window",
                "groups with signal, first window",
                "test accuracy (2 seeds)",
            ],
            [
                [
                    f"all {len(texts)} messages",
                    f"{np.mean([window(r['log'], 'reward', STEPS - width, STEPS) for r in full_runs]):.2f}",
                    f"{np.mean([window(r['log'], 'signal', 0, width) for r in full_runs]):.2f}",
                    ", ".join(f"{r['accuracy']:.1%}" for r in full_runs),
                ],
                [
                    f"only the {len(mixed)} mixed ones",
                    f"{np.mean([window(r['log'], 'reward', STEPS - width, STEPS) for r in filtered]):.2f}",
                    f"{np.mean([window(r['log'], 'signal', 0, width) for r in filtered]):.2f}",
                    ", ".join(f"{r['accuracy']:.1%}" for r in filtered),
                ],
            ],
        )
        report.say(
            "  The filter gives each step more signal at the start and a higher reward on its own\n"
            "  pool, but the pool is smaller and saturates sooner: read the last column for what it\n"
            "  did to held-out accuracy. (Two seeds per row.)"
        )

    sft_model = copy.deepcopy(reference)
    sft_ex = examples(prompts)
    finetune(sft_model, lambda s: sample_batch(sft_ex, s, 16, 0), 20 if QUICK else 200, 1e-4)
    sft_score = intents.score(sft_model, test)
    report.say(
        f"\n  for scale: SFT on the {len(prompts)} correct answers themselves reaches"
        f" {sft_score['accuracy']:.1%} (200 steps, lr 1e-4). RLVR saw one bit per sampled answer, not\n"
        "  the answer; it is the tool for when a checker exists and demonstrations do not."
    )

    report.finish()


if __name__ == "__main__":
    main()
