"""Lab: preference optimization. The DPO loss by hand, its closed-form optimum, and a real DPO run.

Course page: https://dmxsf.com/wiki/post-training/preference-optimization/

The page derives DPO from the KL-regularised reward-maximisation problem. This lab checks each
step with numbers: the hand example, the gradient weight, the closed form (against random
competitors and against DPO fitted on simulated Bradley-Terry comparisons), then fine-tunes a
small byte-level model on preference pairs and looks at what the page warns about: the learning
rate, beta, chosen log-probabilities that fall, LoRA with the adapters switched off as the
reference, and how a proxy reward is exploited as the KL budget grows.

    uv run python labs/post-training/preference-optimization/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import copy
import math
import time
from itertools import pairwise

import numpy as np
import torch

from dmxsf import lab, models
from dmxsf.post_training import intents
from dmxsf.post_training import preference as P
from dmxsf.post_training.lora import adapters_disabled, add_lora, trainable_params
from dmxsf.post_training.sft import build_example, finetune, sample_batch


def t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def main() -> None:
    args, report = lab.start(
        "Preference optimization: the DPO loss, its closed form, and a real DPO run",
        wiki=["post-training/preference-optimization"],
    )
    QUICK = args.quick

    # -----------------------------------------------------------------------------------------
    report.section("1. The loss by hand (the page's numbers)")
    pi_c, pi_r, ref_c, ref_r = t(-12.0), t(-15.0), t(-12.5), t(-14.0)
    rows, losses = [], {}
    for beta in (0.1, 0.5):
        loss, _ = P.dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
        weight = P.dpo_gradient_weight(pi_c, pi_r, ref_c, ref_r, beta)
        chosen, rejected = beta * (pi_c - ref_c).item(), beta * (pi_r - ref_r).item()
        losses[beta] = loss.item()
        rows.append([beta, chosen, rejected, chosen - rejected, loss.item(), weight.item()])
    report.table(
        ["beta", "chosen reward", "rejected reward", "margin", "loss", "gradient weight"], rows
    )
    report.check(
        "loss 0.621 at beta = 0.1 and 0.387 at beta = 0.5 (the page's 0.15 and 0.75 margins)",
        round(losses[0.1], 3) == 0.621 and round(losses[0.5], 3) == 0.387,
        f"{losses[0.1]:.4f} and {losses[0.5]:.4f}",
    )

    a, b = t(-12.0).requires_grad_(), t(-15.0).requires_grad_()
    loss, _ = P.dpo_loss(a, b, ref_c, ref_r, 0.1)
    loss.backward()
    w = P.dpo_gradient_weight(a, b, ref_c, ref_r, 0.1).item()
    err = max(abs(a.grad.item() + 0.1 * w), abs(b.grad.item() - 0.1 * w))
    report.check(
        "autograd gives the page's gradient: -beta * weight on chosen, +beta * weight on rejected",
        err < 1e-12,
        f"weight {w:.4f}, max difference {err:.1e}",
    )

    lp, lq = torch.randn(64, dtype=torch.float64), torch.randn(64, dtype=torch.float64)
    start_loss, _ = P.dpo_loss(lp, lq, lp, lq, 0.1)
    report.check(
        "policy = reference: the loss is exactly ln 2 (the page's self-test)",
        abs(start_loss.item() - math.log(2)) < 1e-12,
        f"{start_loss.item():.6f} vs {math.log(2):.6f}",
    )
    both, _ = P.dpo_loss(t(-2.0), t(-5.0), t(0.0), t(0.0), 0.1)
    report.check(
        "the loss sees only the margin: both log-probabilities can fall (-2 and -5) and it still drops",
        both.item() < math.log(2),
        f"{both.item():.3f} < {math.log(2):.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. The closed form: what DPO is trying to reach")
    ref = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
    reward = np.array([0.0, 0.4, 1.0, 1.4, 2.0, 3.0])
    report.say("  a made-up world: six candidate answers, a reference policy and a hidden reward;")
    report.say("  the best answer (reward 3) has only 3% of the reference's probability.")
    rows, exp_r, kls = [], [], []
    for beta in (10.0, 2.0, 0.5, 0.2, 0.05):
        pi = P.optimal_policy(ref, reward, beta)
        exp_r.append(float(pi @ reward))
        kls.append(P.kl(pi, ref))
        rows.append([beta, float(pi[-1]), exp_r[-1], kls[-1]])
    report.table(["beta", "pi*(best)", "expected reward", "KL(pi* || ref) nats"], rows)
    report.check(
        "a smaller beta buys more reward and costs more KL, monotonically",
        all(x < y for x, y in pairwise(exp_r)) and all(x < y for x, y in pairwise(kls)),
        f"reward {exp_r[0]:.2f} -> {exp_r[-1]:.2f}, KL {kls[0]:.3f} -> {kls[-1]:.2f}",
    )

    beta = 0.5
    star = P.optimal_policy(ref, reward, beta)
    rng = np.random.default_rng(args.seed)

    def objective(pi):
        return float(pi @ reward - beta * P.kl(pi, ref))

    rivals = [objective(rng.dirichlet(np.ones(6) * s)) for s in (0.3, 1.0, 5.0) for _ in range(700)]
    nudged = [np.clip(star + rng.normal(scale=0.01, size=6), 1e-6, None) for _ in range(300)]
    near = [objective(v / v.sum()) for v in nudged]
    report.check(
        "pi* beats 2,100 random distributions and 300 small perturbations of itself",
        max(rivals) < objective(star) and max(near) < objective(star),
        f"objective {objective(star):.4f}; best rival {max(rivals):.4f}, best perturbation {max(near):.4f}",
    )

    sizes = (100, 1_000, 10_000, 100_000) if not QUICK else (100, 10_000)
    seeds = range(5 if not QUICK else 2)
    rows, tv_by_n, err_by_n = [], [], []
    for n in sizes:
        tvs, errs = [], []
        for s in seeds:
            counts = P.bradley_terry_counts(reward, n, seed=1000 * n + s, ref=ref)
            pi = P.fit_dpo_tabular(counts, ref, beta, steps=1500, lr=0.1)
            tvs.append(0.5 * np.abs(pi - star).sum())
            implicit = beta * np.log(pi / ref)
            errs.append(np.abs((implicit - implicit[0]) - (reward - reward[0])).max())
        tv_by_n.append(float(np.mean(tvs)))
        err_by_n.append(float(np.mean(errs)))
        rows.append([f"{n:,}", tv_by_n[-1], f"{min(tvs):.3f} to {max(tvs):.3f}", err_by_n[-1]])
    report.say(
        f"  DPO fitted from simulated comparisons (beta = {beta}; pairs drawn from the reference;"
        f" the winner chosen\n  with probability sigmoid(r_i - r_j)), {len(list(seeds))} draws per size:"
    )
    report.table(
        ["comparisons", "TV(pi_DPO, pi*) mean", "range over draws", "max |implicit - true reward|"],
        rows,
    )
    if QUICK:
        report.skip("with enough comparisons DPO reaches the closed form", "needs the full run")
    else:
        report.check(
            "with enough comparisons DPO reaches the closed form (total variation < 0.05)",
            tv_by_n[-1] < 0.05 and tv_by_n[-1] < tv_by_n[0] / 3,
            f"{tv_by_n[0]:.3f} at {sizes[0]:,} comparisons -> {tv_by_n[-1]:.3f} at {sizes[-1]:,}",
        )
        report.check(
            "the implicit reward recovers the true reward up to a constant (the log Z that cancels)",
            err_by_n[-1] < 0.2,
            f"max error {err_by_n[-1]:.3f} on rewards spanning 3.0",
        )

    # -----------------------------------------------------------------------------------------
    report.section("3. A real DPO run on the intent task")
    STEPS = 20 if QUICK else 200
    SFT_STEPS = 40 if QUICK else 300
    BATCH = 16
    DEFAULT_LR, DEFAULT_BETA = 3e-5, 0.1

    base = models.pretrained("base", quick=QUICK)
    train_pairs = intents.make_dataset(5 if not QUICK else 2, seed=1)
    test = intents.make_dataset(
        25 if not QUICK else 5, seed=99, exclude={x for x, _ in train_pairs}
    )
    seen = {x for x, _ in train_pairs} | {x for x, _ in test}
    prompts = intents.make_dataset(30 if not QUICK else 3, seed=7, exclude=seen)
    texts, gold = [x for x, _ in prompts], [y for _, y in prompts]

    reference = copy.deepcopy(base)
    torch.manual_seed(0)
    examples = [
        build_example(list(intents.render(x)), list(intents.answer_bytes(y)))
        for x, y in train_pairs
    ]
    finetune(reference, lambda s: sample_batch(examples, s, BATCH, 0), SFT_STEPS, 3e-4)
    reference.eval()
    ref_score = intents.score(reference, test)
    ref_bpb = models.shakespeare_bpb(reference)
    report.say(
        f"  reference model: the pretrained model after SFT on {len(train_pairs)} messages"
        f" ({len(train_pairs) // 12} per intent, {SFT_STEPS} steps):"
    )
    report.say(
        f"  accuracy {ref_score['accuracy']:.1%} on {len(test)} test messages, valid labels"
        f" {ref_score['valid']:.0%}, {ref_bpb:.2f} bits/byte on held-out Shakespeare."
    )

    ev = test[: 120 if not QUICK else 24]
    ev_texts = [x for x, _ in ev]
    ev_gold = torch.tensor([intents.LABELS.index(y) for _, y in ev])
    ev_ref_lp = P.label_distribution(reference, ev_texts)
    ev_ref_dist = P.normalise(ev_ref_lp)
    ev_pairs = P.make_pairs(ev_texts, [y for _, y in ev], ev_ref_dist.exp().numpy(), 2, 21)
    ev_ref_c, ev_ref_r = P.pair_logprobs(reference, ev_pairs)

    ref_lp = P.label_distribution(reference, texts)
    ref_probs = P.normalise(ref_lp).exp().numpy()
    pair_cache: dict[int, tuple] = {}

    def pairs_for(seed: int):
        if seed not in pair_cache:
            prs = P.make_pairs(texts, gold, ref_probs, 2, 11 + seed)
            pair_cache[seed] = (prs, *P.pair_logprobs(reference, prs))
        return pair_cache[seed]

    prs0 = pairs_for(0)[0]
    report.say(
        f"  preference pairs: {len(prs0)} = {len(texts)} new messages x 2; chosen = the correct"
        f" label, rejected = a wrong label\n  drawn from the reference's own probabilities over the"
        f" eleven wrong labels (the mistakes it really makes)."
    )

    def evaluate(model, beta=None):
        model.eval()
        lp = P.label_distribution(model, ev_texts)
        dist = P.normalise(lp)
        out = intents.score(model, test)
        out["p_correct"] = dist.exp()[torch.arange(len(ev)), ev_gold].mean().item()
        out["kl"] = (dist.exp() * (dist - ev_ref_dist)).sum(-1).mean().item()
        out["mass"] = lp.logsumexp(-1).exp().mean().item()
        out["bpb"] = models.shakespeare_bpb(model)
        if beta is not None:
            ec, er = P.pair_logprobs(model, ev_pairs)
            out["held"] = (beta * ((ec - ev_ref_c) - (er - ev_ref_r)) > 0).float().mean().item()
        return out

    def run_dpo(beta, lr, *, seed=0, lora=False):
        prs, rc, rr = pairs_for(seed)
        policy = copy.deepcopy(reference)
        if lora:
            torch.manual_seed(seed)
            add_lora(policy, "all", 8, 16.0)
            with adapters_disabled(policy):  # the reference needs no second copy of the weights
                same = P.pair_logprobs(policy.eval(), prs)
            exact = torch.equal(same[0], rc) and torch.equal(same[1], rr)
        policy.train()
        params = [p for p in policy.parameters() if p.requires_grad]
        t0 = time.perf_counter()
        log = P.train_dpo(
            policy,
            prs,
            rc,
            rr,
            beta=beta,
            lr=lr,
            steps=STEPS,
            batch=BATCH,
            seed=seed,
            params=params,
        )
        seconds = time.perf_counter() - t0
        tail = log[-min(50, len(log)) :]
        out = {k: float(np.mean([r[k] for r in tail])) for k in log[0]}
        out |= evaluate(policy, beta) | {"log": log, "seconds": seconds, "model": policy}
        out["trainable"] = (
            trainable_params(policy) if lora else sum(p.numel() for p in policy.parameters())
        )
        if lora:
            out["exact"] = exact
        return out

    def row(label, r):
        return [
            label,
            f"{r['reward_acc']:.2f}",
            f"{r['d_chosen']:+.2f}",
            f"{r['d_rejected']:+.2f}",
            f"{r['accuracy']:.1%}",
            f"{r['valid']:.0%}",
            f"{r['mass']:.2f}",
            f"{r['kl']:.2f}",
            f"{r['bpb']:.2f}",
        ]

    head = ["", "reward acc", "d log p chosen", "d log p rejected", "accuracy", "valid label",
            "mass on 12 labels", "KL to ref", "bits/byte"]  # fmt: skip

    # the default run, as a curve
    default = run_dpo(DEFAULT_BETA, DEFAULT_LR)
    first = default["log"][0]
    report.check(
        "step 1 of a real run: loss ln 2 and mean gradient weight 0.5 (policy = reference)",
        abs(first["loss"] - math.log(2)) < 1e-5 and abs(first["weight"] - 0.5) < 1e-5,
        f"loss {first['loss']:.6f}, weight {first['weight']:.4f}",
    )
    log = default["log"]
    width = max(1, len(log) // 4)
    curve = []
    for s in range(0, len(log), width):
        chunk = log[s : s + width]
        mean = {k: float(np.mean([r[k] for r in chunk])) for k in chunk[0]}
        curve.append(
            [
                f"{s + 1}-{s + len(chunk)}",
                mean["loss"],
                mean["reward_acc"],
                mean["d_chosen"],
                mean["d_rejected"],
                mean["weight"],
            ]
        )
    report.say(f"\n  the default run (beta {DEFAULT_BETA}, lr {DEFAULT_LR:g}), training batches:")
    report.table(
        ["steps", "loss", "reward acc", "d log p chosen", "d log p rejected", "mean weight"], curve
    )
    if QUICK:
        report.skip(
            "pairs ordered correctly weigh less: the mean gradient weight falls",
            "needs the full run",
        )
    else:
        report.check(
            "pairs that are ordered correctly weigh less: the mean gradient weight falls",
            curve[-1][-1] < 0.6 * curve[0][-1],
            f"{curve[0][-1]:.2f} -> {curve[-1][-1]:.2f}",
        )

    # learning rate window
    lrs = (DEFAULT_LR,) if QUICK else (1e-5, 3e-5, 1e-4, 3e-4)
    lr_runs = {lr: (default if lr == DEFAULT_LR else run_dpo(DEFAULT_BETA, lr)) for lr in lrs}
    report.say(f"\n  learning rate (beta {DEFAULT_BETA}, {STEPS} steps, SFT used 3e-4):")
    report.table(head, [row(f"lr {lr:g}", r) for lr, r in lr_runs.items()])
    report.say(
        f"  (the reference: accuracy {ref_score['accuracy']:.1%}, valid {ref_score['valid']:.0%},"
        f" mass on the 12 labels {ev_ref_lp.logsumexp(-1).exp().mean().item():.2f},"
        f" {ref_bpb:.2f} bits/byte)"
    )
    if QUICK:
        report.skip("a smaller learning rate moves the policy far less", "needs the full run")
        report.skip(
            "a too-large learning rate collapses the model while the loss looks fine",
            "needs the full run",
        )
    else:
        small, mid, big = lr_runs[1e-5], lr_runs[3e-5], lr_runs[3e-4]
        report.check(
            "a 3x smaller learning rate moves the policy much less (KL to the reference)",
            small["kl"] < 0.6 * mid["kl"],
            f"KL {small['kl']:.2f} at 1e-5 vs {mid['kl']:.2f} at 3e-5",
        )
        report.check(
            "too large: the chosen answers' log-probability FALLS although the reward accuracy is 100%",
            big["d_chosen"] < 0 and big["reward_acc"] >= 0.95,
            f"chosen {big['d_chosen']:+.1f} nats, rejected {big['d_rejected']:+.1f}, reward acc {big['reward_acc']:.2f}",
        )
        report.check(
            "...and the model stops answering in the format (valid labels collapse)",
            big["valid"] < 0.5,
            f"valid {big['valid']:.0%}, mass on the twelve labels {big['mass']:.3f}",
        )

    # seeds
    seed_runs = [default] + [
        run_dpo(DEFAULT_BETA, DEFAULT_LR, seed=s) for s in ((1,) if QUICK else (1, 2))
    ]
    accs = [r["accuracy"] for r in seed_runs]
    spread = max(accs) - min(accs)
    report.say(
        "\n  the default run with other seeds (pairs resampled, batch order changed): "
        + ", ".join(f"{a:.1%}" for a in accs)
        + f" (range {spread * 100:.1f} points; one test message is {100 / len(test):.2f} points)"
    )
    if QUICK:
        report.skip("DPO beats the reference by more than the seed range", "needs the full run")
    else:
        mean_acc = float(np.mean(accs))
        report.check(
            "DPO beats the reference by more than the seed range",
            mean_acc - ref_score["accuracy"] > spread,
            f"{ref_score['accuracy']:.1%} -> {mean_acc:.1%} on average (range {spread * 100:.1f} points)",
        )

    # beta
    betas = (DEFAULT_BETA,) if QUICK else (0.01, 0.1, 0.5, 2.0)
    beta_runs = {b: (default if b == DEFAULT_BETA else run_dpo(b, DEFAULT_LR)) for b in betas}
    report.say(f"\n  beta (lr {DEFAULT_LR:g}, {STEPS} steps):")
    report.table(head, [row(f"beta {b:g}", r) for b, r in beta_runs.items()])
    if QUICK:
        report.skip("a larger beta keeps the policy closer to the reference", "needs the full run")
    else:
        report.check(
            "a larger beta keeps the policy closer to the reference (KL to it)",
            beta_runs[2.0]["kl"] < 0.5 * beta_runs[0.1]["kl"],
            f"KL {beta_runs[0.01]['kl']:.2f} / {beta_runs[0.1]['kl']:.2f} / {beta_runs[0.5]['kl']:.2f} / {beta_runs[2.0]['kl']:.2f}",
        )

    # LoRA, with the adapters switched off as the reference
    lora_lrs = (1e-3,) if QUICK else (3e-4, 1e-3)
    lora_runs = {lr: run_dpo(DEFAULT_BETA, lr, lora=True) for lr in lora_lrs}
    report.say(
        "\n  LoRA (r = 8, all linear layers), the reference is the same model with the adapters off:"
    )
    report.table(
        [*head, "trained parameters"],
        [[*row(f"lora lr {lr:g}", r), f"{r['trainable']:,}"] for lr, r in lora_runs.items()]
        + [[*row(f"full lr {DEFAULT_LR:g}", default), f"{default['trainable']:,}"]],
    )
    report.check(
        "with the adapters off, the LoRA model's log-probabilities equal the reference's exactly",
        all(r["exact"] for r in lora_runs.values()),
        "no second copy of the weights is needed",
    )

    # SFT on the chosen answers, for comparison
    sft_ex = [
        build_example(list(intents.render(x)), list(intents.answer_bytes(y))) for x, y in prompts
    ]
    sft_model = copy.deepcopy(reference)
    finetune(sft_model, lambda s: sample_batch(sft_ex, s, BATCH, 0), STEPS, 1e-4)
    sft = evaluate(sft_model)
    report.say(
        f"\n  for comparison, plain SFT on the same {len(prompts)} correct answers"
        f" (lr 1e-4, {STEPS} steps; neither method tuned further):"
    )
    report.table(
        ["", "accuracy", "valid label", "mass on 12 labels", "KL to ref", "bits/byte"],
        [
            ["SFT on the chosen answers", f"{sft['accuracy']:.1%}", f"{sft['valid']:.0%}",
             f"{sft['mass']:.2f}", f"{sft['kl']:.2f}", f"{sft['bpb']:.2f}"],
            [f"DPO (default, mean of {len(seed_runs)} seeds)", f"{np.mean(accs):.1%}",
             f"{np.mean([r['valid'] for r in seed_runs]):.0%}",
             f"{np.mean([r['mass'] for r in seed_runs]):.2f}",
             f"{np.mean([r['kl'] for r in seed_runs]):.2f}",
             f"{np.mean([r['bpb'] for r in seed_runs]):.2f}"],
        ],
    )  # fmt: skip
    report.say(
        "  Where a correct answer exists, SFT on it uses the data more directly. DPO is for the\n"
        "  case where only comparisons exist; with a verifiable answer the next page's rule-based\n"
        "  reward is the other route."
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. Reward hacking: a proxy reward and the KL budget (a toy world)")
    k_resp, n_tricks, overrate, penalty, sigma, worlds = 200, 3, 4.5, 2.0, 0.3, 200
    report.say(
        f"  {k_resp} candidate answers with true quality ~ N(0, 1) and a uniform reference. The proxy\n"
        f"  reward sees quality plus noise (sd {sigma}), except for {n_tricks} 'tricks' (think: very long,\n"
        f"  flattering answers) that are really {penalty:g} worse than average but that the proxy rates\n"
        f"  {overrate:g} above their true value. Averages over {worlds} random worlds. The world is built so\n"
        f"  that the proxy can be exploited; it shows the mechanism, not how big the effect is."
    )
    grid = (10.0, 3.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.1, 0.05)
    table = {b: [] for b in grid}
    uniform = np.full(k_resp, 1 / k_resp)
    for w in range(worlds):
        rng = np.random.default_rng((5, w))
        true = rng.normal(size=k_resp)
        tricks = rng.choice(k_resp, size=n_tricks, replace=False)
        true[tricks] -= penalty
        proxy = true + rng.normal(scale=sigma, size=k_resp)
        proxy[tricks] += overrate
        for b in grid:
            pi = P.optimal_policy(uniform, proxy, b)
            table[b].append((P.kl(pi, uniform), pi @ proxy, pi @ true, pi[tricks].sum()))
    mean = {b: np.mean(v, axis=0) for b, v in table.items()}
    report.table(
        ["beta", "KL (nats)", "proxy reward", "true reward", "mass on the tricks"],
        [[b, *map(float, mean[b])] for b in grid],
    )
    proxies = [mean[b][1] for b in grid]
    trues = [mean[b][2] for b in grid]
    peak = int(np.argmax(trues))
    report.check(
        "the proxy reward rises with every bit of KL spent",
        all(x < y for x, y in pairwise(proxies)),
        f"{proxies[0]:.2f} -> {proxies[-1]:.2f}",
    )
    report.check(
        "the true reward peaks at an interior beta, then falls while the proxy keeps rising",
        0 < peak < len(grid) - 1 and trues[peak] - trues[-1] > 0.3,
        f"peak {trues[peak]:.2f} at KL {mean[grid[peak]][0]:.1f} nats (beta {grid[peak]:g}),"
        f" {trues[-1]:.2f} at KL {mean[grid[-1]][0]:.1f}",
    )
    report.say(
        "  In a real run the true reward is not observable. That is why the page recommends a KL\n"
        "  penalty, early stopping on a held-out gold signal, and refreshing the reward model."
    )

    report.finish()


if __name__ == "__main__":
    main()
