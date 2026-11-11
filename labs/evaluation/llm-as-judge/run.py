"""Lab: LLM-as-a-judge. How judges go wrong, how to measure them, and how votes become a ranking.

Course page: https://dmxsf.com/wiki/evaluation/llm-as-judge/

The judge here is a stand-in with biases you choose (see dmxsf.evaluation.judge), so every number
can be compared with an exact answer: swapping the order cancels a position bias and leaves a length
bias alone, Cohen's kappa exposes a judge that always guesses the majority class, and a leaderboard
fitted with Bradley-Terry can be corrected for length when the judge likes long answers.

    uv run python labs/evaluation/llm-as-judge/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import itertools

import numpy as np

from dmxsf import lab
from dmxsf.evaluation.judge import (
    agreement_no_bias,
    bt_design,
    cohens_kappa,
    elo_to_beta,
    exact_agreement,
    fit_bt,
    fit_logistic,
    judge_once,
    judge_swapped,
    labels_needed,
    make_judge,
    norm_cdf,
    online_elo,
    to_elo,
)


def main() -> None:
    args, report = lab.start(
        "LLM-as-a-judge: bias, agreement, and Bradley-Terry leaderboards",
        wiki=["evaluation/llm-as-judge"],
    )
    QUICK = args.quick

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.check(name, ok, detail)

    # -----------------------------------------------------------------------------------------
    report.section("1. Swapping the order: the page's simulation, then the exact answer")
    rng = np.random.default_rng(0)  # the page's seed, independent of --seed
    judge = make_judge(position_bias=0.5, noise=0.8, rng=rng)
    gap = rng.normal(size=20000)  # true quality gap x - y of every pair
    truth = np.where(gap > 0, "x", "y")
    once = np.array([judge_once(judge, g, 0.0, rng) for g in gap])
    swapped = np.array([judge_swapped(judge, g, 0.0) for g in gap])
    decided = swapped != "tie"
    acc_once = float(np.mean(once == truth))
    tie = float(1 - decided.mean())
    acc_swapped = float(np.mean(swapped[decided] == truth[decided]))
    exact = exact_agreement(0.5, 0.8)
    report.table(
        ["procedure", "simulated (20,000 pairs)", "exact (integral over the gap)"],
        [
            [
                "judge once, random order: agrees with truth",
                f"{acc_once:.3f}",
                f"{exact['once']:.3f}",
            ],
            ["swap both orders: pairs left as ties", f"{tie:.3f}", f"{exact['tie']:.3f}"],
            [
                "swap both orders: agreement on the rest",
                f"{acc_swapped:.3f}",
                f"{exact['swapped']:.3f}",
            ],
        ],
    )
    report.check(
        "the page's numbers: 0.753 once; 0.380 ties and 0.914 agreement when swapped",
        (f"{acc_once:.3f}", f"{tie:.3f}", f"{acc_swapped:.3f}") == ("0.753", "0.380", "0.914"),
    )
    report.check(
        "the simulation agrees with the exact integral to within 0.01",
        abs(acc_once - exact["once"]) < 0.01
        and abs(tie - exact["tie"]) < 0.01
        and abs(acc_swapped - exact["swapped"]) < 0.01,
    )

    report.say("\n  The price of swapping grows with the bias (noise 0.8, exact):")
    rows = []
    results = {}
    for bias in (0.0, 0.25, 0.5, 1.0, 2.0):
        e = exact_agreement(bias, 0.8)
        results[bias] = e
        rows.append([bias, f"{e['once']:.3f}", f"{e['tie']:.3f}", f"{e['swapped']:.3f}"])
    report.table(
        ["position bias", "judge once", "ties when swapped", "agreement when swapped"], rows
    )
    report.check(
        "judging once gets worse as the bias grows; swapping keeps agreement high but ties rise",
        all(
            results[a]["once"] > results[b]["once"] for a, b in [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]
        )
        and all(
            results[a]["tie"] < results[b]["tie"] for a, b in [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]
        )
        and min(r["swapped"] for r in results.values()) > 0.9,
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. Averaging both orders, and the closed forms on the page")
    n = 4000 if QUICK else 200_000
    gen = np.random.default_rng(args.seed)
    rows, found = [], {}
    for bias in (0.0, 0.5):
        g = gen.normal(size=n)
        s1 = g + bias + 0.8 * gen.normal(size=n)  # evidence that x wins, x shown first
        s2 = -g + bias + 0.8 * gen.normal(size=n)  # evidence that y wins, y shown first
        single = np.where(gen.random(n) < 0.5, s1 > 0, s2 < 0)  # one judgment, random order
        averaged = (s1 - s2) / 2 > 0  # both orders; the bias drops out
        found[bias] = (float(np.mean(single == (g > 0))), float(np.mean(averaged == (g > 0))))
        rows.append([bias, f"{found[bias][0]:.3f}", f"{found[bias][1]:.3f}"])
    report.say(f"  quality gap sd 1.0, noise 0.8, {n:,} pairs per row:")
    report.table(["position bias", "judge once", "average of both orders"], rows)
    closed_once, closed_avg = agreement_no_bias(1.0, 0.8), agreement_no_bias(1.0, 0.8 / np.sqrt(2))
    report.check(
        "no bias: 1/2 + arctan(s/sigma)/pi = 78.5% once, and 83.6% averaging both orders (noise / sqrt 2)",
        (f"{closed_once:.3f}", f"{closed_avg:.3f}") == ("0.785", "0.836"),
    )
    report.check(
        "the simulations match those closed forms to within 0.01",
        abs(found[0.0][0] - closed_once) < 0.01 and abs(found[0.0][1] - closed_avg) < 0.01,
        f"{found[0.0][0]:.3f} and {found[0.0][1]:.3f}",
    )
    report.check(
        "a position bias lowers the one-shot agreement but leaves the averaged one unchanged",
        found[0.5][0] < found[0.0][0] - 0.01 and abs(found[0.5][1] - found[0.0][1]) < 0.01,
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. A length bias is not cancelled by swapping")
    rng = np.random.default_rng(args.seed + 1)
    trials = 400 if QUICK else 5000
    rows, share = [], {}
    for label, bias, length in [("position bias only", 1.0, 0.0), ("length bias only", 0.0, 0.5)]:
        judge = make_judge(bias, 0.5, rng, length_bias=length)
        x, y = (0.0, 4.0), (0.0, 1.0)  # equal quality; x is four times as long as y
        verdict = [judge_swapped(judge, x, y) for _ in range(trials)]
        once = [judge_once(judge, x, y, rng) for _ in range(trials)]
        share[label] = {k: verdict.count(k) / trials for k in ("x", "y", "tie")}
        rows.append(
            [
                label,
                f"{once.count('x') / trials:.3f}",
                f"{share[label]['x']:.3f}",
                f"{share[label]['y']:.3f}",
                f"{share[label]['tie']:.3f}",
            ]
        )
    report.say("  two answers of equal quality, x four times as long; the right verdict is 'tie':")
    report.table(
        ["judge", "x wins, judged once", "x wins swapped", "y wins swapped", "tie swapped"], rows
    )
    report.check(
        "position bias: swapping turns almost every pair into a tie (the right verdict)",
        share["position bias only"]["tie"] > 0.9,
        f"{share['position bias only']['tie']:.3f} ties",
    )
    report.check(
        "length bias: the longer answer wins in both orders and swapping reports no tie",
        share["length bias only"]["x"] > 0.9 and share["length bias only"]["tie"] < 0.1,
        f"x wins {share['length bias only']['x']:.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. Agreement, kappa, and how many human labels are needed")
    confusion = np.array([[40, 6, 4], [8, 30, 2], [3, 2, 5]])
    p_o, p_e, kappa = cohens_kappa(confusion)
    report.say("  rows: human (A wins, B wins, tie); columns: judge")
    report.say(f"  agreement {p_o:.3f}, chance agreement {p_e:.3f}, Cohen's kappa {kappa:.3f}")
    p_o2, _p_e2, kappa2 = cohens_kappa(np.array([[90, 0], [10, 0]]))
    report.say(f"  a judge that always says 'A': agreement {p_o2:.3f}, kappa {kappa2:.3f}")
    report.check(
        "the page's numbers: 0.750 / 0.418 / 0.570, and 0.900 agreement with kappa 0 for the lazy judge",
        (f"{p_o:.3f}", f"{p_e:.3f}", f"{kappa:.3f}") == ("0.750", "0.418", "0.570")
        and (f"{p_o2:.3f}", f"{kappa2:.3f}") == ("0.900", "0.000"),
    )
    report.check(
        "labels needed for +-5 points: 246 at 80% agreement, 385 at 50%",
        (labels_needed(0.8, 0.05), labels_needed(0.5, 0.05)) == (246, 385),
    )

    # Agreement with the truth depends on how far apart the answers are.
    g = gen.normal(size=n)
    verdict = g + 0.8 * gen.normal(size=n) > 0  # judge without position bias, one judgment
    edges = [0.0, 0.25, 0.5, 1.0, 2.0, 8.0]
    rows, by_bin = [], []
    for lo, hi in itertools.pairwise(edges):
        sel = (np.abs(g) >= lo) & (np.abs(g) < hi)
        agree = float(np.mean(verdict[sel] == (g[sel] > 0)))
        expected = float(np.mean(norm_cdf(np.abs(g[sel]) / 0.8)))
        by_bin.append(agree)
        rows.append([f"{lo:g} to {hi:g}", int(sel.sum()), f"{agree:.3f}", f"{expected:.3f}"])
    overall = float(np.mean(verdict == (g > 0)))
    report.say(
        "\n  Agreement with the truth by size of the quality gap (the same judge throughout):"
    )
    report.table(["|gap|", "pairs", "agreement", "expected"], rows)
    report.say(f"  one overall number would be {overall:.3f}, which hides all of that")
    report.check(
        "agreement rises with the gap, from near a coin flip to near certainty",
        by_bin == sorted(by_bin) and by_bin[0] < 0.6 and by_bin[-1] > 0.99,
        f"{by_bin[0]:.3f} -> {by_bin[-1]:.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("5. From pairwise votes to a leaderboard: Bradley-Terry with a bootstrap")
    rng = np.random.default_rng(0)  # the page's seed
    names = ["A", "B", "C", "D"]
    true_elo = np.array([1000.0, 1050.0, 1120.0, 1250.0])
    beta_true = elo_to_beta(true_elo)
    m, n_battles = 4, 3000
    i = rng.integers(m, size=n_battles)
    j = (i + rng.integers(1, m, size=n_battles)) % m
    win = (rng.random(n_battles) < 1 / (1 + np.exp(-(beta_true[i] - beta_true[j])))).astype(float)
    estimate = to_elo(fit_bt(i, j, win, m))
    boots = 40 if QUICK else 500
    boot = []
    for _ in range(boots):
        k = rng.integers(n_battles, size=n_battles)
        boot.append(to_elo(fit_bt(i[k], j[k], win[k], m)))
    boot = np.array(boot)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    report.table(
        ["model", "true Elo", "estimate", "95% interval"],
        [
            [name, f"{true_elo[q]:.0f}", f"{estimate[q]:.1f}", f"[{low[q]:.1f}, {high[q]:.1f}]"]
            for q, name in enumerate(names)
        ],
    )
    d_lo, d_hi = np.percentile(boot[:, 1] - boot[:, 0], [2.5, 97.5])
    report.say(
        f"  B - A: {estimate[1] - estimate[0]:.1f} points, interval [{d_lo:.1f}, {d_hi:.1f}]"
    )
    if QUICK:
        report.skip(
            "the page's table: B 1031.0, C 1118.6, D 1246.8 and B - A in [6.8, 52.3]",
            "500 bootstraps",
        )
    else:
        report.check(
            "the page's table: B 1031.0 [1006.8, 1052.3], C 1118.6 [1095.0, 1140.9], D 1246.8 [1223.2, 1270.5]",
            [f"{estimate[q]:.1f} [{low[q]:.1f}, {high[q]:.1f}]" for q in (1, 2, 3)]
            == ["1031.0 [1006.8, 1052.3]", "1118.6 [1095.0, 1140.9]", "1246.8 [1223.2, 1270.5]"],
        )
        report.check(
            "the interval of the difference B - A is [6.8, 52.3] and excludes 0",
            f"{d_lo:.1f} {d_hi:.1f}" == "6.8 52.3",
        )
    full_check(
        "every true rating lies inside its interval",
        all(low[q] <= true_elo[q] <= high[q] for q in range(m)),
    )
    # The likelihood is maximised: the gradient of the log-likelihood vanishes at the estimate.
    beta = (estimate - estimate[0]) * np.log(10) / 400
    p = 1 / (1 + np.exp(-(beta[i] - beta[j])))
    grad = np.bincount(i, win - p, m) - np.bincount(j, win - p, m)
    report.check(
        "the Newton fit sits at the maximum of the likelihood (gradient is zero)",
        np.abs(grad).max() < 1e-6,
        f"largest gradient entry {np.abs(grad).max():.1e}",
    )
    report.check(
        "a 400-point Elo gap is 10 to 1 odds, and the Elo scale is 400 / ln 10 per coefficient",
        abs(10 ** (400 / 400) - np.exp(float(elo_to_beta(np.array(400.0))))) < 1e-9,
    )

    # Elo as an online rule against Bradley-Terry as a fit.
    rows, finals = [], []
    for shuffle in range(5):
        order = np.random.default_rng(shuffle).permutation(n_battles)
        elo = online_elo(i[order], j[order], win[order], m)
        bt = to_elo(fit_bt(i[order], j[order], win[order], m))
        finals.append((elo - elo[0] + 1000, bt))
        rows.append([shuffle, f"{finals[-1][0][3] - finals[-1][0][1]:.1f}", f"{bt[3] - bt[1]:.1f}"])
    report.say("\n  The same 3,000 battles in five random orders; D minus B in rating points:")
    report.table(["order", "online Elo (K=4)", "Bradley-Terry"], rows)
    elo_spread = np.ptp([f[0][3] - f[0][1] for f in finals])
    bt_spread = np.ptp([f[1][3] - f[1][1] for f in finals])
    report.check(
        "Bradley-Terry does not depend on the order of the battles; online Elo does",
        bt_spread < 1e-6 < elo_spread,
        f"spread {bt_spread:.1e} against {elo_spread:.1f} points",
    )

    # -----------------------------------------------------------------------------------------
    report.section("6. The judge's bias goes into the leaderboard unless you model it")
    names = ["A", "B", "C", "D", "E"]
    true_elo = np.array([1000.0, 1050.0, 1120.0, 1250.0, 1000.0])
    length = np.array([1.0, 1.0, 1.0, 1.0, 4.0])  # E is as good as A but four times as wordy
    lam = 0.35  # the judge's log-odds bonus per unit of extra length
    m, n_battles = 5, 6000
    rng = np.random.default_rng(args.seed + 5)
    i = rng.integers(m, size=n_battles)
    j = (i + rng.integers(1, m, size=n_battles)) % m
    beta_true = elo_to_beta(true_elo)
    logit = beta_true[i] - beta_true[j] + lam * (length[i] - length[j])
    win = (rng.random(n_battles) < 1 / (1 + np.exp(-logit))).astype(float)
    naive = to_elo(fit_bt(i, j, win, m))
    w = fit_logistic(bt_design(i, j, m, extra=length[i] - length[j]), win)
    controlled = to_elo(np.r_[0.0, w[: m - 1]])
    report.table(
        ["model", "true Elo", "naive Bradley-Terry", "length-controlled"],
        [
            [names[q], f"{true_elo[q]:.0f}", f"{naive[q]:.0f}", f"{controlled[q]:.0f}"]
            for q in range(m)
        ],
    )
    report.say(f"  estimated length coefficient {w[-1]:.3f} (true {lam}); E is the wordy model")
    report.check(
        "a length-loving judge puts the wordy model E above C, though E is as good as A",
        naive[4] > naive[2] > naive[0],
        f"naive E {naive[4]:.0f}, C {naive[2]:.0f}, A {naive[0]:.0f}",
    )
    report.check(
        "adding the length difference as a covariate recovers the true order and the coefficient",
        list(np.argsort(controlled)) in ([0, 4, 1, 2, 3], [4, 0, 1, 2, 3])
        and abs(w[-1] - lam) < 0.06
        and abs(controlled[4] - 1000) < 40,
        f"controlled E {controlled[4]:.0f}, coefficient {w[-1]:.3f}",
    )
    report.say(
        "\n  This is the idea behind length-controlled win rates (Dubois et al., 2024): fit a model of the\n"
        "  judge's preference, then read the leaderboard at a length difference of zero. It only works\n"
        "  to the extent that the effect of length is what the regression says it is."
    )
    report.finish()


if __name__ == "__main__":
    main()
