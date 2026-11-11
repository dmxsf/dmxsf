"""Lab: scaling laws, from the closed forms on the page to a small study you can run yourself.

Course page: https://dmxsf.com/wiki/pretraining/scaling-laws/

1. the page's formulas and tables, recomputed (and the closed form checked against brute force);
2. what serving cost does to the best model;
3. a real, tiny scaling study on TinyShakespeare: tune the learning rate per size, train a grid of
   model sizes and token budgets, measure the noise, fit L(N, D) and try to predict a run it has not seen;
4. what a 12-point fit can and cannot identify, on synthetic data where the truth is known.

    uv run python labs/pretraining/scaling-laws/run.py [--quick]
"""

import math
import statistics as st
from itertools import pairwise

import numpy as np
import torch

from dmxsf import lab
from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.data import ByteCorpus, get_batch
from dmxsf.pretraining import accounting as acc
from dmxsf.pretraining import scaling
from dmxsf.pretraining.train import TrainConfig, bits_per_byte, make_optimizer, train

args, report = lab.start(
    "Scaling laws: the formulas, the serving cost, and a tiny study of our own",
    wiki=["pretraining/scaling-laws"],
)
E, A, B, ALPHA, BETA = scaling.E, scaling.A, scaling.B, scaling.ALPHA, scaling.BETA

# ---------------------------------------------------------------------------
report.section("1. Power laws and the compute-optimal split")
report.check(
    "Kaplan's N exponent 0.076: ten times the parameters multiply the loss by 0.84",
    round(10**-0.076, 2) == 0.84,
    f"{10**-0.076:.4f}",
)
rows, got = [], []
for c in (1e21, 1e22, 1e23, 5.76e23, 1e25):
    n, d = scaling.compute_optimal(c)
    rows.append(
        [
            f"{c:.3g}",
            f"{n / 1e9:.1f}B",
            f"{d / 1e12:.2f}T",
            f"{d / n:.1f}",
            f"{scaling.loss(n, d):.3f}",
        ]
    )
    got.append(rows[-1][1:])
report.table(["C (FLOPs)", "N_opt", "D_opt", "D / N", "loss"], rows)
report.check(
    "the page's table: 2.8B 0.06T 21.6 2.306 ... 312.1B 5.34T 17.1 1.912",
    got
    == [
        ["2.8B", "0.06T", "21.6", "2.306"],
        ["9.0B", "0.18T", "20.4", "2.141"],
        ["29.4B", "0.57T", "19.2", "2.032"],
        ["72.2B", "1.33T", "18.4", "1.974"],
        ["312.1B", "5.34T", "17.1", "1.912"],
    ],
)
a_exp, d_exp = scaling.optimal_exponents()
n1, _ = scaling.compute_optimal(1e22)
n2, _ = scaling.compute_optimal(1e23)
report.check(
    "exponents 0.51 and 0.49: ten times the budget buys about 3.2x the parameters and 3.1x the data",
    f"{a_exp:.2f} {d_exp:.2f} {n2 / n1:.2f} {(1e23 / n2) / (1e22 / n1):.2f}"
    == "0.51 0.49 3.26 3.07",
    f"N x{n2 / n1:.2f}, D x{(1e23 / n2) / (1e22 / n1):.2f}",
)

sizes = np.logspace(8, 12.5, 40001)
numeric = [sizes[np.argmin(scaling.loss(sizes, c / (6 * sizes)))] for c in (1e21, 1e22, 1e23, 1e24)]
closed = [scaling.compute_optimal(c)[0] for c in (1e21, 1e22, 1e23, 1e24)]
report.check(
    "the closed form is the brute-force minimum of L(N, C/6N) (the four IsoFLOP minima of the page's figure)",
    all(abs(x / y - 1) < 1e-3 for x, y in zip(numeric, closed, strict=True)),
    ", ".join(f"{x / 1e9:.1f}B" for x in closed),
)
worse = max(
    scaling.loss(n1 * f, 1e22 / (6 * n1 * f)) - scaling.loss(n1, 1e22 / (6 * n1))
    for f in (0.5, 2.0)
)
report.check(
    "the valley is flat: half or double the optimal size costs less than 0.02 in loss",
    0 < worse < 0.02,
    f"worst +{worse:.4f}",
)
c_llama3 = 6 * 8e9 * 15e12
n_opt, d_opt = scaling.compute_optimal(c_llama3)
gap = scaling.loss(8e9, 15e12) - scaling.loss(n_opt, d_opt)
report.check(
    "Llama 3 8B (D/N = 1875) sits 0.05 above the compute-optimal loss at its own budget",
    round(gap, 2) == 0.05,
    f"optimal at this budget would be {n_opt / 1e9:.0f}B on {d_opt / 1e12:.1f}T tokens; an extrapolation",
)
report.check(
    "D = 20N and C = 6ND give C = 120 N^2: 1.2e21 FLOPs -> 3.2B (quiz), 1e22 -> 9.1B and 180B tokens (exercise)",
    f"{math.sqrt(1.2e21 / 120) / 1e9:.1f} {math.sqrt(1e22 / 120) / 1e9:.1f} "
    f"{round(20 * math.sqrt(1e22 / 120) / 1e9, -1):.0f}" == "3.2 9.1 180",
    f"the last is {20 * math.sqrt(1e22 / 120) / 1e9:.0f}B, which the page rounds to 180B",
)
flops = acc.train_flops(7e9, 3e12)
report.check(
    "exercise 2: a 7B model on 3T tokens is 1.26e23 FLOPs, 88 thousand H100 hours at MFU 40%",
    f"{flops:.3g} {acc.gpu_hours(flops, 989e12, 0.4) / 1e4:.1f}" == "1.26e+23 8.8",
    f"{acc.gpu_hours(flops, 989e12, 0.4):,.0f} GPU hours",
)
report.check(
    "6ND and 184,320 A100 hours put Llama 2 7B at MFU 39%",
    f"{acc.implied_mfu(acc.train_flops(6.74e9, 2e12), 184_320, 312e12):.0%}" == "39%",
)

# ---------------------------------------------------------------------------
report.section("2. Counting the serving cost: 6ND to train plus 2NT to serve")
rows, got = [], []
for served in (0, 1e12, 1e13, 1e14, 1e15):
    _, n, d = scaling.best_model(2.0, served)
    rows.append(
        [f"{served:.0e}" if served else "0", f"{n / 1e9:.1f}B", f"{d / 1e12:.2f}T", f"{d / n:.0f}"]
    )
    got.append(rows[-1][1:])
report.table(["tokens served", "best N", "training tokens", "tokens / parameter"], rows)
report.check(
    "the page's table: 46.9B 0.88T 19 | 34.3B 1.25T 36 | 20.0B 3.01T 150 | 12.7B 11.12T 876 | 9.6B 50.60T 5264",
    got
    == [
        ["46.9B", "0.88T", "19"],
        ["34.3B", "1.25T", "36"],
        ["20.0B", "3.01T", "150"],
        ["12.7B", "11.12T", "876"],
        ["9.6B", "50.60T", "5264"],
    ],
)
report.check(
    "with nothing served the best model is the Chinchilla one, about 19 tokens per parameter",
    round(float(got[0][2])) == 19,
)
report.check(
    "more serving means a smaller model on more data, monotonically",
    all(float(a[0][:-1]) > float(b[0][:-1]) for a, b in pairwise(got)),
)
ratios = {"GPT-3": (175e9, 300e9), "Chinchilla": (70e9, 1.4e12), "LLaMA 7B": (7e9, 1e12),
          "Llama 2 7B": (7e9, 2e12), "Llama 3 8B": (8e9, 15e12)}  # fmt: skip
report.check(
    "tokens per parameter of the page's models: 1.7, 20, 143, 286, 1875",
    [f"{d / n:.1f}" if d / n < 10 else f"{d / n:.0f}" for n, d in ratios.values()]
    == ["1.7", "20", "143", "286", "1875"],
)

# ---------------------------------------------------------------------------
report.section("3. A tiny scaling study on TinyShakespeare")
corpus = ByteCorpus.shakespeare()
BATCH, BLOCK = 8, 64  # 512 tokens per step
SHAPES = [(16, 2, 2), (32, 2, 4), (64, 3, 4), (128, 4, 4)]  # d_model, layers, heads
TOKENS = (32_768, 131_072, 524_288)  # 64, 256 and 1024 steps; the last is half an epoch
LADDER = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
START = {16: 3, 32: 3, 64: 2, 128: 1}  # where the learning-rate search starts: it falls with width
SEEDS = 1 if args.quick else 2
if args.quick:
    SHAPES, TOKENS, START = SHAPES[:2], (16_384, 65_536), {16: 4, 32: 3}


def build(shape: tuple[int, int, int], seed: int) -> GPT:
    d_model, layers, heads = shape
    torch.manual_seed(seed)
    cfg = small_gpt(
        vocab_size=256, block_size=BLOCK, n_layer=layers, n_head=heads, d_model=d_model,
        norm="rmsnorm", pos="rope", mlp="swiglu", bias=False,
    )  # fmt: skip
    return GPT(cfg)


def train_cell(shape: tuple[int, int, int], tokens: int, lr: float, seed: int) -> float:
    """Train one model for ``tokens`` tokens; return its validation loss in nats per byte."""
    model = build(shape, seed)
    steps = tokens // (BATCH * BLOCK)
    cfg = TrainConfig(steps=steps, peak_lr=lr, warmup=min(50, steps // 10 + 1))
    batches = lambda step: get_batch(corpus.train, step, BATCH, BLOCK, seed)  # noqa: E731
    train(model, make_optimizer(model, cfg), batches, cfg)
    return bits_per_byte(model, corpus.val, BLOCK, 800) * math.log(2)


def tune_lr(shape: tuple[int, int, int], tokens: int) -> dict[float, float]:
    """Three learning rates around a starting guess; move the window while the best is at its edge."""
    results: dict[float, float] = {}
    lo, hi = max(START[shape[0]] - 1, 0), min(START[shape[0]] + 1, len(LADDER) - 1)
    if args.quick:
        lo = hi = START[shape[0]]  # one learning rate: the quick run only checks that the code runs
    while True:
        for lr in LADDER[lo : hi + 1]:
            if lr not in results:
                results[lr] = train_cell(shape, tokens, lr, args.seed)
        best = min(results, key=results.get)
        if args.quick or (best != LADDER[lo] and best != LADDER[hi]):
            return results
        if best == LADDER[lo] and lo > 0:
            lo -= 1
        elif best == LADDER[hi] and hi < len(LADDER) - 1:
            hi += 1
        else:
            return results  # the best is at the end of the ladder: say so rather than search on


n_of = {shape[0]: acc.matmul_params(build(shape, 0)) for shape in SHAPES}
report.say(
    f"  {len(SHAPES)} sizes (N = {', '.join(f'{n:,}' for n in n_of.values())}) x {len(TOKENS)} token budgets, "
    f"{BATCH} x {BLOCK} = {BATCH * BLOCK} tokens per step, {SEEDS} seed(s) per cell;"
)
report.say("  validation loss in nats per byte on the held-out 5% of the text")
mid = TOKENS[len(TOKENS) // 2]
sweeps = {shape[0]: tune_lr(shape, mid) for shape in SHAPES}
best_lr = {d: min(r, key=r.get) for d, r in sweeps.items()}
report.say(f"\n  learning-rate sweep at {mid:,} tokens (the best of each row is marked *):")
rows = []
for d_model, lrs in sweeps.items():
    marks = [
        f"{lrs[lr]:.3f}{'*' if lr == best_lr[d_model] else ''}" if lr in lrs else ""
        for lr in LADDER
    ]
    rows.append([f"d_model {d_model}", *marks])
report.table(["", *[f"{lr:g}" for lr in LADDER]], rows)

cells: dict[tuple[int, int], list[float]] = {}
for shape in SHAPES:
    d_model = shape[0]
    for tokens in TOKENS:
        losses = []
        for seed in range(args.seed, args.seed + SEEDS):
            if tokens == mid and seed == args.seed:
                losses.append(
                    sweeps[d_model][best_lr[d_model]]
                )  # the sweep already trained this cell
            else:
                losses.append(train_cell(shape, tokens, best_lr[d_model], seed))
        cells[(n_of[d_model], tokens)] = losses
ns = sorted(n_of.values())
mean = {k: st.mean(v) for k, v in cells.items()}
spread = {k: max(v) - min(v) for k, v in cells.items()}

report.say(
    "\n  mean validation loss over seeds (rows: N = matrix-multiply parameters, columns: training tokens D):"
)
report.table(
    ["N \\ D", *[f"{t:,}" for t in TOKENS]],
    [[f"{n:,}", *[f"{mean[(n, t)]:.3f}" for t in TOKENS]] for n in ns],
)
if SEEDS > 1:
    noise = max(spread.values())
    report.say(
        f"  seed-to-seed range over the {len(cells)} cells: median {st.median(spread.values()):.3f}, "
        f"largest {noise:.3f} nats"
    )
else:
    noise = 0.0
    report.say("  (one seed: no noise estimate, so no difference below is called real)")

report.check(
    "more tokens always help: at every size the loss falls with D",
    all(
        mean[(n, TOKENS[i])] > mean[(n, TOKENS[i + 1])] for n in ns for i in range(len(TOKENS) - 1)
    ),
)
big_d = TOKENS[-1]
if args.quick:
    for name in (
        "with the most data, loss falls with N",
        "with the fewest tokens the biggest model is not the best",
        "the best learning rate falls with model size",
        "an untuned learning rate costs as much as a size step",
        "the fit predicts a run it never saw",
    ):
        report.skip(name, "needs the full grid")
else:
    report.check(
        "with the most data every size step helps: loss falls with N",
        all(mean[(a, big_d)] > mean[(b, big_d)] for a, b in pairwise(ns)),
    )
    winner = min(ns, key=lambda n: mean[(n, TOKENS[0])])
    report.check(
        "with the fewest tokens (64 steps) the biggest model is not the best: it cannot use its capacity",
        winner != ns[-1],
        f"best at {TOKENS[0]:,} tokens: N = {winner:,}",
    )
    first, last = SHAPES[0][0], SHAPES[-1][0]
    report.check(
        "the best learning rate falls with model size",
        best_lr[first] > best_lr[last],
        f"{best_lr[first]:g} for N = {ns[0]:,}, {best_lr[last]:g} for N = {ns[-1]:,}",
    )
    fixed = {}
    for shape in SHAPES:
        lrs = sweeps[shape[0]]
        fixed[shape[0]] = lrs[3e-3] if 3e-3 in lrs else train_cell(shape, mid, 3e-3, args.seed)
    report.say(f"\n  what an untuned learning rate costs (3e-3 for every size, {mid:,} tokens):")
    tuned = {d: sweeps[d][best_lr[d]] for d in sweeps}
    report.table(
        ["N", "tuned lr", "tuned loss", "loss at 3e-3", "cost of not tuning"],
        [
            [
                f"{n_of[d]:,}",
                f"{best_lr[d]:g}",
                f"{tuned[d]:.3f}",
                f"{fixed[d]:.3f}",
                f"+{fixed[d] - tuned[d]:.3f}",
            ]
            for d in sweeps
        ],
    )
    step = mean[(ns[0], mid)] - mean[(ns[1], mid)]
    cost = fixed[first] - tuned[first]
    report.check(
        "not tuning the learning rate costs as much as a size step: the curve then shows the recipe, not the scale",
        cost > 0.5 * step,
        f"+{cost:.3f} for the smallest model, against {step:.3f} gained by the next size up",
    )

    ns_a = np.array([n for n, _ in cells], dtype=float)
    ds_a = np.array([d for _, d in cells], dtype=float)
    ys_a = np.array([mean[k] for k in cells])
    fit = scaling.fit(ns_a, ds_a, ys_a)
    report.say(
        f"\n  fit of L = E + A/N^alpha + B/D^beta to all {len(cells)} cells: E = {fit['E']:.2f}, "
        f"alpha = {fit['alpha']:.2f}, beta = {fit['beta']:.2f}, rms residual {fit['rms']:.3f} nats"
    )
    report.say(
        f"  (the page's coefficients, E = {E}, alpha = {ALPHA}, beta = {BETA}, describe subword tokens of web "
        "text; bytes of one play are another setting, so these are different numbers, not wrong ones)"
    )
    report.check(
        "a loss floor and two decaying terms describe the grid",
        fit["rms"] < 0.25,
        f"rms {fit['rms']:.3f}",
    )

    # Hold out the biggest cell, fit the rest, predict it.
    big_n = ns[-1]
    mask = np.array([not (n == big_n and d == big_d) for n, d in cells])
    rest = scaling.fit(ns_a[mask], ds_a[mask], ys_a[mask])
    coef = (rest["E"], rest["A"], rest["B"], rest["alpha"], rest["beta"])
    predicted = float(scaling.loss(big_n, big_d, *coef))
    actual = mean[(big_n, big_d)]
    neighbour = mean[(big_n, TOKENS[-2])]  # the naive guess: the last cell we trained at this size
    report.say(
        f"\n  held out the biggest run (N = {big_n:,}, D = {big_d:,}, {6 * big_n * big_d:.2g} FLOPs, "
        f"{big_n * big_d / (ns[-2] * big_d):.1f}x the compute of the next smaller model at the same D):"
    )
    report.say(
        f"  predicted {predicted:.3f}, measured {actual:.3f} "
        f"(seeds {min(cells[(big_n, big_d)]):.3f} - {max(cells[(big_n, big_d)]):.3f}), "
        f"the previous cell of this size was {neighbour:.3f}"
    )
    report.check(
        "the fit predicts a run it never saw better than 'same as the previous cell of that size'",
        abs(predicted - actual) < abs(neighbour - actual),
        f"error {abs(predicted - actual):.3f} against {abs(neighbour - actual):.3f}",
    )
    limit = max(noise, rest["rms"])
    report.say(
        f"  the error is {'within' if abs(predicted - actual) <= limit else 'beyond'} the larger of the seed "
        f"range ({noise:.3f}) and the fit's rms ({rest['rms']:.3f}) when the biggest run is left out"
    )

    # Where does the fit put the best model for a budget, and what did the grid measure near it?
    c_near = 6 * ns[1] * big_d
    n_star, d_star = scaling.compute_optimal(c_near, fit["A"], fit["B"], fit["alpha"], fit["beta"])
    report.say(
        f"\n  at C = {c_near:.2g} FLOPs the fit's best split is N = {n_star:,.0f} on D = {d_star:,.0f} tokens "
        f"(D / N = {d_star / n_star:.0f}; the page's rule of thumb is 20; the corpus has {len(corpus.train):,} "
        "training bytes)"
    )
    near = sorted(sorted(cells, key=lambda k: abs(math.log(6 * k[0] * k[1] / c_near)))[:3])
    report.table(
        ["cell near that budget", "FLOPs", "D / N", "loss"],
        [
            [f"N = {n:,}, D = {d:,}", f"{6 * n * d:.2g}", f"{d / n:.1f}", f"{mean[(n, d)]:.3f}"]
            for n, d in near
        ],
    )
    best_near = min(near, key=lambda k: mean[k])
    closest = min(near, key=lambda k: abs(math.log(k[0] / n_star)))
    report.check(
        "among the cells near that budget the fit and the measurements pick the same winner",
        best_near == closest,
        f"N = {best_near[0]:,}",
    )

# ---------------------------------------------------------------------------
report.section("4. What a small grid can and cannot identify (synthetic data, known truth)")
n_grid = np.repeat(np.logspace(7, 9, 4), 3)
d_grid = np.tile(np.logspace(9.5, 11, 3), 4)
truth = scaling.loss(n_grid, d_grid)
draws = 8 if args.quick else 30
# Two draws of N(0, sigma^2) differ by 1.13 sigma on average, so a measured seed-to-seed range maps
# to a noise level; without a measurement use the same order of magnitude.
typical = (st.median(spread.values()) if SEEDS > 1 else 0.03) / 1.128
SIGMAS = (typical, 10 * typical)
n_true = scaling.compute_optimal(1e23)[0]
final_loss = float(scaling.loss(1e9, 1e12))
one_decade = float(
    scaling.loss(*scaling.compute_optimal(1e22)) - scaling.loss(*scaling.compute_optimal(1e23))
)


def study(sigma: float, stream: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng([args.seed, stream])
    fits = [
        scaling.fit(n_grid, d_grid, truth + rng.normal(0, sigma, truth.shape)) for _ in range(draws)
    ]
    n_best = np.array(
        [scaling.compute_optimal(1e23, f["A"], f["B"], f["alpha"], f["beta"])[0] for f in fits]
    )
    return {
        **{k: np.array([f[k] for f in fits]) for k in ("E", "A", "alpha", "B", "beta")},
        "loss at N=1e9, D=1e12": np.array(
            [scaling.loss(1e9, 1e12, f["E"], f["A"], f["B"], f["alpha"], f["beta"]) for f in fits]
        ),
        "N_opt at C=1e23": n_best,
        "penalty": scaling.loss(n_best, 1e23 / (6 * n_best))
        - scaling.loss(n_true, 1e23 / (6 * n_true)),
    }


def half_width(v: np.ndarray, true: float) -> float:
    return float((np.percentile(v, 90) - np.percentile(v, 10)) / 2 / abs(true))


def pct(x: float) -> str:
    return ">1000%" if x > 10 else f"{x:.0%}"


studies = [study(s, k) for k, s in enumerate(SIGMAS)]
truths = {
    "E": E,
    "A": A,
    "alpha": ALPHA,
    "B": B,
    "beta": BETA,
    "loss at N=1e9, D=1e12": final_loss,
    "N_opt at C=1e23": n_true,
}
report.say(
    f"  12 cells of the page's law (N 1e7 - 1e9, D 3e9 - 1e11), {draws} noisy copies of the grid, fitted each time;"
)
report.say(
    f"  noise sigma = {SIGMAS[0]:.3f} (the median seed-to-seed range of section 3) and ten times that; "
    "10th - 90th percentile of the fitted values:"
)
report.table(
    ["quantity", "truth", f"sigma {SIGMAS[0]:.3f}", "+-", f"sigma {SIGMAS[1]:.3f}", "+-"],
    [
        [
            name,
            f"{true:.4g}",
            *[
                x
                for st_ in studies
                for x in (
                    f"{np.percentile(st_[name], 10):.4g} - {np.percentile(st_[name], 90):.4g}",
                    pct(half_width(st_[name], true)),
                )
            ],
        ]
        for name, true in truths.items()
    ],
)
for res, label in zip(studies, ("measured noise", "ten times the noise"), strict=True):
    loose = max(half_width(res["A"], A), half_width(res["B"], B))
    tight = half_width(res["loss at N=1e9, D=1e12"], final_loss)
    report.check(
        f"{label}: the coefficients are far less certain than the predictions they combine into",
        loose > 5 * tight,
        f"uncertainty of A and B {pct(loose)}, of the extrapolated loss {tight:.1%} (half-width of the 10 - 90% range)",
    )
report.check(
    "measured noise: a typical fitted N_opt costs a few hundredths of a nat (the valley is flat)",
    float(np.median(studies[0]["penalty"])) < 0.05,
    f"loss penalty at C = 1e23 under the true law: median {np.median(studies[0]['penalty']):.4f}, "
    f"worst tenth {np.percentile(studies[0]['penalty'], 90):.4f} nats or more",
)
report.say(
    f"  with ten times the noise the median penalty is {np.median(studies[1]['penalty']):.3f} and the worst tenth "
    f"lose {np.percentile(studies[1]['penalty'], 90):.3f} or more; "
    f"for scale, a tenfold cut in the compute budget costs {one_decade:.3f}"
)
report.say(
    "  so a fitted 'alpha = 0.34' deserves less trust than the curve it came from; compare predictions, not coefficients"
)

report.finish()
