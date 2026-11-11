"""Lab: the pretraining pipeline, shrunk to a laptop (offline, bytes, two domains).

Course page: https://dmxsf.com/wiki/pretraining/training-pipeline/

The page runs these experiments on WikiText-2 and the Python standard library with a 0.86M-parameter
GPT and 2000 steps. Here the prose is the bundled TinyShakespeare and the model is smaller, so the
numbers differ; what has to hold is the *pattern* each experiment shows.

1. two checks before you spend any compute: the first loss, and overfitting one batch;
2. a baseline run and how to read its monitoring log;
3. experiment 1, the data mix: what each domain's loss does when its share changes;
4. experiment 2, resuming: four ways to continue from a checkpoint, only one of them exact;
5. experiment 3, leakage: validation text inside the training data.

    uv run python labs/pretraining/training-pipeline/run.py [--quick]
"""

import math
import shutil
import statistics as st
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from dmxsf import lab
from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.data import (
    ByteCorpus,
    as_array,
    get_batch,
    get_mixed_batch,
    load_python_code,
    split_documents,
)
from dmxsf.pretraining.train import (
    TrainConfig,
    bits_per_byte,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    train,
)

args, report = lab.start(
    "The pretraining pipeline on a laptop: monitoring, data mix, resuming, leakage",
    wiki=["pretraining/training-pipeline"],
)
QUICK = args.quick
B, T = (8, 128) if QUICK else (16, 128)
STEPS = 150 if QUICK else 1200
HALF = STEPS // 2
SEEDS = (0,) if QUICK else (0, 1)
BITS = math.log(2)

prose, code = ByteCorpus.shakespeare(), ByteCorpus.python_code()
SRC = {"prose": prose.train, "code": code.train}
VAL = {"prose": prose.val, "code": code.val}
CFG = TrainConfig(steps=STEPS, warmup=30 if not QUICK else 10, peak_lr=6e-3)
BASE = {"prose": 0.5, "code": 0.5}


def make(seed: int = 0):
    """The model and optimizer; the seed fixes the initialisation."""
    torch.manual_seed(seed)
    model = GPT(small_gpt(block_size=T, d_model=64, n_layer=2))
    return model, make_optimizer(model, CFG)


def batches(mix=None, seed: int = 0, src=None):
    mix = mix or BASE
    return lambda step: get_mixed_batch(src or SRC, mix, step, B, T, seed)


def scores(model, data=None) -> dict[str, float]:
    data = data or VAL
    return {k: bits_per_byte(model, v, T) for k, v in data.items()}


def run(mix=None, seed: int = 0, src=None):
    model, opt = make(seed)
    rows = train(model, opt, batches(mix, seed, src), CFG)
    return model, rows


def weight_gap(a, b) -> float:
    pairs = zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    return max((x - y).abs().max().item() for x, y in pairs)


def mean_loss_bits(rows) -> float:
    return sum(r["loss"] for r in rows) / len(rows) / BITS


# ---------------------------------------------------------------------------
report.section("1. Two free checks before you train")
page = small_gpt()  # the page's 0.86M-parameter model
pcfg = TrainConfig(steps=60, warmup=10, peak_lr=6e-3)
n_page = sum(p.numel() for p in GPT(page).parameters())
report.say(
    f"  the page's model: {n_page:,} parameters; one batch of {B} x {T} bytes, 60 steps of AdamW"
)
table, first_loss, to_eight = [], {}, {}
for name, std in (("N(0, 0.02^2), GPT-2 style", 0.02), ("N(0, 1), nn.Embedding's default", 1.0)):
    torch.manual_seed(0)
    model = GPT(small_gpt(init_std=std))
    opt = make_optimizer(model, pcfg)
    rows = train(model, opt, lambda s: get_batch(code.train, s, B, T), pcfg, 0, 60)
    losses = [r["loss"] / BITS for r in rows]
    first_loss[std] = losses[0]
    below = next((i for i, v in enumerate(losses) if v < 8.0), None)
    to_eight[std] = below
    clipped = sum(r["grad_norm"] > 1.0 for r in rows[:50])
    table.append(
        [
            name,
            f"{losses[0]:.2f}",
            "never (60 steps)" if below is None else below,
            f"{clipped} of 50",
        ]
    )
report.table(
    ["initialisation", "loss at step 0 (bits)", "first step below 8 bits", "clipped steps"], table
)
report.check(
    "with a sensible initialisation the first loss is log2(256) = 8 bits",
    abs(first_loss[0.02] - 8.0) < 0.3,
    f"{first_loss[0.02]:.2f}",
)
report.check(
    "with N(0, 1) and tied embeddings the model starts far worse than uniform guessing",
    first_loss[1.0] > 3 * 8.0 and (to_eight[1.0] is None or to_eight[1.0] > 10),
    f"{first_loss[1.0]:.1f} bits; the page's 0.86M model starts at 108.0",
)

torch.manual_seed(0)
model = GPT(small_gpt(block_size=64, d_model=64, n_layer=2))
flat = TrainConfig(steps=300, warmup=0, peak_lr=3e-3, floor=1.0, schedule="linear")  # constant lr
batch = get_batch(code.train, 0, 4, 64)  # the same batch, again and again
rows = train(model, make_optimizer(model, flat), lambda s: batch, flat)
overfit = [rows[i]["loss"] / BITS for i in (0, 99, 199, 299)]
report.say(
    "\n  one batch, 300 steps, constant learning rate: loss "
    + " -> ".join(f"{v:.2f}" for v in overfit)
    + " bits (steps 1, 100, 200, 300)"
)
report.check(
    "the model can memorise a single batch: the code, the loss and the optimizer all work",
    overfit[-1] < 0.5 and overfit[0] > 7,
    f"{overfit[0]:.2f} -> {overfit[-1]:.2f} bits",
)

# ---------------------------------------------------------------------------
report.section(f"2. A baseline run: {STEPS} steps, 50% prose + 50% code, and how to read its log")
nparams = sum(p.numel() for p in make()[0].parameters())
report.say(
    f"  {nparams:,}-parameter byte-level GPT, {B * T:,} bytes per step ({STEPS * B * T / 1e6:.2f} MB in all); "
    f"prose {len(prose.train) / 1e6:.2f} MB (TinyShakespeare, last 5% held out), "
    f"code {len(code.train) / 1e6:.2f} MB (Python standard library, held-out files)"
)
tmp = Path(tempfile.mkdtemp())
checkpoint = tmp / "mid.pt"
model, opt = make()
every = max(1, STEPS // 6)


def val_fn(m):
    return {f"val_{k}": v for k, v in scores(m).items()}


rows = train(model, opt, batches(), CFG, 0, HALF, eval_fn=val_fn, eval_every=every)
save_checkpoint(checkpoint, model, opt, HALF)  # weights + Adam state + step, as the page says
rows += train(model, opt, batches(), CFG, HALF, STEPS, eval_fn=val_fn, eval_every=every)
base = model
base_scores = scores(base)
table = []
for end in range(every, STEPS + 1, every):
    window = rows[end - every : end]
    ev = next(r for r in window if r["step"] == end and "val_prose" in r)
    norms = [r["grad_norm"] for r in window]
    table.append(
        [
            end,
            f"{window[-1]['lr']:.1e}",
            f"{mean_loss_bits(window[-20:]):.2f}",
            f"{ev['val_prose']:.2f}",
            f"{ev['val_code']:.2f}",
            f"{st.median(norms):.2f}",
            f"{sum(n > 1.0 for n in norms)} of {len(norms)}",
        ]
    )
report.table(
    ["step", "lr", "train loss", "val prose", "val code", "grad norm (median)", "clipped"], table
)
norms = [r["grad_norm"] for r in rows]
peak_at = int(np.argmax(norms))
early = sum(n > 1.0 for n in norms[:100]) / 100
late = sum(n > 1.0 for n in norms[100:]) / max(1, len(norms) - 100)
report.say(
    f"  largest gradient norm before clipping: {max(norms):.1f} at step {peak_at + 1}; "
    f"clipped in {early:.0%} of the first 100 steps against {late:.1%} of the rest"
)
val_curve = [r["val_prose"] + r["val_code"] for r in rows if "val_prose" in r]
final_train = mean_loss_bits(rows[-20:])
report.check(
    "the loss starts at log2(256) = 8 bits",
    abs(rows[0]["loss"] / BITS - 8.0) < 0.3,
    f"{rows[0]['loss'] / BITS:.2f}",
)
report.check(
    "validation loss falls at every evaluation (prose + code)",
    all(a > b for a, b in pairwise(val_curve)),
    " > ".join(f"{v / 2:.2f}" for v in val_curve),
)
gap = sum(base_scores.values()) / 2 - final_train
report.check(
    "no overfitting yet: the validation mean is close to the training loss",
    abs(gap) < 0.3,
    f"validation mean {sum(base_scores.values()) / 2:.2f}, training {final_train:.2f} bits",
)
report.check(
    "clipping is a start-up event: clipped steps are more frequent early than later",
    early > late,
    f"{early:.0%} against {late:.1%}",
)

# ---------------------------------------------------------------------------
report.section("3. Experiment 1: the data mix decides what the model is good at")
# One run per share with seed 0, plus the 50:50 run again with seed 1: two seeds of the same
# configuration are what tells how big a "real" difference has to be.
shares = (0.9, 0.5, 0.1)
results: dict[tuple[float, int], dict[str, float]] = {(0.5, 0): base_scores}
for share, seed in [(0.9, 0), (0.1, 0), (0.5, SEEDS[-1])]:
    if (share, seed) not in results:
        results[(share, seed)] = scores(run({"prose": share, "code": 1 - share}, seed)[0])
table_rows = [
    [
        f"{share:.0%} : {1 - share:.0%}",
        seed,
        f"{r['prose']:.3f}",
        f"{r['code']:.3f}",
        f"{(r['prose'] + r['code']) / 2:.3f}",
    ]
    for (share, seed), r in sorted(results.items(), key=lambda kv: (kv[0][1], -kv[0][0]))
]
report.table(["prose : code", "seed", "prose bpb", "code bpb", "mean"], table_rows)
if QUICK:
    report.skip("a 90% share beats a 10% share for that domain", "needs the full run")
else:
    gains = {
        "prose": results[(0.1, 0)]["prose"] - results[(0.9, 0)]["prose"],
        "code": results[(0.9, 0)]["code"] - results[(0.1, 0)]["code"],
    }
    report.check(
        "a 90% share beats a 10% share for that domain",
        min(gains.values()) > 0,
        f"gain {gains['prose']:.2f} bits on prose, {gains['code']:.2f} bits on code",
    )
    noise = {d: abs(results[(0.5, 0)][d] - results[(0.5, SEEDS[-1])][d]) for d in ("prose", "code")}
    report.check(
        "...by more than the seed range of the 50:50 runs",
        all(gains[d] > noise[d] for d in gains),
        f"seed range {noise['prose']:.2f} bits on prose, {noise['code']:.2f} bits on code",
    )
    mean = {sh: (results[(sh, 0)]["prose"] + results[(sh, 0)]["code"]) / 2 for sh in shares}
    report.say(
        "  the mean over both domains hardly moves ("
        + " / ".join(f"{mean[sh]:.3f}" for sh in shares)
        + f" for 90:10 / 50:50 / 10:90, seed range {abs(mean[0.5] - sum(results[(0.5, SEEDS[-1])].values()) / 2):.3f}):"
        " one number would have hidden the trade"
    )

# ---------------------------------------------------------------------------
report.section(f"4. Experiment 2: resuming from step {HALF}")


def resume(keep_optimizer=True, data_from=HALF, lr_from=HALF):
    """Continue to the end from the checkpoint. data_from / lr_from: the step the batches and the
    learning-rate schedule restart at (HALF is correct)."""
    model, opt = make()
    load_checkpoint(checkpoint, model, opt if keep_optimizer else None)
    rows = train(
        model,
        opt,
        batches(),
        CFG,
        data_from,
        data_from + (STEPS - HALF),
        lr_shift=lr_from - data_from,
    )
    return model, rows


cases = {
    "weights + optimizer + step": {},
    "weights only": dict(keep_optimizer=False),
    "forgot the step (data and lr restart)": dict(data_from=0, lr_from=0),
    "data right, lr restarts": dict(lr_from=0),
}
base_early = mean_loss_bits(rows[HALF : HALF + 20])
table, gaps, means = [], {}, {}
for name, kwargs in cases.items():
    m, r = resume(**kwargs)
    s = scores(m)
    gaps[name] = weight_gap(m, base)
    means[name] = (s["prose"] + s["code"]) / 2
    table.append(
        [
            name,
            f"{s['prose']:.3f}",
            f"{s['code']:.3f}",
            f"{means[name]:.3f}",
            f"{gaps[name]:.2e}",
            f"{mean_loss_bits(r[:20]):.3f}",
        ]
    )
base_mean = sum(base_scores.values()) / 2
table.insert(
    0,
    [
        "uninterrupted baseline",
        f"{base_scores['prose']:.3f}",
        f"{base_scores['code']:.3f}",
        f"{base_mean:.3f}",
        "0",
        f"{base_early:.3f}",
    ],
)
report.table(
    ["resume", "prose bpb", "code bpb", "mean", "max |w - baseline|", "loss, first 20 steps"], table
)
names = list(cases)
report.check(
    "with weights, optimizer state and step restored, the resumed run equals the uninterrupted one bit for bit",
    gaps[names[0]] == 0.0 and abs(means[names[0]] - base_mean) == 0.0,
    f"largest weight difference {gaps[names[0]]}",
)
report.check(
    "each of the other three ends with different weights",
    all(gaps[n] > 1e-3 for n in names[1:]),
    ", ".join(f"{gaps[n]:.2f}" for n in names[1:]),
)
worst = max(abs(means[n] - base_mean) for n in names[1:])
if len(SEEDS) < 2:
    report.skip(
        "...and the validation score cannot tell them apart", "needs two seeds for the noise"
    )
else:
    noise = abs(sum(results[(0.5, 0)].values()) / 2 - sum(results[(0.5, SEEDS[-1])].values()) / 2)
    report.check(
        "...and the validation score cannot tell them apart: no shift is twice a change of seed",
        worst < 2 * noise,
        f"largest shift {worst:.3f} bits for the three, {noise:.3f} between seeds 0 and {SEEDS[-1]}",
    )
    better = [n for n in names[1:] if means[n] < base_mean - noise / 2]
    if better:
        report.say(
            "  the runs that restart the learning rate even score *better* than the correct resume ("
            + "; ".join(f"{n.strip()}: {means[n] - base_mean:+.3f}" for n in better)
            + "): at this step the model is still under-trained, so a higher learning rate helps.\n"
            "  The score cannot tell a bug from a benefit."
        )

# ---------------------------------------------------------------------------
report.section("5. Experiment 3: validation text in the training data")
docs = load_python_code()
_, val_docs = split_documents(docs, 0.05)
k = 0
while sum(map(len, val_docs[:k])) < 4096:  # the first held-out files, until there are 4 KB
    k += 1
chunk = as_array(val_docs[:k])[:4096]
rest = as_array(val_docs[k:])
SHARE = 0.2
report.say(
    f"  leaked: the first {len(chunk) / 1e3:.1f} KB of the held-out code; the remaining {len(rest) / 1e3:.0f} KB "
    f"never enter training. {SHARE:.0%} of the training windows come from the extra data."
)


def retrain_with(extra):
    src = dict(SRC, extra=extra)
    mix = {"prose": (1 - SHARE) / 2, "code": (1 - SHARE) / 2, "extra": SHARE}
    model, _ = run(mix, 0, src)
    return model


control = retrain_with(SRC["code"][: len(chunk)])  # as much extra data, but from the training split
leaked = retrain_with(chunk)
out = {
    "baseline: no extra data": base,
    "control: 4 KB of training code": control,
    "leaked: the 4 KB of held-out code": leaked,
}
table = {n: (bits_per_byte(m, chunk, T, 32), bits_per_byte(m, rest, T)) for n, m in out.items()}
report.table(
    ["model", "leaked chunk (bpb)", "rest of held-out code (bpb)"],
    [[n, f"{a:.3f}", f"{b:.3f}"] for n, (a, b) in table.items()],
)
c_chunk, c_rest = table["control: 4 KB of training code"]
l_chunk, l_rest = table["leaked: the 4 KB of held-out code"]
if QUICK:
    report.skip("the leaked chunk scores far better than under the control", "needs the full run")
else:
    report.check(
        "the leaked text scores far better than the same model without the leak",
        c_chunk - l_chunk > 0.5,
        f"{c_chunk:.3f} -> {l_chunk:.3f} bits: {(c_chunk - l_chunk) / c_chunk:.0%} lower, "
        "and the model did not become 30% better at code",
    )
    report.check(
        "the rest of the held-out code stays where the control had it (no spillover at this size)",
        abs(l_rest - c_rest) < 0.1,
        f"{c_rest:.3f} -> {l_rest:.3f}; the page's larger leak (60 KB into 0.86M parameters) lifted "
        "sibling files in the same package",
    )
shutil.rmtree(tmp)
report.finish()
