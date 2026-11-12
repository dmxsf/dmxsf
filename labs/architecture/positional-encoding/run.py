"""Lab: position encodings, and what happens beyond the training length.

Course pages: https://dmxsf.com/wiki/architecture/positional-encoding/
              https://dmxsf.com/wiki/architecture/long-context/

    uv run python labs/architecture/positional-encoding/run.py [--quick] [--seed N]
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from dmxsf import lab
from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.architecture.position import (
    alibi_slopes,
    apply_rope,
    rope_cos_sin,
    rope_inv_freq,
    rope_llama3,
    rope_ntk,
    rope_scaled_inv_freq,
    rope_yarn,
    sinusoidal_table,
)
from dmxsf.data import ByteCorpus, get_batch
from dmxsf.foundations.optim import lr_factor

args, report = lab.start(
    "Position encodings and length extrapolation",
    wiki=["architecture/positional-encoding", "architecture/long-context"],
)

# ---------------------------------------------------------------------------
report.section("1. Sinusoidal: every pair of dimensions is a clock with its own period")
table = sinusoidal_table(2048, 64)
periods = [2 * math.pi * 10_000 ** (2 * i / 64) for i in range(32)]
report.say(f"  32 clocks: periods from {periods[0]:.1f} to {periods[-1]:,.0f} positions")
k, i = 7, 5  # pair i advanced by k positions is a rotation by w_i * k
w = 1 / 10_000 ** (2 * i / 64)
pair = table[:, 2 * i : 2 * i + 2].double()
rot = torch.tensor(
    [[math.cos(w * k), math.sin(w * k)], [-math.sin(w * k), math.cos(w * k)]]
).double()
report.check(
    "moving k positions rotates each pair by a fixed angle (so offsets are easy to learn)",
    torch.allclose(pair[:1000] @ rot.T, pair[k : 1000 + k], atol=1e-5),
)

# ---------------------------------------------------------------------------
report.section("2. RoPE: the score depends only on the distance")
torch.manual_seed(0)
d, t = 64, 200
q, k = torch.randn(d), torch.randn(d)
cos, sin = rope_cos_sin(torch.arange(t), rope_inv_freq(d))
qs = apply_rope(q.expand(t, d), cos, sin, interleaved=True)
ks = apply_rope(k.expand(t, d), cos, sin, interleaved=True)
scores = []
for m, n in [(3, 10), (13, 20), (103, 110), (190, 197)]:
    scores.append(float(qs[m] @ ks[n]))
    report.say(f"  m={m:3d}  n={n:3d}  n-m={n - m}  score {scores[-1]:.4f}")
report.check(
    "the same offset gives the same score at any position (page: -7.5584)",
    max(scores) - min(scores) < 1e-3 and abs(scores[0] + 7.5584) < 2e-3,
)
x = torch.randn(10, 64)
perm = torch.cat([torch.arange(0, 64, 2), torch.arange(1, 64, 2)])
cos10, sin10 = rope_cos_sin(torch.arange(10), rope_inv_freq(64))
report.check(
    "RoFormer's adjacent pairing and Llama's rotate_half differ only by a permutation",
    torch.allclose(
        apply_rope(x, cos10, sin10, interleaved=True)[:, perm],
        apply_rope(x[:, perm], cos10, sin10),
        atol=1e-5,
    ),
)
wavelengths = [2 * math.pi * 10_000 ** (2 * i / 128) for i in range(64)]
report.say(
    f"  d=128, base 10,000: wavelengths run from {wavelengths[0]:.2f} to {wavelengths[-1]:,.0f} positions"
)
report.check(
    "the fastest pair turns every 6.28 positions, the slowest every 54,410",
    round(wavelengths[0], 2) == 6.28 and round(wavelengths[-1], -1) == 54_410,
)

# ---------------------------------------------------------------------------
report.section("3. Stretching RoPE: PI, NTK-aware, YaRN and the Llama 3.1 rule")
dim, base, trained, s = 128, 10_000.0, 4096, 8.0
freqs = rope_inv_freq(dim, base)
short = [g for g, f in enumerate(freqs.tolist()) if trained * f < 2 * math.pi]
report.say(
    f"  d={dim}, base {base:,.0f}, trained on {trained}, target {int(s * trained)} (s = {s:g})"
)
report.say(
    f"  pairs that turn less than once inside the training length: {len(short)} of {dim // 2}"
    f" (pairs {short[0]} to {short[-1]})"
)
methods = {
    "RoPE (no change)": freqs,
    "PI": rope_scaled_inv_freq("linear", dim, base, s, trained)[0],
    "NTK-aware": rope_ntk(dim, base, s),
    "YaRN": rope_yarn(dim, base, s, trained)[0],
}
rows = []
for name, new in methods.items():
    new_phase = max(
        float(s * new[g] / freqs[g]) for g in short
    )  # > 1: angles never seen in training
    rows.append([name, float(new[0]), float(freqs[-1] / new[-1]), new_phase])
report.table(
    ["method", "pair 0 frequency", "slowest pair divided by", "largest unseen phase"], rows
)
report.check(
    "the course page's table: 8.00 / 1.00 / 1.75 / 1.00",
    [round(r[3], 2) for r in rows] == [8.00, 1.00, 1.75, 1.00],
)
yarn, factor = rope_yarn(dim, base, s, trained)
ratio = yarn / freqs
unchanged, divided = int((ratio > 1 - 1e-9).sum()), int((ratio < 1 / s + 1e-9).sum())
report.say(
    f"  YaRN: {unchanged} pairs unchanged, {dim // 2 - unchanged - divided} blended, {divided} divided by s;"
    f" attention factor {factor:.3f} (q.k grows {factor**2:.2f}x)"
)
l3 = rope_llama3(rope_inv_freq(128, 500_000.0), 8.0, 1.0, 4.0, 8192) / rope_inv_freq(128, 500_000.0)
l3_unchanged, l3_divided = int((l3 > 1 - 1e-9).sum()), int((l3 < 1 / 8 + 1e-9).sum())
report.say(
    f"  Llama 3.1 (base 500,000): {l3_divided} pairs divided by 8, {64 - l3_unchanged - l3_divided} blended, "
    f"{l3_unchanged} unchanged"
)
report.check(
    "segments match the course page: YaRN 21/25/18, Llama 3.1 29/6/29",
    (unchanged, 64 - unchanged - divided, divided) == (21, 25, 18)
    and (l3_divided, 64 - l3_unchanged - l3_divided, l3_unchanged) == (29, 6, 29),
)
report.say(
    f"  NTK-aware base: {base * s ** (dim / (dim - 2)):,.0f}   ALiBi slopes (8 heads): "
    f"1/2 ... 1/{round(1 / float(alibi_slopes(8)[-1]))}"
)

# ---------------------------------------------------------------------------
report.section("4. Train short, test long: five position schemes, same model otherwise")
TRAIN_T, EVAL_T = 64, 256
steps = 40 if args.quick else 1000
seeds = [args.seed] if args.quick else [args.seed, args.seed + 1]
data = ByteCorpus.shakespeare()


def val_windows(length: int, n: int = 60):
    rows = np.stack([data.val[j * length : (j + 1) * length + 1] for j in range(n)]).astype(
        np.int64
    )
    rows = torch.from_numpy(rows)
    return rows[:, :-1], rows[:, 1:]


def loss_by_position(model, length: int) -> torch.Tensor:
    x, y = val_windows(length)
    with torch.no_grad():
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1), reduction="none")
    return loss.view(x.shape[0], -1).mean(0)


def buckets(model, length: int = EVAL_T) -> list[float]:
    by_pos = loss_by_position(model, length)
    return [by_pos[:64].mean().item(), by_pos[64:128].mean().item(), by_pos[128:].mean().item()]


def train(pos: str, seed: int) -> GPT:
    torch.manual_seed(seed)
    cfg = small_gpt(
        n_layer=2,
        d_model=64,
        n_head=4,
        block_size=TRAIN_T,
        pos=pos,
        norm="rmsnorm",
        mlp="swiglu",
        bias=False,
    )
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = 3e-3 * lr_factor("cosine", step, steps, warmup=min(50, steps // 4))
        x, y = get_batch(data.train, step, 32, TRAIN_T, seed=seed)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


SCHEMES = ["learned", "sinusoidal", "rope", "alibi", "none"]
STRETCH = ["linear", "ntk", "yarn", "llama3"]
results: dict[str, list[list[float]]] = {}  # name -> one [0-63, 64-127, 128-255] per seed
for seed in seeds:
    for pos in SCHEMES:
        model = train(pos, seed)
        if pos == "learned":
            inside = loss_by_position(model, TRAIN_T).mean().item()
            results.setdefault(pos, []).append([inside, math.nan, math.nan])
            try:
                model(torch.zeros(1, TRAIN_T + 1, dtype=torch.long))
                refused = False
            except ValueError:
                refused = True
            if seed == seeds[0]:
                learned_refuses = refused
        else:
            results.setdefault(pos, []).append(buckets(model))
        if pos == "rope":
            for kind in STRETCH:
                model.set_rope_scaling(kind, EVAL_T / TRAIN_T, TRAIN_T)
                results.setdefault(f"rope + {kind} x4", []).append(buckets(model))
            model.set_rope_scaling("none")


def mean(name: str) -> list[float]:
    return (
        np.nanmean(np.array(results[name]), axis=0).tolist()
        if name != "learned"
        else results[name][0]
    )


report.say(
    f"  2 layers, d_model 64, {steps} steps on Shakespeare bytes with {TRAIN_T}-byte windows, "
    f"{len(seeds)} seed(s); loss in nats per byte"
)
report.say(
    f"  evaluated on {EVAL_T}-byte windows, split by position (position < {TRAIN_T} was seen in training)"
)
rows = [[name, *[("n/a" if math.isnan(v) else v) for v in mean(name)]] for name in results]
report.table(["scheme", "pos 0-63", "pos 64-127", "pos 128-255"], rows)
report.say("  (learned positions exist only for the first 64 positions; the four rows below RoPE")
report.say("   reuse the RoPE models, stretched 4x at test time with no further training)")

if args.quick:
    report.skip("the claims about extrapolation", "40 steps teach nothing")
else:
    report.check("learned positions cannot even run past the training length", learned_refuses)

    def per_seed(name: str, pred) -> bool:
        return all(pred(row) for row in results[name])

    report.check(
        "ALiBi is flat: far positions cost the same as near ones (within 0.1 nats)",
        per_seed("alibi", lambda r: abs(r[2] - r[0]) < 0.1),
    )
    report.check(
        "RoPE collapses beyond about twice the training length (+1 nat)",
        per_seed("rope", lambda r: r[2] > r[0] + 1.0),
    )
    report.check(
        "sinusoidal degrades beyond the training length too (+0.5 nats)",
        per_seed("sinusoidal", lambda r: r[2] > r[0] + 0.5),
    )
    report.check(
        "PI without fine-tuning damages even the positions it was trained on (+0.5 nats)",
        all(
            a[0] > b[0] + 0.5
            for a, b in zip(results["rope + linear x4"], results["rope"], strict=True)
        ),
    )
    report.check(
        "NTK-aware and YaRN stretch RoPE 4x with no training: far positions improve by 0.5+ nats",
        all(
            a[2] < b[2] - 0.5
            for kind in ("ntk", "yarn")
            for a, b in zip(results[f"rope + {kind} x4"], results["rope"], strict=True)
        ),
    )

report.finish()
