"""Lab: optimizers, learning-rate schedules, mixed precision and loss spikes.

Course page: https://dmxsf.com/wiki/foundations/optimizers-and-training-stability/

    uv run python labs/foundations/optimizers-and-training-stability/run.py [--quick] [--seed N]
"""

import math

import torch
import torch.nn.functional as F

from dmxsf import lab
from dmxsf.foundations.optim import SCHEDULES, SGD, Adam, AdamW, clip_grad_norm, lr_factor

args, report = lab.start(
    "Optimizers, schedules, mixed precision and loss spikes",
    wiki=["foundations/optimizers-and-training-stability"],
)


def regression(seed: int = 0):
    """The course page's toy problem: y = x @ w_true, fitted by Linear(16, 1)."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(16, 1)  # created first, as on the page: the data depends on the order
    x = torch.randn(256, 16)
    y = x @ torch.randn(16, 1)
    return x, y, model


# ---------------------------------------------------------------------------
report.section("1. Our optimizers against PyTorch's (float64, 100 steps)")
torch.manual_seed(args.seed)
x64, y64 = torch.randn(64, 8, dtype=torch.float64), torch.randn(64, 1, dtype=torch.float64)
pairs = {
    "SGD+momentum": (
        lambda p: SGD(p, lr=0.05, momentum=0.9, weight_decay=0.01),
        lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=0.01),
    ),
    "Adam": (
        lambda p: Adam(p, lr=0.01, weight_decay=0.01),
        lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.01),
    ),
    "AdamW": (
        lambda p: AdamW(p, lr=0.01, weight_decay=0.1),
        lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1),
    ),
}
rows = []
for name, (ours_cls, ref_cls) in pairs.items():
    models = []
    for make in (ours_cls, ref_cls):
        torch.manual_seed(1)
        model = torch.nn.Linear(8, 1).double()
        opt = make(model.parameters())
        for _ in range(100):
            loss = F.mse_loss(model(x64), y64)
            opt.zero_grad()
            loss.backward()
            opt.step()
        models.append(model)
    diff = max(
        (a - b).abs().max().item()
        for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True)
    )
    rows.append([name, diff])
report.table(["optimizer", "max |ours - torch|"], [[n, f"{d:.1e}"] for n, d in rows])
report.check("all three agree with torch.optim to 1e-10", all(d < 1e-10 for _, d in rows))

# ---------------------------------------------------------------------------
report.section("2. Why Adam: an ill-conditioned quadratic f = (100 x^2 + y^2) / 2")


def steps_to_converge(make_opt, limit=3000, target=1e-3):
    w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    opt = make_opt([w])
    for step in range(limit):
        loss = 0.5 * (100 * w[0] ** 2 + w[1] ** 2)
        if loss.item() < target:
            return step
        opt.zero_grad()
        loss.backward()
        opt.step()
    return None


rows = [
    [
        "SGD, lr 0.019 (the largest stable one is 2/100)",
        steps_to_converge(lambda p: SGD(p, lr=0.019)),
    ],
    ["SGD + momentum 0.9, lr 0.019", steps_to_converge(lambda p: SGD(p, lr=0.019, momentum=0.9))],
    ["Adam, lr 0.05", steps_to_converge(lambda p: Adam(p, lr=0.05))],
]
report.table(
    ["optimizer", "steps until f < 1e-3"],
    [[name, "never" if steps is None else steps] for name, steps in rows],
)
report.say(
    "  The curvature differs 100x between the axes: one learning rate cannot suit both for SGD,"
)
report.say("  while Adam divides each coordinate by its own gradient scale.")
needed = [math.inf if steps is None else steps for _, steps in rows]  # "never" is the slowest
report.check("Adam gets there first", needed[2] < needed[1] < needed[0])

# ---------------------------------------------------------------------------
report.section("3. AdamW decouples weight decay; Adam + L2 does not")
result = {}
for name, cls in [("AdamW", AdamW), ("Adam + L2", Adam)]:
    w = torch.nn.Parameter(torch.ones(2))
    opt = cls([w], lr=1e-2, weight_decay=0.1)
    gen = torch.Generator().manual_seed(0)
    for _ in range(300):
        opt.zero_grad()
        # the same noise for both weights, one of them 1000x larger
        w.grad = torch.tensor([0.01, 10.0]) * torch.randn(1, generator=gen)
        opt.step()
    result[name] = w.detach().tolist()
    report.say(
        f"  {name:10s} weights after 300 steps (both started at 1): "
        f"{result[name][0]:+.4f} (gradient x0.01)  {result[name][1]:+.4f} (gradient x10)"
    )
report.check(
    "AdamW gives the two weights the same trajectory (scale-invariant)",
    abs(result["AdamW"][0] - result["AdamW"][1]) < 1e-5,
)
report.check(
    "Adam + L2 shrinks the small-gradient weight much more than the large-gradient one",
    result["Adam + L2"][1] - result["Adam + L2"][0] > 0.5,
)

# ---------------------------------------------------------------------------
report.section("4. Learning-rate schedules")
total, warmup = 20_000, 1_000
rows = []
for name in SCHEDULES:
    factors = [lr_factor(name, s, total, warmup) for s in range(total)]
    at = lambda frac, f=factors: f[int(frac * total) - 1]  # noqa: E731
    rows.append([name, sum(factors) / total, at(0.25), at(0.5), at(0.75), factors[-1]])
report.table(["schedule", "mean", "at 25%", "at 50%", "at 75%", "last step"], rows)
report.say(
    f"  {total:,} steps, {warmup:,} warmup steps, floor 10% of the peak, WSD decays over the last 20%"
)
means = {r[0]: r[1] for r in rows}
report.check(
    "course page: cosine and linear average 0.548, WSD 0.885 (about 1.6x cosine)",
    round(means["cosine"], 3) == round(means["linear"], 3) == 0.548
    and round(means["wsd"], 3) == 0.885,
)
report.check(
    "course page: inverse square root is 0.447 / 0.316 / 0.258 and ends at 0.224",
    [round(v, 3) for v in rows[3][2:5]] == [0.447, 0.316, 0.258] and round(rows[3][5], 3) == 0.224,
)

# ---------------------------------------------------------------------------
report.section("5. The course page's training loop: AdamW + warmup/cosine + clipping")
steps, warm, floor = 1000, 100, 0.1


def page_lr_scale(step: int) -> float:
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / (steps - 1 - warm)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def train(ours: bool):
    x, y, model = regression(0)
    if ours:
        opt = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, page_lr_scale)
    log = {}
    for step in range(steps):
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        if ours:
            grad_norm = clip_grad_norm(model.parameters(), 1.0)
            for group in opt.param_groups:
                group["lr"] = 3e-4 * page_lr_scale(step)
        else:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        lr = opt.param_groups[0]["lr"]
        opt.step()
        if not ours:
            sched.step()
        if step in (0, 99, 100, 500, 999):
            log[step] = (lr, grad_norm, loss.item())
    return log, model


ours_log, ours_model = train(True)
ref_log, ref_model = train(False)
for step, (lr, grad_norm, loss) in ours_log.items():
    report.say(f"  step {step:3d}  lr {lr:.2e}  grad_norm {grad_norm:.3f}  loss {loss:.4f}")
page = {
    0: ("3.00e-06", "7.770", "16.4080"),
    99: ("3.00e-04", "7.666", "16.0073"),
    100: ("3.00e-04", "7.664", "15.9993"),
    500: ("1.88e-04", "6.959", "13.4174"),
    999: ("3.00e-05", "6.674", "12.4340"),
}
ours_text = {s: (f"{lr:.2e}", f"{g:.3f}", f"{loss:.4f}") for s, (lr, g, loss) in ours_log.items()}
report.check("same five lines as the course page", ours_text == page)
gap = max(
    (a - b).abs().max().item()
    for a, b in zip(ours_model.parameters(), ref_model.parameters(), strict=True)
)
report.check(
    "our loop and torch's end with the same weights", gap < 1e-5, f"max difference {gap:.1e}"
)

# ---------------------------------------------------------------------------
report.section("6. Mixed precision: why the master weights stay in FP32")
rows = []
for name, dtype in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
    info = torch.finfo(dtype)
    rows.append([name, f"{info.max:.3g}", f"{info.smallest_normal:.2g}", f"{info.eps:.2g}"])
report.table(["format", "max", "smallest normal", "machine eps"], rows)
report.check(
    "values on the course page",
    [r[1:] for r in rows]
    == [
        ["3.4e+38", "1.2e-38", "1.2e-07"],
        ["6.55e+04", "6.1e-05", "0.00098"],
        ["3.39e+38", "1.2e-38", "0.0078"],
    ],
)
tiny = 1e-8
scaled = (torch.tensor(tiny * 65536).half().float() / 65536).item()
report.say(
    f"  a gradient of {tiny:g}: FP16 stores {torch.tensor(tiny).half().item():g}; "
    f"with loss scaling x65536 it survives as {scaled:.4g}"
)
report.check("FP16 underflows to 0 without loss scaling", torch.tensor(tiny).half().item() == 0.0)
report.check("loss scaling brings it back (to 3 digits)", abs(scaled / tiny - 1) < 1e-2)

report.say("")
report.say("  Same regression, same AdamW (lr 1e-3), three ways to hold the weights:")
n_steps = 1500 if args.quick else 3000


def fit(variant: str) -> float:
    x, y, model = regression(0)
    if variant == "fp32":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        for _ in range(n_steps):
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()
    master = [torch.nn.Parameter(p.detach().clone()) for p in model.parameters()]
    model = model.bfloat16()
    xb, yb = x.bfloat16(), y.bfloat16()
    if variant == "bf16":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(master, lr=1e-3, weight_decay=0.0)
    for _ in range(n_steps):
        loss = F.mse_loss(model(xb), yb)
        model.zero_grad()
        loss.backward()
        if variant == "bf16+master":
            for m, p in zip(master, model.parameters(), strict=True):
                m.grad = p.grad.float()
        opt.step()
        if variant == "bf16+master":
            with torch.no_grad():
                for m, p in zip(master, model.parameters(), strict=True):
                    p.copy_(m)
    return loss.float().item()


x0, y0, m0 = regression(0)
losses = {v: fit(v) for v in ("fp32", "bf16", "bf16+master")}
report.say(f"  loss at step 0: {F.mse_loss(m0(x0), y0).item():.2f}")
report.table(
    ["weights held in", "loss after training"],
    [
        ["FP32", losses["fp32"]],
        ["BF16 only (the update is smaller than half a BF16 step: rounded away)", losses["bf16"]],
        ["BF16 copy for compute + FP32 master weights", losses["bf16+master"]],
    ],
)
report.check(
    "pure BF16 stalls far above FP32",
    losses["bf16"] > 3 * losses["fp32"],
    f"{losses['bf16'] / losses['fp32']:,.0f}x the FP32 loss"
    + ("; the shorter --quick run has not converged in FP32 either" if args.quick else ""),
)
report.check(
    "BF16 compute with FP32 master weights matches FP32",
    abs(losses["bf16+master"] / losses["fp32"] - 1) < 0.05,
)
per_param = 2 + 2 + 4 + 4 + 4
report.say(
    f"  model state per parameter: bf16 weight 2 + bf16 grad 2 + fp32 master 4 + m 4 + v 4 = {per_param} bytes"
)
report.say(
    f"  a 7B model: {per_param * 7e9 / 1e9:.0f} GB ({per_param * 7e9 / 2**30:.0f} GiB) before activations"
)
report.check("16 bytes per parameter, 112 GB for 7B", per_param == 16 and per_param * 7 == 112)

# ---------------------------------------------------------------------------
report.section("7. One corrupted batch: what clipping and beta2 do")
total_steps, spike_at = (500, 300) if args.quick else (700, 400)


def run_spike(clip, beta2: float, spike: bool):
    x, y, model = regression(0)
    opt = Adam(model.parameters(), lr=1e-2, betas=(0.9, beta2))
    losses, norm_at_spike = [], 0.0
    for step in range(total_steps):
        target = y * 1000 if (spike and step == spike_at) else y  # the bad batch: labels x1000
        loss = F.mse_loss(model(x), target)
        opt.zero_grad()
        loss.backward()
        norm = clip_grad_norm(model.parameters(), clip or 1e9)
        if step == spike_at:
            norm_at_spike = norm
        opt.step()
        with torch.no_grad():
            losses.append(F.mse_loss(model(x), y).item())  # loss on the clean data
    return losses, norm_at_spike


rows, outcome = [], {}
for clip, beta2 in [(None, 0.999), (1.0, 0.999), (None, 0.95), (1.0, 0.95)]:
    hit, norm = run_spike(clip, beta2, True)
    ctl, _ = run_spike(clip, beta2, False)
    outcome[clip, beta2] = (hit, ctl)
    rows.append(
        [
            f"clip {clip or 'off'}, beta2 {beta2}",
            f"{max(hit[spike_at:]):.1e}",
            f"{hit[spike_at + 50]:.1e}",
            f"{ctl[spike_at + 50]:.1e}",
            f"{hit[-1]:.1e}",
            f"{ctl[-1]:.1e}",
        ]
    )
report.say(f"  one batch at step {spike_at} has labels x1000: its gradient norm is {norm:,.0f}")
report.say("  (loss on clean data; 'control' = the same run without the bad batch)")
report.table(
    ["setting", "peak", "+50 steps", "control", f"step {total_steps - 1}", "control"], rows
)
unclipped = outcome[None, 0.999]
clipped = outcome[1.0, 0.999]
report.check(
    "without clipping the bad batch is not forgotten: 1000x the control loss 250 steps on",
    unclipped[0][-1] > 1000 * unclipped[1][-1],
)
report.check(
    "clipping to norm 1 keeps the peak below 1% of the unclipped peak",
    max(clipped[0][spike_at:]) < 0.01 * max(unclipped[0][spike_at:]),
)
report.check(
    "beta2 = 0.95 forgets the spike sooner than 0.999 (no clipping)",
    outcome[None, 0.95][0][-1] < unclipped[0][-1],
)

report.finish()
