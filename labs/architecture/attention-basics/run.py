"""Lab: attention, by hand and in a model that has to use it.

Course page: https://dmxsf.com/wiki/architecture/attention-basics/

    uv run python labs/architecture/attention-basics/run.py [--quick]
"""

import copy
import math

import torch
import torch.nn.functional as F

from dmxsf import lab
from dmxsf.architecture.attention import Attention, attention, causal_mask
from dmxsf.architecture.gpt import GPT, GPTConfig
from dmxsf.utils import count_params

args, report = lab.start(
    "Attention, by hand and in a model", wiki=["architecture/attention-basics"]
)

# ---------------------------------------------------------------------------
report.section("1. The course page's three-position example")
q = torch.tensor([[1.0, 0.0]])
k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
v = torch.tensor([[10.0, 0.0], [0.0, 10.0], [6.0, 6.0]])
out, weights = attention(q, k, v)
report.say(f"  scores / sqrt(2) = {[round(s, 2) for s in ((q @ k.T) / math.sqrt(2))[0].tolist()]}")
report.say(f"  weights          = {[round(w, 2) for w in weights[0].tolist()]}")
report.say(f"  output           = {[round(o, 1) for o in out[0].tolist()]}")
report.check(
    "weights [0.40, 0.20, 0.40] and output [6.4, 4.4], as on the page",
    [round(w, 2) for w in weights[0].tolist()] == [0.40, 0.20, 0.40]
    and [round(o, 1) for o in out[0].tolist()] == [6.4, 4.4],
)

# ---------------------------------------------------------------------------
report.section("2. Why divide by sqrt(d): the dot product has variance d")
gen = torch.Generator().manual_seed(args.seed)
rows = []
for d in (4, 16, 64, 256, 1024):
    qs = torch.randn(4000, 1, d, generator=gen)
    ks = torch.randn(4000, 8, d, generator=gen)
    raw = (qs * ks).sum(-1)  # 4000 queries, 8 keys each
    row = [d, raw.var().item(), (raw / math.sqrt(d)).var().item()]
    for scores in (raw, raw / math.sqrt(d)):
        row.append(scores.softmax(-1).max(-1).values.mean().item())
    rows.append(row)
report.table(["d", "var(q.k)", "var after scaling", "max weight, raw", "max weight, scaled"], rows)
report.check("var(q.k) is about d", all(abs(r[1] / r[0] - 1) < 0.1 for r in rows))
report.check("scaling brings the variance back to about 1", all(abs(r[2] - 1) < 0.1 for r in rows))
report.check(
    "without scaling the softmax saturates (largest weight -> 1); with it nothing changes with d",
    rows[-1][3] > 0.9 and abs(rows[-1][4] - rows[0][4]) < 0.15,
)

# ---------------------------------------------------------------------------
report.section("3. Masks, heads and positions")
torch.manual_seed(args.seed)
x = torch.randn(5, 8)
_, w = attention(x, x, x, mask=causal_mask(5, 5))
report.say("  causal attention weights of a 5-token sequence (row = the token that is looking):")
for row in w:
    report.say("    " + " ".join(f"{p:.2f}" for p in row.tolist()))
report.check("a token never attends to the future", bool((w.triu(1) == 0).all()))
params = {h: count_params(Attention(64, h)) for h in (1, 4, 8)}
report.check(
    "parameters: 4 d^2 whatever the number of heads (d = 64: 16,384)",
    set(params.values()) == {4 * 64 * 64},
    f"1, 4 and 8 heads: {sorted(set(params.values()))}",
)
perm = torch.randperm(5)
out, _ = attention(x, x, x)
out_perm, _ = attention(x[perm], x[perm], x[perm])
report.check(
    "shuffling the tokens only shuffles the outputs: attention alone cannot see order",
    torch.allclose(out[perm], out_perm, atol=1e-6),
    f"max difference {(out[perm] - out_perm).abs().max().item():.1e}",
)

# ---------------------------------------------------------------------------
report.section("4. A model that has to use attention: copy what came after the last time")
VOCAB, T, M = 128, 40, 10  # random tokens; a block of M of them appears twice
steps = 120 if args.quick else 2000
batch_size = 64


def make_batch(n: int, gen: torch.Generator):
    """Random tokens, with the block seq[s1:s1+M] repeated at s2 > s1 (positions differ per row)."""
    seq = torch.randint(0, VOCAB, (n, T), generator=gen)
    s1 = torch.randint(0, T - 2 * M + 1, (n,), generator=gen)
    room = T - M - (s1 + M) + 1
    s2 = s1 + M + (torch.rand(n, generator=gen) * room).long()
    offset = torch.arange(M)
    seq.scatter_(1, s2[:, None] + offset, seq.gather(1, s1[:, None] + offset))
    return seq, s1, s2


def per_token_loss(model, seq):
    logits = model(seq[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1), reduction="none"
    ).view(seq.shape[0], -1)


def split_loss(loss, s2):
    """Loss on the predictable tokens (the repeat, after its first token) and on all the others."""
    j = torch.arange(T - 1)[None, :]
    predictable = (j >= s2[:, None]) & (j <= s2[:, None] + M - 2)
    return loss[predictable].mean().item(), loss[~predictable].mean().item()


torch.manual_seed(args.seed)
cfg = GPTConfig(
    vocab_size=VOCAB,
    block_size=T,
    n_layer=2,
    n_head=4,
    d_model=64,
    pos="rope",
    norm="rmsnorm",
    mlp="swiglu",
    bias=False,
)
model = GPT(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
train_gen = torch.Generator().manual_seed(100 + args.seed)
eval_seq, eval_s1, eval_s2 = make_batch(256, torch.Generator().manual_seed(99))
report.say(
    f"  2 layers x 4 heads, d_model 64, {model.num_params():,} parameters; "
    f"vocabulary {VOCAB}, ln(vocab) = {math.log(VOCAB):.3f}"
)
history = []
log_every = steps // 40
for step in range(1, steps + 1):
    seq, _, _ = make_batch(batch_size, train_gen)
    loss = per_token_loss(model, seq).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % log_every == 0:
        with torch.no_grad():
            history.append((step, *split_loss(per_token_loss(model, eval_seq), eval_s2)))
report.table(["step", "loss on the repeat", "loss elsewhere"], history[3::4])
final_copy, final_other = history[-1][1], history[-1][2]
# the biggest fall over two consecutive evaluations (a tenth of the run)
falls = [
    (history[i - 2][1] - history[i][1], history[i - 2][0], history[i][0])
    for i in range(2, len(history))
]
fall, fall_from, fall_to = max(falls)
report.say(
    f"  the repeat's loss fell by {fall:.2f} nats between step {fall_from} and step {fall_to}"
)
if args.quick:
    report.skip("loss falls abruptly", "needs about 1,000 steps")
else:
    report.check("it starts at chance", history[0][1] > math.log(VOCAB) - 0.3)
    report.check("then falls abruptly, by more than 1.5 nats in a twentieth of the run", fall > 1.5)
    report.check(
        "the repeat is predicted far better than chance", final_copy < 1.0, f"{final_copy:.2f} nats"
    )
    report.check(
        "everything else stays at chance (it is random)", abs(final_other - math.log(VOCAB)) < 0.5
    )

# What did the heads learn to look at?
maps = model.attention_maps(eval_seq[:, :-1])
b = torch.arange(256)[:, None]
j = torch.arange(M - 1)[None, :]
query_pos = eval_s2[:, None] + j  # inside the repeat
prev_key = query_pos - 1
induction_key = eval_s1[:, None] + j + 1  # the token after the earlier copy of the current token
rows, prev_score, ind_score = [], {}, {}
for layer, w in enumerate(maps):
    prev = w[b, :, query_pos, prev_key].mean(dim=(0, 1))
    ind = w[b, :, query_pos, induction_key].mean(dim=(0, 1))
    for h in range(cfg.n_head):
        prev_score[layer, h], ind_score[layer, h] = prev[h].item(), ind[h].item()
        rows.append([f"layer {layer} head {h}", prev[h].item(), ind[h].item()])
report.say("")
report.say("  attention paid by the tokens inside the repeat, averaged over 256 sequences:")
report.table(["head", "to the previous token", "to the token after the earlier copy"], rows)
if args.quick:
    report.skip(
        "a previous-token head in layer 0 and induction heads in layer 1", "needs the full run"
    )
else:
    report.check(
        "layer 0 has a previous-token head (> 0.3)",
        max(prev_score[0, h] for h in range(cfg.n_head)) > 0.3,
    )
    report.check(
        "layer 1 has induction heads (> 0.7)", max(ind_score[1, h] for h in range(cfg.n_head)) > 0.7
    )

# Switch one head off at a time and see what the repeat costs.
base_copy = split_loss(per_token_loss(model, eval_seq), eval_s2)[0]
ablations = []
for layer in range(cfg.n_layer):
    for h in range(cfg.n_head):
        broken = copy.deepcopy(model)
        hd = cfg.dim_per_head
        with torch.no_grad():
            broken.blocks[layer].attn.o_proj.weight[:, h * hd : (h + 1) * hd] = 0
            ablations.append(
                [
                    f"layer {layer} head {h}",
                    split_loss(per_token_loss(broken, eval_seq), eval_s2)[0],
                ]
            )
report.say("")
report.say(f"  loss on the repeat when one head is switched off (all heads on: {base_copy:.3f}):")
report.table(["head switched off", "loss on the repeat"], ablations)
if args.quick:
    report.skip("switching off the circuit's heads hurts", "needs the full run")
else:
    worst = max(a[1] for a in ablations)
    report.check(
        "switching off the right head costs at least 1 nat",
        worst > base_copy + 1.0,
        f"worst {worst:.2f}",
    )

report.finish()
