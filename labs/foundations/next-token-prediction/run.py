"""Lab: next-token prediction and information theory.

Course pages:
  https://dmxsf.com/wiki/foundations/next-token-prediction/
  https://dmxsf.com/wiki/foundations/information-theory/

    uv run python labs/foundations/next-token-prediction/run.py [--quick] [--seed N]
"""

import math

import numpy as np
import torch

from dmxsf import lab
from dmxsf.data import ByteCorpus, decode_ids
from dmxsf.foundations.information import (
    bits_per_byte,
    cross_entropy,
    entropy,
    kl_divergence,
    perplexity,
)
from dmxsf.foundations.language_model import (
    NGramLM,
    bigram_counts,
    conditional_entropy,
    fit_neural_bigram,
)

args, report = lab.start(
    "Next-token prediction and information theory",
    wiki=["foundations/next-token-prediction", "foundations/information-theory"],
)

# ---------------------------------------------------------------------------
report.section("1. A bigram model on the course page's toy corpus")
corpus = "the cat sat on the mat . the dog sat on the log .".split()
corpus += "the cat saw the dog .".split()
lm = NGramLM(corpus, n=2, k=0.1)
natural = lm.perplexity("the cat sat on the mat .".split())
shuffled = lm.perplexity("mat the on . sat cat the".split())
report.say(f"  V = {len(lm.vocab)}")
report.say(f"  P(sat | cat) = {lm.prob('sat', 'cat'):.3f}")
report.say(f"  P(mat | the) = {lm.prob('mat', 'the'):.3f}")
report.say(f"  P(cat | the) = {lm.prob('cat', 'the'):.3f}")
report.say(f"  perplexity, natural order  = {natural:.2f}")
report.say(f"  perplexity, shuffled words = {shuffled:.2f}")
report.check(
    "numbers match the course page",
    (round(natural, 2), round(shuffled, 2)) == (2.37, 31.23),
)

# ---------------------------------------------------------------------------
report.section("2. Entropy, cross-entropy and KL divergence (nats)")
p = [0.50, 0.25, 0.10, 0.10, 0.05]  # the truth
q = [0.02, 0.02, 0.02, 0.02, 0.92]  # a model that is confident and wrong
report.table(
    ["quantity", "value"],
    [
        ["H(p)", entropy(p)],
        ["H(p, q)", cross_entropy(p, q)],
        ["KL(p || q)", kl_divergence(p, q)],
        ["KL(q || p)", kl_divergence(q, p)],
        ["perplexity exp(H(p, q))", perplexity(cross_entropy(p, q))],
    ],
)
report.check(
    "H(p, q) = H(p) + KL(p || q)",
    math.isclose(cross_entropy(p, q), entropy(p) + kl_divergence(p, q)),
)
report.check("KL is not symmetric", abs(kl_divergence(p, q) - kl_divergence(q, p)) > 0.05)
one_hot, model = [1, 0, 0, 0, 0], [0.6, 0.2, 0.1, 0.05, 0.05]
report.check(
    "with a one-hot target, cross-entropy is -ln(probability of the true token)",
    math.isclose(cross_entropy(one_hot, model), -math.log(0.6)),
    f"{cross_entropy(one_hot, model):.3f} nats",
)
report.say("  Same perplexity is not same quality when tokenizers differ; bits per byte is:")
for name, loss, bytes_per_token in [("A", 2.0, 4.0), ("B", 1.0, 2.0)]:
    report.say(
        f"    model {name}: perplexity {perplexity(loss):.2f}, "
        f"bits per byte {bits_per_byte(loss, bytes_per_token):.3f}"
    )

# ---------------------------------------------------------------------------
report.section("3. Character n-grams on Shakespeare: context helps until the counts run out")
data = ByteCorpus.shakespeare()
train = data.train[:100_000] if args.quick else data.train
val = data.val[:10_000] if args.quick else data.val
train_list, val_list = train.tolist(), val.tolist()
rows = []
for n in range(1, 5 if args.quick else 7):
    ngram = NGramLM(train_list, n=n, k=0.05)
    rows.append(
        [
            n,
            len(ngram.counts),
            ngram.nll(train_list[:20_000]) / math.log(2),
            ngram.nll(val_list) / math.log(2),
        ]
    )
report.table(["n", "contexts seen", "train bits/char", "val bits/char"], rows)
val_bits = [r[3] for r in rows]
report.check(
    "each of the first orders beats the one before", val_bits[0] > val_bits[1] > val_bits[2]
)
if not args.quick:
    report.check(
        "long contexts overfit: order 6 is worse than the best order",
        val_bits[-1] > min(val_bits) + 0.2,
        f"best {min(val_bits):.2f}, order 6 {val_bits[-1]:.2f}",
    )
rng = np.random.default_rng(args.seed)
for n in (2, 5):
    ngram = NGramLM(train_list, n=n, k=0.05)
    context = train_list[1000 : 1000 + n - 1]
    text = decode_ids(context + ngram.sample(context, 100, rng))
    report.say(f"  order {n} samples: {text!r}")

# ---------------------------------------------------------------------------
report.section("4. A neural bigram, trained by gradient descent, learns the count table")
counts = bigram_counts(train, 256)
best = conditional_entropy(counts)
model, history = fit_neural_bigram(counts, steps=150 if args.quick else 300, lr=0.5)
report.say(f"  loss at step 0     : {history[0]:.4f}   (ln 256 = {math.log(256):.4f})")
report.say(f"  loss at the end    : {history[-1]:.4f}")
report.say(f"  H(next | prev)     : {best:.4f}   (the best any bigram can do)")
report.check("starts at ln V: the uniform guess", abs(history[0] - math.log(256)) < 1e-3)
report.check(
    "ends at the maximum-likelihood optimum",
    abs(history[-1] - best) < (0.02 if args.quick else 2e-3),
)
learned = torch.softmax(model.table.weight, dim=-1).detach().numpy()
for char in "q", "t":
    row = learned[ord(char)]
    top = [i for i in np.argsort(-row)[:3] if row[i] > 0.01]
    shown = ", ".join(f"{decode_ids([int(i)])!r}: {row[i]:.2f}" for i in top)
    report.say(f"  after {char!r} the model expects  {shown}")

report.finish()
