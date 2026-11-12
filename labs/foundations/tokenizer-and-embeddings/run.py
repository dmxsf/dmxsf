"""Lab: tokenizers and embeddings.

Course page: https://dmxsf.com/wiki/foundations/tokenizer-and-embeddings/

    uv run python labs/foundations/tokenizer-and-embeddings/run.py [--quick] [--seed N]
"""

import itertools
import time

import torch
import torch.nn.functional as F
from parallel_text import HELD_OUT, TRAIN

from dmxsf import lab
from dmxsf.architecture.configs import PUBLIC_MODELS, small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.data import load_shakespeare
from dmxsf.foundations.tokenizer import BPETokenizer, learn_merges, merge_pair, pretokenize

args, report = lab.start("Tokenizers and embeddings", wiki=["foundations/tokenizer-and-embeddings"])

# ---------------------------------------------------------------------------
report.section("1. BPE by hand: the course page's toy corpus")
corpus = {"▁low": 5, "▁lower": 2, "▁newest": 6, "▁widest": 3}
chars = sorted({c for word in corpus for c in word})
ident = {c: i for i, c in enumerate(chars)}
words = {tuple(ident[c] for c in w): n for w, n in corpus.items()}
names = {i: c for c, i in ident.items()}
pair_counts: list[int] = []
merges = learn_merges(words, 10, first_id=len(ident), counts=pair_counts)
learned = []
for step, ((a, b), count) in enumerate(zip(merges, pair_counts, strict=True), start=1):
    names[len(ident) + step - 1] = names[a] + names[b]
    learned.append((names[a], names[b]))
    report.say(f"  {step:2d}  {names[a]} + {names[b]} -> {names[a] + names[b]}  ({count} times)")


def encode_toy(word: str) -> list[str]:
    symbols = [ident[c] for c in word]
    for i, pair in enumerate(merges):
        symbols = merge_pair(symbols, pair, len(ident) + i)
    return [names[s] for s in symbols]


report.say(f"  {encode_toy('▁lowest')} {encode_toy('▁newer')} {encode_toy('▁wider')}")
report.check(
    "same merge table and segmentations as the course page",
    learned[:3] == [("e", "s"), ("es", "t"), ("▁", "l")]
    and encode_toy("▁lowest") == ["▁low", "est"]
    and encode_toy("▁newer") == ["▁new", "e", "r"],
)

# ---------------------------------------------------------------------------
report.section("2. Byte-level BPE: pre-tokenization and a tokenizer that never says <unk>")
sample = "Hello, world! It's 12345678."
report.say(f"  chunks: {pretokenize(sample)}")
shakespeare = load_shakespeare().decode()
train_text = shakespeare[: (60_000 if args.quick else 300_000)]
held_text = shakespeare[600_000:620_000]
max_vocab = 384 if args.quick else 1024
t0 = time.perf_counter()
tok = BPETokenizer.train(train_text, max_vocab)
report.say(
    f"  trained {tok.vocab_size - 256} merges on {len(train_text):,} characters "
    f"in {time.perf_counter() - t0:.1f}s"
)
first = [tok.vocab[256 + i].decode("utf-8", "replace") for i in range(8)]
longest = sorted({t for t in tok.vocab.values() if len(t) > 1}, key=len, reverse=True)[:5]
report.say(f"  first merges: {first}")
report.say(f"  longest tokens: {[t.decode('utf-8', 'replace') for t in longest]}")
tricky = [
    "Hello, world!",
    " Hello",
    "naïve café",
    "你好，世界",
    "emoji 🙂🚀",
    "tabs\tand\nnewlines  ",
    "",
]
report.check(
    "encode -> decode is lossless for any string, including emoji and CJK",
    all(tok.decode(tok.encode(t)) == t for t in tricky),
)
report.say(f"  'Hello' -> {tok.tokens('Hello')}     ' Hello' -> {tok.tokens(' Hello')}")
report.say(f"  '你好' -> {len(tok.encode('你好'))} tokens for {len('你好'.encode())} bytes")

report.say("")
report.say(
    "  A bigger vocabulary means shorter sequences (held-out Shakespeare, 20,000 characters):"
)
rows = []
for size in [256, 320, 384, 512, 768, 1024]:
    if size > max_vocab:
        continue
    prefix = BPETokenizer(tok.merges[: size - 256])  # BPE merges are prefix-consistent
    n_tokens = len(prefix.encode(held_text))
    n_bytes = len(held_text.encode())
    rows.append([size, n_tokens, n_bytes / n_tokens, (n_tokens / n_bytes) ** 2, size * 128])
report.table(
    ["vocab", "tokens", "bytes/token", "attention cost (x bytes)", "embedding params (d=128)"], rows
)
report.check(
    "each larger vocabulary compresses better",
    all(a[2] < b[2] for a, b in itertools.pairwise(rows)),
)
report.check(
    "a byte-level vocabulary of 256 is one token per byte", rows[0][1] == len(held_text.encode())
)

# ---------------------------------------------------------------------------
report.section("3. What the tokenizer saw decides what a language costs")
en_train = " ".join(en for _, en in TRAIN)
zh_train = " ".join(zh for zh, _ in TRAIN)
zh_test = " ".join(zh for zh, _ in HELD_OUT)
en_test = " ".join(en for _, en in HELD_OUT)
report.say(
    f"  held-out text: {len(zh_test)} Chinese characters ({len(zh_test.encode())} bytes) and "
    f"{len(en_test.split())} English words ({len(en_test.encode())} bytes)"
)
report.say(
    "  E = trained on English only, E+Z = English and Chinese, Z = Chinese only; "
    "same vocabulary size within a row"
)
rows, zh_cost, en_cost = [], {}, {}
for size in (320, 384, 512):
    for name, text in [("E", en_train), ("E+Z", en_train + " " + zh_train), ("Z", zh_train)]:
        tk = BPETokenizer.train(text, size)
        zh_cost[size, name] = len(tk.encode(zh_test)) / len(zh_test)
        en_cost[size, name] = len(en_test.encode()) / len(tk.encode(en_test))
    rows.append(
        [size]
        + [zh_cost[size, n] for n in ("E", "E+Z", "Z")]
        + [en_cost[size, n] for n in ("E", "E+Z", "Z")]
    )
report.table(["vocab", "zh tok/char E", "E+Z", "Z", "en bytes/tok E", "E+Z", "Z"], rows)
report.check(
    "an English-only tokenizer pays about 2.75 tokens per Chinese character at any vocab size",
    all(abs(zh_cost[size, "E"] - 2.75) < 0.05 for size in (320, 384, 512)),
)
report.check(
    "putting Chinese in the training data makes Chinese cheaper ...",
    all(zh_cost[size, "E+Z"] < zh_cost[size, "E"] - 0.3 for size in (320, 384, 512)),
)
report.check(
    "... and English more expensive: the vocabulary is a shared budget",
    all(en_cost[size, "E+Z"] < en_cost[size, "E"] for size in (320, 384, 512)),
)
report.say("  (24 sentence pairs written for this lab; the direction matters, not the exact ratio.")
report.say(
    "   On the course page GPT-2 needs 50 tokens for a 24-character sentence, Qwen2.5 only 16.)"
)

# ---------------------------------------------------------------------------
report.section("4. From token id to vector: an embedding table is a matrix multiply")
torch.manual_seed(args.seed)
vocab, dim = 1000, 64
table = torch.nn.Embedding(vocab, dim)
ids = torch.tensor([[5, 17, 17, 999], [0, 1, 2, 3]])
looked_up = table(ids)
one_hot = F.one_hot(ids, vocab).float() @ table.weight
report.check("lookup == one-hot @ weight matrix", torch.equal(looked_up, one_hot))
loss = looked_up.sum()
loss.backward()
touched = (table.weight.grad.abs().sum(1) > 0).sum().item()
report.say(
    f"  one backward pass touched {touched} of {vocab} rows: only tokens that appeared are trained"
)
report.check("only the rows of the tokens that appeared get gradient", touched == 7)

# ---------------------------------------------------------------------------
report.section("5. How much of a model is its embedding table?")
rows = []
for model in PUBLIC_MODELS:
    if model.name not in ("GPT-2 small", "Qwen2.5-0.5B", "Llama 3 8B"):
        continue
    with torch.device("meta"):
        net = GPT(model.config)
    copies = 1 if model.config.tie_embeddings else 2
    emb = net.tok_emb.weight.numel()
    rows.append([model.name, net.num_params(), emb, copies, copies * emb / net.num_params()])
with torch.device("meta"):
    ours = GPT(small_gpt())
rows.append(["our byte-level GPT", ours.num_params(), 256 * 128, 1, 256 * 128 / ours.num_params()])
report.table(
    ["model", "total params", "table V x d", "copies", "share"],
    [[r[0], f"{r[1]:,}", f"{r[2]:,}", r[3], f"{r[4]:.1%}"] for r in rows],
)
shares = [f"{r[4]:.1%}" for r in rows[:3]]
report.check("matches the course page: 31.0%, 27.6%, 13.1%", shares == ["31.0%", "27.6%", "13.1%"])

report.finish()
