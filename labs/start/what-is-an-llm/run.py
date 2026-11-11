"""Lab: What is a large language model? The whole loop, small enough to read.

Course pages: https://dmxsf.com/wiki/start/what-is-an-llm/ and
              https://dmxsf.com/wiki/start/learning-roadmap/

A tour of the page's pipeline on a model small enough to open: a 4-layer, 842,496-parameter GPT that has
read TinyShakespeare as bytes (trained once on your machine, a few minutes, then cached). Predict, append,
repeat; text becomes tokens; ids become vectors and the last vector becomes a probability distribution; the
training objective is the chain rule; the three numbers people ask about (parameters, tokens, context).
The last section is the code course's own map: which lab belongs to which stage of the pipeline, and the
three study routes of the learning roadmap with the labs to run each week.

    uv run python labs/start/what-is-an-llm/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dmxsf import course, lab, models
from dmxsf.data import ByteCorpus, decode_ids, encode_text
from dmxsf.foundations.information import entropy, softmax
from dmxsf.foundations.tokenizer import BPETokenizer
from dmxsf.inference.generate import generate

PROMPT = "ROMEO:\nWhat is "


def main() -> None:
    args, report = lab.start(
        "What is an LLM? The whole loop, small enough to read",
        wiki=["start/what-is-an-llm", "start/learning-roadmap"],
    )
    quick = args.quick
    root = course.repo_root()

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if quick:
            report.skip(name, "--quick uses a barely trained model")
        else:
            report.check(name, ok, detail)

    model = models.pretrained("base", quick=quick, seed=args.seed)
    model.eval()
    corpus = ByteCorpus.shakespeare()
    n_params = model.num_params()
    steps, batch = models.SPECS["base"][1:]
    steps = 60 if quick else steps
    report.say(
        f"  model: {model.cfg.n_layer} layers, width {model.cfg.d_model}, {model.cfg.n_head} heads, "
        f"{n_params:,} parameters, context {models.BLOCK} bytes; read {steps * batch * models.BLOCK:,} "
        "bytes of Shakespeare"
    )

    # ------------------------------------------------------------------------------------------------
    report.section("1. The loop: read everything so far, predict one token, append, repeat")
    ids = torch.tensor([encode_text(PROMPT)])
    n_new = 60

    def loop(prompt: torch.Tensor, steps: int) -> tuple[torch.Tensor, int]:
        """The page's loop, written out. No cache: every round feeds the model the whole context."""
        out, read = prompt, 0
        for _ in range(steps):
            logits = model(out)[0, -1]  # a score for every byte, at the last position
            read += out.shape[1]
            out = torch.cat([out, logits.argmax().view(1, 1)], dim=1)
        return out, read

    with torch.no_grad():
        by_hand, read = loop(ids, n_new)
        cached = torch.cat([ids, generate(model, ids, n_new, temperature=0.0)], dim=1)
    report.say(f"  {decode_ids(by_hand[0])!r}")
    p = ids.shape[1]
    report.check(
        "the hand-written loop and the library's KV-cached `generate` produce the same 60 bytes",
        torch.equal(by_hand, cached),
    )
    report.check(
        "round i feeds the model the prompt plus everything generated: the bytes read add up to "
        f"{p} x {n_new} + {n_new} x {n_new - 1} / 2",
        read == p * n_new + n_new * (n_new - 1) // 2,
        f"{read:,} bytes",
    )
    rows = [
        [
            n,
            f"{p * n + n * (n - 1) // 2:,}",
            f"{p + n - 1:,}",
            f"{(p * n + n * (n - 1) // 2) / (p + n - 1):.1f}x",
        ]
        for n in (16, 32, 64, 128)
    ]
    report.say(f"\n  Bytes the model must read to write n bytes after a {p}-byte prompt:")
    report.table(["n", "re-read every round", "with a KV cache", "ratio"], rows)
    report.say(
        "  (A KV cache keeps the keys and values of what was already read; see labs/inference/kv-cache.)"
    )

    # ------------------------------------------------------------------------------------------------
    report.section("2. Stop 1: text becomes tokens")
    size, vocab = (40_000, 400) if quick else (100_000, 600)
    code = ByteCorpus.python_code()

    def as_text(data: np.ndarray, n: int) -> str:
        return data[:n].tobytes().decode("utf-8", errors="ignore")

    tokenizers = {
        "bytes only": None,
        "BPE on Shakespeare": BPETokenizer.train(as_text(corpus.train, size), vocab),
        "BPE on Python code": BPETokenizer.train(as_text(code.train, size), vocab),
    }

    def tokens_of(tok: BPETokenizer | None, text: str) -> list[int]:
        return list(text.encode()) if tok is None else tok.encode(text)

    bpes = list(tokenizers.values())[1:]
    sentences = [
        "unbelievably good",
        "To be, or not to be",
        "def main():",
        "Hello, world",
        "你好，世界",
    ]
    counts = {s: [len(s.encode()), *(len(tokens_of(tok, s)) for tok in bpes)] for s in sentences}
    report.say(
        f"  Tokens needed by two tokenizers (vocabulary {vocab}, trained on {size:,} bytes each):"
    )
    report.table(
        ["text", "bytes", "BPE Shakespeare", "BPE code"], [[repr(s), *c] for s, c in counts.items()]
    )
    shown = "unbelievably good"
    report.say(f"\n  {shown!r}, piece by piece:")
    for name, tok in tokenizers.items():
        pieces = [bytes([b]) for b in shown.encode()] if tok is None else tok.tokens(shown)
        report.say(f"    {name:<20} {' | '.join(x.decode(errors='replace') for x in pieces)}")
        report.say(f"    {'':<20} ids {tokens_of(tok, shown)}")
    ids_of = [tokens_of(tok, shown) for tok in tokenizers.values()]
    report.check(
        "ids belong to a tokenizer: the same text becomes different ids under each of the three",
        len({tuple(i) for i in ids_of}) == 3,
    )
    held = {
        "Shakespeare": as_text(corpus.val, 20_000),
        "Python code": as_text(code.val, 20_000),
    }
    rows = [
        [name, *(f"{len(t.encode()) / len(tok.encode(t)):.2f}" for t in held.values())]
        for name, tok in list(tokenizers.items())[1:]
    ]
    report.say(
        "\n  Bytes per token on held-out text (more is better: fewer tokens for the same text):"
    )
    report.table(["tokenizer", *held], rows)
    report.check(
        "a tokenizer is best on the text it was trained on: each BPE beats the other one on its own domain",
        float(rows[0][1]) > float(rows[1][1]) and float(rows[1][2]) > float(rows[0][2]),
    )
    zh, hello = counts["你好，世界"], counts["Hello, world"]
    report.check(
        "text the tokenizer never saw falls back to bytes: 5 Chinese characters are 15 UTF-8 bytes and "
        "15 tokens under both BPEs, while English gets shorter",
        zh == [15, 15, 15] and hello[1] < hello[0] and hello[2] < hello[0],
        f"Chinese {zh[1:]} tokens; English {hello[1:]} tokens for {hello[0]} bytes",
    )

    # ------------------------------------------------------------------------------------------------
    report.section("3. Stop 2: ids become vectors, and the Transformer reads them")
    parts = {
        "embedding table (256 x d)": model.tok_emb.weight.numel(),
        "position table (128 x d)": model.pos_emb.weight.numel(),
        f"{model.cfg.n_layer} Transformer blocks": sum(
            p.numel() for p in model.blocks.parameters()
        ),
        "final norm": sum(p.numel() for p in model.final_norm.parameters()),
        "output matrix (shared with the table)": 0,
    }
    report.table(["part", "parameters"], [[k, f"{v:,}"] for k, v in parts.items()])
    report.check(
        "the parts add up to the model's size; the output matrix is the embedding table used backwards "
        "(tied), so it is counted once",
        sum(parts.values()) == n_params and model.lm_head.weight is model.tok_emb.weight,
        f"{sum(parts.values()):,} = {n_params:,}",
    )
    ids_demo = torch.tensor([[ord("c"), ord("a"), ord("t")]])
    report.check(
        "an id is a row of the embedding table",
        torch.equal(model.tok_emb(ids_demo)[0, 1], model.tok_emb.weight[ord("a")]),
    )
    window = torch.from_numpy(corpus.val[:64].astype(np.int64))[None]
    changed = window.clone()
    changed[0, 40:] = torch.from_numpy(np.random.default_rng(args.seed).integers(97, 123, 24))
    with torch.no_grad():
        before, after = model(window), model(changed)
    report.check(
        "the model only looks backwards: changing bytes 40-63 changes no prediction at positions 0-39, "
        "and does change those after",
        torch.allclose(before[0, :40], after[0, :40], atol=1e-5)
        and not torch.allclose(before[0, 41:], after[0, 41:], atol=1e-3),
    )

    # ------------------------------------------------------------------------------------------------
    report.section("4. Stop 3: the output is a probability distribution")
    with torch.no_grad():
        logits = model(ids)[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(6)
    report.say(f"  After {PROMPT!r} the six most likely next bytes:")
    report.table(
        ["byte", "probability"],
        [
            [repr(chr(i)), f"{v:.3f}"]
            for v, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)
        ],
    )
    manual = torch.exp(logits - logits.max()) / torch.exp(logits - logits.max()).sum()
    report.check(
        "softmax of the 256 scores is a distribution: exp(z_v) / sum of exp(z_u), and it sums to 1",
        torch.allclose(manual, probs, atol=1e-6) and abs(probs.sum().item() - 1) < 1e-5,
    )
    with torch.no_grad():
        first = int(generate(model, ids, 1, temperature=0.0)[0, 0])
    report.check(
        "greedy decoding takes the most likely byte",
        first == int(probs.argmax()),
        f"{chr(first)!r} with probability {probs.max():.3f}",
    )

    rows = []
    for temperature in (0.3, 0.7, 1.0, 1.5):
        q = softmax(logits.numpy() / temperature)
        rows.append([f"{temperature:g}", f"{entropy(q) / math.log(2):.2f}", f"{q.max():.3f}"])
    report.say(
        "\n  The same scores at different temperatures (decoding does not change the model):"
    )
    report.table(["temperature", "entropy (bits)", "top probability"], rows)
    entropies = [float(r[1]) for r in rows]
    report.check(
        "a higher temperature flattens the distribution: entropy up, top probability down",
        entropies == sorted(entropies)
        and [r[2] for r in rows] == sorted((r[2] for r in rows), reverse=True),
    )
    with torch.no_grad():
        runs = [
            decode_ids(
                generate(
                    model, ids, 30, temperature=1.0, generator=torch.Generator().manual_seed(s)
                )[0]
            )
            for s in range(3)
        ]
        greedy = [decode_ids(generate(model, ids, 30, temperature=0.0)[0]) for _ in range(2)]
    report.say("  Sampling at temperature 1 with three seeds, then greedy twice:")
    for s, r in enumerate(runs):
        report.say(f"    seed {s}: {r!r}")
    report.say(f"    greedy : {greedy[0]!r}")
    report.check(
        "the same question gets different answers when sampled, and the same answer when greedy",
        len(set(runs)) == 3 and greedy[0] == greedy[1],
    )

    # ------------------------------------------------------------------------------------------------
    report.section("5. Why predicting the next byte is the whole training objective")
    passage = torch.from_numpy(corpus.val[2000:2128].astype(np.int64))[None]
    x, y = passage[:, :-1], passage[:, 1:]
    with torch.no_grad():
        log_p = F.log_softmax(model(x)[0], dim=-1).gather(1, y[0][:, None]).squeeze(1)
        step_by_step = sum(
            F.log_softmax(model(passage[:, :t])[0, -1], dim=-1)[passage[0, t]].item()
            for t in range(1, passage.shape[1])
        )
        loss = F.cross_entropy(model(x)[0], y[0]).item()
    report.check(
        "the probability of a text is the product of the next-byte probabilities: one forward pass over "
        "the whole text gives the same log-probability as predicting byte by byte",
        abs(log_p.sum().item() - step_by_step) < 1e-3,
        f"log P = {log_p.sum().item():.3f} nats ({log_p.sum().item() / math.log(2):.1f} bits for 127 bytes)",
    )
    report.check(
        "the training loss is the average of -log p over positions",
        abs(loss + log_p.mean().item()) < 1e-6,
        f"{loss:.3f} nats = {loss / math.log(2):.3f} bits per byte",
    )
    bpb = models.shakespeare_bpb(model)
    report.say(
        f"  held-out Shakespeare: {bpb:.3f} bits per byte; knowing nothing, 256 bytes equally likely: 8.000"
    )
    full_check(
        "reading Shakespeare for a few minutes cuts the cost of a byte from 8 bits to under 2.5",
        bpb < 2.5,
        f"{bpb:.3f}",
    )

    val = torch.from_numpy(
        np.stack([corpus.val[i * 128 : i * 128 + 129] for i in range(100)]).astype(np.int64)
    )
    with torch.no_grad():
        bits = -F.log_softmax(model(val[:, :-1]), dim=-1).gather(2, val[:, 1:, None]).squeeze(
            2
        ) / math.log(2)
    prev, cur = val[:, :-1], val[:, 1:]

    def letter(b: torch.Tensor) -> torch.Tensor:
        return ((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122))

    classes = {
        "inside a word": letter(prev) & letter(cur),
        "first letter of a word": (prev == 32) & letter(cur),
        "first byte of a line": prev == 10,
        "punctuation": ~letter(cur) & (cur != 32) & (cur != 10),
    }
    rows = [[k, int(m.sum()), f"{bits[m].mean():.2f}"] for k, m in classes.items()]
    report.say(
        "\n  Where the bits go: the average cost of a byte, by the kind of byte (held-out text):"
    )
    report.table(["byte", "count", "bits"], rows)
    full_check(
        "the model has learned spelling: a byte inside a word is far cheaper than the first letter of the "
        "next word, where what is said next has to be guessed",
        float(rows[0][2]) < 0.5 * float(rows[1][2]),
        f"{rows[0][2]} against {rows[1][2]} bits",
    )

    # ------------------------------------------------------------------------------------------------
    report.section("6. Parameters, tokens, context: the three numbers")

    def size(n_bytes: float) -> str:
        return f"{n_bytes / 1e9:.1f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.2f} MB"

    def stored(m: torch.nn.Module) -> int:
        return sum(q.numel() * q.element_size() for q in m.parameters())

    bf16 = copy.deepcopy(model).to(torch.bfloat16)
    formats = {"FP32": 4, "BF16": 2, "INT8": 1, "4-bit": 0.5}
    rows = [
        [name, *(size(count * width) for width in formats.values())]
        for name, count in (("this model", n_params), ("a 7B model", 7_000_000_000))
    ]
    report.say("  Memory for the weights alone (parameters x bytes per parameter):")
    report.table(["model", *formats], rows)
    report.check(
        "weights take parameters x bytes per parameter: measured on this model's own tensors, FP32 is "
        "4 bytes each and BF16 is 2",
        (stored(model), stored(bf16)) == (4 * n_params, 2 * n_params),
        f"{stored(model):,} and {stored(bf16):,} bytes",
    )
    report.check(
        "the same formula gives the page's numbers for a 7B model: 14 GB in BF16 and 3.5 GB at 4 bits "
        "(the KV cache and activations come on top)",
        (rows[1][2], rows[1][4]) == ("14.0 GB", "3.5 GB"),
    )
    read = steps * batch * models.BLOCK
    passes = read / len(corpus.train)
    report.say(
        f"\n  Training tokens: {steps} steps x {batch} windows x {models.BLOCK} bytes = {read:,} bytes read, "
        f"{passes:.1f} passes over\n  the {len(corpus.train):,}-byte training text, {read / n_params:.1f} bytes per parameter."
    )
    try:
        with torch.no_grad():
            model(torch.zeros(1, models.BLOCK + 1, dtype=torch.long))
        refused = False
    except ValueError as exc:
        refused = "stop at" in str(exc)
    report.check(
        f"the context window is a hard limit: this model's positions stop at {models.BLOCK}, and a "
        f"{models.BLOCK + 1}-byte input is refused",
        refused,
    )

    # ------------------------------------------------------------------------------------------------
    report.section("7. The stages of the page, and where each lives in this repository")
    stages = {
        "Pre-training": ["foundations", "architecture", "pretraining"],
        "Post-training": ["post-training"],
        "Inference and serving": ["inference"],
        "Using the model": ["applications", "multimodal", "evaluation"],
    }
    rows = []
    for stage, sections in stages.items():
        labs = [x for x in course.LABS if x.section in sections]
        rows.append([stage, ", ".join(sections), len(labs), len({k for x in labs for k in x.wiki})])
    report.table(["stage", "sections", "labs", "course pages"], rows)
    report.check(
        "the 30 labs cover all 35 pages of the course, each page at least once",
        len(course.LABS) == 30 and len(course.by_key()) == 35,
    )
    for route, (title, weeks) in course.ROUTES.items():
        labs = course.route_labs(route)
        report.say(f"\n  {title}: {len(labs)} labs, in the order of the weeks")
        for week in range(1, len(weeks) + 1):
            names = ", ".join(x.slug for w, x in labs if w == week)
            if names:
                report.say(f"    week {week:>2}: {names}")
    if root is not None:
        missing = [x.key for x in course.LABS if not (Path(root) / x.path / "run.py").is_file()]
        report.check("every lab on the map exists in this checkout", not missing, str(missing))
    else:
        report.skip("every lab on the map exists", "not running from a source checkout")
    report.say(
        "\n  List them all: dmxsf list. Run one: dmxsf run kv-cache --quick. A route: dmxsf roadmap engineer."
    )
    report.finish()


if __name__ == "__main__":
    main()
