"""Lab: SFT data on a laptop: what the loss sees, data health checks, and four controlled experiments.

Course page: https://dmxsf.com/wiki/post-training/sft-data/

The page fine-tunes Qwen2.5-0.5B on banking77. This lab keeps the questions and the code shapes and
swaps the scale: a 0.86M-parameter byte-level GPT (pretrained on TinyShakespeare, see
dmxsf.models) learns a *generated* twelve-intent customer-message task (dmxsf.post_training.intents).
The numbers therefore differ from the page; the mechanisms and the way to measure them do not.

    uv run python labs/post-training/sft-data/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import copy
import math
import random
import statistics as st

import numpy as np
import torch

from dmxsf import lab, models
from dmxsf.data import ByteCorpus, get_batch, load_shakespeare
from dmxsf.post_training import data_quality as dq
from dmxsf.post_training import intents
from dmxsf.post_training.sft import (
    IGNORE,
    build_example,
    collate,
    finetune,
    forward_packed,
    packing_share,
    padding_share,
    sample_batch,
    sft_loss,
)
from dmxsf.pretraining.train import lm_loss


def main() -> None:
    args, report = lab.start(
        "SFT data: the loss mask, a data health check, and what data size, noise, masking and replay do",
        wiki=["post-training/sft-data"],
    )
    QUICK = args.quick
    STEPS = 40 if QUICK else 300
    SIZES = (1, 5) if QUICK else (1, 2, 5, 10, 20, 50)
    DEFAULT_K = 5 if QUICK else 20
    LR = 3e-4
    BATCH = 16

    # -----------------------------------------------------------------------------------------
    report.section("1. One sample: ids, labels and the -100 mask")
    text, label = intents.make_dataset(1, seed=0)[0]
    prompt, answer = list(intents.render(text)), list(intents.answer_bytes(label))
    ids, labels = build_example(prompt, answer)
    share = len(answer) / len(ids)
    report.say(f"  message {text!r} -> {label}")
    report.say(
        f"  {len(prompt)} prompt tokens, {len(answer)} answer tokens (supervised share {share:.0%})"
    )
    rows = [
        (pos, repr(chr(ids[pos])), labels[pos])
        for pos in range(len(prompt) - 3, min(len(ids), len(prompt) + 4))
    ]
    report.table(["position", "token", "label"], rows)
    report.check(
        "every prompt position is -100 and every answer position carries its token",
        all(v == IGNORE for v in labels[: len(prompt)]) and labels[len(prompt) :] == answer,
    )
    report.check(
        "the stop token (newline) is supervised: it is the only signal for where to stop",
        labels[-1] == intents.STOP,
    )

    # the template is part of the data: a model trained on one prompt, asked with another
    train_k = intents.make_dataset(DEFAULT_K, seed=1)
    test = intents.make_dataset(25 if not QUICK else 5, seed=99, exclude={t for t, _ in train_k})
    test_texts = {t for t, _ in test}
    base = models.pretrained("base", quick=QUICK)
    base_bpb = models.shakespeare_bpb(base)

    def examples(pairs, mask=True):
        return [
            build_example(list(intents.render(t)), list(intents.answer_bytes(y)), mask_prompt=mask)
            for t, y in pairs
        ]

    def noisy(pairs, p, seed):
        """Replace the label of a fraction p of the pairs by a different, random label."""
        rng = random.Random(seed)
        return [
            (t, rng.choice([x for x in intents.LABELS if x != y]) if rng.random() < p else y)
            for t, y in pairs
        ]

    def run(k, *, noise=0.0, mask=True, replay=False, seed=0, data_seed=1, steps=STEPS):
        """Fine-tune a copy of the base on k examples per intent; score accuracy and forgetting."""
        pairs = intents.make_dataset(k, seed=data_seed, exclude=test_texts)
        if noise:  # an independent random stream: not the data's, not the training's
            pairs = noisy(pairs, noise, seed=10_000 + 7 * data_seed)
        ex = examples(pairs, mask)
        model = copy.deepcopy(base)
        torch.manual_seed(seed)
        shakes = ByteCorpus.shakespeare().train
        if replay:  # every other step is plain next-byte prediction on pretraining-style text

            def batch(step):
                if step % 2:
                    return ("lm", get_batch(shakes, step, BATCH, 115, seed))
                return ("sft", sample_batch(ex, step // 2, BATCH, seed))

            def loss(m, item):
                kind, b = item
                return sft_loss(m, b) if kind == "sft" else lm_loss(m, b)

            finetune(model, batch, 2 * steps, LR, loss_fn=loss)
        else:
            finetune(model, lambda s: sample_batch(ex, s, BATCH, seed), steps, LR)
        out = intents.score(model, test)
        out["bpb"] = models.shakespeare_bpb(model)
        out["model"] = model
        return out

    default = run(DEFAULT_K)
    variants = {
        "the trained prompt": intents.PROMPT,
        "one extra space before the answer": intents.PROMPT + " ",
        "another wording of the prompt": "Which intent is this?\nCustomer: {text:<56}\nAnswer:",
    }
    rows = []
    for name, template in variants.items():
        sc = intents.score(default["model"], test, template)
        rows.append((name, f"{sc['accuracy']:.1%}", f"{sc['valid']:.1%}"))
    report.say(
        f"\n  the model fine-tuned on {DEFAULT_K} messages per intent, asked in other words:"
    )
    report.table(["prompt at inference time", "accuracy", "valid label"], rows)
    report.say(
        "  (one model, one run: the pattern is the point, not the exact gap. The template is part of\n"
        "   the data and should be versioned with it.)"
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. What the loss averages over")
    pairs = intents.make_dataset(3, seed=5)[:8]
    batch = collate(examples(pairs))
    x, y = batch
    n_sup = (y != IGNORE).sum(dim=1)
    token_mean = sft_loss(base, batch).item()
    seq_mean = sft_loss(base, batch, per_sequence=True).item()
    full_batch = collate(examples(pairs, mask=False))
    report.say(
        f"  a batch of {len(pairs)} examples, answers of {n_sup.min().item()} to {n_sup.max().item()} "
        f"bytes; the base model's loss on it:"
    )
    report.table(
        ["averaging", "loss (nats)"],
        [
            ["per supervised token (default)", f"{token_mean:.4f}"],
            ["per sequence, then over sequences", f"{seq_mean:.4f}"],
            ["prompt not masked (all positions)", f"{sft_loss(base, full_batch).item():.4f}"],
        ],
    )
    with torch.no_grad():
        logits = base(x)
    ce = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), y, ignore_index=IGNORE, reduction="none"
    )
    manual = (ce.sum() / n_sup.sum()).item()
    report.check(
        "the default loss is the sum over supervised positions divided by their count",
        abs(manual - token_mean) < 1e-6,
        f"{manual:.6f} vs {token_mean:.6f}",
    )
    report.check(
        "unequal answer lengths make the two averages differ (equal lengths would not)",
        abs(token_mean - seq_mean) > 1e-3,
        f"{token_mean:.4f} vs {seq_mean:.4f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. Packing and padding")
    docs = [list(p.encode()[:32]) for p in load_shakespeare().decode().split("\n\n") if p.strip()]
    docs = [d for d in docs if len(d) >= 8][:12]
    cut = [len(d) for d in docs[:4]]
    ids_packed = torch.tensor([[b for d in docs[:4] for b in d]])
    with torch.no_grad():
        packed = forward_packed(base, ids_packed, cut)[0]
        alone = torch.cat([base(torch.tensor([d]))[0] for d in docs[:4]])
        naive = base(ids_packed)[0]
    err_exact = (packed - alone).abs().max().item()
    err_naive = (naive[cut[0] :] - alone[cut[0] :]).abs().max().item()
    report.say(
        f"  four speeches ({', '.join(map(str, cut))} bytes) packed into one row of {sum(cut)}:"
    )
    report.table(
        ["packing", "max |logit difference| vs four separate passes"],
        [
            ["block-diagonal mask + positions restart", f"{err_exact:.1e}"],
            ["plain concatenation (no mask, positions run on)", f"{err_naive:.1e}"],
        ],
    )
    report.check(
        "packing with a document mask and restarted positions equals separate passes",
        err_exact < 1e-4,
        f"{err_exact:.1e}",
    )
    report.check(
        "plain concatenation changes the numbers: documents see each other",
        err_naive > 1e-2,
        f"{err_naive:.1e}",
    )

    paras = [len(p.encode()) for p in load_shakespeare().decode().split("\n\n") if p.strip()]
    paras = [min(n, 1024) for n in paras]
    report.say(
        f"\n  lengths of the {len(paras):,} speeches in TinyShakespeare (cut at 1,024 bytes): "
        f"mean {st.mean(paras):.0f}, median {st.median(paras):.0f}, longest {max(paras)}"
    )
    rows = []
    shares = {}
    for bs in (16, 64):
        shares[bs] = (padding_share(paras, bs, False), padding_share(paras, bs, True))
        rows.append([f"batch {bs}", f"{shares[bs][0]:.1%}", f"{shares[bs][1]:.1%}"])
    pack = packing_share(paras, 1024)
    rows.append(["packed into 1,024-byte rows", f"{pack:.1%}", "-"])
    report.table(["padding share of all positions", "random batches", "grouped by length"], rows)
    report.check(
        "random batches waste far more positions than length-grouped ones",
        all(r > 3 * g for r, g in shares.values()),
        f"{shares[16][0]:.0%} vs {shares[16][1]:.0%} at batch 16",
    )
    report.check(
        "packing (with the mask above) wastes less than either",
        pack < min(g for _, g in shares.values()),
        f"{pack:.1%}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. Health check: rules, near-duplicates, contamination (planted ground truth)")
    rng = random.Random(0)
    speeches = [p.replace("\n", " ") for p in load_shakespeare().decode().split("\n\n")]
    speeches = [p for p in speeches if 40 <= len(p.split()) <= 120]
    ask = [
        "Summarize the passage in one sentence.",
        "Rewrite the passage in plain English.",
        "Who is speaking in this passage?",
        "Continue the passage.",
        "Give the passage a title.",
        "List the names that appear in the passage.",
        "What is the mood of the passage?",
        "Explain the passage to a ten-year-old.",
    ]
    n_rec = 200 if QUICK else 500
    pool = rng.sample(speeches, 2 * n_rec)
    recs = [
        {
            "instruction": rng.choice(ask),
            "input": pool[i],
            "output": " ".join(pool[n_rec + i].split()[:25]),
            "tag": "",
        }
        for i in range(n_rec)
    ]
    planted = {"placeholder": 0.05, "refusal": 0.03, "empty": 0.02, "url": 0.02, "short": 0.03}
    for tag, frac in planted.items():
        for r in rng.sample([r for r in recs if not r["tag"]], int(frac * n_rec)):
            r["tag"] = tag
            if tag == "placeholder":
                r["input"] = "<noinput>"
            elif tag == "refusal":
                r["output"] = "As an AI language model, I cannot help with that."
            elif tag == "empty":
                r["output"] = ""
            elif tag == "url":
                r["input"] = "https://example.com/page"
            else:  # a short answer that is exactly right
                r["output"] = rng.choice(["Hamlet.", "Yes.", "Sad.", "Lady Macbeth."])
    for _ in range(int(0.08 * n_rec)):  # near-duplicates: one word of the input changed
        src = rng.choice([r for r in recs if not r["tag"]])
        words = src["input"].split()
        words[rng.randrange(len(words))] = "banana"
        recs.append({**src, "input": " ".join(words), "tag": "near-dup"})
    defect = {  # which planted tags each rule is meant to find
        "placeholder text": {"placeholder"},
        "refusal-like phrase": {"refusal"},
        "empty output": {"empty"},
        "input is only a URL": {"url"},
        "output <= 2 words": {"empty", "short"},
    }
    rows, misses = [], 0
    for name, rule in dq.RULES.items():
        hits = {i for i, r in enumerate(recs) if rule(r)}
        want = {i for i, r in enumerate(recs) if r["tag"] in defect[name]}
        misses += len(want - hits)
        rows.append([name, len(hits), len(want & hits), len(want), len(hits - want)])
    report.say(f"  {len(recs):,} records made of Shakespeare passages; planted defects:")
    report.table(["rule", "flagged", "planted found", "planted", "flagged, not planted"], rows)
    report.check("each rule finds every defect planted for it", misses == 0)
    extra = {r[0]: r[4] for r in rows}
    report.say(
        f"  'output <= 2 words' counts the {len([r for r in recs if r['tag'] == 'short'])} correct one-word"
        " answers as well: rules only produce candidates, and\n"
        f"  a person has to read them. The refusal rule also flags {extra['refusal-like phrase']} passages"
        " where a character simply says 'I cannot'."
    )

    texts = [r["instruction"] + " " + r["input"][:200] for r in recs]
    found = {(i, j) for i, j, _ in dq.near_duplicates(texts)}
    sets = [dq.shingles(t) for t in texts]
    exact = {
        (i, j)
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
        if dq.jaccard(sets[i], sets[j]) >= 0.8
    }
    planted_pairs = sum(r["tag"] == "near-dup" for r in recs)
    report.say(
        f"\n  pairs with Jaccard >= 0.8 by brute force over all "
        f"{len(texts) * (len(texts) - 1) // 2:,} pairs: {len(exact)} ({planted_pairs} planted);\n"
        f"  MinHash + LSH (16 bands x 4 rows) found {len(found & exact)} of them and "
        f"{len(found - exact)} pairs that are not"
    )
    report.check(
        "LSH finds (almost) every pair brute force finds, and invents none",
        len(found - exact) == 0 and len(found & exact) >= 0.98 * len(exact),
        f"{len(found & exact)} of {len(exact)}",
    )

    trials = 100 if QUICK else 400
    rows, worst = [], 0.0
    for jac in (0.3, 0.5, 0.6, 0.7, 0.8):
        shared = round(2 * jac * 100 / (1 + jac))  # two sets of 100 hashes, `shared` in common
        hit, real = 0, []
        for t in range(trials):
            ids = np.random.default_rng((int(jac * 100), t)).integers(0, 2**32, 200 - shared)
            a, b = set(ids[:100].tolist()), set(ids[:shared].tolist()) | set(ids[100:].tolist())
            real.append(dq.jaccard(a, b))
            hit += bool(dq.candidate_pairs(dq.signatures([a, b], 64, seed=t), 16, 4))
        measured, theory = hit / trials, dq.candidate_probability(st.mean(real), 16, 4)
        # a gap of three binomial standard deviations (plus a little slack) would be suspicious
        worst = max(worst, abs(measured - theory) - 3 * math.sqrt(theory * (1 - theory) / trials))
        rows.append([f"{st.mean(real):.2f}", f"{measured:.3f}", f"{theory:.3f}"])
    report.say("\n  the LSH S-curve: how often two texts with a given Jaccard share a band")
    report.table(["Jaccard", "measured", "1-(1-J^4)^16"], rows)
    report.check(
        "the measured candidate rate follows the S-curve formula",
        worst < 0.01,
        f"{trials} trials per row; every row within three binomial standard deviations (+0.01)",
    )

    rng2 = random.Random(1)
    mix = rng2.sample(speeches, 60 + 300 + 200)
    bench, corpus_docs, clean = mix[:60], mix[60:360], mix[360:]
    questions = [" ".join(b.split()[:40]) for b in bench]
    train_docs = list(corpus_docs)
    for q in questions[:20]:  # copied verbatim into a longer document
        train_docs.append("Some chapter text. " + q + " And the chapter goes on.")
    for q in questions[20:40]:  # every 8th word replaced: no 13-gram survives
        w = q.split()
        w[7::8] = ["xx"] * len(w[7::8])
        train_docs.append(" ".join(w))
    for q in questions[40:]:  # only a 10-word stretch survives
        train_docs.append(" ".join(q.split()[5:15]))
    clean_q = [" ".join(c.split()[:40]) for c in clean]
    rows = []
    for n in (5, 8, 13):
        hit = dq.contaminated(train_docs, questions, n)
        false_rate = len(dq.contaminated(train_docs, clean_q, n)) / len(clean_q)
        rows.append(
            [
                n,
                f"{len(hit & set(range(20)))}/20",
                f"{len(hit & set(range(20, 40)))}/20",
                f"{len(hit & set(range(40, 60)))}/20",
                f"{false_rate:.1%}",
            ]
        )
    report.say(
        "\n  contamination: 60 'benchmark questions'; 20 copied verbatim into the training text, 20 with\n"
        "  every 8th word replaced, 20 with only a 10-word stretch left; 'clean' questions were never copied"
    )
    report.table(
        ["n-gram", "verbatim found", "edited found", "10-word overlap found", "clean flagged"],
        rows,
    )
    by_n = {r[0]: r for r in rows}
    report.check(
        "13-grams find every verbatim copy and miss the edited ones",
        by_n[13][1] == "20/20" and by_n[13][2] == "0/20",
        f"verbatim {by_n[13][1]}, edited {by_n[13][2]}",
    )
    report.check(
        "shorter n-grams catch the partial overlaps that 13-grams (almost) all miss",
        by_n[8][3] == "20/20" and int(by_n[13][3].split("/")[0]) <= 2,
        f"10-word overlap found: n=8 {by_n[8][3]}, n=13 {by_n[13][3]}",
    )
    report.say(
        f"  the price of a short n: clean questions flagged {by_n[5][4]} at n=5, {by_n[8][4]} at n=8, "
        f"{by_n[13][4]} at n=13."
    )

    # -----------------------------------------------------------------------------------------
    report.section("5. Experiment: how much does a run vary when nothing is changed?")
    noise_runs = [default]
    for seed, data_seed in [(1, 1)] if QUICK else [(1, 1), (2, 1), (0, 2), (0, 3)]:
        noise_runs.append(run(DEFAULT_K, seed=seed, data_seed=data_seed))
    accs = [r["accuracy"] for r in noise_runs]
    labels_ = [
        "default",
        *(
            ["training seed 1"]
            if QUICK
            else ["training seed 1", "training seed 2", "data subset 2", "data subset 3"]
        ),
    ]
    report.table(
        ["run", "accuracy", "shakespeare bits/byte"],
        [
            [n, f"{r['accuracy']:.1%}", f"{r['bpb']:.2f}"]
            for n, r in zip(labels_, noise_runs, strict=True)
        ],
    )
    lo, hi = min(accs), max(accs)
    spread = hi - lo
    report.say(
        f"  accuracy {st.mean(accs):.1%} on average, range {lo:.1%}-{hi:.1%}; {len(test)} test messages,"
        f" so one message is {1 / len(test):.2%}.\n"
        f"  The pretrained model scores {base_bpb:.2f} bits/byte on held-out Shakespeare: the 'forgetting' baseline."
    )
    report.say("  Differences below the range are not findings.")

    # -----------------------------------------------------------------------------------------
    report.section(f"6. Experiment: examples per intent ({STEPS} steps each, all else equal)")
    size_runs = {}
    for k in SIZES:
        size_runs[k] = default if k == DEFAULT_K else run(k)
    report.table(
        ["per intent", "examples", "accuracy", "valid label", "shakespeare bits/byte"],
        [
            [k, 12 * k, f"{r['accuracy']:.1%}", f"{r['valid']:.0%}", f"{r['bpb']:.2f}"]
            for k, r in size_runs.items()
        ],
    )
    if QUICK:
        report.skip(
            "accuracy rises with the number of examples, steeply at first", "needs the full run"
        )
    else:
        a = [size_runs[k]["accuracy"] for k in SIZES]
        report.check(
            "accuracy rises with data, and each step up past 20 per intent is worth less than the first ones",
            a[2] - a[0] > 0.2 and a[-1] - a[-2] < a[2] - a[1],
            "  ->  ".join(f"{v:.0%}" for v in a),
        )

    # -----------------------------------------------------------------------------------------
    report.section("7. Experiment: wrong labels")
    rates = (0.2,) if QUICK else (0.1, 0.2, 0.4)
    noise_table = []
    noise_res = {}
    for p in rates:
        noise_res[p] = run(DEFAULT_K, noise=p)
        noise_table.append(
            [f"{p:.0%}", f"{noise_res[p]['accuracy']:.1%}", f"{noise_res[p]['bpb']:.2f}"]
        )
    report.table(
        ["labels replaced", "accuracy", "shakespeare bits/byte"],
        [["0%", f"{default['accuracy']:.1%}", f"{default['bpb']:.2f}"], *noise_table],
    )
    if QUICK:
        report.skip("wrong labels cost accuracy, more of them cost more", "needs the full run")
    else:
        mean_acc = st.mean(accs)
        drops = [mean_acc - noise_res[p]["accuracy"] for p in rates]
        report.check(
            "every noise level costs more than the run-to-run range, and more noise costs more",
            all(d > spread for d in drops) and drops == sorted(drops),
            "drops "
            + ", ".join(f"{d * 100:.1f} pts" for d in drops)
            + f"; range {spread * 100:.1f} pts",
        )
        report.say(
            f"  about {drops[-1] * 100 / (rates[-1] * 100):.2f} accuracy points per 1% of wrong labels at "
            f"{rates[-1]:.0%}; the page's 0.5B model lost 0.4"
        )

    # -----------------------------------------------------------------------------------------
    report.section("8. Experiment: loss on the prompt, and replaying general text")
    unmasked = run(DEFAULT_K, mask=False)
    replay = run(DEFAULT_K, replay=True)
    report.table(
        ["setting", "accuracy", "shakespeare bits/byte", "optimizer steps"],
        [
            [
                "default: loss on the answer only",
                f"{default['accuracy']:.1%}",
                f"{default['bpb']:.2f}",
                STEPS,
            ],
            [
                "loss on the whole sequence",
                f"{unmasked['accuracy']:.1%}",
                f"{unmasked['bpb']:.2f}",
                STEPS,
            ],
            [
                "answer-only + every other step plain Shakespeare",
                f"{replay['accuracy']:.1%}",
                f"{replay['bpb']:.2f}",
                f"{2 * STEPS} ({STEPS} of them SFT)",
            ],
        ],
    )
    report.say(f"  (the pretrained model: {base_bpb:.2f} bits/byte before any fine-tuning)")
    if QUICK:
        report.skip("replay keeps the pretraining distribution", "needs the full run")
        report.skip("loss on the prompt changes accuracy by more than noise", "needs the full run")
    else:
        report.check(
            "replaying pretraining-style text keeps held-out Shakespeare much closer to the base model",
            replay["bpb"] - base_bpb < 0.5 * (default["bpb"] - base_bpb),
            f"{replay['bpb']:.2f} vs {default['bpb']:.2f} bits/byte (base {base_bpb:.2f})",
        )
        report.observe(
            "...without costing target accuracy beyond the run-to-run range (same number of SFT steps)",
            replay["accuracy"] > lo - spread,
            f"{replay['accuracy']:.1%} vs runs {lo:.1%}-{hi:.1%}",
        )
        if replay["accuracy"] <= lo - spread:
            report.say(
                "  Here replay did cost accuracy. A likely reason, not isolated here: with 300 SFT steps\n"
                "  this small model is near the point where it takes off, and that point moves with the\n"
                "  seed and the machine, so one run per setting cannot say whether replay costs accuracy."
            )
        diff = default["accuracy"] - unmasked["accuracy"]
        report.say(
            f"  loss on the prompt: accuracy {diff * 100:+.1f} points for the answer-only default "
            f"(run-to-run range {spread * 100:.1f}).\n"
            "  A likely reason, not isolated here: the prompt is about eight times longer than the answer,\n"
            "  so it dominates the loss and the 300 steps. On the page's 0.5B model the same switch moved\n"
            "  accuracy by 1.3 points (85.8% to 84.5%)."
        )

    report.finish()


if __name__ == "__main__":
    main()
