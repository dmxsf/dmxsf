"""Lab: decoding and sampling. The page's example, then what each knob does to real text.

Course page: https://dmxsf.com/wiki/inference/decoding-and-sampling/

Part one reproduces the page's four-token example exactly (temperature, top-k, top-p, min-p, and
why the order matters). Part two generates text with the byte-level GPT of dmxsf.models, trained
on TinyShakespeare for 2,400 steps, and measures: how often a setting writes words that exist in
the training text, how often it loops, and how surprised the model itself is by its own output.
Beam search and repetition penalties are compared on the same prompts. Everything is small enough
for a CPU; the model is tiny, so the numbers show the mechanisms, not what a chat model does.

    uv run python labs/inference/decoding-and-sampling/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import re

import numpy as np
import torch

from dmxsf import lab, models
from dmxsf.data import ByteCorpus
from dmxsf.inference.generate import beam_search, generate, process_logits, sample_logits

Z = torch.tensor([[4.0, 3.0, 1.0, 0.0]])


def dist(**kw) -> list[float]:
    return process_logits(Z, **kw).softmax(-1)[0].tolist()


def close(got: list[float], want: list[float], tol: float = 5e-4) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(got, want, strict=True))


def fmt(p: list[float]) -> list[str]:
    return [f"{x:.3f}" for x in p]


def main() -> None:
    args, report = lab.start(
        "Decoding and sampling: temperature, top-k, top-p, min-p, beam search, penalties",
        wiki=["inference/decoding-and-sampling"],
    )
    QUICK = args.quick
    gen_seed = args.seed

    # -----------------------------------------------------------------------------------------
    report.section("1. The page's example: logits z = [4, 3, 1, 0]")
    cases = [
        ("T = 0.5", {"temperature": 0.5}, [0.879, 0.119, 0.002, 0.000]),
        ("T = 1", {"temperature": 1.0}, [0.696, 0.256, 0.035, 0.013]),
        ("T = 2", {"temperature": 2.0}, [0.509, 0.309, 0.114, 0.069]),
        ("top-k 2", {"top_k": 2}, [0.731, 0.269, 0, 0]),
        ("top-p 0.9", {"top_p": 0.9}, [0.731, 0.269, 0, 0]),
        ("T = 0.5, top-p 0.9", {"temperature": 0.5, "top_p": 0.9}, [0.881, 0.119, 0, 0]),
        ("min-p 0.1", {"min_p": 0.1}, [0.731, 0.269, 0, 0]),
    ]
    rows = [[name, *fmt(dist(**kw))] for name, kw, _ in cases]
    report.table(["setting", "token 1", "token 2", "token 3", "token 4"], rows)
    report.check(
        "every row matches the page's numbers (three decimals)",
        all(close(dist(**kw), want) for _, kw, want in cases),
    )

    cold_first = dist(temperature=0.5, top_p=0.99)
    nucleus_first = (process_logits(Z, 1.0, top_p=0.99) / 0.5).softmax(-1)[0].tolist()
    report.say("\n  the order matters (top-p 0.99):")
    report.table(
        ["order", "token 1", "token 2", "token 3", "token 4"],
        [
            ["temperature 0.5, then top-p", *fmt(cold_first)],
            ["top-p at T = 1, then T = 0.5", *fmt(nucleus_first)],
        ],
    )
    report.check(
        "temperature first keeps 2 tokens, nucleus first keeps 4: the order changes the result",
        sum(x > 0 for x in cold_first) == 2 and sum(x > 0 for x in nucleus_first) == 4,
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. Do the samples follow the processed distribution?")
    n = 20_000 if QUICK else 200_000
    g = torch.Generator().manual_seed(gen_seed)
    rows, worst = [], 0.0
    for name, kw in (
        ("top-k 2", {"top_k": 2}),
        ("T = 0.5, top-p 0.9", {"temperature": 0.5, "top_p": 0.9}),
        ("T = 2, min-p 0.2", {"temperature": 2.0, "min_p": 0.2}),
    ):
        want = torch.tensor(dist(**kw))
        draws = sample_logits(Z.expand(n, -1), generator=g, **kw)
        freq = torch.bincount(draws, minlength=4) / n
        sigma = (want * (1 - want) / n).sqrt()
        z = ((freq - want).abs() / sigma.clamp(min=1e-12)).where(want > 0, freq.abs() * 1e9).max()
        worst = max(worst, z.item())
        rows.append([name, *fmt(want.tolist()), *fmt(freq.tolist()), f"{z.item():.1f}"])
    report.table(
        ["setting", "p1", "p2", "p3", "p4", "freq 1", "freq 2", "freq 3", "freq 4", "max |z|"], rows
    )
    report.check(
        f"{n:,} draws per setting: frequencies within 4 standard errors, dropped tokens never drawn",
        worst < 4,
        f"worst z = {worst:.1f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. What the settings do to text (byte-level GPT on TinyShakespeare)")
    model = models.pretrained("base", quick=QUICK)
    corpus = ByteCorpus.shakespeare()
    vocabulary = set(re.findall(r"[a-z']+", bytes(corpus.train).decode().lower()))
    n_prompts, plen, new = (8, 48, 24) if QUICK else (64, 48, 80)
    starts = np.random.default_rng(gen_seed).integers(0, len(corpus.val) - plen - new, n_prompts)
    prompts = torch.tensor(np.stack([corpus.val[s : s + plen] for s in starts]).astype(np.int64))
    truth = torch.tensor(
        np.stack([corpus.val[s + plen : s + plen + new] for s in starts]).astype(np.int64)
    )

    def words(text: str) -> list[str]:
        found = re.findall(r"[A-Za-z']+", text)
        return [w.lower() for w in found[1:-1]]  # the first and last word are cut by the window

    @torch.no_grad()
    def metrics(continuations: torch.Tensor) -> dict[str, float]:
        ids = torch.cat([prompts[: len(continuations)], continuations], dim=1)
        logp = torch.log_softmax(model(ids[:, :-1]), -1).gather(-1, ids[:, 1:, None])[..., 0]
        bits = -logp[:, plen - 1 :].mean().item() / np.log(2)
        texts = [bytes(row).decode("utf-8", errors="replace") for row in continuations.tolist()]
        ws = [w for t in texts for w in words(t)]
        d4 = []
        for row in continuations.tolist():
            grams = [tuple(row[i : i + 4]) for i in range(len(row) - 3)]
            d4.append(len(set(grams)) / len(grams))
        return {
            "valid": float(np.mean([w in vocabulary for w in ws])) if ws else float("nan"),
            "distinct4": float(np.mean(d4)),
            "bits": bits,
            "words": len(ws),
        }

    def sample(seed=gen_seed, **kw) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return generate(model, prompts, new, generator=g, **kw)

    settings = {
        "greedy (T = 0)": {"temperature": 0.0},
        "T = 0.5": {"temperature": 0.5},
        "T = 1": {"temperature": 1.0},
        "T = 1.5": {"temperature": 1.5},
        "T = 2": {"temperature": 2.0},
        "T = 1, top-p 0.9": {"temperature": 1.0, "top_p": 0.9},
        "T = 1, min-p 0.05": {"temperature": 1.0, "min_p": 0.05},
        "T = 1.5, top-p 0.9": {"temperature": 1.5, "top_p": 0.9},
    }
    results = {name: metrics(sample(**kw)) for name, kw in settings.items()}
    real = metrics(truth)
    rows = [
        [name, f"{m['valid']:.1%}", f"{m['distinct4']:.2f}", f"{m['bits']:.2f}"]
        for name, m in results.items()
    ] + [
        [
            "(the real continuation)",
            f"{real['valid']:.1%}",
            f"{real['distinct4']:.2f}",
            f"{real['bits']:.2f}",
        ]
    ]
    report.say(
        f"  {n_prompts} prompts of {plen} bytes from the held-out text, {new} new bytes each"
        f" (about {results['T = 1']['words']} words per setting):"
    )
    report.table(
        ["setting", "words in the training text", "distinct 4-grams", "bits/byte under the model"],
        rows,
    )
    report.say(
        "  distinct 4-grams: share of byte 4-grams in a continuation that are unique (low = looping).\n"
        "  bits/byte: how surprised the model (at T = 1) is by the text, which includes its own output."
    )
    if QUICK:
        for name in (
            "greedy loops more than any sampled setting at T >= 1",
            "heating up lowers the share of real words (T = 2 against T = 1)",
            "cutting the tail at T = 1.5 recovers real words",
        ):
            report.skip(name, "needs the full run")
    else:
        hot = [results[k]["distinct4"] for k in ("T = 1", "T = 1.5", "T = 2", "T = 1, top-p 0.9")]
        report.check(
            "greedy loops more than any sampled setting at T >= 1 (distinct 4-grams)",
            results["greedy (T = 0)"]["distinct4"] < min(hot),
            f"{results['greedy (T = 0)']['distinct4']:.2f} against {min(hot):.2f}",
        )
        report.check(
            "heating up lowers the share of real words (T = 2 against T = 1, by more than 10 points)",
            results["T = 2"]["valid"] < results["T = 1"]["valid"] - 0.10,
            f"{results['T = 2']['valid']:.1%} against {results['T = 1']['valid']:.1%}",
        )
        report.check(
            "cutting the tail at T = 1.5 (top-p 0.9) recovers real words",
            results["T = 1.5, top-p 0.9"]["valid"] > results["T = 1.5"]["valid"] + 0.10,
            f"{results['T = 1.5, top-p 0.9']['valid']:.1%} against {results['T = 1.5']['valid']:.1%}",
        )
    seed2 = metrics(sample(seed=gen_seed + 1, temperature=1.0))
    report.say(
        f"  noise: T = 1 again with another seed gives {seed2['valid']:.1%} real words,"
        f" {seed2['distinct4']:.2f} distinct, {seed2['bits']:.2f} bits/byte"
        f" (first run {results['T = 1']['valid']:.1%}, {results['T = 1']['distinct4']:.2f},"
        f" {results['T = 1']['bits']:.2f})."
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. Beam search: the most probable text is not the most natural")
    nb, beams, bnew = (2, 3, 24) if QUICK else (8, 4, 80)
    sampled_bits = metrics(sample(temperature=1.0)[:nb])["bits"]
    beam_scores, greedy_scores, real_bits = [], [], []
    d4 = {"greedy": [], "beam": []}

    def distinct4(row: torch.Tensor) -> float:
        grams = [tuple(row.tolist()[j : j + 4]) for j in range(len(row) - 3)]
        return len(set(grams)) / len(grams)

    for i in range(nb):
        prompt = prompts[i : i + 1]
        seqs, scores = beam_search(model, prompt, beams, bnew)
        greedy, greedy_score = beam_search(model, prompt, 1, bnew)
        beam_scores.append(scores[0].item())
        greedy_scores.append(greedy_score[0].item())
        d4["beam"].append(distinct4(seqs[0]))
        d4["greedy"].append(distinct4(greedy[0]))
        full = torch.cat([prompt, truth[i : i + 1, :bnew]], dim=1)
        with torch.no_grad():
            lp = torch.log_softmax(model(full[:, :-1]), -1).gather(-1, full[:, 1:, None])[..., 0]
        real_bits.append(-lp[:, plen - 1 :].mean().item() / np.log(2))
    per_byte = lambda xs: -float(np.mean(xs)) / bnew / np.log(2)  # noqa: E731
    bits = {
        "real": float(np.mean(real_bits)),
        "greedy": per_byte(greedy_scores),
        "beam": per_byte(beam_scores),
    }
    wins = sum(b > g + 1e-6 for b, g in zip(beam_scores, greedy_scores, strict=True))
    ties = sum(abs(b - g) <= 1e-6 for b, g in zip(beam_scores, greedy_scores, strict=True))
    report.table(
        ["continuation", "bits/byte under the model", "distinct 4-grams"],
        [
            ["real text", f"{bits['real']:.2f}", "-"],
            ["sampled, T = 1", f"{sampled_bits:.2f}", f"{results['T = 1']['distinct4']:.2f}"],
            ["greedy", f"{bits['greedy']:.2f}", f"{np.mean(d4['greedy']):.2f}"],
            [f"beam search, {beams} beams", f"{bits['beam']:.2f}", f"{np.mean(d4['beam']):.2f}"],
        ],
    )
    report.say(
        f"  beam search found a strictly more probable sequence than greedy on {wins} of {nb}"
        f" prompts ({ties} ties)."
    )
    report.check(
        "greedy and beam output are far more probable than the real text (open-ended text is not the mode)",
        bits["greedy"] < bits["real"] - 0.5 and bits["beam"] < bits["real"] - 0.5,
        f"real {bits['real']:.2f}, greedy {bits['greedy']:.2f}, beam {bits['beam']:.2f} bits/byte",
    )
    beam_name = "no beam is worse than greedy (not a guarantee of the algorithm, only what these prompts show)"
    if QUICK:
        report.skip(beam_name, "needs the full run")
    else:
        report.check(
            beam_name,
            all(b >= g - 1e-6 for b, g in zip(beam_scores, greedy_scores, strict=True)),
            f"beam better on {wins}, equal on {ties} of {nb}",
        )

    # -----------------------------------------------------------------------------------------
    report.section("5. Repetition penalties on a 256-symbol vocabulary")
    pen = {
        "greedy": {},
        "repetition 1.1": {"repetition": 1.1},
        "repetition 1.3": {"repetition": 1.3},
        "repetition 2.0": {"repetition": 2.0},
        "frequency 0.5": {"frequency": 0.5},
        "presence 0.5": {"presence": 0.5},
    }
    pres = {name: metrics(sample(temperature=0.0, **kw)) for name, kw in pen.items()}
    report.table(
        [
            "greedy decoding with",
            "words in the training text",
            "distinct 4-grams",
            "bits/byte under the model",
        ],
        [
            [n_, f"{m['valid']:.1%}", f"{m['distinct4']:.2f}", f"{m['bits']:.2f}"]
            for n_, m in pres.items()
        ],
    )
    report.say(
        "  A byte-level model has 256 symbols and the prompt already contains every common letter,\n"
        "  so 'seen' is nearly everything: these penalties are far harsher than on a vocabulary of\n"
        "  100k tokens. Read the table as a warning about strength, not as a recipe."
    )
    if QUICK:
        report.skip("a strong penalty damages the text", "needs the full run")
    else:
        report.check(
            "a very strong repetition penalty (2.0) costs real words against plain greedy, by more than 2 points",
            pres["repetition 2.0"]["valid"] < pres["greedy"]["valid"] - 0.02,
            f"{pres['repetition 2.0']['valid']:.1%} against {pres['greedy']['valid']:.1%}",
        )

    # -----------------------------------------------------------------------------------------
    report.section("6. Does the batch shape change greedy decoding?")
    batch = prompts if not QUICK else prompts[:4]
    with torch.no_grad():
        full = model(batch)[:, -1]
        alone = torch.cat([model(batch[i : i + 1])[:, -1] for i in range(len(batch))])
    diff = (full - alone).abs().max().item()
    top2 = full.topk(2, dim=-1).values
    gap = (top2[:, 0] - top2[:, 1]).min().item()
    flips = int((full.argmax(-1) != alone.argmax(-1)).sum())
    report.say(
        f"  the same {len(batch)} prompts, run as one batch and one at a time:"
        f" max |logit difference| {diff:.1e},"
    )
    report.say(
        f"  smallest gap between the best and second-best logit {gap:.1e}, greedy choices that"
        f" differ: {flips}."
    )
    if diff == 0:
        report.say(
            "  bit-identical here: on this CPU the arithmetic is the same for every batch shape, so nothing flips.\n"
            "  The page's warning (a library may pick different kernels, and a different summation order, per\n"
            "  batch size) is about other hardware; this lab cannot show it."
        )
    else:
        report.say(
            f"  rounding moved the logits by up to {diff:.1e}; that changes a greedy choice only when the top two\n"
            f"  logits are closer than that, and the smallest gap among these {len(batch)} prompts is {gap:.1e}."
        )
    report.check(
        "batch shape changes the logits by at most float rounding (relative to the logit scale)",
        diff < 1e-3 * full.abs().max().item(),
        f"{diff:.1e}",
    )

    report.finish()


if __name__ == "__main__":
    main()
