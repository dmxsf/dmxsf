"""SFT mechanics, the intent task, and generation with a KV cache."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.inference.generate import generate, sample_logits
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


def tiny(pos="learned", **kw):
    torch.manual_seed(0)
    return GPT(small_gpt(block_size=64, n_layer=2, n_head=2, d_model=32, pos=pos, **kw)).eval()


def test_build_example_masks_only_the_prompt():
    ids, labels = build_example([1, 2, 3], [7, 8], mask_prompt=True)
    assert ids == [1, 2, 3, 7, 8] and labels == [IGNORE] * 3 + [7, 8]
    assert build_example([1, 2, 3], [7, 8], mask_prompt=False)[1] == [1, 2, 3, 7, 8]


def test_collate_shifts_and_ignores_padding():
    a = build_example([1, 2], [3])
    b = build_example([4, 5, 6], [7, 8])
    x, y = collate([a, b])
    assert x.shape == y.shape == (2, 4)
    assert y[0].tolist() == [IGNORE, 3, IGNORE, IGNORE]  # the answer; then padding
    assert y[1].tolist() == [IGNORE, IGNORE, 7, 8]
    assert x[0].tolist() == [1, 2, 3, 0]  # padded with token 0, and never supervised


def test_sft_loss_is_a_mean_over_supervised_tokens():
    model = tiny()
    batch = collate([build_example([1, 2, 3, 4], [9]), build_example([5, 6], [7, 8, 9, 10, 11])])
    x, y = batch
    logits = model(x)
    ce = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=IGNORE, reduction="none")
    n = (y != IGNORE).sum(dim=1)
    assert n.tolist() == [1, 5]
    token_mean = ce.sum() / n.sum()
    sequence_mean = (ce.sum(dim=1) / n).mean()
    assert torch.allclose(sft_loss(model, batch), token_mean, atol=1e-6)
    assert torch.allclose(sft_loss(model, batch, per_sequence=True), sequence_mean, atol=1e-6)
    assert not torch.allclose(token_mean, sequence_mean, atol=1e-4)  # unequal answers: they differ


def test_equal_length_answers_make_the_two_averages_agree():
    model = tiny()
    batch = collate([build_example([1, 2, 3], [9, 8]), build_example([4, 5], [7, 6])])
    assert torch.allclose(
        sft_loss(model, batch), sft_loss(model, batch, per_sequence=True), atol=1e-6
    )


def test_sample_batch_depends_only_on_seed_and_step():
    examples = [build_example([i, i + 1], [i + 2]) for i in range(20)]
    a = sample_batch(examples, 3, 4, seed=1)
    b = sample_batch(examples, 3, 4, seed=1)
    c = sample_batch(examples, 4, 4, seed=1)
    assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))
    assert not torch.equal(a[0], c[0])


@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_packing_with_masks_and_restarted_positions_is_exact(pos):
    model = tiny(pos)
    docs = [torch.randint(0, 256, (n,)) for n in (7, 12, 5)]
    with torch.no_grad():
        packed = forward_packed(model, torch.cat(docs)[None], [len(d) for d in docs])[0]
        separate = torch.cat([model(d[None])[0] for d in docs])
    assert torch.allclose(packed, separate, atol=1e-5)


def test_packing_without_the_mask_changes_the_numbers():
    model = tiny("rope")
    docs = [torch.randint(0, 256, (n,)) for n in (7, 12)]
    with torch.no_grad():
        naive = model(torch.cat(docs)[None])[0]
        separate = torch.cat([model(d[None])[0] for d in docs])
    assert not torch.allclose(naive[7:], separate[7:], atol=1e-3)  # the second document leaks
    assert torch.allclose(naive[:7], separate[:7], atol=1e-5)  # the first has nothing before it


def test_forward_packed_rejects_unsupported_positions():
    with pytest.raises(ValueError):
        forward_packed(tiny("alibi"), torch.zeros(1, 4, dtype=torch.long), [4])


def test_padding_and_packing_shares():
    rng = np.random.default_rng(0)
    lens = np.clip(rng.lognormal(4.3, 0.7, 4000), 8, 500).astype(int).tolist()
    random_share = padding_share(lens, 16, grouped=False)
    grouped_share = padding_share(lens, 16, grouped=True)
    assert random_share > 0.4 > 0.1 > grouped_share > 0
    assert padding_share([10] * 64, 16, grouped=False) == 0
    assert packing_share(lens, 512) < 0.05  # short documents pack almost perfectly
    assert packing_share([300, 300], 512) == pytest.approx(1 - 600 / 1024)


def test_finetune_learns_a_tiny_task():
    model = tiny()
    examples = [build_example(list(b"ab"), list(b"c")), build_example(list(b"xy"), list(b"z"))]
    losses = finetune(model, lambda s: sample_batch(examples, s, 4, 0), 60, 3e-3)
    assert losses[-1] < 0.2 * losses[0]


def test_intents_are_well_formed():
    texts = [intents.all_texts(label) for label in intents.LABELS]
    assert len(intents.LABELS) == 12 and all(len(t) == 1600 == len(set(t)) for t in texts)
    flat = [t for group in texts for t in group]
    assert len(flat) == len(set(flat))  # no message belongs to two intents
    assert len({len(intents.render(t)) for t in flat}) == 1  # fixed-width prompts
    words = [w for group in (*intents.TOPICS.values(), *intents.PROBLEMS.values()) for w in group]
    assert len(words) == len(set(words)) == 56  # no phrase means two things


def test_make_dataset_respects_size_and_exclusion():
    train = intents.make_dataset(5, seed=0)
    assert len(train) == 60 and len({t for t, _ in train}) == 60
    test = intents.make_dataset(7, seed=1, exclude={t for t, _ in train})
    assert not {t for t, _ in train} & {t for t, _ in test}
    assert train == intents.make_dataset(5, seed=0)  # deterministic


def test_generate_with_cache_matches_recomputing_everything():
    model = tiny()
    prompt = torch.randint(0, 256, (3, 9))
    cached = generate(model, prompt, 10)
    ids = prompt
    with torch.no_grad():
        for _ in range(10):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(cached, ids[:, 9:])


def test_generate_stops_and_pads_with_the_stop_token():
    model = tiny()
    prompt = torch.randint(0, 256, (2, 5))
    free = generate(model, prompt, 6)
    stop = int(free[0, 2])  # make the third token of row 0 the stop token
    out = generate(model, prompt, 6, stop=stop)
    first = (free[0] == stop).nonzero()[0].item()
    assert torch.equal(out[0, : first + 1], free[0, : first + 1])
    assert (out[0, first:] == stop).all()


def test_sample_logits_filters():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    assert sample_logits(logits, temperature=0).item() == 0
    g = torch.Generator().manual_seed(0)
    assert all(sample_logits(logits, 1.0, top_k=1, generator=g).item() == 0 for _ in range(20))
    # top_p smaller than the best token's probability still keeps that token
    assert all(sample_logits(logits, 1.0, top_p=0.01, generator=g).item() == 0 for _ in range(20))
    seen = {sample_logits(logits, 1.0, top_k=2, generator=g).item() for _ in range(200)}
    assert seen == {0, 1}
    # nucleus: probabilities of logits [2, 1, .5, -1] are about .60 .22 .13 .03 -> top_p .7 keeps two
    seen = {sample_logits(logits, 1.0, top_p=0.7, generator=g).item() for _ in range(300)}
    assert seen == {0, 1}
