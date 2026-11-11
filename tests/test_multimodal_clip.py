"""CLIP in miniature: the loss, the toy dataset, the dual encoder and a short training run."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dmxsf.multimodal import clip


def unit(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_loss_matches_the_formula_and_is_symmetric():
    a, b = unit(6, 16, 0), unit(6, 16, 1)
    assert clip.clip_loss(a, b, 14.3).item() == pytest.approx(
        clip.clip_loss_by_hand(a, b, 14.3), abs=1e-5
    )
    assert clip.clip_loss(a, b, 7.0).item() == pytest.approx(
        clip.clip_loss(b, a, 7.0).item(), abs=1e-6
    )


def test_loss_limits():
    a = unit(8, 16, 2)
    assert clip.clip_loss(a, unit(8, 16, 3), 0.0).item() == pytest.approx(math.log(8), abs=1e-6)
    eye = torch.eye(8)  # orthogonal pairs: matched cosine 1, the rest 0
    assert clip.clip_loss(eye, eye, 100.0).item() < 1e-6
    assert clip.clip_loss(eye, eye.roll(1, dims=0), 100.0).item() > 50  # all pairs wrong


def test_class_words_cover_every_caption_once():
    words = clip.class_words(np.arange(clip.N_CLASSES))
    assert words.shape == (clip.N_CLASSES, 4)
    assert len({tuple(w) for w in words}) == clip.N_CLASSES
    assert words.min() == 0 and words.max() == clip.N_WORDS - 1
    # each column draws from its own range of word ids: colours 0-5, shapes 6-9, sizes 10-11, sides 12-13
    assert [(words[:, i].min(), words[:, i].max()) for i in range(4)] == [
        (0, 5),
        (6, 9),
        (10, 11),
        (12, 13),
    ]
    assert clip.caption(0) == "a small red square on the left"
    assert clip.caption(clip.N_CLASSES - 1) == "a large cyan plus on the right"


def class_of(color, shape, size, side):
    return int(np.ravel_multi_index((color, shape, size, side), clip.SHAPE_OF_CLASS))


def test_render_is_deterministic_and_in_range():
    classes = np.arange(clip.N_CLASSES)
    one = clip.render(classes, np.random.default_rng(5))
    two = clip.render(classes, np.random.default_rng(5))
    assert one.shape == (clip.N_CLASSES, 3, clip.CANVAS, clip.CANVAS)
    assert torch.equal(one, two)
    assert one.min() >= 0 and one.max() <= 1


def test_render_follows_the_caption():
    rng = np.random.default_rng(0)

    def draw(*attrs, n=20):
        img = clip.render(np.full(n, class_of(*attrs)), rng).mean(0)  # average out the noise
        mask = img.max(0).values > 0.5
        xs = torch.nonzero(mask)[:, 1].float()
        return img, mask.sum().item(), xs.mean().item()

    red, red_area, _ = draw(0, 0, 0, 0)
    cyan, _, _ = draw(5, 0, 0, 0)
    _, large_area, _ = draw(0, 0, 1, 0)
    _, circle_area, _ = draw(0, 1, 0, 0)
    _, _, left_x = draw(0, 0, 0, 0)
    _, _, right_x = draw(0, 0, 0, 1)
    assert red[0].max() > 0.8 and red[2].max() < 0.3  # red: strong red channel, weak blue
    assert cyan[0].max() < 0.3 and cyan[2].max() > 0.6  # cyan: no red, strong blue
    assert large_area > 3 * red_area  # a large shape covers more than three times the area
    assert circle_area < red_area  # a circle fills less of its bounding square
    assert left_x < clip.CANVAS / 2 < right_x


def test_dual_encoder_outputs_and_temperature():
    learned = clip.DualEncoder()
    assert learned.scale().item() == pytest.approx(1 / 0.07, rel=1e-5)
    assert learned.logit_scale.requires_grad
    fixed = clip.DualEncoder(scale=100.0)
    assert (
        fixed.scale().item() == pytest.approx(100.0, rel=1e-5)
        and not fixed.logit_scale.requires_grad
    )
    capped = clip.DualEncoder()
    capped.logit_scale.data.fill_(10.0)  # e^10 is far above the cap
    assert capped.scale().item() == 100.0
    img = learned.encode_image(clip.render(np.arange(5), np.random.default_rng(0)))
    txt = learned.encode_text(torch.from_numpy(clip.class_words(np.arange(5))))
    assert img.shape == txt.shape == (5, 32)
    assert torch.allclose(img.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(txt.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_training_is_deterministic_and_learns():
    _, first = clip.train(60, 32, None, seed=0)
    _, again = clip.train(60, 32, None, seed=0)
    assert first.history == again.history and len(first.history) == 60
    assert first.history[-1] < first.history[0]
    assert first.accuracy > 5 / clip.N_CLASSES  # well above chance after 60 steps
    assert first.scale != pytest.approx(1 / 0.07)  # the temperature was learned


def test_untrained_model_is_at_chance_and_batches_hold_distinct_captions():
    torch.manual_seed(0)
    assert clip.zero_shot_accuracy(clip.DualEncoder(), per_class=2) < 0.1
    # a batch of 96 distinct captions is every caption once: batch size cannot exceed the 96 classes
    with pytest.raises(ValueError):
        clip.train(1, clip.N_CLASSES + 1, None, seed=0)
