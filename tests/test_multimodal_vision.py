"""Multimodal: patch merging, dynamic resolution, KV cache, FLOPs, LLaVA's stages."""

import pytest
import torch
import torch.nn as nn

from dmxsf.multimodal import vision


def test_merge_patches_groups_two_by_two_blocks_in_row_major_order():
    grid_h, grid_w, d = 4, 6, 3
    x = torch.arange(grid_h * grid_w * d, dtype=torch.float32).view(grid_h * grid_w, d)
    merged = vision.merge_patches(x, grid_h, grid_w, 2)
    assert merged.shape == (6, 12)
    patch = lambda r, c: x[r * grid_w + c]  # noqa: E731
    # the second block of the first block-row covers patches (0,2) (0,3) (1,2) (1,3)
    assert torch.equal(merged[1], torch.cat([patch(0, 2), patch(0, 3), patch(1, 2), patch(1, 3)]))
    # the first block of the second block-row covers patches (2,0) (2,1) (3,0) (3,1)
    assert torch.equal(merged[3], torch.cat([patch(2, 0), patch(2, 1), patch(3, 0), patch(3, 1)]))


def test_merge_with_a_factor_of_one_is_the_identity():
    x = torch.randn(12, 5)
    assert torch.equal(vision.merge_patches(x, 3, 4, 1), x)


def test_projector_parameter_count():
    proj = vision.projector(4096, 2560)
    assert sum(p.numel() for p in proj.parameters()) == 27_269_632
    assert sum(p.numel() for p in vision.projector(8, 4, hidden=16).parameters()) == (
        8 * 16 + 16
    ) + (16 * 4 + 4)


def test_splice_replaces_exactly_the_placeholder_rows():
    embed = nn.Embedding(10, 4)
    visual = torch.randn(3, 4)
    ids = torch.tensor([1, 2, -1, -1, -1, 3])
    out = vision.splice(ids, embed, visual, -1)
    assert torch.equal(out[2:5], visual)
    assert torch.equal(out[[0, 1, 5]], embed(torch.tensor([1, 2, 3])))


@pytest.mark.parametrize(
    ("cfg", "size", "expected"),
    [
        # (height, width) -> (resized height, resized width, tokens), the numbers on the page
        (vision.QWEN2_VL, (768, 1024), (756, 1036, 999)),
        (vision.QWEN2_VL, (1080, 1920), (1092, 1932, 2691)),
        (vision.QWEN2_VL, (3024, 4032), (3024, 4032, 15552)),
        (vision.QWEN3_VL, (224, 224), (256, 256, 64)),  # below min_pixels: enlarged
        (vision.QWEN3_VL, (1080, 1920), (1088, 1920, 2040)),
        (vision.QWEN3_VL, (3024, 4032), (3008, 4032, 11844)),
    ],
)
def test_visual_tokens_match_the_page(cfg, size, expected):
    t = vision.visual_tokens(cfg, *size)
    assert (t.height, t.width, t.tokens) == expected
    assert t.patches == t.tokens * cfg.merge**2


def test_fixed_resolution_ignores_the_image_size():
    sizes = [(100, 100), (1080, 1920), (4000, 300)]
    assert {vision.visual_tokens(vision.LLAVA_15, *s).tokens for s in sizes} == {576}


def test_smart_resize_rounds_halves_to_even_like_python():
    assert vision.smart_resize(42, 100, 28, 1, 10**9)[0] == 56  # 1.5 -> 2
    assert vision.smart_resize(70, 100, 28, 1, 10**9)[0] == 56  # 2.5 -> 2, not 3


def test_smart_resize_stays_in_budget_and_on_the_grid():
    for h in (30, 333, 1234, 6000):
        for w in (45, 500, 2048, 9000):
            for lo, hi in ((65_536, 16_777_216), (3136, 1_000_000)):
                hb, wb = vision.smart_resize(h, w, 32, lo, hi)
                assert hb % 32 == 0 and wb % 32 == 0 and hb > 0 and wb > 0
                assert hb * wb <= hi
                assert hb * wb >= min(lo, hi) - 32 * (
                    hb + wb
                )  # rounding may undershoot by a margin


def test_cap_tokens_limits_the_token_count():
    for cap in (64, 256, 1024):
        assert vision.visual_tokens(vision.QWEN3_VL, 3024, 4032, cap).tokens <= cap
    uncapped = vision.visual_tokens(vision.QWEN3_VL, 3024, 4032)
    assert vision.visual_tokens(vision.QWEN3_VL, 3024, 4032, 10**6).tokens == uncapped.tokens


def test_kv_bytes_per_token():
    assert vision.LLAVA_15.kv_bytes_per_token == 512 * 1024
    assert vision.QWEN2_VL.kv_bytes_per_token == 56 * 1024
    assert vision.QWEN3_VL.kv_bytes_per_token == 144 * 1024


def test_flops_grow_and_the_vision_share_rises_with_tokens():
    shares = [vision.vit_share(n) for n in (64, 256, 1024, 4096, 16384)]
    assert shares == sorted(shares)
    assert round(100 * shares[0]) == 26 and round(100 * shares[-1]) == 70
    # the quadratic term makes doubling the tokens more than double the work, at most quadruple it
    for fn in (vision.vit_flops, vision.llm_flops):
        assert 2 < fn(4096) / fn(2048) < 4


def test_llama_parameter_count():
    assert vision.llama_params(32, 4096, 11008, 32000) == 6_738_415_616
    assert vision.llama_params(32, 4096, 11008, 32000, tied=True) == 6_738_415_616 - 32000 * 4096


def test_llava_stages():
    v, p, lm = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)
    n = sum(q.numel() for q in p.parameters())
    assert vision.set_llava_stage(v, p, lm, 1) == n
    assert not any(q.requires_grad for q in v.parameters())
    assert not any(q.requires_grad for q in lm.parameters())
    assert vision.set_llava_stage(v, p, lm, 2) == 2 * n
    assert not any(q.requires_grad for q in v.parameters())
    with pytest.raises(ValueError):
        vision.set_llava_stage(v, p, lm, 3)
