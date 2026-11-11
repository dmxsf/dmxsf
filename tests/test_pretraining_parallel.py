"""The page's parallel-training arithmetic: ring all-reduce, pipeline bubbles, communication costs."""

from itertools import pairwise

import numpy as np
import pytest

from dmxsf.pretraining import parallel as pl

PSI = 6.738e9  # Llama 2 7B, the page's example
H, LAYERS, SEQ, BF16, GB = 4096, 32, 4096, 2, 1e9
PEAK, MFU, NVLINK, IB = 989e12, 0.40, 160 * GB, 50 * GB


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_ring_all_reduce_is_a_sum_and_sends_two_n_minus_one_over_n(n):
    rng = np.random.default_rng(0)
    vectors = [rng.standard_normal(1600) for _ in range(n)]
    out, sent = pl.ring_all_reduce(vectors)
    assert all(np.allclose(o, sum(vectors)) for o in out)
    assert sent == [2 * (n - 1) * 1600 // n] * n


def test_ring_handles_one_rank_and_uneven_chunks():
    out, sent = pl.ring_all_reduce([np.arange(5.0)])
    assert out[0].tolist() == [0, 1, 2, 3, 4] and sent == [0]
    vectors = [np.arange(7.0) * (r + 1) for r in range(3)]
    out, _ = pl.ring_all_reduce(vectors)
    assert all(np.allclose(o, sum(vectors)) for o in out)


# the page: (p, m) -> (total time, idle share) with 1 time unit forward and 2 backward
PAGE_GPIPE = {(4, 4): (21, 0.429), (4, 8): (33, 0.273), (4, 16): (57, 0.158), (4, 32): (105, 0.086)}


@pytest.mark.parametrize(("p", "m"), [*PAGE_GPIPE, (8, 32)])
@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_pipeline_idle_share_is_p_minus_one_over_m_plus_p_minus_one(p, m, schedule):
    t = pl.pipeline(p, m, schedule)
    assert t.total == (m + p - 1) * 3
    assert t.idle == pytest.approx((p - 1) / (m + p - 1))
    assert t.idle == pytest.approx(pl.idle_fraction(p, m))
    if (p, m) in PAGE_GPIPE:
        assert (t.total, round(t.idle, 3)) == PAGE_GPIPE[(p, m)]


def test_1f1b_keeps_at_most_p_micro_batches_alive_gpipe_keeps_all():
    assert pl.pipeline(4, 8, "gpipe").peak == [8, 8, 8, 8]
    assert pl.pipeline(4, 8, "1f1b").peak == [
        4,
        3,
        2,
        1,
    ]  # the page: 8 against 4 on the first stage
    assert pl.pipeline(4, 2, "1f1b").peak == [2, 2, 2, 1]  # fewer micro-batches than stages


def test_pipeline_respects_dependencies_and_one_job_at_a_time():
    t = pl.pipeline(3, 5, "1f1b", t_fwd=1.0, t_bwd=1.5)
    end = {(s, k, i): e for s, k, i, _, e in t.ops}
    for s, kind, i, start, _ in t.ops:
        if kind == "F" and s:
            assert start >= end[(s - 1, "F", i)]
        if kind == "B" and s < 2:
            assert start >= end[(s + 1, "B", i)]
        if kind == "B":
            assert start >= end[(s, "F", i)]
    for s in range(3):
        spans = sorted((a, b) for st, _, _, a, b in t.ops if st == s)
        assert all(x[1] <= y[0] + 1e-12 for x, y in pairwise(spans))


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="schedule"):
        pl.pipeline(2, 2, "zero-bubble")


def test_interleaving_divides_the_bubble_and_min_micro_batches():
    assert pl.bubble_ratio(8, 16, v=2) == pl.bubble_ratio(8, 16) / 2
    assert pl.min_microbatches(8, 0.10) == 63  # the page's exercise 2
    assert pl.idle_fraction(8, 63) <= 0.10 < pl.idle_fraction(8, 62)


def test_page_communication_numbers():
    grads = PSI * BF16
    assert round(pl.ring_bytes(64, grads) / GB, 1) == 26.5
    assert round(pl.zero3_bytes(64, grads) / GB, 1) == 39.8
    assert pl.zero3_bytes(64, grads) == pytest.approx(1.5 * pl.ring_bytes(64, grads))
    assert [round(pl.ring_bytes(n, grads) / GB, 1) for n in (8, 1024)] == [23.6, 26.9]  # exercise 1
    act = SEQ * H * BF16
    rows = []
    for t in (2, 4, 8):
        tp = pl.tensor_parallel_bytes(t, LAYERS, act)
        compute = 6 * PSI * SEQ / t / (PEAK * MFU)
        rows.append(
            (
                round(tp / GB, 1),
                round(tp / NVLINK * 1e3),
                round(tp / IB * 1e3),
                round(compute * 1e3),
            )
        )
    assert rows == [(4.3, 27, 86, 209), (6.4, 40, 129, 105), (7.5, 47, 150, 52)]
    assert round(pl.pipeline_boundary_bytes(act) / 1e6, 1) == 67.1


def test_page_scaling_table_and_tokens_to_hide():
    comm = pl.ring_bytes(64, PSI * BF16) / IB
    assert round(comm, 2) == 0.53
    eff = {}
    for gpus in (64, 128, 256, 512, 1024):
        compute, exposed, e = pl.scaling_row(gpus, 4 * 2**20, PSI, comm, PEAK, MFU)
        eff[gpus] = (round(compute, 2), round(exposed, 2), round(e, 3))
    assert eff[64] == (6.70, 0.0, 1.0)
    assert eff[512] == (0.84, 0.0, 1.0)
    assert eff[1024] == (0.42, 0.25, 0.625)
    assert round(pl.tokens_to_hide(comm, PSI, PEAK, MFU)) == 7788


def test_tensor_parallel_cost_grows_with_degree_and_matches_the_per_token_formula():
    act = SEQ * H * BF16
    bytes_ = [pl.tensor_parallel_bytes(t, LAYERS, act) for t in (2, 4, 8)]
    assert bytes_ == sorted(bytes_) and bytes_[-1] < 2 * bytes_[0]  # 2(t-1)/t tends to 2
    for t in (2, 4, 8):  # one token, one layer: 16 (t - 1) / t h bytes
        assert pl.tensor_parallel_bytes(t, 1, H * BF16) == pytest.approx(16 * (t - 1) / t * H)
