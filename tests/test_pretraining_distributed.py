"""Real multi-process runs (gloo, CPU): gradient averaging, ZeRO and tensor parallelism."""

import pytest
import torch

from dmxsf.pretraining import distributed as D

EQUAL = [16] * 8
UNEQUAL = [16, 14, 12, 12, 10, 8, 8, 4]  # the page's lengths


def test_equal_batches_make_every_way_of_averaging_agree():
    r = D.ddp_weighting("toy", EQUAL)
    assert r["tokens"] == [30, 30, 30, 30]
    assert max(r["error"].values()) < 1e-6  # float32 rounding, as on the page (2.45e-07)


def test_unequal_batches_bias_the_default_average_and_the_recipe_fixes_it():
    r = D.ddp_weighting("toy", UNEQUAL)
    e = r["error"]
    assert r["tokens"] == [28, 22, 16, 10]
    assert 0.36 < e["A: mean of the ranks' means"] < 0.37  # the page: 3.65e-01
    assert e["DDP as shipped"] == pytest.approx(e["A: mean of the ranks' means"], rel=1e-3)
    assert e["gradient accumulation"] == pytest.approx(e["A: mean of the ranks' means"], rel=1e-3)
    assert e["B: divide by all valid tokens"] < 1e-6
    assert e["DDP, loss x world / all tokens"] < 1e-6


def test_data_parallel_zero1_and_zero3_train_the_same_model():
    steps, rows, world = 3, 8, 2
    ref = D.train_reference(steps, rows)
    runs = {stage: D.train_sharded(stage, world, steps, rows) for stage in (0, 1, 3)}
    n = runs[0][0]["n_params"]
    assert n == 139_584 == ref.numel()
    for stage, ranks in runs.items():
        for r in ranks:  # every rank, not only the first, ends with the reference weights
            assert (r["flat"] - ref).abs().max() < 1e-10, stage
    held = {s: ranks[0]["held"] for s, ranks in runs.items()}
    assert held[0] == {"weights": n, "gradients": n, "optimizer": 2 * n}
    assert held[1] == {"weights": n, "gradients": n, "optimizer": n}  # the optimizer state halves
    assert held[3] == {"weights": n // 2, "gradients": n // 2, "optimizer": n}  # everything halves
    sent = {s: ranks[0]["sent"] / n for s, ranks in runs.items()}
    assert sent[0] == pytest.approx(2 * (world - 1) / world)  # an all-reduce
    assert sent[1] == pytest.approx(sent[0])  # reduce-scatter plus all-gather: the same
    assert sent[3] == pytest.approx(1.5 * sent[0], rel=1e-3)  # one more collective's worth
    assert runs[1][0]["calls"] == {"all_reduce": 0, "reduce_scatter": 1, "all_gather": 1}
    assert runs[3][0]["calls"] == {"all_reduce": 0, "reduce_scatter": 4, "all_gather": 8}
    assert runs[0][0]["loss"] == pytest.approx(runs[3][0]["loss"], abs=1e-10)


@pytest.mark.parametrize("world", [2, 4])
def test_tensor_parallel_blocks_equal_the_unsplit_blocks(world):
    for r in D.tensor_parallel_check(world):
        assert r["out"] < 1e-12 and r["x_grad"] < 1e-12 and r["weight_grad"] < 1e-12
        assert (r["forward_all_reduces"], r["backward_all_reduces"]) == (4, 4)  # 2 per layer each
        assert r["matrix_params"] * world == r["full_matrix_params"]


def test_workers_that_fail_raise_in_the_parent():
    with pytest.raises(Exception, match="boom"):
        D.launch(_boom, 2)


def _boom(rank, world):
    raise RuntimeError("boom")


def test_launch_returns_one_result_per_rank_and_all_reduce_adds():
    out = D.launch(_rank_sum, 3)
    assert [o.item() for o in out] == [3.0, 3.0, 3.0]  # 0 + 1 + 2 on every rank
    assert isinstance(out[0], torch.Tensor)


def _rank_sum(rank, world):
    x = torch.tensor(float(rank))
    torch.distributed.all_reduce(x)
    return x
