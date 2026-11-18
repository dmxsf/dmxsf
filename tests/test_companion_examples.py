"""Boundary cases in the CPU exercises that the printed examples do not cover."""

import numpy as np
import pytest

from dmxsf.applications.mcp_messages import catalog_tools, handle, wire_size
from dmxsf.applications.tool_authorization import ReadGrant, ToolProposal, read_record
from dmxsf.architecture.state_space import causal_matrix, recurrent
from dmxsf.evaluation.calibration import selective_rates
from dmxsf.evaluation.privacy import canary_rank, exposure
from dmxsf.evaluation.test_time import at_least_one
from dmxsf.pretraining.dedup import jaccard, shingles
from dmxsf.systems.collectives import ring_all_reduce
from dmxsf.systems.roofline import tiled_bytes


@pytest.mark.parametrize("ranks", [1, 3, 4, 8])
def test_ring_returns_a_global_sum_without_mutating_input(ranks):
    values = np.arange(ranks * ranks * 2).reshape(ranks, -1)
    before = values.copy()
    result, traffic = ring_all_reduce(values)
    assert np.all(result == values.sum(axis=0))
    assert np.array_equal(values, before)
    assert traffic == [2 * values.itemsize] * (2 * (ranks - 1))


def test_ring_and_tiling_reject_incompatible_shapes():
    with pytest.raises(ValueError, match="divisible"):
        ring_all_reduce(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="divide"):
        tiled_bytes(10, 3)


def test_a_zero_decay_erases_earlier_writes_in_both_forms():
    decay = np.array([0.9, 0.5, 0.0, 0.5])
    writes = np.array([[1.0], [2.0], [3.0], [4.0]])
    expected = np.array([[1.0], [2.5], [3.0], [5.5]])
    np.testing.assert_array_equal(recurrent(decay, writes), expected)
    np.testing.assert_array_equal(causal_matrix(decay) @ writes, expected)


def test_mcp_messages_keep_the_identifier_and_count_utf8_bytes():
    request = {"id": 3, "method": "tools/missing"}
    error = handle(request, catalog_tools(1), {})
    assert error["id"] == 3
    assert error["error"]["code"] == -32601
    assert wire_size({"text": "字"}) - wire_size({"text": "a"}) == 2


def test_denied_tool_proposal_never_reads_the_store():
    class Unreadable(dict):
        def get(self, *_):
            pytest.fail("the store was queried before authorization")

    grant = ReadGrant(scope="public")
    assert read_record(ToolProposal("read", "private", "item"), grant, Unreadable()) == (
        "denied",
        None,
    )


def test_empty_data_and_endpoint_cases_have_distinct_meanings():
    assert selective_rates(np.array([0.7]), np.array([True]), 0.8) == (0.0, None)
    assert jaccard(shingles("short", 2), shingles("tiny", 2)) == 0
    assert at_least_one(1, 0) == 0
    assert at_least_one(1, 1) == 1
    assert at_least_one(0, 10) == 0
    assert canary_rank([1.0, 1.0], 1) == 2
    assert exposure(2, 2) == 0
