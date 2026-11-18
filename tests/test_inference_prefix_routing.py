import pytest

from dmxsf.inference.prefix_routing import RoutingResult, replay


def test_burst_splits_despite_losing_one_hit():
    arrivals = [0, 0, 0, 1, 2, 20]
    assert replay(arrivals, hit_ms=4, miss_ms=11, affinity=True) == RoutingResult(6, 10, 18)
    assert replay(arrivals, hit_ms=4, miss_ms=11, affinity=False) == RoutingResult(5, 9.5, 13)


def test_reject_unsorted_arrivals():
    with pytest.raises(ValueError, match="ordered arrivals"):
        replay([2, 1], hit_ms=4, miss_ms=11, affinity=True)
