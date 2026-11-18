import numpy as np
import pytest

from dmxsf.systems.kernels import tiled_matmul, traffic_elements


@pytest.mark.parametrize("tiles", [(1, 1, 1), (3, 2, 4), (3, 2, 20), (9, 9, 9)])
def test_edge_tiles_match_matmul_and_expected_traffic(tiles):
    left = np.arange(30).reshape(5, 6)
    right = np.arange(42).reshape(6, 7)
    result, moved = tiled_matmul(left, right, tiles)
    np.testing.assert_array_equal(result, left @ right)
    assert moved == traffic_elements(5, 7, 6, *tiles[:2])


def test_reduction_tile_size_does_not_change_logical_traffic():
    assert traffic_elements(5, 7, 6, 3, 2) == 239
