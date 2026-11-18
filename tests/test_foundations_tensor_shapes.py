import pytest

from dmxsf.foundations.tensor_shapes import linear, relu


def test_linear_reduces_feature_axis_and_broadcasts_bias():
    values = linear([[1, 2, 0], [-1, 0, 2]], [[2, 0], [1, -1], [0, 2]], [1, -1])
    assert values == [[5, -3], [-1, 3]]
    assert relu(values) == [[5, 0], [0, 3]]


def test_shape_mismatch_is_explicit():
    with pytest.raises(ValueError, match="input width"):
        linear([[1]], [[1], [2]], [0])
