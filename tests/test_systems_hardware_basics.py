import pytest

from dmxsf.systems.hardware_basics import copy_seconds, tensor_bytes


def test_shape_and_precision_both_determine_storage():
    assert tensor_bytes((1024, 1024), 2) == 2**21
    assert copy_seconds(2**21, 2**29) == pytest.approx(1 / 256)


def test_scalar_and_empty_axis():
    assert tensor_bytes((), 4) == 4
    assert tensor_bytes((2, 0, 3), 4) == 0
