import pytest

from dmxsf.inference.first_service import generate


def test_stops_at_end_or_bound():
    assert generate("one", 1) == ["two"]
    assert generate("one", 8) == ["two", "three"]
    assert generate("three", 8) == []


def test_json_boolean_is_not_a_token_count():
    with pytest.raises(ValueError, match="integer limit"):
        generate("one", True)
