import pytest

from dmxsf.applications.context_memory import Record, recent_groups, resolve


@pytest.mark.parametrize(
    ("budget", "expected"), [(0, []), (25, [2]), (30, [1, 2]), (48, [0, 1, 2])]
)
def test_budget_trims_a_whole_suffix(budget, expected):
    assert recent_groups([18, 22, 8], budget) == expected


def test_current_value_and_latest_record_have_precedence():
    records = [Record(3, None), Record(1, "morning"), Record(2, "evening")]
    assert resolve("afternoon", records) == "afternoon"
    assert resolve(None, records) is None
    assert resolve(None, records[1:]) == "evening"
