import pytest

from dmxsf.applications.agent_metrics import matches_calendar, pass_all


def test_tasks_are_equally_weighted_even_with_different_trial_counts():
    assert pass_all([(True, True), (False, False, False, False)], 2) == 0.5
    assert pass_all([(True, True, False)], 2) == pytest.approx(1 / 3)


def test_require_enough_trials_for_each_task():
    with pytest.raises(ValueError, match="at least k"):
        pass_all([(True,), (True, True)], 2)


def test_calendar_preserves_multiplicity():
    expected = [("a", "10:00"), ("b", "14:00")]
    assert matches_calendar(expected[::-1], expected)
    assert not matches_calendar([*expected, expected[0]], expected)
