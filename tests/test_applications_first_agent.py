import pytest

from dmxsf.applications.first_agent import run, scripted_policy


def test_success_returns_tool_evidence():
    assert run("atlas", scripted_policy, 2) == ("shelf A", (("atlas", "shelf A"),))


def test_unknown_tool_and_exhausted_steps():
    with pytest.raises(ValueError, match="unrecognized"):
        run("atlas", lambda task, history: ("other", task), 2)
    with pytest.raises(RuntimeError, match="budget"):
        run("atlas", scripted_policy, 1)
