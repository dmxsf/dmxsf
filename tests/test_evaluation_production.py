import pytest

from dmxsf.evaluation.production import Pair, Replay, release_blockers, summarize


def test_replay_counts_only_actual_version_changes():
    replay = summarize([Pair(True, True), Pair(False, False), Pair(False, True)])
    assert replay == Replay(3, 1, 0)
    assert replay.change == pytest.approx(1 / 3)


def test_release_gate_distinguishes_insufficient_evidence_and_loss():
    results = {"healthy": Replay(10, 1, 1), "small": Replay(2, 0, 0), "loss": Replay(10, 0, 2)}
    assert release_blockers(results, min_cases=10, max_loss=0.10) == {
        "small": "insufficient samples",
        "loss": "observed regression",
    }


def test_empty_replay_requires_data():
    with pytest.raises(ValueError, match="paired request"):
        summarize([])
