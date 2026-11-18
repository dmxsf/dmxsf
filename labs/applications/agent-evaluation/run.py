"""Compare task reliability and verify simulated calendar outcomes.

Wiki pages
  Chinese: https://dmxsf.com/wiki/applications/agent-evaluation/
  English: https://dmxsf.com/en/wiki/applications/agent-evaluation/
"""

from dmxsf import lab
from dmxsf.applications.agent_metrics import matches_calendar, pass_all


def main() -> None:
    _, report = lab.start(
        "Agent evaluation: repeatability and terminal state", wiki=["applications/agent-evaluation"]
    )
    report.section("Repeated attempts")
    concentrated = [(True, True), (False, False)]
    spread = [(True, False), (False, True)]
    report.check(
        "equal one-attempt success", pass_all(concentrated, 1) == pass_all(spread, 1) == 0.5
    )
    report.check(
        "two-attempt reliability differs",
        pass_all(concentrated, 2) == 0.5 and pass_all(spread, 2) == 0,
    )
    report.section("Four calendar traces")
    expected = [("a", "10:00"), ("b", "14:00")]
    outcomes = [
        expected[::-1],
        [("a", "09:00"), ("b", "14:00")],
        [*expected, ("a", "10:00")],
        [("a", "10:00"), ("b", "15:00")],
    ]
    verified = [matches_calendar(events, expected) for events in outcomes]
    report.check("already-ordered and reordered responses are equivalent", verified[0])
    report.check("only one terminal state is correct", verified == [True, False, False, False])
    report.finish()


if __name__ == "__main__":
    main()
