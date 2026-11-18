"""Replay a synthetic release and check that aggregate improvement cannot hide a slice regression.

Wiki pages
  Chinese: https://dmxsf.com/wiki/evaluation/production-evaluation/
  English: https://dmxsf.com/en/wiki/evaluation/production-evaluation/
"""

from dmxsf import lab
from dmxsf.evaluation.production import Pair, release_blockers, summarize


def main() -> None:
    _, report = lab.start(
        "Production evaluation: replay and release gates",
        wiki=["evaluation/production-evaluation"],
    )
    groups = {
        "common": [Pair(True, True)] * 70
        + [Pair(False, True)] * 12
        + [Pair(True, False)] * 3
        + [Pair(False, False)] * 5,
        "critical": [Pair(True, True)] * 5
        + [Pair(False, True)]
        + [Pair(True, False)] * 3
        + [Pair(False, False)],
        "new locale": [Pair(True, True)] * 2,
    }
    results = {name: summarize(pairs) for name, pairs in groups.items()}
    overall = summarize(pair for pairs in groups.values() for pair in pairs)
    report.section("Overall and per-slice changes")
    for name, result in {"overall": overall, **results}.items():
        report.say(f"{name}: n={result.count}, change={result.change:+.1%}")
    blockers = release_blockers(results, min_cases=10, max_loss=0.10)
    report.check("overall change is positive", overall.change > 0)
    report.check("critical slice blocks release", blockers.get("critical") == "observed regression")
    report.check(
        "missing evidence blocks release", blockers.get("new locale") == "insufficient samples"
    )
    report.finish()


if __name__ == "__main__":
    main()
