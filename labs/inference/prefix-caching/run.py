"""Replay a shared prefix under different arrival patterns.

Wiki pages
  Chinese: https://dmxsf.com/wiki/inference/prefix-caching/
  English: https://dmxsf.com/en/wiki/inference/prefix-caching/
"""

from dmxsf import lab
from dmxsf.inference.prefix_routing import RoutingResult, replay


def main() -> None:
    _, report = lab.start(
        "Prefix caching: cache reuse versus queues", wiki=["inference/prefix-caching"]
    )
    report.section("Burst and spaced requests")
    burst = [0, 0, 0, 1, 2, 20]
    affinity = replay(burst, hit_ms=4, miss_ms=11, affinity=True)
    balanced = replay(burst, hit_ms=4, miss_ms=11, affinity=False)
    report.check(
        "fixed affinity matches the wiki", affinity == RoutingResult(6, 10, 18), str(affinity)
    )
    report.check(
        "finish-time routing matches the wiki", balanced == RoutingResult(5, 9.5, 13), str(balanced)
    )
    spaced = list(range(0, 60, 10))
    left = replay(spaced, hit_ms=4, miss_ms=11, affinity=True)
    right = replay(spaced, hit_ms=4, miss_ms=11, affinity=False)
    report.check(
        "with idle warm workers both routes agree", left == right == RoutingResult(6, 4, 4)
    )
    report.finish()


if __name__ == "__main__":
    main()
