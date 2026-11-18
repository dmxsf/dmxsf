"""Measure simulated overlap, policy lag, and filtered batches.

Wiki pages
  Chinese: https://dmxsf.com/wiki/post-training/rollout-and-training/
  English: https://dmxsf.com/en/wiki/post-training/rollout-and-training/
"""

from dmxsf import lab
from dmxsf.post_training.rollout_pipeline import simulate


def main() -> None:
    _, report = lab.start(
        "Reasoning RL systems: rollout and learner scheduling",
        wiki=["post-training/rollout-and-training"],
    )
    report.section("One actor and one learner")
    durations = [3, 3, 9, 2, 2, 2]
    serial = simulate(durations, 6, overlap=False, max_lag=0)
    bounded = simulate(durations, 6, overlap=True, max_lag=1)
    permissive = simulate(durations, 6, overlap=True, max_lag=9)
    for label, trace in (("serial", serial), ("bounded", bounded), ("permissive", permissive)):
        report.say(f"{label}: t={trace.elapsed}, used={trace.accepted}, lag={trace.lags}")
    report.check(
        "serial accepts every batch with zero lag", serial.accepted == 6 and max(serial.lags) == 0
    )
    report.check("bounded run drops stale work", bounded.dropped > 0)
    report.check("overlap saves wall time at permissive lag", permissive.elapsed < serial.elapsed)
    report.finish()


if __name__ == "__main__":
    main()
