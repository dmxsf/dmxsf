"""Lab: Collective communication: simulate a ring all-reduce.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/collective-communication/
  English: https://dmxsf.com/en/wiki/systems/collective-communication/
"""

import numpy as np

from dmxsf import lab
from dmxsf.systems.collectives import alpha_beta_seconds, ring_all_reduce


def main() -> None:
    args, report = lab.start(
        "Collective communication: simulate a ring all-reduce",
        wiki=["systems/collective-communication"],
    )
    report.section("1. Reduce-scatter then all-gather")
    ranks, width = 8, 64
    values = np.random.default_rng(args.seed).integers(-10, 10, size=(ranks, width), dtype=np.int32)
    out, traffic = ring_all_reduce(values)
    expected = values.sum(axis=0)
    report.check("every rank receives the elementwise global sum", bool(np.all(out == expected)))
    report.check(
        "eight ranks need 14 rounds and send 1.75 vectors per rank",
        len(traffic) == 14 and sum(traffic) == width * values.itemsize * 1.75,
        f"{len(traffic)} rounds, {sum(traffic)} bytes per rank",
    )
    report.section("2. Count rounds and bytes separately")
    cost = alpha_beta_seconds(traffic, 5e-6, 100e9)
    report.check(
        "small messages pay mainly the 14 startup latencies",
        abs(cost - 70.00448e-6) < 1e-12,
        f"{cost * 1e6:.6f} us",
    )
    report.finish()


if __name__ == "__main__":
    main()
