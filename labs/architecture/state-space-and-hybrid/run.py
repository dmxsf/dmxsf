"""Lab: State space and hybrids: equivalent scans and memory.

Wiki pages
  Chinese: https://dmxsf.com/wiki/architecture/state-space-and-hybrid/
  English: https://dmxsf.com/en/wiki/architecture/state-space-and-hybrid/
"""

import numpy as np

from dmxsf import lab
from dmxsf.architecture.state_space import causal_matrix, hybrid_memory, recurrent


def main() -> None:
    args, report = lab.start(
        "State space and hybrids: equivalent scans and memory",
        wiki=["architecture/state-space-and-hybrid"],
    )
    report.section("1. Recurrence and explicit causal mixing")
    rng = np.random.default_rng(args.seed)
    decay, writes = rng.uniform(0.7, 0.99, 32), rng.normal(size=(32, 8))
    loop = recurrent(decay, writes)
    explicit = causal_matrix(decay) @ writes
    gap = np.max(np.abs(loop - explicit))
    report.check(
        "the two evaluation orders agree in float64", gap < 1e-12, f"largest gap {gap:.1e}"
    )
    decay[16] = 0
    report.check(
        "a zero decay resets history at that step",
        np.array_equal(recurrent(decay, writes)[16], writes[16]),
    )
    report.section("2. A hybrid cache ledger")
    short = hybrid_memory(4096, 4096, 8, 32768, 24)
    long = hybrid_memory(8192, 4096, 8, 32768, 24)
    report.check(
        "doubling sequence length doubles KV storage and leaves the state fixed",
        short == (134_217_728, 786_432) and long == (2 * short[0], short[1]),
        f"{short} vs {long} bytes",
    )
    report.finish()


if __name__ == "__main__":
    main()
