"""Lab: From RNN to attention: gradients and retrieval paths.

Wiki pages
  Chinese: https://dmxsf.com/wiki/architecture/from-rnn-to-attention/
  English: https://dmxsf.com/en/wiki/architecture/from-rnn-to-attention/
"""

import math

import torch

from dmxsf import lab
from dmxsf.architecture.recurrent import attention_read, recurrent_gradient


def main() -> None:
    args, report = lab.start(
        "From RNN to attention: gradients and retrieval paths",
        wiki=["architecture/from-rnn-to-attention"],
    )
    report.section("1. Gradient paths through time")
    plain = recurrent_gradient(100, 0.5)
    gated = recurrent_gradient(100, 0.99, gated=True)
    report.check(
        "backprop agrees with multiplying 100 local derivatives",
        math.isclose(plain, 0.5**100) and math.isclose(gated, 0.99**100),
        f"tanh {plain:.3e}, gated {gated:.3f}",
    )
    report.section("2. Read an earlier value without the recurrent chain")
    keys = torch.eye(16, dtype=torch.float64)
    values = torch.arange(16, dtype=torch.float64)
    target = args.seed % len(keys)
    read, weights = attention_read(keys, values, target, 12)
    report.check(
        "the most relevant earlier key receives over 99.9% of the mass",
        int(weights.argmax()) == target and float(weights[target]) > 0.999,
        f"target {target}, recovered {float(read):.4f}",
    )
    read_error = abs(float(read) - target)
    report.check(
        "the returned value is near the selected value",
        read_error < 0.002,
        f"absolute error {read_error:.6f}",
    )
    report.finish()


if __name__ == "__main__":
    main()
