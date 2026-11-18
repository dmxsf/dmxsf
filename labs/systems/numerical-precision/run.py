"""Lab: Numerical precision: rounding, overflow and loss scaling.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/numerical-precision/
  English: https://dmxsf.com/en/wiki/systems/numerical-precision/
"""

import math

import torch

from dmxsf import lab
from dmxsf.systems.precision import round_scaled_gradient, stalled_sum


def main() -> None:
    _args, report = lab.start(
        "Numerical precision: rounding, overflow and loss scaling",
        wiki=["systems/numerical-precision"],
    )
    report.section("1. Range and rounding")
    report.check(
        "adding one stalls at 256 in BF16 and 2048 in FP16",
        (stalled_sum(torch.bfloat16), stalled_sum(torch.float16)) == (256, 2048),
        "256 and 2048",
    )
    x = torch.tensor([12.0, 11.0, 0.0], dtype=torch.float16)
    unstable = x.exp() / x.exp().sum()
    stable = x.softmax(dim=0)
    report.check(
        "FP16 naive softmax overflows, shifted softmax is finite",
        bool(torch.isnan(unstable[0]) and torch.isfinite(stable).all())
        and math.isclose(float(stable[0]), 0.731, abs_tol=0.001),
        str(stable.tolist()),
    )
    report.section("2. Small gradients and master weights")
    tiny = 2**-27
    unscaled, scaled = round_scaled_gradient(tiny, 1), round_scaled_gradient(tiny, 2**12)
    report.check(
        "power-of-two scaling preserves this tiny FP16 update",
        unscaled == 0 and scaled == tiny,
        f"{unscaled:.2e} vs {scaled:.2e}",
    )
    bf16 = torch.tensor(1.0, dtype=torch.bfloat16)
    fp32 = torch.tensor(1.0, dtype=torch.float32)
    for _ in range(1000):
        bf16 += 1e-4
        fp32 += 1e-4
    report.check(
        "FP32 accumulates small steps lost by BF16 per-step rounding",
        float(bf16) == 1 and abs(float(fp32) - 1.1) < 1e-4,
        f"{float(bf16):.4f} vs {float(fp32):.4f}",
    )
    report.finish()


if __name__ == "__main__":
    main()
