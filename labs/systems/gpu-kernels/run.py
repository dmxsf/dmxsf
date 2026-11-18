"""Check logical loads against tiled matrix multiplication, including edge tiles.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/gpu-kernels/
  English: https://dmxsf.com/en/wiki/systems/gpu-kernels/
"""

import numpy as np

from dmxsf import lab
from dmxsf.systems.kernels import tiled_matmul, traffic_elements


def main() -> None:
    args, report = lab.start(
        "GPU kernels: tiled GEMM and memory traffic", wiki=["systems/gpu-kernels"]
    )
    rng = np.random.default_rng(args.seed)
    rows, cols, depth = (7, 5, 9) if args.quick else (65, 49, 33)
    left = rng.integers(-4, 5, size=(rows, depth)).astype(np.float32)
    right = rng.integers(-4, 5, size=(depth, cols)).astype(np.float32)
    reference = left @ right
    report.section(f"Multiply ({rows}, {depth}) by ({depth}, {cols})")
    for tiles in ((4, 4, 4), (8, 8, 4), (8, 8, 16)):
        result, moved = tiled_matmul(left, right, tiles)
        report.check(f"{tiles} matches the reference", bool(np.array_equal(reference, result)))
        report.check(
            f"{tiles} matches the traffic formula",
            moved == traffic_elements(rows, cols, depth, *tiles[:2]),
            f"{moved * left.itemsize:,} logical bytes",
        )
    report.finish()


if __name__ == "__main__":
    main()
