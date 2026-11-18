"""Lab: GPU roofline: bytes, FLOPs and tile reuse.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/gpu-and-roofline/
  English: https://dmxsf.com/en/wiki/systems/gpu-and-roofline/
"""

from dmxsf import lab
from dmxsf.systems.roofline import linear_intensity, roofline, tiled_bytes


def main() -> None:
    _args, report = lab.start(
        "GPU roofline: bytes, FLOPs and tile reuse", wiki=["systems/gpu-and-roofline"]
    )
    report.section("1. Compute ceilings and ridges")
    for name, peak, bandwidth, ridge in [
        ("A100 80GB SXM", 312e12, 2039e9, 153.0),
        ("H100 SXM", 989e12, 3.35e12, 295.2),
    ]:
        value = peak / bandwidth
        report.check(f"{name} ridge", round(value, 1) == ridge, f"{value:.1f} FLOP/byte")
    report.section("2. Weights shared across tokens")
    sizes = (1, 16, 64, 153, 295, 512, 2048, 32768)
    values = [linear_intensity(n, 4096, 4096) for n in sizes]
    report.table(
        ["batch", "FLOP/byte", "H100 ceiling TFLOP/s"],
        [
            [n, f"{i:.1f}", f"{roofline(i, 989e12, 3.35e12) / 1e12:.1f}"]
            for n, i in zip(sizes, values, strict=True)
        ],
    )
    report.check(
        "one token remains bandwidth-bound and a large batch reaches the compute ceiling",
        round(values[0], 1) == 1.0 and roofline(values[-1], 989e12, 3.35e12) == 989e12,
    )
    report.check(
        "INT4 weight-only intensity is about four FLOP/byte for one token",
        round(linear_intensity(1, 4096, 4096, 0.5), 1) == 4.0,
    )
    report.section("3. Tiling changes traffic with the same FLOPs")
    report.check(
        "128-square matrices, one output store and 32-square input tiles",
        tiled_bytes(128, 32) == 294_912 and tiled_bytes(128, 128) == 98_304,
        "294,912 vs 98,304 bytes",
    )
    report.finish()


if __name__ == "__main__":
    main()
