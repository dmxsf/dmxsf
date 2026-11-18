"""Check storage and ideal copy time without requiring a GPU.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/hardware-basics/
  English: https://dmxsf.com/en/wiki/systems/hardware-basics/
"""

from dmxsf import lab
from dmxsf.systems.hardware_basics import copy_seconds, tensor_bytes


def main() -> None:
    _, report = lab.start(
        "Hardware basics: storage and copy time", wiki=["systems/hardware-basics"]
    )
    report.section("One square matrix")
    fp16 = tensor_bytes((1024, 1024), 2)
    fp32 = tensor_bytes((1024, 1024), 4)
    report.check("two element sizes", (fp16, fp32) == (2**21, 2**22), f"{fp16}, {fp32} bytes")
    ideal_ms = 1000 * copy_seconds(fp16, 512 * 2**20)
    report.check("ideal copy bound", ideal_ms == 3.90625, f"{ideal_ms:.3f} ms")
    report.finish()


if __name__ == "__main__":
    main()
