"""Track the shapes of a tiny linear layer and activation.

Wiki pages
  Chinese: https://dmxsf.com/wiki/foundations/tensors-and-shapes/
  English: https://dmxsf.com/en/wiki/foundations/tensors-and-shapes/
"""

from dmxsf import lab
from dmxsf.foundations.tensor_shapes import linear, relu


def main() -> None:
    _, report = lab.start(
        "Tensor shapes: a linear layer by hand", wiki=["foundations/tensors-and-shapes"]
    )
    report.section("Two positions and three features")
    result = linear([[1, 2, 0], [-1, 0, 2]], [[2, 0], [1, -1], [0, 2]], [1, -1])
    report.check("projection and broadcast bias", result == [[5, -3], [-1, 3]], str(result))
    report.check(
        "ReLU keeps row and column counts", relu(result) == [[5, 0], [0, 3]], str(relu(result))
    )
    report.finish()


if __name__ == "__main__":
    main()
