"""Compare open-loop execution with replanning after a map error.

Wiki pages
  Chinese: https://dmxsf.com/wiki/multimodal/world-models/
  English: https://dmxsf.com/en/wiki/multimodal/world-models/
"""

from dmxsf import lab
from dmxsf.multimodal.world_models import traverse


def main() -> None:
    _, report = lab.start(
        "World models: plan, observe, and replan", wiki=["multimodal/world-models"]
    )
    report.section("Hidden wall")
    fixed = traverse(False)
    feedback = traverse(True)
    report.say(f"open loop: {fixed}")
    report.say(f"replanned: {feedback}")
    report.check("open loop stalls", fixed[-1] == (0, 1))
    report.check(
        "feedback reaches the goal in six actions", feedback[-1] == (0, 3) and len(feedback) == 7
    )
    report.finish()


if __name__ == "__main__":
    main()
