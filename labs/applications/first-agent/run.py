"""Run a scripted, read-only tool loop and check its stopping behavior.

Wiki pages
  Chinese: https://dmxsf.com/wiki/applications/first-agent/
  English: https://dmxsf.com/en/wiki/applications/first-agent/
"""

from dmxsf import lab
from dmxsf.applications.first_agent import run, scripted_policy


def main() -> None:
    _, report = lab.start("First agent: a bounded tool loop", wiki=["applications/first-agent"])
    report.section("A local library lookup")
    answer, trace = run("atlas", scripted_policy, 2)
    report.check("answer follows the tool observation", answer == "shelf A", str(trace))
    answer, _ = run("missing", scripted_policy, 2)
    report.check("missing input gets an explicit result", answer == "not in the catalog")
    try:
        run("atlas", scripted_policy, 1)
    except RuntimeError as exc:
        report.check("bounded run stops", str(exc) == "step budget exhausted")
    else:
        report.check("bounded run stops", False)
    report.finish()


if __name__ == "__main__":
    main()
