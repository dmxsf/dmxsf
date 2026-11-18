"""Preserve tool groups and apply explicit, versioned memory precedence.

Wiki pages
  Chinese: https://dmxsf.com/wiki/applications/context-and-memory/
  English: https://dmxsf.com/en/wiki/applications/context-and-memory/
"""

from dmxsf import lab
from dmxsf.applications.context_memory import Record, recent_groups, resolve


def main() -> None:
    _, report = lab.start(
        "Agent context and memory: budget and precedence", wiki=["applications/context-and-memory"]
    )
    report.section("Context budget")
    selected = recent_groups([18, 22, 8], 25)
    report.check("no partial tool group enters the context", selected == [2], str(selected))
    report.check(
        "the whole tool group fits with enough budget", recent_groups([18, 22, 8], 30) == [1, 2]
    )
    report.section("Memory precedence")
    records = [Record(3, None), Record(1, "morning"), Record(2, "evening")]
    report.check("current request takes precedence", resolve("afternoon", records) == "afternoon")
    report.check("a deletion does not resurrect an older value", resolve(None, records) is None)
    report.finish()


if __name__ == "__main__":
    main()
