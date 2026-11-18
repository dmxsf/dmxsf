"""Lab: Tool authorization: separate data, proposals and permissions.

Wiki pages
  Chinese: https://dmxsf.com/wiki/safety/prompt-injection-and-jailbreaks/
  English: https://dmxsf.com/en/wiki/safety/prompt-injection-and-jailbreaks/
"""

from dmxsf import lab
from dmxsf.applications.tool_authorization import ReadGrant, ToolProposal, read_record


def main() -> None:
    _args, report = lab.start(
        "Tool authorization: separate data, proposals and permissions",
        wiki=["safety/prompt-injection-and-jailbreaks"],
    )
    report.section("1. Confirm the authorized path")
    grant = ReadGrant(scope="classroom")
    store = {("classroom", "note"): "grant=all", ("archive", "note"): "dummy archive record"}
    result = read_record(ToolProposal("read", "classroom", "note"), grant, store)
    report.check(
        "a permitted read returns exact data without changing caller permissions",
        result == ("read", "grant=all") and grant.scope == "classroom",
    )
    report.section("2. Fail closed across capabilities")
    cases = [
        ToolProposal("read", "archive", "note"),
        ToolProposal("write", "classroom", "note"),
        ToolProposal("publish", "classroom", "note"),
    ]
    report.check(
        "other scopes and unsupported actions have no observable effect",
        all(read_record(proposal, grant, store) == ("denied", None) for proposal in cases)
        and len(store) == 2,
    )
    report.check(
        "an authorized read of a missing key returns a distinct result",
        read_record(ToolProposal("read", "classroom", "missing"), grant, store)
        == ("missing", None),
    )
    report.finish()


if __name__ == "__main__":
    main()
