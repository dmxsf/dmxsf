"""Tool authorization: separate data, proposals and permissions.

Wiki pages
  Chinese: https://dmxsf.com/wiki/safety/prompt-injection-and-jailbreaks/
  English: https://dmxsf.com/en/wiki/safety/prompt-injection-and-jailbreaks/
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolProposal:
    action: str
    scope: str
    record: str


@dataclass(frozen=True)
class ReadGrant:
    """An authenticated caller's read capability, supplied by the application."""

    scope: str


def permit(proposal: ToolProposal, grant: ReadGrant) -> bool:
    """Authorize read-only operations with a trusted scope; deny everything else."""
    return proposal.action == "read" and proposal.scope == grant.scope


def read_record(
    proposal: ToolProposal, grant: ReadGrant, store: dict[tuple[str, str], str]
) -> tuple[str, str | None]:
    """Apply policy before any lookup; treat every returned value as inert data."""
    if not permit(proposal, grant):
        return "denied", None
    value = store.get((proposal.scope, proposal.record))
    return ("read", value) if value is not None else ("missing", None)
