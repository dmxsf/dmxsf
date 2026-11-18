"""A bounded, read-only tool loop for /wiki/applications/first-agent/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/applications/first-agent/
  English: https://dmxsf.com/en/wiki/applications/first-agent/
"""

from __future__ import annotations

from collections.abc import Callable

Action = tuple[str, str]
Observation = tuple[str, str]
Policy = Callable[[str, tuple[Observation, ...]], Action]


def lookup(book: str) -> str:
    catalog = {"atlas": "shelf A", "algebra": "shelf B"}
    return catalog.get(book, "not in the catalog")


def scripted_policy(task: str, history: tuple[Observation, ...]) -> Action:
    """Stand in for a model so each transition is reproducible."""
    if history:
        return "finish", history[-1][1]
    return "lookup", task


def run(
    task: str,
    policy: Policy,
    max_steps: int,
) -> tuple[str, tuple[Observation, ...]]:
    history: list[Observation] = []
    for _ in range(max_steps):
        action, value = policy(task, tuple(history))
        if not isinstance(value, str):
            raise ValueError("tool arguments and final answers must be strings")
        if action == "finish":
            return value, tuple(history)
        if action != "lookup":
            raise ValueError("unrecognized tool or argument")
        history.append((value, lookup(value)))
    raise RuntimeError("step budget exhausted")
