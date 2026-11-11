"""Boilerplate shared by the ``labs/*/run.py`` scripts: arguments and a tidy console report.

Every lab script looks the same::

    args, report = lab.start("Lab title", wiki=["architecture/attention-basics"])
    report.section("1. something")
    report.check("matches torch", ok, detail="max error 1e-7")
    report.observe("the effect is large", big, detail="noise can move this one")  # [differs], no failure
    report.finish()          # exits with status 1 if any check failed

``--quick`` shrinks the experiment so that CI can run every lab in seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence

SITE = "https://dmxsf.com"


def wiki_url(key: str, lang: str = "zh") -> str:
    """URL of a wiki page on dmxsf.com, e.g. ``architecture/attention-basics``."""
    prefix = "" if lang == "zh" else f"/{lang}"
    return f"{SITE}{prefix}/wiki/{key}/"


class Report:
    """Console output of a lab: headings, tables and pass/fail checks."""

    def __init__(self, title: str, wiki: Sequence[str] = ()) -> None:
        if hasattr(sys.stdout, "reconfigure"):  # keep non-ASCII symbols safe on any terminal
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        self.failures: list[str] = []
        self._start = time.perf_counter()
        print(f"# {title}")
        for key in wiki:
            print(f"  course: {wiki_url(key)}")
            print(f"          {wiki_url(key, 'en')}")

    def section(self, text: str) -> None:
        print(f"\n## {text}")

    def say(self, text: str = "") -> None:
        print(text)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        """Print one pass/fail line and remember failures for ``finish``."""
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            self.failures.append(name)
        return ok

    def observe(self, name: str, ok: bool, detail: str = "") -> bool:
        """A claim about the size of an effect in one training run, which noise can move.

        The effect depends on the seed and on the machine's rounding, so a miss is printed as
        ``[differs]`` and never fails the lab. Use ``check`` for what has to hold on every machine.
        """
        print(f"  [{'ok' if ok else 'differs'}] {name}" + (f"  ({detail})" if detail else ""))
        return ok

    def skip(self, name: str, reason: str) -> None:
        """A check that only makes sense in the full run (``--quick`` trains too little)."""
        print(f"  [skip] {name}  ({reason})")

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        """Print an aligned plain-text table. Floats get four significant digits."""
        cells = [[_fmt(c) for c in row] for row in rows]
        widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
        line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
        print(f"  {line}")
        print(f"  {'  '.join('-' * w for w in widths)}")
        for row in cells:
            print("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)))

    def finish(self) -> None:
        seconds = time.perf_counter() - self._start
        if self.failures:
            print(f"\n{len(self.failures)} check(s) failed: {', '.join(self.failures)}")
            print(f"({seconds:.1f}s)")
            sys.exit(1)
        print(f"\nAll checks passed. ({seconds:.1f}s)")


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return format(value, "#.4g").rstrip(".")  # "#" keeps trailing zeros: 2.500, not 2.5
    return str(value)


def start(
    title: str,
    wiki: Sequence[str] = (),
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, Report]:
    """Parse the common flags (``--quick``, ``--seed``) and print the lab header."""
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--quick", action="store_true", help="tiny sizes, for a fast smoke run")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    if add_args:
        add_args(parser)
    args = parser.parse_args(argv)
    return args, Report(title, wiki)
