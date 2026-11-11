"""The ``dmxsf`` command: list the labs, run one, print a study route, check the environment.

    dmxsf list                      every lab, the course pages it belongs to, how long it takes
    dmxsf run kv-cache --quick      run a lab (everything after the name goes to the lab)
    dmxsf roadmap engineer          the weeks of a route of the learning roadmap, with the labs to run
    dmxsf doctor                    Python, PyTorch, the device and the optional extras

The labs live in the source tree (``labs/``), so ``run`` needs a checkout of the repository, installed
with ``uv sync`` or ``pip install -e .``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import sys

from dmxsf import course


def _duration(seconds: int) -> str:
    return f"{seconds // 60} min" if seconds >= 120 else f"{seconds} s"


def cmd_list(args: argparse.Namespace) -> int:
    labs = [x for x in course.LABS if args.section in (None, x.section)]
    if not labs:
        print(
            f"no section {args.section!r}; sections: "
            + ", ".join(dict.fromkeys(x.section for x in course.LABS))
        )
        return 1
    width = max(len(x.key) for x in labs)
    print(f"{'lab':<{width}}  {'time':>6}  course pages")
    for x in labs:
        extra = f"  [needs --extra {x.extra} for --hf]" if x.extra else ""
        print(f"{x.key:<{width}}  {_duration(x.seconds):>6}  {', '.join(x.wiki)}{extra}")
    plural = "" if len(labs) == 1 else "s"
    print(
        f"\n{len(labs)} lab{plural}. Run one with: dmxsf run <lab> [--quick]. Course: {course.SITE}"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        x = course.find(args.lab)
    except KeyError as exc:
        print(exc.args[0])
        return 2
    root = course.repo_root()
    if root is None:
        print(
            "the labs are in the source tree: clone https://github.com/dmxsf/dmxsf and install it with uv sync"
        )
        return 2
    script = root / x.path / "run.py"
    print(f"$ python {script.relative_to(root)} {' '.join(args.rest)}".rstrip(), flush=True)
    return subprocess.call([sys.executable, str(script), *args.rest], cwd=root)


def cmd_roadmap(args: argparse.Namespace) -> int:
    names = [args.route] if args.route else list(course.ROUTES)
    for name in names:
        title, weeks = course.ROUTES[name]
        print(f"{title}   ({course.SITE}/wiki/start/learning-roadmap/)")
        for week, pages in enumerate(weeks, start=1):
            labs = [course.by_key()[p] for p in pages]
            cells = (
                ", ".join(dict.fromkeys(x.key for x in labs))
                or "(read the papers and write up: no lab)"
            )
            print(f"  week {week:>2}: {cells}")
        print()
    return 0


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def cmd_doctor(_: argparse.Namespace) -> int:
    import torch

    ok = sys.version_info >= (3, 10)
    print(f"python      {platform.python_version()} {'ok' if ok else '(need 3.10 or newer)'}")
    print(f"torch       {torch.__version__}")
    print(
        f"device      cpu (default); cuda: {torch.cuda.is_available()}, mps: {torch.backends.mps.is_available()}"
    )
    print(f"labs        {'found' if course.repo_root() else 'not found (not a source checkout)'}")
    cache = os.environ.get("DMXSF_CACHE", "~/.cache/dmxsf")
    print(f"cache       {cache}  (the small pretrained models are trained once and kept here)")
    for module, what in (
        ("transformers", "the --hf and --bge paths"),
        ("datasets", "CLIP on CIFAR-10"),
        ("PIL", "CLIP on CIFAR-10"),
    ):
        print(
            f"{module:<11} {'installed' if _has(module) else 'missing'}  ({what}; optional: uv sync --extra hf)"
        )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="dmxsf", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="list the labs")
    p.add_argument("--section", help="only this section, e.g. inference")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("run", help="run a lab; arguments after its name go to the lab")
    p.add_argument("lab", help="section/slug, or just the slug")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="e.g. --quick --seed 1")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("roadmap", help="a route of the learning roadmap, week by week")
    p.add_argument("route", nargs="?", choices=list(course.ROUTES))
    p.set_defaults(fn=cmd_roadmap)
    p = sub.add_parser("doctor", help="check the environment")
    p.set_defaults(fn=cmd_doctor)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
