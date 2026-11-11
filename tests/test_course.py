"""The course map: every page of the course has a lab, every lab is declared once and runs.

``course.LABS`` is the single source of truth for the relation between https://dmxsf.com and this
repository. These tests keep it honest: the list of pages is copied from the site (a page added to the
course shows up here as a failing test until it has a lab), and each lab's own ``run.py`` has to declare
the same pages and title as the table.
"""

import ast
import importlib.util
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from dmxsf import cli, course

ROOT = Path(__file__).resolve().parents[1]

# The 35 articles of the course (https://dmxsf.com/wiki/), by wiki key.
PAGES = """
start/what-is-an-llm start/learning-roadmap
foundations/next-token-prediction foundations/information-theory
foundations/tokenizer-and-embeddings foundations/optimizers-and-training-stability
architecture/attention-basics architecture/positional-encoding architecture/transformer-block
architecture/modern-llm-architecture architecture/long-context architecture/efficient-attention
architecture/mixture-of-experts
pretraining/training-pipeline pretraining/scaling-laws pretraining/compute-and-memory
pretraining/distributed-training
post-training/sft-data post-training/lora-finetuning post-training/preference-optimization
post-training/reasoning-rl
inference/decoding-and-sampling inference/kv-cache inference/quantization inference/serving
inference/continuous-batching inference/speculative-decoding
applications/prompt-engineering applications/rag
applications/structured-output-and-function-calling applications/agents-and-tool-calling
multimodal/multimodal-llms
evaluation/llm-evaluation evaluation/eval-statistics evaluation/llm-as-judge
""".split()

SLOW_LABS = [lab.key for lab in course.LABS]


def declared(lab: course.Lab) -> tuple[str, list[str]]:
    """The title and the wiki keys a lab's ``run.py`` passes to ``lab.start``, read without running it."""
    tree = ast.parse((ROOT / lab.path / "run.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start"
            and getattr(node.func.value, "id", None) == "lab"
        ):
            wiki = next(k.value for k in node.keywords if k.arg == "wiki")
            return ast.literal_eval(node.args[0]), ast.literal_eval(wiki)
    raise AssertionError(f"{lab.path}/run.py never calls lab.start(...)")


def test_the_list_of_pages_is_the_whole_course():
    assert len(PAGES) == 35 and len(set(PAGES)) == 35


def test_every_page_has_exactly_one_lab():
    counts = Counter(key for lab in course.LABS for key in lab.wiki)
    assert not [k for k, n in counts.items() if n > 1], "a page is covered by two labs"
    assert set(counts) == set(PAGES)


def test_labs_are_unique_and_the_slug_is_the_primary_page():
    assert len({lab.key for lab in course.LABS}) == len(course.LABS)
    for lab in course.LABS:
        assert lab.wiki[0] == lab.key, (
            lab.key
        )  # labs/<section>/<slug> is built around <section>/<slug>
        assert lab.title and lab.seconds > 0


def test_the_directories_match_the_table():
    on_disk = {
        f"{p.parent.parent.name}/{p.parent.name}" for p in (ROOT / "labs").glob("*/*/run.py")
    }
    assert on_disk == {lab.key for lab in course.LABS}


@pytest.mark.parametrize("lab", course.LABS, ids=lambda x: x.key)
def test_each_lab_declares_what_the_table_says(lab):
    title, wiki = declared(lab)
    assert title == lab.title
    assert wiki == list(lab.wiki)


@pytest.mark.parametrize("lab", course.LABS, ids=lambda x: x.key)
def test_each_lab_has_both_readmes_linking_every_page_it_covers(lab):
    for name, prefix in (
        ("README.md", "https://dmxsf.com"),
        ("README.en.md", "https://dmxsf.com/en"),
    ):
        text = (ROOT / lab.path / name).read_text(encoding="utf-8")
        for key in lab.wiki:
            assert f"{prefix}/wiki/{key}/" in text, f"{lab.path}/{name} does not link {key}"


@pytest.mark.parametrize("lab", course.LABS, ids=lambda x: x.key)
def test_the_library_modules_of_each_lab_exist(lab):
    for module in lab.modules:
        assert importlib.util.find_spec(f"dmxsf.{module}") is not None, module


def test_extras_are_declared_in_pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert {lab.extra for lab in course.LABS if lab.extra} <= set(extras)


@pytest.mark.parametrize("route", list(course.ROUTES))
def test_routes_only_point_to_real_labs(route):
    title, weeks = course.ROUTES[route]
    assert len(weeks) == int(title.split("(")[1].split()[0])  # "(8 weeks)"
    labs = course.route_labs(route)
    assert labs and [w for w, _ in labs] == sorted(w for w, _ in labs)
    assert len({lab.key for _, lab in labs}) == len(labs)  # each lab once, at its first week
    assert all(1 <= w <= len(weeks) for w, _ in labs)


def test_the_three_routes_of_the_roadmap():
    assert {r: len(w) for r, (_, w) in course.ROUTES.items()} == {
        "application": 8,
        "engineer": 12,
        "researcher": 16,
    }


def test_find_accepts_the_slug_and_rejects_the_unknown():
    assert course.find("kv-cache") is course.find("inference/kv-cache")
    with pytest.raises(KeyError):
        course.find("no-such-lab")


def test_cli_list_and_roadmap(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert all(lab.key in out for lab in course.LABS)
    assert cli.main(["list", "--section", "nope"]) == 1
    assert cli.main(["roadmap", "engineer"]) == 0
    assert "week  1:" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.parametrize("key", SLOW_LABS)
def test_lab_quick_run_passes(key):
    """Every lab runs end to end in ``--quick`` mode and none of its checks fails."""
    lab = course.find(key)
    done = subprocess.run(
        [sys.executable, str(ROOT / lab.path / "run.py"), "--quick"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=900,
    )
    failed = "\n".join(line for line in done.stdout.splitlines() if "[FAIL]" in line)
    assert done.returncode == 0, f"{failed}\n{done.stdout[-2000:]}{done.stderr[-2000:]}"
    assert "[FAIL]" not in done.stdout
    # a lab that never calls report.finish() would exit 0 whatever its checks said
    assert "All checks passed." in done.stdout.splitlines()[-1]
