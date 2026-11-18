"""The course map: every page of the course has a lab, every lab is declared once and runs.

``course.LABS`` is the single source of truth for the relation between https://dmxsf.com and this
repository. These tests keep it honest: the list of pages is copied from the site (a page added to the
course shows up here as a failing test until it has a lab), and each lab's own ``run.py`` has to declare
the same pages and title as the table.
"""

import ast
import importlib.util
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from dmxsf import cli, course

ROOT = Path(__file__).resolve().parents[1]

# Published Wiki articles with runnable content, from the site. The resources page is a reading list.
PAGES = """
applications/agent-evaluation
applications/agents-and-tool-calling
applications/context-and-memory
applications/first-agent
applications/mcp-and-tool-ecosystem
applications/prompt-engineering
applications/rag
applications/structured-output-and-function-calling
architecture/attention-basics
architecture/efficient-attention
architecture/from-rnn-to-attention
architecture/long-context
architecture/mixture-of-experts
architecture/modern-llm-architecture
architecture/positional-encoding
architecture/state-space-and-hybrid
architecture/transformer-block
evaluation/benchmark-contamination
evaluation/building-eval-sets
evaluation/eval-statistics
evaluation/llm-as-judge
evaluation/llm-evaluation
evaluation/production-evaluation
foundations/information-theory
foundations/neural-network-basics
foundations/next-token-prediction
foundations/optimizers-and-training-stability
foundations/tensors-and-shapes
foundations/tokenizer-and-embeddings
frontier/interpretability
frontier/synthetic-data-and-self-improvement
frontier/test-time-compute
inference/continuous-batching
inference/decoding-and-sampling
inference/first-inference-service
inference/kv-cache
inference/prefix-caching
inference/quantization
inference/serving
inference/speculative-decoding
multimodal/diffusion-and-flow-matching
multimodal/multimodal-llms
multimodal/world-models
post-training/lora-finetuning
post-training/preference-optimization
post-training/reasoning-rl
post-training/reward-models
post-training/rl-fundamentals
post-training/rollout-and-training
post-training/sft-data
pretraining/compute-and-memory
pretraining/data-engineering
pretraining/distributed-training
pretraining/scaling-laws
pretraining/training-pipeline
safety/hallucination-and-calibration
safety/memorization-and-privacy
safety/prompt-injection-and-jailbreaks
start/cheat-sheet
start/learning-roadmap
start/what-is-an-llm
systems/collective-communication
systems/gpu-and-roofline
systems/gpu-kernels
systems/hardware-basics
systems/numerical-precision
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
    assert len(PAGES) == 66 and len(set(PAGES)) == 66


def test_every_page_has_exactly_one_lab():
    counts = Counter(key for lab in course.LABS for key in lab.wiki)
    assert not [k for k, n in counts.items() if n > 1], "a page is covered by two labs"
    assert set(counts) == set(PAGES)


def test_labs_are_unique_and_the_slug_is_the_primary_page():
    assert len({lab.key for lab in course.LABS}) == len(course.LABS)
    assert len({lab.slug for lab in course.LABS}) == len(course.LABS)
    assert set(course.TEXTS) == {lab.key for lab in course.LABS}
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
def test_source_docstrings_link_both_wiki_languages(lab):
    paths = [ROOT / lab.path / "run.py"]
    paths.extend(ROOT / "src/dmxsf" / (module.replace(".", "/") + ".py") for module in lab.modules)
    for path in paths:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        for key in lab.wiki:
            for lang in ("zh", "en"):
                assert course.wiki_url(key, lang) in doc, path


@pytest.mark.parametrize("lab", course.LABS, ids=lambda x: x.key)
def test_each_lab_has_both_readmes_linking_every_page_it_covers(lab):
    for lang, name, other in (
        ("zh", "README.md", "README.en.md"),
        ("en", "README.en.md", "README.md"),
    ):
        text = (ROOT / lab.path / name).read_text(encoding="utf-8")
        links = set(re.findall(r"\]\(([^\s)]+)\)", text))
        assert other in links, f"{lab.path}/{name} has no language switch"
        for key in lab.wiki:
            assert course.wiki_url(key, lang) in links, f"{lab.path}/{name} does not link {key}"


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
    _, weeks = course.ROUTES[route]
    assert all(page in course.by_key() for pages in weeks for page in pages)
    labs = course.route_labs(route)
    assert labs and [w for w, _ in labs] == sorted(w for w, _ in labs)
    assert len({lab.key for _, lab in labs}) == len(labs)  # each lab once, at its first week
    assert all(1 <= w <= len(weeks) for w, _ in labs)


def test_the_four_routes_of_the_roadmap():
    assert {r: len(w) for r, (_, w) in course.ROUTES.items()} == {
        "weekend": 7,
        "application": 8,
        "engineer": 14,
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
    capsys.readouterr()
    assert cli.main(["list", "--section", "foundations", "--links"]) == 0
    linked = capsys.readouterr().out
    expected = {
        course.wiki_url(key, lang)
        for lab in course.LABS
        if lab.section == "foundations"
        for key in lab.wiki
        for lang in ("zh", "en")
    }
    assert set(re.findall(r"https://dmxsf\.com(?:/en)?/wiki/[^\s]+", linked)) == expected
    assert cli.main(["roadmap", "engineer"]) == 0
    assert "week  1:" in capsys.readouterr().out
    assert cli.main(["roadmap", "weekend"]) == 0
    assert "step  1:" in capsys.readouterr().out


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
