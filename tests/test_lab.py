"""The console report every lab uses: it must exit with status 1 when a check fails."""

import pytest

from dmxsf import lab


def test_start_parses_the_common_flags_and_prints_the_course_links(capsys) -> None:
    args, _ = lab.start(
        "A lab", wiki=["architecture/attention-basics"], argv=["--quick", "--seed", "3"]
    )
    assert args.quick is True and args.seed == 3
    out = capsys.readouterr().out
    assert "# A lab" in out
    assert "https://dmxsf.com/wiki/architecture/attention-basics/" in out
    assert "https://dmxsf.com/en/wiki/architecture/attention-basics/" in out


def test_a_lab_with_only_passing_checks_finishes_normally(capsys) -> None:
    _, report = lab.start("A lab", argv=[])
    assert report.check("two plus two", 2 + 2 == 4, detail="obviously")
    report.finish()
    out = capsys.readouterr().out
    assert "[ok] two plus two  (obviously)" in out
    assert "All checks passed." in out


def test_a_failing_check_makes_finish_exit_with_status_1(capsys) -> None:
    _, report = lab.start("A lab", argv=[])
    report.check("fine", True)
    assert report.check("broken", False) is False
    with pytest.raises(SystemExit) as exc:
        report.finish()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL] broken" in out
    assert "1 check(s) failed: broken" in out


def test_skip_is_printed_and_does_not_count_as_a_failure(capsys) -> None:
    _, report = lab.start("A lab", argv=[])
    report.skip("needs the full run", "--quick trains too little")
    report.finish()
    assert "[skip] needs the full run" in capsys.readouterr().out


def test_tables_are_aligned_and_floats_get_four_significant_digits(capsys) -> None:
    _, report = lab.start("A lab", argv=[])
    capsys.readouterr()
    report.table(["name", "value"], [["a", 1.23456], ["long name", 2.0], ["big", 12345.678]])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["name", "value"]
    assert [line.split()[-1] for line in lines[2:]] == ["1.235", "2.000", "12,345.7"]
    assert len({len(line.rstrip()) for line in lines}) <= 2  # right-justified columns line up
