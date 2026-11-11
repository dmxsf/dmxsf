"""Prompting: the chat template, the scorers, the word-problem parser and the small-eval-set statistics."""

import json
from collections import Counter
from itertools import product

import pytest

from dmxsf.applications import prompt_data, prompting

OK = '{"task_type": "faq", "requires_human": false}'


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_render_matches_what_the_real_tokenizer_printed_on_the_page(lang):
    d = prompt_data.DATA[lang]
    msgs = [
        {"role": "system", "content": d["demo_system"]},
        {"role": "user", "content": d["demo_ticket"]},
    ]
    assert prompting.render_chatml(msgs) == d["render_with"]
    assert prompting.render_chatml(msgs[1:]) == d["render_without"]


def test_render_adds_the_default_system_message_only_when_there_is_none():
    plain = prompting.render_chatml([{"role": "user", "content": "hi"}])
    assert plain.startswith(f"<|im_start|>system\n{prompting.DEFAULT_SYSTEM}<|im_end|>\n")
    own = prompting.render_chatml(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    )
    assert prompting.DEFAULT_SYSTEM not in own and own.count("<|im_start|>system") == 1


def test_render_generation_prompt_flag_and_history():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    open_turn = prompting.render_chatml(msgs)
    closed = prompting.render_chatml(msgs, add_generation_prompt=False)
    assert open_turn == closed + "<|im_start|>assistant\n"
    assert closed.endswith("<|im_start|>user\nc<|im_end|>\n")
    assert "<|im_start|>assistant\nb<|im_end|>\n" in closed


def test_common_prefix():
    assert prompting.common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert prompting.common_prefix("abc", "abc") == 3
    assert prompting.common_prefix([], [1]) == 0
    assert prompting.common_prefix([1, 2], [1, 2, 3]) == 2


def test_parse_accepts_only_a_clean_json_object():
    assert prompting.parse(OK) == {"task_type": "faq", "requires_human": False}
    assert prompting.parse("\n  " + OK + "\n") is not None
    assert prompting.parse("```json\n" + OK + "\n```") is None
    assert prompting.parse('{"task_type": "faq", "requires_human": "false"}') is None
    assert prompting.parse('{"task_type": "faq", "requires_human": 0}') is None
    assert prompting.parse('{"task_type": "refund", "requires_human": false}') is None
    assert prompting.parse('{"task_type": "faq"}') is None
    assert prompting.parse("[1, 2]") is None
    assert prompting.parse("") is None
    # extra fields are allowed: how strict to be is a choice
    assert prompting.parse(OK[:-1] + ', "confidence": 0.9}') is not None


def test_label_of_reads_the_last_label_word():
    assert prompting.label_of("faq or escalation? Final: order_lookup") == "order_lookup"
    assert prompting.label_of("no class here") is None
    # the blind spot: a remark after the answer wins
    assert prompting.label_of(OK + " (this is not an escalation)") == "escalation"


def test_score_reply_keeps_format_and_judgment_apart():
    s = prompting.score_reply("```json\n" + OK + "\n```", "faq")
    assert (s["valid"], s["label"], s["both"]) == (False, True, False)
    s = prompting.score_reply(OK, "escalation")
    assert (s["valid"], s["label"], s["both"]) == (True, False, False)
    s = prompting.score_reply(OK, "faq")
    assert (s["valid"], s["label"], s["both"]) == (True, True, True)


def test_build_writes_the_examples_as_turns():
    make = prompting.build("sys", [("t1", "faq"), ("t2", "escalation")])
    msgs = make("ticket")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert json.loads(msgs[2]["content"]) == {"task_type": "faq", "requires_human": False}
    assert json.loads(msgs[4]["content"]) == {"task_type": "escalation", "requires_human": True}
    assert msgs[-1]["content"] == "ticket"
    assert [m["role"] for m in prompting.build("sys")("x")] == ["system", "user"]


def test_evaluate_and_summarize_with_a_scripted_model():
    replies = {
        "t1": OK,
        "t2": "```json\n" + OK + "\n```",
        "t3": '{"task_type": "faq", "requires_human": "false"}',
        "t4": '{"task_type": "escalation", "requires_human": true}',
    }
    cases = [("t1", "faq"), ("t2", "faq"), ("t3", "faq"), ("t4", "faq")]

    def chat(messages, max_new_tokens):
        return replies[messages[-1]["content"]], 7

    rows = prompting.evaluate(chat, prompting.build("sys"), cases)
    s = prompting.summarize(rows)
    assert (s["valid"], s["label"], s["both"], s["n"], s["tokens"]) == (2, 3, 1, 4, 7)
    assert prompting.format_slips(rows) == (1, 1)


def test_answer_number():
    assert prompting.answer_number("so 12 + 30 = 42.\nAnswer: 1,280 dollars") == 1280.0
    assert prompting.answer_number("It ends at 640 liters.") == 640.0
    assert prompting.answer_number("Answer: about 3.5 (2 + 1.5)") == 3.5
    assert prompting.answer_number("Answer: 1\nAnswer: 2") == 2.0  # the last marker counts
    assert prompting.answer_number("Net change: -7") == -7.0
    assert prompting.answer_number("Answer: unknown") is None
    assert prompting.answer_number("no digits") is None
    assert prompting.answer_number("答案：20 棵", "答案：") == 20.0
    # a decimal comma is read as a thousands separator
    assert prompting.answer_number("Answer: 2,5") == 25.0


def test_expected_best_of_matches_brute_force():
    n, m, p = 3, 2, 0.5
    pmf = prompting.binomial_pmf(n, p)
    brute = sum(pmf[a] * pmf[b] * max(a, b) for a, b in product(range(n + 1), repeat=2)) / n
    assert prompting.expected_best_of(m, n, p) == pytest.approx(brute)
    # one prompt: unbiased; certain prompts stay certain; more prompts only raise the best
    assert prompting.expected_best_of(1, 12, 0.7) == pytest.approx(0.7)
    assert prompting.expected_best_of(5, 12, 1.0) == pytest.approx(1.0)
    assert prompting.expected_best_of(5, 12, 0.0) == pytest.approx(0.0)
    values = [prompting.expected_best_of(m, 12, 0.6) for m in (1, 2, 5, 20)]
    assert values == sorted(values) and values[-1] < 1


def test_p_gap_at_least_is_a_tail_probability():
    assert prompting.p_gap_at_least(12, 0.5, 0) == pytest.approx(1.0)
    assert prompting.p_gap_at_least(12, 0.5, 13) == 0.0
    tails = [prompting.p_gap_at_least(12, 2 / 3, g) for g in range(0, 8)]
    assert tails == sorted(tails, reverse=True)
    # two independent 3-item splits at p = 1/2: P(X = Y) = 20/64
    assert prompting.p_gap_at_least(3, 0.5, 1) == pytest.approx(1 - 20 / 64)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_page_data_is_what_the_page_says(lang):
    d = prompt_data.DATA[lang]
    # 24 tickets, 8 per class, split 12 / 12, with the same count per class in each half
    for split in ("dev", "test"):
        assert Counter(g for _, g in d[split]) == dict.fromkeys(prompting.LABELS, 4)
    # the three examples are one per class and are not copies of an evaluation ticket
    assert sorted(g for _, g in d["shots"]) == sorted(prompting.LABELS)
    eval_texts = {t for t, _ in d["dev"] + d["test"]}
    assert not eval_texts & {t for t, _ in d["shots"]}
    assert len(d["math"]) == 12
    for key in ("v1", "rules", "strict", "cot", "sys_direct", "sys_cot"):
        assert d[key].strip()
    for label in prompting.LABELS:
        assert label in d["rules"]
