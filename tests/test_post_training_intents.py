"""The synthetic customer-message task behind the post-training labs."""

from dmxsf.post_training import intents


def test_there_are_twelve_intents_and_each_has_many_distinct_messages() -> None:
    assert len(intents.LABELS) == 12
    texts = intents.all_texts(intents.LABELS[0])
    assert len(texts) == len(set(texts)) == 1600
    assert max(len(t) for t in texts) <= intents.WIDTH


def test_a_dataset_has_per_class_examples_of_every_label_and_is_reproducible() -> None:
    a = intents.make_dataset(per_class=5, seed=0)
    assert len(a) == 5 * len(intents.LABELS)
    assert {label for _, label in a} == set(intents.LABELS)
    assert a == intents.make_dataset(per_class=5, seed=0)
    assert a != intents.make_dataset(per_class=5, seed=1)


def test_excluded_messages_never_appear_so_a_test_set_can_be_kept_clean() -> None:
    train = intents.make_dataset(per_class=50, seed=0)
    test = intents.make_dataset(per_class=50, seed=1, exclude={t for t, _ in train})
    assert {t for t, _ in train}.isdisjoint({t for t, _ in test})


def test_every_prompt_has_the_same_length_and_the_answer_ends_in_the_stop_byte() -> None:
    lengths = {len(intents.render(t)) for t, _ in intents.make_dataset(per_class=3, seed=0)}
    assert len(lengths) == 1
    answer = intents.answer_bytes(intents.LABELS[3])
    assert answer.endswith(b"\n") and answer[-1] == intents.STOP
