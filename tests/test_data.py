"""Byte corpora and batching: batches depend only on (seed, step), and held-out data is held out."""

import numpy as np
import pytest
import torch

from dmxsf import data


def test_shakespeare_is_split_into_train_and_a_held_out_tail() -> None:
    corpus = data.ByteCorpus.shakespeare()
    whole = data.as_array(data.load_shakespeare())
    assert len(corpus.train) + len(corpus.val) == len(whole)
    assert np.array_equal(np.concatenate([corpus.train, corpus.val]), whole)
    assert len(corpus.val) == pytest.approx(0.05 * len(whole), abs=1)


def test_a_batch_is_shifted_by_one_and_depends_only_on_seed_and_step() -> None:
    corpus = data.ByteCorpus.shakespeare().train
    x, y = data.get_batch(corpus, step=3, batch_size=4, block_size=16, seed=1)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x moved one byte to the left
    x2, _ = data.get_batch(corpus, step=3, batch_size=4, block_size=16, seed=1)
    assert torch.equal(x, x2)
    assert not torch.equal(x, data.get_batch(corpus, 4, 4, 16, seed=1)[0])
    assert not torch.equal(x, data.get_batch(corpus, 3, 4, 16, seed=2)[0])


def test_a_mixed_batch_takes_every_row_from_the_source_it_picked() -> None:
    sources = {"zeros": np.zeros(500, dtype=np.uint8), "ones": np.ones(500, dtype=np.uint8)}
    x, _ = data.get_mixed_batch(sources, {"zeros": 1.0, "ones": 0.0}, 0, 8, 16)
    assert int(x.sum()) == 0
    x, _ = data.get_mixed_batch(sources, {"zeros": 0.0, "ones": 1.0}, 0, 8, 16)
    assert int(x.min()) == 1
    x, _ = data.get_mixed_batch(sources, {"zeros": 0.5, "ones": 0.5}, 0, 64, 16)
    rows_of_ones = int((x.min(dim=1).values == 1).sum())
    assert 16 < rows_of_ones < 48  # both sources appear


def test_text_round_trips_through_bytes_including_chinese() -> None:
    text = "To be, or not to be; 你好，世界"
    assert data.decode_ids(data.encode_text(text)) == text
    assert data.decode_ids(torch.tensor(data.encode_text(text))) == text
    assert len(data.encode_text("你好")) == 6  # three bytes per character


def test_invalid_utf8_shows_up_as_the_replacement_character() -> None:
    assert data.decode_ids([0xFF, 0x41]) == "\ufffdA"


def test_documents_are_held_out_whole_from_the_end() -> None:
    docs = [bytes([i]) * 100 for i in range(20)]
    train, val = data.split_documents(docs, val_frac=0.25)
    assert train + val == docs  # nothing lost, nothing shuffled
    assert set(train).isdisjoint(val)
    # whole documents only, so the held-out share is rounded up to the next document boundary
    assert sum(map(len, val)) >= 0.25 * 2000
    assert sum(map(len, train)) < 0.75 * 2000
