"""The small shared models: their size, and the cache that keeps the labs from retraining them."""

import torch

from dmxsf import models
from dmxsf.utils import count_params


def test_the_base_model_is_the_842k_parameter_byte_level_gpt_of_the_labs() -> None:
    base, draft = models.build("base"), models.build("draft")
    assert count_params(base) == 842_496
    assert count_params(draft) < count_params(base) / 4
    assert (
        base.cfg.n_layer == 4 and base.cfg.vocab_size == 256 and base.cfg.block_size == models.BLOCK
    )


def test_quick_models_are_trained_briefly_and_never_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DMXSF_CACHE", str(tmp_path))
    model = models.pretrained("draft", quick=True)
    assert count_params(model) == count_params(models.build("draft"))
    assert list(tmp_path.iterdir()) == []


def test_a_cached_model_is_loaded_instead_of_retrained(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DMXSF_CACHE", str(tmp_path))
    torch.manual_seed(123)
    saved = models.build("draft")
    steps = models.SPECS["draft"][1]
    torch.save(saved.state_dict(), tmp_path / f"draft-seed0-{steps}steps.pt")
    loaded = models.pretrained("draft")  # would take minutes if it trained
    for (name, a), (_, b) in zip(
        saved.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
