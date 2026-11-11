"""A thin wrapper around a Hugging Face chat model, for the optional ``--hf`` paths of the labs.

The labs of the *applications* section are built so that every claim about the harness (templates,
scorers, guards, statistics) can be checked offline. The pages' own numbers, however, come from real
chat models, and this wrapper is how a lab runs the same model on the same prompts to compare. It imports
``torch`` and ``transformers`` only when built; the models are downloaded from the Hugging Face Hub on first
use (several GB), unless they are already in the local cache.
"""

from __future__ import annotations

from collections.abc import Sequence


class HFChat:
    """Greedy (or sampled) chat generation with the model's own chat template."""

    def __init__(self, name: str, dtype: str = "float32") -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "this needs the 'hf' extra: uv sync --extra hf (or pip install transformers)"
            ) from exc
        transformers.logging.disable_progress_bar()  # keep the lab output readable
        self.torch = torch
        self.name = name
        self.tok = transformers.AutoTokenizer.from_pretrained(name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            name, dtype=getattr(torch, dtype)
        ).eval()

    def render(self, messages: Sequence[dict], tools=None, **template_kwargs) -> str:
        """The prompt text the model actually reads (ends where the model starts writing)."""
        return self.tok.apply_chat_template(
            list(messages),
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    def count(self, text: str) -> int:
        return len(self.tok(text).input_ids)

    def generate(
        self,
        messages: Sequence[dict],
        max_new_tokens: int = 60,
        tools=None,
        sample: bool = False,
        seed: int | None = None,
        **template_kwargs,
    ) -> tuple[str, int]:
        """Returns the generated text and its length in tokens. Greedy unless ``sample`` is true, in
        which case it samples at temperature 0.7 with top-p 0.8 and top-k 20 (seeded by ``seed``)."""
        ids = self.tok(self.render(messages, tools=tools, **template_kwargs), return_tensors="pt")
        knobs = (
            dict(temperature=0.7, top_p=0.8, top_k=20)
            if sample
            else dict(temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
        )
        if sample and seed is not None:
            self.torch.manual_seed(seed)
        with self.torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=sample, **knobs
            )
        new = out[0, ids.input_ids.shape[1] :]
        return self.tok.decode(new, skip_special_tokens=True), len(new)
