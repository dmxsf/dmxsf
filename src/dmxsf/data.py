"""Tiny corpora and batching. Everything here works offline.

Models in these labs read *bytes*: the vocabulary is the 256 byte values, so no tokenizer is
needed and ``loss / ln 2`` is bits per byte, which is comparable across models and domains.
"""

from __future__ import annotations

import glob
import sysconfig
from dataclasses import dataclass
from importlib import resources

import numpy as np
import torch

BYTE_VOCAB = 256


def load_shakespeare() -> bytes:
    """The bundled 1.1 MB TinyShakespeare text (public domain), as bytes."""
    return (resources.files("dmxsf") / "data_files" / "tinyshakespeare.txt").read_bytes()


def load_python_code() -> list[bytes]:
    """Sources of the Python standard library, one document per file (tests excluded).

    The result depends on the interpreter version, so numbers computed on it can differ
    slightly between machines. It is a convenient second domain that needs no download.
    """
    root = sysconfig.get_paths()["stdlib"]
    paths = sorted(glob.glob(f"{root}/**/*.py", recursive=True))
    paths = [p for p in paths if "/test" not in p and "site-packages" not in p]
    docs = []
    for path in paths:
        with open(path, "rb") as f:
            docs.append(f.read())
    return docs


def split_documents(docs: list[bytes], val_frac: float = 0.05) -> tuple[list[bytes], list[bytes]]:
    """Hold out whole documents (the last ones) for validation, never random windows.

    Random windows would put neighbouring, overlapping text on both sides of the split and
    the validation score would no longer measure generalisation.
    """
    ends = np.cumsum([len(d) for d in docs])
    k = int(np.searchsorted(ends, ends[-1] * (1 - val_frac)))
    return docs[:k], docs[k:]


def as_array(data: bytes | list[bytes]) -> np.ndarray:
    """Bytes (or a list of documents) as a flat ``uint8`` array."""
    if isinstance(data, list):
        data = b"".join(data)
    return np.frombuffer(data, dtype=np.uint8)


@dataclass(frozen=True)
class ByteCorpus:
    """A training and a validation array of bytes."""

    train: np.ndarray
    val: np.ndarray

    @classmethod
    def shakespeare(cls, val_frac: float = 0.05) -> ByteCorpus:
        """TinyShakespeare, with the last ``val_frac`` of the text held out."""
        data = as_array(load_shakespeare())
        cut = int(len(data) * (1 - val_frac))
        return cls(train=data[:cut], val=data[cut:])

    @classmethod
    def python_code(cls, val_frac: float = 0.05) -> ByteCorpus:
        train, val = split_documents(load_python_code(), val_frac)
        return cls(train=as_array(train), val=as_array(val))


def get_batch(
    data: np.ndarray, step: int, batch_size: int, block_size: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch number ``step``: inputs ``x`` and next-byte targets ``y``, both ``(B, T)``.

    The batch depends only on ``(seed, step)``, so a run that is stopped and resumed at
    step ``k`` sees exactly the batches an uninterrupted run would have seen.
    """
    rng = np.random.default_rng((seed, step))
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    rows = np.stack([data[i : i + block_size + 1] for i in starts]).astype(np.int64)
    rows = torch.from_numpy(rows)
    return rows[:, :-1], rows[:, 1:]


def get_mixed_batch(
    sources: dict[str, np.ndarray],
    mix: dict[str, float],
    step: int,
    batch_size: int,
    block_size: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`get_batch`, but every row first picks a source with probability ``mix[name]``.

    A pretraining data mixture is exactly this: the shares of prose, code and so on in the batches.
    """
    rng = np.random.default_rng((seed, step))
    names = rng.choice(list(mix), size=batch_size, p=list(mix.values()))
    rows = []
    for name in names:
        data = sources[name]
        start = int(rng.integers(0, len(data) - block_size - 1))
        rows.append(data[start : start + block_size + 1])
    rows = torch.from_numpy(np.stack(rows).astype(np.int64))
    return rows[:, :-1], rows[:, 1:]


def encode_text(text: str) -> list[int]:
    """UTF-8 bytes of ``text`` as token ids."""
    return list(text.encode("utf-8"))


def decode_ids(ids: list[int] | torch.Tensor) -> str:
    """Inverse of :func:`encode_text`; invalid UTF-8 shows up as the replacement character."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return bytes(ids).decode("utf-8", errors="replace")
