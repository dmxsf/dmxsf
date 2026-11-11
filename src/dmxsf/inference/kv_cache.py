"""KV caches: remember every layer's keys and values so each new token costs one step, not a re-run.

Course page: https://dmxsf.com/wiki/inference/kv-cache/

Both caches follow the two-member protocol that dmxsf.architecture.gpt.GPT expects:
    cache.length                      number of tokens cached before the current forward pass
    cache.update(layer, k, v) -> k, v  store the new keys/values, return all of them
Tensors are (batch, n_kv_heads, tokens, head_dim): the cache holds the *kv* heads, not the query heads.
"""

import torch


class KVCache:
    """Grows by concatenation: simple, but every step copies the whole cache."""

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    def crop(self, length: int) -> None:
        """Forget everything after the first ``length`` tokens (speculative decoding: a rejected
        draft token must not stay in the cache)."""
        self.k = [None if k is None else k[:, :, :length] for k in self.k]
        self.v = [None if v is None else v[:, :, :length] for v in self.v]

    def reorder(self, index: torch.Tensor) -> None:
        """Beam search: row i of the new batch continues row ``index[i]`` of the old one."""
        self.k = [k[index] for k in self.k]
        self.v = [v[index] for v in self.v]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class StaticKVCache:
    """Preallocated for max_len tokens; update writes in place and returns views (no copying)."""

    def __init__(
        self, n_layers, batch, n_kv_heads, head_dim, max_len, dtype=torch.float32, device=None
    ):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self._len = [0] * n_layers

    @property
    def length(self) -> int:
        return self._len[0]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        start, t = self._len[layer], k.shape[2]
        if start + t > self.max_len:
            raise ValueError(f"cache holds {self.max_len} tokens; need {start + t}")
        self.k[layer, :, :, start : start + t] = k
        self.v[layer, :, :, start : start + t] = v
        self._len[layer] = start + t
        return self.k[layer, :, :, : start + t], self.v[layer, :, :, : start + t]

    def nbytes(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


class SlidingKVCache(KVCache):
    """For sliding-window attention: keeps only the last window - 1 tokens of every layer.

    The next token needs those plus itself, so older entries can never be read again. Memory stops
    growing once the window is full. `length` still counts every token seen, so position encodings
    (RoPE) keep advancing.
    """

    def __init__(self, n_layers: int, window: int):
        super().__init__(n_layers)
        self.window = window
        self._seen = [0] * n_layers

    @property
    def length(self) -> int:
        return self._seen[0]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        k_all, v_all = super().update(layer, k, v)
        self._seen[layer] += k.shape[2]
        keep = max(self.window - 1, 0)
        start = k_all.shape[2] - min(keep, k_all.shape[2])
        self.k[layer], self.v[layer] = k_all[:, :, start:], v_all[:, :, start:]
        return k_all, v_all
