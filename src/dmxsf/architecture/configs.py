"""Architecture configs of released checkpoints, with the parameter counts they report.

The same table (minus GPT-2 and Qwen2.5-0.5B) drives the parameter-counter widget on dmxsf.com
(src/data/llm-configs.ts); the tests check that this code builds models with exactly these totals.
"""

from dataclasses import dataclass

from dmxsf.architecture.gpt import GPTConfig


@dataclass(frozen=True)
class PublicModel:
    name: str
    config: GPTConfig
    params: int  # the total reported for the checkpoint (Hugging Face model.num_parameters())


def _llama_like(**kw) -> GPTConfig:
    base = dict(
        norm="rmsnorm", norm_eps=1e-5, pos="rope", mlp="swiglu", bias=False, tie_embeddings=False
    )
    return GPTConfig(**{**base, **kw})


PUBLIC_MODELS: list[PublicModel] = [
    PublicModel(
        "GPT-2 small",
        GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, d_model=768),
        124_439_808,
    ),
    PublicModel(
        "Qwen2.5-0.5B",
        _llama_like(
            vocab_size=151936,
            block_size=32768,
            n_layer=24,
            n_head=14,
            n_kv_head=2,
            d_model=896,
            d_ff=4864,
            qkv_bias=True,
            tie_embeddings=True,
            norm_eps=1e-6,
            rope_base=1_000_000.0,
        ),
        494_032_768,
    ),
    PublicModel(
        "Llama 2 7B",
        _llama_like(
            vocab_size=32000,
            block_size=4096,
            n_layer=32,
            n_head=32,
            d_model=4096,
            d_ff=11008,
        ),
        6_738_415_616,
    ),
    PublicModel(
        "Llama 3 8B",
        _llama_like(
            vocab_size=128256,
            block_size=8192,
            n_layer=32,
            n_head=32,
            n_kv_head=8,
            d_model=4096,
            d_ff=14336,
            rope_base=500_000.0,
        ),
        8_030_261_248,
    ),
    PublicModel(
        "Mistral 7B",
        _llama_like(
            vocab_size=32000,
            block_size=8192,
            n_layer=32,
            n_head=32,
            n_kv_head=8,
            d_model=4096,
            d_ff=14336,
        ),
        7_241_732_096,
    ),
    PublicModel(
        "Qwen2.5 7B",
        _llama_like(
            vocab_size=152064,
            block_size=32768,
            n_layer=28,
            n_head=28,
            n_kv_head=4,
            d_model=3584,
            d_ff=18944,
            qkv_bias=True,
            norm_eps=1e-6,
            rope_base=1_000_000.0,
        ),
        7_615_616_512,
    ),
    PublicModel(
        "Llama 3 70B",
        _llama_like(
            vocab_size=128256,
            block_size=8192,
            n_layer=80,
            n_head=64,
            n_kv_head=8,
            d_model=8192,
            d_ff=28672,
            rope_base=500_000.0,
        ),
        70_553_706_496,
    ),
]

BY_NAME = {m.name: m for m in PUBLIC_MODELS}


def small_gpt(**overrides) -> GPTConfig:
    """The 0.86M-parameter byte-level GPT used by the course's training-pipeline page and most labs."""
    base = dict(vocab_size=256, block_size=256, n_layer=4, n_head=4, d_model=128)
    return GPTConfig(**{**base, **overrides})


def param_breakdown(model) -> dict[str, int]:
    """Parameters grouped by role: embedding (token and learned position tables), attention, mlp,
    norm and head (the output matrix, 0 when it is tied to the embedding)."""
    groups = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "head": 0}
    for name, p in model.named_parameters():  # a tied matrix is listed once, as tok_emb
        if name.startswith(("tok_emb", "pos_emb")):
            groups["embedding"] += p.numel()
        elif name.startswith("lm_head"):
            groups["head"] += p.numel()
        elif ".attn." in name:
            groups["attention"] += p.numel()
        elif ".mlp." in name:
            groups["mlp"] += p.numel()
        else:
            groups["norm"] += p.numel()
    return groups


def kv_bytes_per_token(cfg: GPTConfig, dtype_bytes: int = 2) -> int:
    """Keys and values of every layer for one token: 2 * layers * kv_heads * head_dim * bytes."""
    return 2 * cfg.n_layer * cfg.kv_heads * cfg.dim_per_head * dtype_bytes
