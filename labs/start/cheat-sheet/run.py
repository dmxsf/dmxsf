"""Lab: LLM formula sheet: parameters, memory and compute.

Wiki pages
  Chinese: https://dmxsf.com/wiki/start/cheat-sheet/
  English: https://dmxsf.com/en/wiki/start/cheat-sheet/
"""

import math

import torch

from dmxsf import lab
from dmxsf.architecture.configs import PUBLIC_MODELS, kv_bytes_per_token
from dmxsf.architecture.gpt import GPT
from dmxsf.evaluation.stats import pass_at_k
from dmxsf.pretraining.accounting import gpu_hours, train_flops


def main() -> None:
    _args, report = lab.start(
        "LLM formula sheet: parameters, memory and compute", wiki=["start/cheat-sheet"]
    )
    report.section("1. Count from actual model shapes without allocating storage")
    models = [
        entry
        for entry in PUBLIC_MODELS
        if entry.name in {"Llama 2 7B", "Llama 3 8B", "Mistral 7B", "Qwen2.5 7B", "Llama 3 70B"}
    ]
    for entry in models:
        with torch.device("meta"):
            count = sum(parameter.numel() for parameter in GPT(entry.config).parameters())
        report.check(entry.name, count == entry.params, f"{count:,}")
    report.section("2. KV cache and training budget")
    llama = next(model for model in models if model.name == "Llama 3 8B")
    kv = kv_bytes_per_token(llama.config)
    report.check(
        "Llama 3 8B at BF16 needs 128 KiB/token and 1 GiB at 8192 tokens",
        kv == 128 * 1024 and kv * 8192 == 2**30,
    )
    compute = train_flops(405e9, 15.6e12)
    hours = gpu_hours(compute, 989e12, 0.40)
    report.check(
        "405B x 15.6T tokens is about 3.8e25 FLOP",
        f"{compute:.2e}" == "3.79e+25" and round(hours / 1e6, 1) == 26.6,
        f"{compute:.2e}, {hours / 1e6:.1f}M GPU-hours",
    )
    report.check(
        "pass@k matches direct combinations",
        math.isclose(pass_at_k(20, 5, 4), 1 - math.comb(15, 4) / math.comb(20, 4)),
    )
    report.finish()


if __name__ == "__main__":
    main()
