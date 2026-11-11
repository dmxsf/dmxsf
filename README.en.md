# DMXSF companion code

The companion code of the [DMXSF](https://dmxsf.com/en/) course on large language model algorithms: the core algorithms of the course rebuilt from scratch in Python and PyTorch, with one runnable lab per wiki page that checks its own results.

[中文](README.md) · [Course site](https://dmxsf.com/en/) · [Code and course map](https://dmxsf.com/en/code/)

## What it is

- **One-to-one with the course.** 30 labs cover all 35 wiki pages: a lab's directory `labs/<section>/<page>/` has the same name as the wiki path, each lab's README links back to its course pages, and every wiki page ends with a "Companion code" card that points here. The mapping is written in one place only: [`src/dmxsf/course.py`](src/dmxsf/course.py).
- **It reproduces the numbers on the pages.** Formulas, tables and experimental results are recomputed and compared one by one; a pass prints `[ok]`, a mismatch prints `[FAIL]` and exits non-zero. Everything that can be reproduced offline is; what needs a real large model is an optional path and the rest does not depend on it (see [Optional: real models](#optional-real-models)).
- **A laptop is enough.** Everything runs on the CPU, with no GPU and no API. Training, fine-tuning, quantization, speculative decoding and continuous batching really run on a byte-level GPT of 842,496 parameters (vocabulary 256, the TinyShakespeare corpus; the first run spends about 3 minutes training it and caches it), they are not mock-ups.
- **The code is the textbook.** The library is in `src/dmxsf/`, one module for one or two course pages, written to be read; the core modules have unit tests, and the lab scripts carry their own checks.

## Quick start

Python 3.10 or newer. [uv](https://docs.astral.sh/uv/) is recommended:

```bash
git clone https://github.com/dmxsf/dmxsf.git
cd dmxsf
uv sync                                # install the dependencies (PyTorch and NumPy)
uv run dmxsf list                      # list all labs
uv run dmxsf run attention-basics      # run one lab; add --quick for a fast first pass
uv run dmxsf roadmap engineer          # which labs a study route asks for, week by week
uv run dmxsf doctor                    # check the environment
```

Without uv: `pip install -e .` and drop `uv run` from the commands. You can also run a script directly: `python labs/architecture/attention-basics/run.py`.

## Course and labs

All labs in the order of the course's sections, which is also the study order. Times are measured full runs on an Apple M2 Max (CPU) with the models already cached.

### Getting Started (`start`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`what-is-an-llm`](labs/start/what-is-an-llm/) | What is an LLM? The whole loop, small enough to read | [What Is a Large Language Model? From Next-Word Prediction to General Assistants](https://dmxsf.com/en/wiki/start/what-is-an-llm/)<br>[LLM Learning Roadmap: Application, Engineering and Research Paths](https://dmxsf.com/en/wiki/start/learning-roadmap/) | a few seconds |

### Foundations (`foundations`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`next-token-prediction`](labs/foundations/next-token-prediction/) | Next-token prediction and information theory | [From Language Modeling to Next-Token Prediction](https://dmxsf.com/en/wiki/foundations/next-token-prediction/)<br>[Information Theory Refresher: Entropy, Cross-Entropy and KL Divergence](https://dmxsf.com/en/wiki/foundations/information-theory/) | a few seconds |
| [`tokenizer-and-embeddings`](labs/foundations/tokenizer-and-embeddings/) | Tokenizers and embeddings | [Tokenizers and Embeddings: BPE, Vocabulary Size and the Embedding Table](https://dmxsf.com/en/wiki/foundations/tokenizer-and-embeddings/) | a few seconds |
| [`optimizers-and-training-stability`](labs/foundations/optimizers-and-training-stability/) | Optimizers, schedules, mixed precision and loss spikes | [Optimizers and Training Stability: AdamW, Warmup, Mixed Precision, Loss Spikes](https://dmxsf.com/en/wiki/foundations/optimizers-and-training-stability/) | a few seconds |

### Model Architecture (`architecture`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`attention-basics`](labs/architecture/attention-basics/) | Attention, by hand and in a model | [Attention Basics: How a Transformer Reads Context](https://dmxsf.com/en/wiki/architecture/attention-basics/) | about 35 seconds |
| [`positional-encoding`](labs/architecture/positional-encoding/) | Position encodings and length extrapolation | [Positional Encoding Explained: Absolute, Relative and RoPE](https://dmxsf.com/en/wiki/architecture/positional-encoding/)<br>[Long Context: Position Extrapolation, Training Recipes and Cost](https://dmxsf.com/en/wiki/architecture/long-context/) | about 2 minutes |
| [`transformer-block`](labs/architecture/transformer-block/) | The Transformer block and the modern recipe | [Transformer Block Explained: Attention, FFN, Residuals and LayerNorm](https://dmxsf.com/en/wiki/architecture/transformer-block/)<br>[Modern LLM Architecture: From GPT-2 to Llama, Qwen and DeepSeek](https://dmxsf.com/en/wiki/architecture/modern-llm-architecture/) | about 3 minutes |
| [`efficient-attention`](labs/architecture/efficient-attention/) | Efficient attention | [Efficient Attention: MQA, GQA, MLA and FlashAttention](https://dmxsf.com/en/wiki/architecture/efficient-attention/) | about 55 seconds |
| [`mixture-of-experts`](labs/architecture/mixture-of-experts/) | Mixture of experts: routing, load balance and the parameter accounts | [Mixture of Experts (MoE): Routing, Load Balancing and Sparse Activation](https://dmxsf.com/en/wiki/architecture/mixture-of-experts/) | about 3 minutes |

### Pre-training (`pretraining`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`training-pipeline`](labs/pretraining/training-pipeline/) | The pretraining pipeline on a laptop: monitoring, data mix, resuming, leakage | [The LLM Training Pipeline: From Data to Deployment](https://dmxsf.com/en/wiki/pretraining/training-pipeline/) | about 3 minutes |
| [`scaling-laws`](labs/pretraining/scaling-laws/) | Scaling laws: the formulas, the serving cost, and a tiny study of our own | [Scaling Laws: How to Split Compute Between Parameters and Data](https://dmxsf.com/en/wiki/pretraining/scaling-laws/) | about 2 minutes |
| [`compute-and-memory`](labs/pretraining/compute-and-memory/) | Training compute and memory: 6ND, MFU, model state and activations | [The Compute and Memory Ledger of Training: FLOPs, MFU and Memory Budgets](https://dmxsf.com/en/wiki/pretraining/compute-and-memory/) | a few seconds |
| [`distributed-training`](labs/pretraining/distributed-training/) | Distributed training: gradients, ZeRO, tensor and pipeline parallelism | [Distributed Training: Data, Tensor and Pipeline Parallelism](https://dmxsf.com/en/wiki/pretraining/distributed-training/) | about 15 seconds |

### Post-training and Alignment (`post-training`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`sft-data`](labs/post-training/sft-data/) | SFT data: the loss mask, a data health check, and what data size, noise, masking and replay do | [SFT Data: Construction and Quality Control](https://dmxsf.com/en/wiki/post-training/sft-data/) | about 3 minutes |
| [`lora-finetuning`](labs/post-training/lora-finetuning/) | LoRA: the parameter bill, the mechanics, LoRA against full fine-tuning, merging, NF4 | [LoRA Fine-Tuning in Practice](https://dmxsf.com/en/wiki/post-training/lora-finetuning/) | about 4 minutes |
| [`preference-optimization`](labs/post-training/preference-optimization/) | Preference optimization: the DPO loss, its closed form, and a real DPO run | [Preference Optimization: From RLHF to DPO](https://dmxsf.com/en/wiki/post-training/preference-optimization/) | about 6 minutes |
| [`reasoning-rl`](labs/post-training/reasoning-rl/) | Reasoning RL: verifiers, GRPO advantages and loss, and a real RLVR run | [Reinforcement Learning for Reasoning: RLVR and GRPO](https://dmxsf.com/en/wiki/post-training/reasoning-rl/) | about 3 minutes |

### Inference and Serving (`inference`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`decoding-and-sampling`](labs/inference/decoding-and-sampling/) | Decoding and sampling: temperature, top-k, top-p, min-p, beam search, penalties | [Decoding and Sampling: Temperature, Top-k, Top-p and Beam Search](https://dmxsf.com/en/wiki/inference/decoding-and-sampling/) | about 10 seconds |
| [`kv-cache`](labs/inference/kv-cache/) | KV cache: the memory formula, exactness, the saving, prefill against decode | [KV Cache and Inference Performance](https://dmxsf.com/en/wiki/inference/kv-cache/) | a few seconds |
| [`quantization`](labs/inference/quantization/) | Quantization: the ledger, the grid, weights, activations, calibration, the KV cache | [Model Quantization: INT8, INT4, GPTQ and AWQ Explained and Measured](https://dmxsf.com/en/wiki/inference/quantization/) | about 20 seconds |
| [`speculative-decoding`](labs/inference/speculative-decoding/) | Speculative decoding: the rule, the formulas, a real draft and target | [Speculative Decoding: Draft with a Small Model, Verify with a Large One](https://dmxsf.com/en/wiki/inference/speculative-decoding/) | about 35 seconds |
| [`continuous-batching`](labs/inference/continuous-batching/) | Serving: continuous batching, paged KV memory, prefix cache, TTFT, capacity | [Continuous Batching and PagedAttention: How a Server Handles Many Requests at Once](https://dmxsf.com/en/wiki/inference/continuous-batching/)<br>[LLM Inference Serving: Latency Metrics, Measuring, and Choosing an Engine](https://dmxsf.com/en/wiki/inference/serving/) | about 10 seconds |

### Applications (`applications`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`prompt-engineering`](labs/applications/prompt-engineering/) | Prompting, measured | [Prompt Engineering: A Systematic Guide to Specs, Layers, Examples and Evaluation](https://dmxsf.com/en/wiki/applications/prompt-engineering/) | a few seconds |
| [`rag`](labs/applications/rag/) | RAG, piece by piece | [RAG in Practice: Chunking, Hybrid Retrieval, Reranking and Evaluation](https://dmxsf.com/en/wiki/applications/rag/) | a few seconds |
| [`structured-output-and-function-calling`](labs/applications/structured-output-and-function-calling/) | Structured output: constrained decoding, its cost, and the function-calling loop | [Structured Outputs and Function Calling: From JSON Mode to Constrained Decoding](https://dmxsf.com/en/wiki/applications/structured-output-and-function-calling/) | about 2 minutes |
| [`agents-and-tool-calling`](labs/applications/agents-and-tool-calling/) | Agents and tool calling: the loop, its guards, and the trace scorer | [Agents and Tool Calling in Practice: From Function Calling to ReAct Workflows](https://dmxsf.com/en/wiki/applications/agents-and-tool-calling/) | a few seconds |

### Multimodal (`multimodal`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`multimodal-llms`](labs/multimodal/multimodal-llms/) | Multimodal LLMs: images as tokens | [Multimodal LLMs: How an Image Becomes Tokens](https://dmxsf.com/en/wiki/multimodal/multimodal-llms/) | about 110 seconds |

### Evaluation (`evaluation`)

| Lab | What it does | Course pages | Full run |
| --- | --- | --- | --- |
| [`llm-evaluation`](labs/evaluation/llm-evaluation/) | Benchmark contamination: an n-gram detector, and a leak you can measure | [LLM Evaluation Methodology: Benchmarks, Contamination and a Workflow](https://dmxsf.com/en/wiki/evaluation/llm-evaluation/) | about 2 minutes |
| [`eval-statistics`](labs/evaluation/eval-statistics/) | Evaluation statistics: pass@k, the error of a score, paired tests, sample sizes | [The Statistics of Evaluation: pass@k, Confidence Intervals and Sample Size](https://dmxsf.com/en/wiki/evaluation/eval-statistics/) | a few seconds |
| [`llm-as-judge`](labs/evaluation/llm-as-judge/) | LLM-as-a-judge: bias, agreement, and Bradley-Terry leaderboards | [LLM-as-a-Judge: Using a Model as the Grader, Its Biases and Calibration](https://dmxsf.com/en/wiki/evaluation/llm-as-judge/) | a few seconds |

## Study routes

The course's [learning roadmap](https://dmxsf.com/en/wiki/start/learning-roadmap/) has three routes: application development (8 weeks), LLM engineering (12 weeks) and research (16 weeks). `uv run dmxsf roadmap application|engineer|researcher` lists the labs of each route week by week, and the site's [companion code page](https://dmxsf.com/en/code/) has the same tables. The labs do not depend on one another, except that the labs after pretraining share one cached small model; pick them at your own pace.

## How a lab is written

- A lab is `labs/<section>/<page>/run.py` plus two READMEs (`README.md` in Chinese, `README.en.md` in English). The script prints its results section by section and checks them one by one with `[ok]` / `[FAIL]` / `[skip]`; the exit status is 0 when everything passes.
- Every lab supports `--quick` (a shrunk smoke run; checks that only hold for the full run are marked `[skip]`) and `--seed N`.
- The small model: `models.pretrained("base")` is a byte-level GPT with 4 layers, width 128, 4 heads and 842,496 parameters, trained for 2400 steps on TinyShakespeare and cached in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); `"draft"` is the smaller one used for speculative decoding. The labs after pretraining (SFT, LoRA, preference optimization, quantization, ...) start from it.
- Why bytes and why so small: the course's conclusions (loss curves, quantization error, LoRA ranks, speculative-decoding acceptance, ...) have to be repeatable in minutes on any computer; scale is traded for verifiability. A phenomenon a small model does not show is not faked in a lab, and the README says so.

## Optional: real models

Three labs have an optional path that reproduces the course page's numbers with real models:

```bash
uv sync --extra hf       # transformers, datasets, pillow
uv run python labs/applications/prompt-engineering/run.py --hf
```

| Lab | Flag | Models and data |
| --- | --- | --- |
| `applications/prompt-engineering` | `--hf` | Qwen/Qwen2.5-1.5B-Instruct |
| `applications/rag` | `--bge` | BAAI/bge-small-en-v1.5, BAAI/bge-small-zh-v1.5, BAAI/bge-reranker-base |
| `multimodal/multimodal-llms` | `--hf` | openai/clip-vit-base-patch32, CIFAR-10 (`uoft-cs/cifar10`) |

The models and the dataset come from Hugging Face; see each model card for its licence. Once downloaded, add `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` to run offline.

## What is not reproduced here

- The course's Agents page reports results of a real model (Qwen3-4B), and the multimodal page latency and OCR measurements of Qwen3-VL; those need specific models and MLX on Apple silicon and are not reproduced here. The matching labs verify the evaluation harness and the arithmetic with scripted stand-ins for the model, and their READMEs say so.
- Numbers trained on a small model (losses, perplexities, accuracies) are not the numbers of a real large model on the page: the labs test methods, formulas and trends, not absolute values.

## Reproducibility

- Reference machine: Apple M2 Max (12 cores, 64 GB, CPU only), Python 3.12, PyTorch 2.15.0. The "Output" of each lab's README is a complete run on this machine.
- All randomness is seeded. Repeating a run on the same machine with the same PyTorch version gives the same output; on another machine or version the numbers that involve training can differ in the last digits, so the checks use thresholds and tolerances rather than bit-for-bit equality. Labs with large seed-to-seed spread (SFT, LoRA, preference optimization, ...) measure the noise floor first and do not interpret differences smaller than it.
- Timing numbers (the KV-cache speed-up, continuous-batching throughput, speculative-decoding speed) depend on the machine and its load; the checks look at ratios and directions only, with wide thresholds; do not run something else CPU-heavy at the same time.

## Development

```bash
uv sync --extra hf --group dev   # development dependencies
uv run pytest -m "not slow"      # unit tests of the library modules and the course-map checks
uv run pytest                    # plus a --quick smoke run of every lab
uv run ruff check . && uv run ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs the same checks on every push and pull request. Conventions and the steps for adding a lab are in [AGENTS.md](AGENTS.md).

## Licence

MIT, see [LICENSE](LICENSE). `src/dmxsf/data_files/tinyshakespeare.txt` is the public-domain text of Shakespeare's works (the TinyShakespeare file of Andrej Karpathy's char-rnn project).
