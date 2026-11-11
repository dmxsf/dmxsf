# DMXSF 配套代码

[DMXSF](https://dmxsf.com) 大模型算法教程的配套代码：用 Python 和 PyTorch 把课程里的核心算法从零重新实现一遍，每篇 Wiki 都有一个能运行、会自己检查结果的实验。

[![CI](https://github.com/dmxsf/dmxsf/actions/workflows/ci.yml/badge.svg)](https://github.com/dmxsf/dmxsf/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.en.md) · [课程网站](https://dmxsf.com) · [代码与课程的对应表](https://dmxsf.com/code/)

## 它是什么

- **和课程一一对应。** 30 个实验覆盖课程的全部 35 篇 Wiki：实验目录 `labs/<章节>/<页面>/` 与 Wiki 路径同名，每个实验的 README 链回它的课程页面，每篇 Wiki 的页尾也有“配套代码”卡片指向这里。这份对应关系只写在一处：[`src/dmxsf/course.py`](src/dmxsf/course.py)。
- **复现页面上的数字。** 公式、表格和实验结果在实验里重新算一遍并逐项核对，通过打印 `[ok]`，对不上打印 `[FAIL]` 并以非零状态退出。能离线复现的都复现了；必须用真实大模型才能得到的数字是可选路径，不装也不影响其余内容（见[可选：真实模型](#可选真实模型)）。
- **一台笔记本就够。** 全部在 CPU 上运行，不要 GPU，不调用任何 API。训练、微调、量化、投机解码、连续批处理都是真的在一个 84 万参数的字节级小 GPT 上跑出来的（词表 256，TinyShakespeare 语料，首次运行要花约 3 分钟训练并缓存），不是示意。
- **代码就是教材。** 库代码在 `src/dmxsf/`，一个模块对应一到两篇课程页面，写法偏教学；核心模块有单元测试，实验脚本自己还有逐项检查。

## 快速开始

需要 Python 3.10 以上。推荐用 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/dmxsf/dmxsf.git
cd dmxsf
uv sync                                # 安装依赖（PyTorch 与 NumPy）
uv run dmxsf list                      # 列出全部实验
uv run dmxsf run attention-basics      # 运行一个实验；先加 --quick 快速跑一遍也行
uv run dmxsf roadmap engineer          # 一条学习路线按周该跑哪些实验
uv run dmxsf doctor                    # 检查环境
```

不用 uv 也可以：`pip install -e .` 之后把命令里的 `uv run` 去掉。也可以直接运行脚本：`python labs/architecture/attention-basics/run.py`。Linux 上没有 GPU 的话，可以只装 CPU 版的 PyTorch（下载小得多，CI 就是这样装的）：`uv venv && uv pip install --torch-backend cpu -e .`，然后 `source .venv/bin/activate`，同样去掉命令里的 `uv run`。

## 课程与实验的对应

下表是全部实验，按课程的章节顺序排列（也是学习顺序）。耗时是完整运行在 Apple M2 Max（CPU）上的实测，模型已缓存。

### 入门指南（`start`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`what-is-an-llm`](labs/start/what-is-an-llm/) | 大模型是什么：一个小到能读完的完整循环 | [大模型是什么：从“预测下一个词”到通用助手](https://dmxsf.com/wiki/start/what-is-an-llm/)<br>[大模型学习路线图：应用、工程与研究三条路线](https://dmxsf.com/wiki/start/learning-roadmap/) | 几秒 |

### 基础概念（`foundations`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`next-token-prediction`](labs/foundations/next-token-prediction/) | 下一个词预测与信息论 | [从语言建模到下一个 token 预测](https://dmxsf.com/wiki/foundations/next-token-prediction/)<br>[信息论回顾：熵、交叉熵与 KL 散度](https://dmxsf.com/wiki/foundations/information-theory/) | 几秒 |
| [`tokenizer-and-embeddings`](labs/foundations/tokenizer-and-embeddings/) | 分词器与词嵌入 | [词嵌入与 Tokenizer 原理](https://dmxsf.com/wiki/foundations/tokenizer-and-embeddings/) | 几秒 |
| [`optimizers-and-training-stability`](labs/foundations/optimizers-and-training-stability/) | 优化器与训练稳定性 | [优化器与训练稳定性：AdamW、学习率调度、混合精度与损失尖峰](https://dmxsf.com/wiki/foundations/optimizers-and-training-stability/) | 几秒 |

### 模型架构（`architecture`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`attention-basics`](labs/architecture/attention-basics/) | 注意力机制 | [Transformer 注意力机制入门](https://dmxsf.com/wiki/architecture/attention-basics/) | 约 35 秒 |
| [`positional-encoding`](labs/architecture/positional-encoding/) | 位置编码与长上下文 | [位置编码详解（绝对 → 相对 → RoPE）](https://dmxsf.com/wiki/architecture/positional-encoding/)<br>[长上下文：位置外推、训练配方与成本](https://dmxsf.com/wiki/architecture/long-context/) | 约 2 分钟 |
| [`transformer-block`](labs/architecture/transformer-block/) | Transformer Block 与现代架构 | [Transformer Block 完整拆解](https://dmxsf.com/wiki/architecture/transformer-block/)<br>[现代大模型架构：从 GPT-2 到 Llama、Qwen 与 DeepSeek](https://dmxsf.com/wiki/architecture/modern-llm-architecture/) | 约 3 分钟 |
| [`efficient-attention`](labs/architecture/efficient-attention/) | 高效注意力：GQA、MLA 与 FlashAttention | [高效注意力：MQA、GQA、MLA 与 FlashAttention](https://dmxsf.com/wiki/architecture/efficient-attention/) | 约 55 秒 |
| [`mixture-of-experts`](labs/architecture/mixture-of-experts/) | 混合专家（MoE） | [混合专家（MoE）：路由、负载均衡与稀疏激活](https://dmxsf.com/wiki/architecture/mixture-of-experts/) | 约 3 分钟 |

### 预训练（`pretraining`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`training-pipeline`](labs/pretraining/training-pipeline/) | 预训练流水线 | [大模型训练流水线总览](https://dmxsf.com/wiki/pretraining/training-pipeline/) | 约 3 分钟 |
| [`scaling-laws`](labs/pretraining/scaling-laws/) | 扩展定律 | [扩展定律：算力、参数与数据如何分配](https://dmxsf.com/wiki/pretraining/scaling-laws/) | 约 2 分钟 |
| [`compute-and-memory`](labs/pretraining/compute-and-memory/) | 训练的算力与显存 | [训练的算力与显存账：FLOPs、MFU 与显存预算](https://dmxsf.com/wiki/pretraining/compute-and-memory/) | 几秒 |
| [`distributed-training`](labs/pretraining/distributed-training/) | 分布式训练 | [分布式训练入门：数据并行、张量并行与流水线并行](https://dmxsf.com/wiki/pretraining/distributed-training/) | 约 15 秒 |

### 后训练与对齐（`post-training`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`sft-data`](labs/post-training/sft-data/) | SFT 数据：损失看到什么 | [SFT 数据构造与质量控制](https://dmxsf.com/wiki/post-training/sft-data/) | 约 3 分钟 |
| [`lora-finetuning`](labs/post-training/lora-finetuning/) | LoRA 微调 | [LoRA 微调实战](https://dmxsf.com/wiki/post-training/lora-finetuning/) | 约 4 分钟 |
| [`preference-optimization`](labs/post-training/preference-optimization/) | 偏好优化（DPO） | [偏好优化：从 RLHF 到 DPO](https://dmxsf.com/wiki/post-training/preference-optimization/) | 约 6 分钟 |
| [`reasoning-rl`](labs/post-training/reasoning-rl/) | 推理强化学习（GRPO 与可验证奖励） | [推理强化学习：RLVR 与 GRPO](https://dmxsf.com/wiki/post-training/reasoning-rl/) | 约 3 分钟 |

### 推理与部署（`inference`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`decoding-and-sampling`](labs/inference/decoding-and-sampling/) | 解码与采样 | [解码与采样：温度、Top-k、Top-p 与束搜索](https://dmxsf.com/wiki/inference/decoding-and-sampling/) | 约 10 秒 |
| [`kv-cache`](labs/inference/kv-cache/) | KV Cache | [KV Cache 与推理性能优化](https://dmxsf.com/wiki/inference/kv-cache/) | 几秒 |
| [`quantization`](labs/inference/quantization/) | 量化 | [模型量化：INT8、INT4、GPTQ 与 AWQ 的原理与实测](https://dmxsf.com/wiki/inference/quantization/) | 约 20 秒 |
| [`speculative-decoding`](labs/inference/speculative-decoding/) | 投机解码 | [投机解码：用小模型打草稿，让大模型一次验证多个 token](https://dmxsf.com/wiki/inference/speculative-decoding/) | 约 35 秒 |
| [`continuous-batching`](labs/inference/continuous-batching/) | 推理服务：连续批处理、分页 KV 与前缀缓存 | [连续批处理与 PagedAttention：推理服务如何同时服务很多请求](https://dmxsf.com/wiki/inference/continuous-batching/)<br>[LLM 推理服务：延迟指标、实测方法与引擎选型](https://dmxsf.com/wiki/inference/serving/) | 约 10 秒 |

### 应用工程（`applications`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`prompt-engineering`](labs/applications/prompt-engineering/) | Prompt Engineering：可测量的提示词 | [Prompt Engineering 系统指南：任务规格、消息分层、示例与评测回归](https://dmxsf.com/wiki/applications/prompt-engineering/) | 几秒 |
| [`rag`](labs/applications/rag/) | RAG：逐个零件 | [RAG 系统搭建实战](https://dmxsf.com/wiki/applications/rag/) | 几秒 |
| [`structured-output-and-function-calling`](labs/applications/structured-output-and-function-calling/) | 结构化输出与函数调用 | [结构化输出与函数调用：从 JSON 模式到受约束解码](https://dmxsf.com/wiki/applications/structured-output-and-function-calling/) | 约 2 分钟 |
| [`agents-and-tool-calling`](labs/applications/agents-and-tool-calling/) | Agent 与工具调用 | [Agent 与工具调用实战：从 Function Calling 到 ReAct 工作流](https://dmxsf.com/wiki/applications/agents-and-tool-calling/) | 几秒 |

### 多模态（`multimodal`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`multimodal-llms`](labs/multimodal/multimodal-llms/) | 多模态大模型 | [多模态 LLM 原理与应用：图像如何变成 token](https://dmxsf.com/wiki/multimodal/multimodal-llms/) | 约 110 秒 |

### 评测（`evaluation`）

| 实验 | 做什么 | 课程页面 | 完整运行 |
| --- | --- | --- | --- |
| [`llm-evaluation`](labs/evaluation/llm-evaluation/) | 基准污染 | [LLM 评测方法论：基准、数据污染与评测流程](https://dmxsf.com/wiki/evaluation/llm-evaluation/) | 约 2 分钟 |
| [`eval-statistics`](labs/evaluation/eval-statistics/) | 评测统计 | [评测的统计学：pass@k、置信区间与所需题数](https://dmxsf.com/wiki/evaluation/eval-statistics/) | 几秒 |
| [`llm-as-judge`](labs/evaluation/llm-as-judge/) | LLM 评委 | [LLM-as-a-Judge：用模型当评委，以及它的偏差与校准](https://dmxsf.com/wiki/evaluation/llm-as-judge/) | 几秒 |

## 学习路线

课程的[学习路线图](https://dmxsf.com/wiki/start/learning-roadmap/)有三条路线：应用开发（8 周）、LLM 工程（12 周）和研究（16 周）。`uv run dmxsf roadmap application|engineer|researcher` 会按周列出各路线该跑的实验，网站的[配套代码页](https://dmxsf.com/code/)有同样的表。实验之间没有依赖，只是预训练之后的那些实验共用同一个缓存的小模型；按自己的节奏挑着做即可。

## 实验是怎么写的

- 每个实验是 `labs/<章节>/<页面>/run.py` 加两份 README（`README.md` 中文，`README.en.md` 英文）。脚本分节打印结果，用 `[ok]` / `[FAIL]` / `[skip]` 逐条检查，全部通过退出码为 0。
- 所有实验都支持 `--quick`（缩小规模的冒烟运行；只有完整运行才成立的检查标 `[skip]`）和 `--seed N`。
- 小模型：`models.pretrained("base")` 是 4 层、宽 128、4 个头、842,496 个参数的字节级 GPT，在 TinyShakespeare 上训练 2400 步，缓存在 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置）；`"draft"` 是投机解码用的更小的一个。预训练之后的实验（SFT、LoRA、偏好优化、量化……）都从它出发。
- 为什么是字节级、为什么这么小：课程里的结论（损失曲线、量化误差、LoRA 的秩、投机解码的接受率……）要能在几分钟内、在任何一台电脑上重复，规模换来的是可检验。小模型学不到的现象，实验里不假装演示，README 会写明。

## 可选：真实模型

三个实验有可选路径，用真实模型复现课程页里的数字：

```bash
uv sync --extra hf       # transformers、datasets、pillow
uv run python labs/applications/prompt-engineering/run.py --hf
```

| 实验 | 参数 | 用到的模型与数据 |
| --- | --- | --- |
| `applications/prompt-engineering` | `--hf` | Qwen/Qwen2.5-1.5B-Instruct |
| `applications/rag` | `--bge` | BAAI/bge-small-en-v1.5、BAAI/bge-small-zh-v1.5、BAAI/bge-reranker-base |
| `multimodal/multimodal-llms` | `--hf` | openai/clip-vit-base-patch32、CIFAR-10（`uoft-cs/cifar10`） |

模型和数据集从 Hugging Face 下载，许可证见各自的模型卡；下载好之后可以加 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 离线运行。

## 哪些数字不在这里复现

- 课程里 Agent 页用真实模型（Qwen3-4B）跑出的成绩，以及多模态页用 Qwen3-VL 测的延迟与 OCR 结果，需要特定的模型和 Apple 芯片上的 MLX，这里没有复现；对应的实验用脚本化的“模型替身”验证评测框架与算术本身，README 里写明了。
- 在小模型上训练出来的数字（损失、困惑度、准确率）与页面上真实大模型的数字不是一回事：实验检验的是方法、公式和趋势，不是绝对数值。

## 复现性

- 参考机器：Apple M2 Max（12 核，64 GB，仅 CPU），Python 3.12，PyTorch 2.15.0。每个实验 README 的“输出”是在这台机器上的完整运行结果。
- 随机性全部固定。同一台机器、同一个 PyTorch 版本上重复运行，得到同样的输出；换机器或版本，训练出来的数字可能在末位上不同，所以检查用的是阈值与容差，不要求逐位相同。种子之间波动较大的实验（SFT、LoRA、偏好优化……）会先量噪声底，小于噪声的差别不解读。
- 计时类数字（KV Cache 的加速、连续批处理的吞吐、投机解码的速度）取决于机器和当时的负载，检查只看比值与方向，阈值放得很宽；跑这几个实验时不要同时做别的吃 CPU 的事。

## 开发

```bash
uv sync --extra hf --group dev   # 开发依赖
uv run pytest -m "not slow"      # 库模块的单元测试与对照表检查
uv run pytest                    # 另含每个实验的 --quick 冒烟运行
uv run ruff check . && uv run ruff format --check .
```

CI（`.github/workflows/ci.yml`）在推送与 PR 时运行同样的检查。维护约定与新增实验的步骤见 [AGENTS.md](AGENTS.md)。

## 许可

MIT，见 [LICENSE](LICENSE)。`src/dmxsf/data_files/tinyshakespeare.txt` 是莎士比亚作品的公版文本（取自 Andrej Karpathy 的 char-rnn 项目所用的 TinyShakespeare）。
