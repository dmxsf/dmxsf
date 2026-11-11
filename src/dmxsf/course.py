"""The course map: which lab belongs to which page of https://dmxsf.com, and the three study routes.

This file is the single source of truth for the relation between the course and the code. Every lab is
a directory ``labs/<section>/<slug>/`` whose ``<slug>`` is the wiki page it is built around (the labs of
the ``start`` section are the exception, see below); a lab may cover several pages. ``tests/test_course.py``
checks that every page of the course is covered, that each lab's ``run.py`` declares the same pages, and
that the three routes of the learning roadmap only point to labs that exist.

The same table lives in the site repository (``src/data/code-labs.ts``), which draws the "companion
code" block under every wiki page; ``dmxsf list`` prints it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SITE = "https://dmxsf.com"
REPO = "https://github.com/dmxsf/dmxsf"


@dataclass(frozen=True)
class Lab:
    section: str  # the wiki section, also the directory under labs/
    slug: str  # the primary wiki page, also the lab's directory
    wiki: tuple[str, ...]  # every wiki key the lab covers, the primary page first
    title: str  # what the lab prints as its heading
    modules: tuple[str, ...]  # the library code under src/dmxsf/ the lab is built on
    seconds: int  # about how long the full run takes on a laptop CPU (see the lab's README)
    extra: str | None = None  # an optional extra (``uv sync --extra hf``) for the ``--hf`` paths

    @property
    def key(self) -> str:
        return f"{self.section}/{self.slug}"

    @property
    def path(self) -> str:
        return f"labs/{self.section}/{self.slug}"

    @property
    def url(self) -> str:
        return f"{REPO}/tree/main/{self.path}"

    def text(self, lang: str) -> tuple[str, str]:
        """(title, summary) in ``"en"`` or ``"zh"``."""
        zh_title, summary, zh_summary = TEXTS[self.key]
        return (self.title, summary) if lang == "en" else (zh_title, zh_summary)


LABS: tuple[Lab, ...] = (
    Lab(
        "start",
        "what-is-an-llm",
        ("start/what-is-an-llm", "start/learning-roadmap"),
        "What is an LLM? The whole loop, small enough to read",
        ("models", "inference.generate", "foundations.tokenizer", "foundations.information"),
        4,
    ),
    Lab(
        "foundations",
        "next-token-prediction",
        ("foundations/next-token-prediction", "foundations/information-theory"),
        "Next-token prediction and information theory",
        ("foundations.information", "foundations.language_model"),
        4,
    ),
    Lab(
        "foundations",
        "tokenizer-and-embeddings",
        ("foundations/tokenizer-and-embeddings",),
        "Tokenizers and embeddings",
        ("foundations.tokenizer",),
        7,
    ),
    Lab(
        "foundations",
        "optimizers-and-training-stability",
        ("foundations/optimizers-and-training-stability",),
        "Optimizers, schedules, mixed precision and loss spikes",
        ("foundations.optim",),
        3,
    ),
    Lab(
        "architecture",
        "attention-basics",
        ("architecture/attention-basics",),
        "Attention, by hand and in a model",
        ("architecture.attention",),
        33,
    ),
    Lab(
        "architecture",
        "positional-encoding",
        ("architecture/positional-encoding", "architecture/long-context"),
        "Position encodings and length extrapolation",
        ("architecture.position",),
        127,
    ),
    Lab(
        "architecture",
        "transformer-block",
        ("architecture/transformer-block", "architecture/modern-llm-architecture"),
        "The Transformer block and the modern recipe",
        ("architecture.gpt", "architecture.layers", "architecture.configs"),
        166,
    ),
    Lab(
        "architecture",
        "efficient-attention",
        ("architecture/efficient-attention",),
        "Efficient attention",
        ("architecture.attention", "architecture.mla", "inference.kv_cache"),
        57,
    ),
    Lab(
        "architecture",
        "mixture-of-experts",
        ("architecture/mixture-of-experts",),
        "Mixture of experts: routing, load balance and the parameter accounts",
        ("architecture.moe", "architecture.deepseek"),
        172,
    ),
    Lab(
        "pretraining",
        "training-pipeline",
        ("pretraining/training-pipeline",),
        "The pretraining pipeline on a laptop: monitoring, data mix, resuming, leakage",
        ("pretraining.train", "data"),
        165,
    ),
    Lab(
        "pretraining",
        "scaling-laws",
        ("pretraining/scaling-laws",),
        "Scaling laws: the formulas, the serving cost, and a tiny study of our own",
        ("pretraining.scaling", "pretraining.accounting"),
        138,
    ),
    Lab(
        "pretraining",
        "compute-and-memory",
        ("pretraining/compute-and-memory",),
        "Training compute and memory: 6ND, MFU, model state and activations",
        ("pretraining.accounting",),
        4,
    ),
    Lab(
        "pretraining",
        "distributed-training",
        ("pretraining/distributed-training",),
        "Distributed training: gradients, ZeRO, tensor and pipeline parallelism",
        ("pretraining.parallel", "pretraining.distributed"),
        17,
    ),
    Lab(
        "post-training",
        "sft-data",
        ("post-training/sft-data",),
        "SFT data: the loss mask, a data health check, and what data size, noise, masking and replay do",
        ("post_training.sft", "post_training.data_quality", "post_training.intents"),
        207,
    ),
    Lab(
        "post-training",
        "lora-finetuning",
        ("post-training/lora-finetuning",),
        "LoRA: the parameter bill, the mechanics, LoRA against full fine-tuning, merging, NF4",
        ("post_training.lora",),
        224,
    ),
    Lab(
        "post-training",
        "preference-optimization",
        ("post-training/preference-optimization",),
        "Preference optimization: the DPO loss, its closed form, and a real DPO run",
        ("post_training.preference",),
        359,
    ),
    Lab(
        "post-training",
        "reasoning-rl",
        ("post-training/reasoning-rl",),
        "Reasoning RL: verifiers, GRPO advantages and loss, and a real RLVR run",
        ("post_training.reasoning",),
        197,
    ),
    Lab(
        "inference",
        "decoding-and-sampling",
        ("inference/decoding-and-sampling",),
        "Decoding and sampling: temperature, top-k, top-p, min-p, beam search, penalties",
        ("inference.generate",),
        10,
    ),
    Lab(
        "inference",
        "kv-cache",
        ("inference/kv-cache",),
        "KV cache: the memory formula, exactness, the saving, prefill against decode",
        ("inference.kv_cache",),
        6,
    ),
    Lab(
        "inference",
        "quantization",
        ("inference/quantization",),
        "Quantization: the ledger, the grid, weights, activations, calibration, the KV cache",
        ("inference.quantization",),
        20,
    ),
    Lab(
        "inference",
        "speculative-decoding",
        ("inference/speculative-decoding",),
        "Speculative decoding: the rule, the formulas, a real draft and target",
        ("inference.speculative",),
        34,
    ),
    Lab(
        "inference",
        "continuous-batching",
        ("inference/continuous-batching", "inference/serving"),
        "Serving: continuous batching, paged KV memory, prefix cache, TTFT, capacity",
        ("inference.serving",),
        12,
    ),
    Lab(
        "applications",
        "prompt-engineering",
        ("applications/prompt-engineering",),
        "Prompting, measured",
        ("applications.prompting", "applications.prompt_data", "applications.hf"),
        1,
        extra="hf",
    ),
    Lab(
        "applications",
        "rag",
        ("applications/rag",),
        "RAG, piece by piece",
        ("applications.rag", "applications.rag_data"),
        1,
        extra="hf",
    ),
    Lab(
        "applications",
        "structured-output-and-function-calling",
        ("applications/structured-output-and-function-calling",),
        "Structured output: constrained decoding, its cost, and the function-calling loop",
        ("applications.constrained",),
        136,
    ),
    Lab(
        "applications",
        "agents-and-tool-calling",
        ("applications/agents-and-tool-calling",),
        "Agents and tool calling: the loop, its guards, and the trace scorer",
        ("applications.agent", "applications.policies"),
        1,
    ),
    Lab(
        "multimodal",
        "multimodal-llms",
        ("multimodal/multimodal-llms",),
        "Multimodal LLMs: images as tokens",
        ("multimodal.vision", "multimodal.clip"),
        113,
        extra="hf",
    ),
    Lab(
        "evaluation",
        "llm-evaluation",
        ("evaluation/llm-evaluation",),
        "Benchmark contamination: an n-gram detector, and a leak you can measure",
        ("evaluation.contamination",),
        144,
    ),
    Lab(
        "evaluation",
        "eval-statistics",
        ("evaluation/eval-statistics",),
        "Evaluation statistics: pass@k, the error of a score, paired tests, sample sizes",
        ("evaluation.stats",),
        2,
    ),
    Lab(
        "evaluation",
        "llm-as-judge",
        ("evaluation/llm-as-judge",),
        "LLM-as-a-judge: bias, agreement, and Bradley-Terry leaderboards",
        ("evaluation.judge",),
        1,
    ),
)

# --- the three routes of the learning roadmap (https://dmxsf.com/wiki/start/learning-roadmap/) --------
# What each lab does, in the two languages of the course: (Chinese title, English summary, Chinese
# summary). The English title is Lab.title. The site's "companion code" blocks are generated from this.
TEXTS: dict[str, tuple[str, str, str]] = {
    "start/what-is-an-llm": (
        "大模型是什么：一个小到能读完的完整循环",
        "Predict one byte, append it, repeat: the whole loop on an 842,496-parameter GPT you can open, from tokens and embeddings to the softmax, the training objective and the three numbers (parameters, tokens, context). It ends with the map from every page of the course to its lab.",
        "在一个 842,496 参数、能整个打开的 GPT 上跑通完整循环：预测一个字节、接上去、再预测；从分词、嵌入到 softmax、训练目标和三个关键数字（参数、token、上下文）。末尾是全部课程页面与实验的对照表。",
    ),
    "foundations/next-token-prediction": (
        "下一个词预测与信息论",
        "Bigram counts on the page's toy corpus, entropy, cross-entropy and KL divergence checked against each other, character n-grams on Shakespeare until the counts run out, and a neural bigram that learns the count table by gradient descent.",
        "在课程页的玩具语料上数二元组，核对熵、交叉熵与 KL 散度的关系；在莎士比亚文本上看字符 n 元组在数据耗尽前的收益；再用梯度下降训练一个神经二元组，让它学出同一张计数表。",
    ),
    "foundations/tokenizer-and-embeddings": (
        "分词器与词嵌入",
        "BPE by hand on the page's toy corpus, a byte-level BPE that never says <unk>, what the training text decides about a language's cost, an embedding table as a matrix multiply, and how much of a model the embedding table is.",
        "手写 BPE（课程页的玩具语料），实现永不出现 <unk> 的字节级 BPE，看训练文本如何决定一种语言的 token 成本；嵌入表等价于一次矩阵乘法；以及嵌入表在整个模型里占多大比例。",
    ),
    "foundations/optimizers-and-training-stability": (
        "优化器与训练稳定性",
        "SGD, Adam and AdamW written out and checked against PyTorch in float64; why Adam wins on an ill-conditioned quadratic; decoupled weight decay; learning-rate schedules; mixed precision and master weights; one corrupted batch and what clipping and beta2 do about it.",
        "手写 SGD、Adam、AdamW 并在 float64 下与 PyTorch 对拍；Adam 为何在病态二次函数上更快；解耦的权重衰减；学习率调度；混合精度与 FP32 主权重；一个被污染的批次里梯度裁剪和 beta2 各起什么作用。",
    ),
    "architecture/attention-basics": (
        "注意力机制",
        "The page's three-position example by hand, why scores are divided by sqrt(d) (the dot product has variance d), masks, heads and positions, and a small model that has to use attention to copy what followed the last occurrence.",
        "手算课程页的三位置例子；为什么要除以 sqrt(d)（点积的方差是 d）；掩码、多头与位置；再训练一个必须靠注意力才能完成“复制上次出现之后的内容”的小模型。",
    ),
    "architecture/positional-encoding": (
        "位置编码与长上下文",
        "Sinusoidal encodings as clocks, RoPE's score depending only on distance, the stretching recipes (position interpolation, NTK-aware, YaRN, the Llama 3.1 rule) computed side by side, and five position schemes trained on short text and tested on long text.",
        "正弦编码是一组周期不同的时钟；RoPE 的分数只依赖相对距离；位置插值、NTK-aware、YaRN 与 Llama 3.1 的拉伸配方并排计算；五种位置方案在短文本上训练、在长文本上测试。",
    ),
    "architecture/transformer-block": (
        "Transformer Block 与现代架构",
        "Count a GPT-2 small block (12d²+13d parameters), see where real models keep their parameters, measure signal and gradient by depth for pre-norm and post-norm, then walk from GPT-2 to Llama one change at a time.",
        "数清一个 GPT-2 small 的 Block（12d²+13d 个参数）；看真实模型的参数分布；比较 Pre-Norm 与 Post-Norm 在不同深度的信号与梯度；再从 GPT-2 到 Llama，一次只改一处。",
    ),
    "architecture/efficient-attention": (
        "高效注意力：GQA、MLA 与 FlashAttention",
        "KV-cache formulas against the page's table and against real tensors, MLA caching a latent vector, FlashAttention's tiles and online softmax, sliding-window attention, and converting an MHA model to GQA by mean-pooling heads.",
        "KV Cache 公式与课程页的表格、真实张量对拍；MLA 缓存潜向量；FlashAttention 的分块与在线 softmax；滑动窗口注意力；把 MHA 模型通过头的均值池化转成 GQA 并微调。",
    ),
    "architecture/mixture-of-experts": (
        "混合专家（MoE）",
        "A MoE layer that computes only the tokens an expert was chosen for, total against active parameters counted on real modules, a DeepSeek-V3-style toy model, expert capacity, the balance loss, how many experts a decoding batch touches, and a trained MoE against dense models of equal cost and equal size.",
        "只计算被路由到的 token 的 MoE 层；在真实模块上数总参数与激活参数；能跑起来的 DeepSeek-V3 风格玩具模型；专家容量；负载均衡损失；解码时一个批次会激活多少专家；以及在字节数据上训练的 MoE 与同算力、同参数量稠密模型的对比。",
    ),
    "pretraining/training-pipeline": (
        "预训练流水线",
        "Two free checks before spending compute, a baseline run and how to read its log, then three experiments on a laptop: the data mix, four ways to resume from a checkpoint (only one is exact), and validation text leaking into the training data.",
        "花算力之前的两个免费检查；基线训练与监控日志的读法；然后是三个实验：数据配比、从检查点恢复的四种做法（只有一种严格精确）、验证集泄漏进训练数据。",
    ),
    "pretraining/scaling-laws": (
        "扩展定律",
        "The page's power laws and compute-optimal split recomputed, what serving cost does to the best model, a real tiny scaling study (tune the learning rate per size, train a grid, fit L(N, D), predict an unseen run), and what a 12-point fit can and cannot identify.",
        "重算课程页的幂律与计算最优配比；推理成本如何改变最优模型；一次真实的小型扩展实验（按规模调学习率、训练网格、拟合 L(N, D)、预测没跑过的配置）；以及 12 个点的拟合能识别什么、不能识别什么。",
    ),
    "pretraining/compute-and-memory": (
        "训练的算力与显存",
        "Every formula on the page checked against a measurement: PyTorch's FLOP counter, the tensors autograd saves for the backward pass, and the optimizer's state; then MFU, GPU hours and a full Llama 2 7B account on 8 x 80 GiB.",
        "页面上的每个公式都用实测核对：PyTorch 的 FLOP 计数器、自动求导为反向保存的张量、优化器状态；再到 MFU、GPU 小时，以及 8 × 80 GiB 上训练 Llama 2 7B 的完整账本。",
    ),
    "pretraining/distributed-training": (
        "分布式训练",
        "Built and measured in a few processes on one machine: data-parallel gradient averaging, the ring all-reduce, data parallelism, ZeRO-1 and ZeRO-3 trained side by side (same weights, measured memory and communication), Megatron-style tensor parallelism, and pipeline schedules.",
        "在一台机器的多个进程里搭起来并实测：数据并行的梯度平均、环形 all-reduce、数据并行与 ZeRO-1 / ZeRO-3 并排训练（权重一致，实测显存与通信量）、Megatron 式张量并行、流水线调度。",
    ),
    "post-training/sft-data": (
        "SFT 数据：损失看到什么",
        "What the SFT loss averages over (the -100 mask, packing and padding), data health checks against planted ground truth, and four controlled experiments: run-to-run noise, examples per intent, wrong labels, loss on the prompt and replaying general text.",
        "SFT 损失到底在平均什么（-100 掩码、打包与填充）；用埋好答案的数据做健康检查；四个对照实验：同配置重复运行的波动、每类样本数、错误标签、提示部分算不算损失与回放通用文本。",
    ),
    "post-training/lora-finetuning": (
        "LoRA 微调",
        "LoRA written from scratch and compared with full fine-tuning: the exact parameter bill, learning-rate windows, rank and alpha, which matrices to adapt, how low-rank the learned update is, merging into the weights and into 4-bit (NF4) weights, memory and speed.",
        "从零实现 LoRA 并与全量微调对比：精确的参数账本、两者各自的学习率窗口、秩与 alpha、该适配哪些矩阵、学到的更新有多低秩、合并进权重和 4-bit（NF4）权重、显存与速度。",
    ),
    "post-training/preference-optimization": (
        "偏好优化（DPO）",
        "The DPO loss by hand, its closed-form optimum checked against random competitors and fitted comparisons, a real DPO run on a small byte-level model (learning rate, beta, chosen log-probabilities that fall, the LoRA-off reference), and reward hacking as the KL budget grows.",
        "手算 DPO 损失；闭式最优解对照随机竞争者与拟合结果；在小型字节模型上做真实 DPO（学习率、beta、被选回答的对数概率下降、以关掉 LoRA 作参考模型）；以及 KL 预算增大时的奖励黑客。",
    ),
    "post-training/reasoning-rl": (
        "推理强化学习（GRPO 与可验证奖励）",
        "The verifier, group advantages, the GRPO loss, clipping and the k3 KL estimator checked exactly; then a real RLVR run in which a tiny model is updated from nothing but exact-match 0/1 rewards.",
        "验证器、组内优势、GRPO 损失、裁剪与 k3 KL 估计量的精确核对；然后是真实的 RLVR 训练：小模型仅凭精确匹配给出的 0/1 奖励完成策略更新。",
    ),
    "inference/decoding-and-sampling": (
        "解码与采样",
        "The page's four-token example reproduced exactly (temperature, top-k, top-p, min-p, and why the order matters), whether samples follow the processed distribution, what each knob does to real text, beam search, repetition penalties, and whether batch shape changes greedy output.",
        "精确复现课程页的四 token 例子（温度、top-k、top-p、min-p 以及顺序为何重要）；采样是否服从处理后的分布；各个旋钮对真实文本的影响；束搜索、重复惩罚；批次形状是否会改变贪心输出。",
    ),
    "inference/kv-cache": (
        "KV Cache",
        "The memory formula against real cache tensors, cached and uncached decoding agreeing exactly, and the saving counted two ways: token positions processed (exact) and time on a CPU (read the ratios), plus the prefill and decode split.",
        "显存公式对照真实缓存张量；有缓存与无缓存的解码结果完全一致；节省的量用两种方式衡量：处理的 token 位置数（精确）与 CPU 计时（看比值）；再看预填充与解码的差别。",
    ),
    "inference/quantization": (
        "量化",
        "The 80 GB ledger for Llama 3 8B, the rounding grid, and experiments on a small model: round-to-nearest, SmoothQuant, an AWQ-style scale search, GPTQ and a quantized KV cache, with outliers dialled in and labelled as constructed.",
        "Llama 3 8B 在 80 GB 显卡上的账本；取整网格；以及在小模型上的实验：就近取整、SmoothQuant、AWQ 式缩放搜索、GPTQ、KV Cache 量化，其中的离群值是人为构造并明确标注的。",
    ),
    "inference/speculative-decoding": (
        "投机解码",
        "The accept/reject rule that keeps the target's exact distribution, checked on the page's toy distribution, by exact algebra and on real models; the formulas of Leviathan et al.; a draft and target byte-level pair run with KV caches; and when it does not pay.",
        "保持目标模型精确分布的接受/拒绝规则：在课程页的玩具分布上、用精确代数、在真实模型上逐一验证；Leviathan 等人的公式；带 KV Cache 的草稿/目标字节模型对；以及什么时候不划算。",
    ),
    "inference/continuous-batching": (
        "推理服务：连续批处理、分页 KV 与前缀缓存",
        "Roofline and exact batched decoding, static against continuous scheduling on paper and on a real model, a paged KV pool with copy-on-write, a hash-based prefix cache, and what a request costs (TTFT, TPOT, capacity). Read the ratios, not the milliseconds.",
        "Roofline 与精确的批量解码；静态与连续批处理的纸面推演和真实模型对比；带写时复制的分页 KV 池；基于哈希的前缀缓存；一个请求的成本（TTFT、TPOT、容量）。看比值，不看毫秒。",
    ),
    "applications/prompt-engineering": (
        "Prompt Engineering：可测量的提示词",
        "The chat template rendered by hand, what a prefix cache can reuse, scorers that separate format failures from wrong judgments, the twelve word problems re-solved in code, and how far a 12-ticket eval set can be trusted. With --hf the page's experiments are rerun on Qwen2.5-1.5B-Instruct.",
        "手工渲染对话模板；前缀缓存能复用什么；把格式错误与判断错误分开的评分器；用代码重新解十二道应用题；12 张工单的评测集有多可信。加 --hf 可在 Qwen2.5-1.5B-Instruct 上重跑页面里的实验。",
    ),
    "applications/rag": (
        "RAG：逐个零件",
        "Chunking, similarity and index size, BM25 from the formula, rank fusion (RRF), retrieval metrics and context assembly, reproducing the page's numbers offline; with --bge the dense retriever and the reranker run too.",
        "切块、相似度与索引大小、按公式实现的 BM25、排名融合（RRF）、检索指标与上下文拼装，离线复现课程页的数字；加 --bge 还会运行稠密检索器与重排序模型。",
    ),
    "applications/structured-output-and-function-calling": (
        "结构化输出与函数调用",
        "Constrained decoding from the automaton up: the page's demo token for token, why the stepwise mask is not the model's distribution over legal outputs, the mask on a byte-level GPT that has and has not seen the format, schemas, validation and retries.",
        "从自动机开始的约束解码：逐 token 复现课程页的演示；逐步掩码为什么不等于模型在合法输出上的分布；在见过与没见过该格式的字节模型上加掩码；Schema、校验与重试。",
    ),
    "applications/agents-and-tool-calling": (
        "Agent 与工具调用",
        "The tool-calling loop with its guards (format, business rules, repeats, confirmation, step limit) and the scorer, with the model swapped for a scripted stand-in that replays the page's decisions; then pass^k against pass@k. No language model runs.",
        "带护栏（格式、业务规则、重复调用、确认、步数上限）的工具调用循环与评分器；模型换成回放页面决策的脚本替身；最后比较 pass^k 与 pass@k。这里不运行任何语言模型。",
    ),
    "multimodal/multimodal-llms": (
        "多模态大模型",
        "Patches to visual tokens, what the tokens cost in KV cache and FLOPs, LLaVA's projector and two training stages, CLIP's contrastive loss and a small CLIP trained from scratch on rendered shapes. With --hf, CLIP ViT-B/32 on CIFAR-10.",
        "从图像块到视觉 token；这些 token 在 KV Cache 与 FLOPs 上的成本；LLaVA 的投影层与两阶段训练；CLIP 的对比损失，以及在渲染图形上从零训练的小型 CLIP。加 --hf 可在 CIFAR-10 上运行 CLIP ViT-B/32。",
    ),
    "evaluation/llm-evaluation": (
        "基准污染",
        "The page's n-gram contamination detector on its own example; a benchmark whose answers can only be memorised; a model trained with a third of it leaked verbatim, a third paraphrased and a third absent; what the detector flags and what the clean-subset check concludes.",
        "课程页的 n-gram 污染检测器；一个答案只能靠记忆得到的基准；模型的训练流里三分之一原样泄漏、三分之一改写、三分之一没有；检测器能抓到什么，以及“只在干净子集上重测”会得出什么结论。",
    ),
    "evaluation/eval-statistics": (
        "评测统计",
        "pass@k against brute-force enumeration, the 95% interval against how often it actually covers, paired comparisons and the sample-size table against the power of a real test, then two real models compared on held-out text.",
        "pass@k 对照暴力枚举；95% 区间对照真实覆盖率；配对比较与样本量表对照真实检验的功效；最后在留出文本上比较两个真实模型。",
    ),
    "evaluation/llm-as-judge": (
        "LLM 评委",
        "A judge stand-in with chosen biases, so every number has an exact answer: swapping the order cancels position bias but not length bias, Cohen's kappa exposes a judge that always guesses the majority class, and a Bradley-Terry leaderboard is corrected for length.",
        "一个偏差可调的评委替身，所以每个数字都有精确答案：交换顺序能抵消位置偏差却抵消不了长度偏差；Cohen's kappa 能识破永远猜多数类的评委；按长度校正的 Bradley-Terry 排行榜。",
    ),
}


# Each route is a list of weeks; a week lists the wiki pages the roadmap sets for it (weeks that only write
# things up have none), in reading order. Where a week's paper or deliverable has a lab of its own, that
# page is listed too (each is marked in a comment): the roadmap page names only the papers there.

ROUTES: dict[str, tuple[str, tuple[tuple[str, ...], ...]]] = {
    "application": (
        "AI application developer (8 weeks)",
        (
            ("start/what-is-an-llm", "applications/prompt-engineering"),
            (
                "inference/decoding-and-sampling",
                "evaluation/llm-evaluation",
                "evaluation/eval-statistics",  # the week's prompt A/B comparison needs error bars
            ),
            (),
            ("applications/rag",),
            ("applications/agents-and-tool-calling",),
            ("inference/serving", "inference/kv-cache"),
            ("inference/quantization",),
            (),
        ),
    ),
    "engineer": (
        "LLM engineer (12 weeks)",
        (
            ("foundations/next-token-prediction", "foundations/tokenizer-and-embeddings"),
            ("architecture/attention-basics",),
            ("architecture/transformer-block", "architecture/positional-encoding"),
            ("architecture/modern-llm-architecture",),
            ("pretraining/training-pipeline", "pretraining/scaling-laws"),
            ("pretraining/distributed-training",),
            ("post-training/sft-data", "post-training/lora-finetuning"),
            ("post-training/preference-optimization",),
            (
                "inference/kv-cache",
                "architecture/efficient-attention",
            ),  # FlashAttention is that week's paper
            (
                "inference/quantization",
                "inference/serving",
                "inference/continuous-batching",
            ),
            ("evaluation/llm-evaluation", "evaluation/eval-statistics"),
            (),
        ),
    ),
    "researcher": (
        "LLM researcher (16 weeks)",
        (
            (),
            (
                "architecture/attention-basics",
            ),  # the paper of the week is Attention Is All You Need
            ("pretraining/scaling-laws",),
            ("architecture/modern-llm-architecture",),
            ("architecture/mixture-of-experts",),  # Switch Transformers
            (),  # Mamba: no lab yet
            ("architecture/efficient-attention", "inference/serving"),  # FlashAttention and vLLM
            ("architecture/positional-encoding",),
            ("post-training/sft-data",),  # InstructGPT starts from SFT data
            ("post-training/preference-optimization",),
            ("post-training/reasoning-rl",),
            ("evaluation/llm-evaluation",),
            ("evaluation/eval-statistics",),  # a reproduction needs a noise floor
            ("evaluation/eval-statistics",),
            (),
            (),
        ),
    ),
}


def by_key() -> dict[str, Lab]:
    """Every wiki key the labs cover, mapped to the lab that covers it."""
    return {key: lab for lab in LABS for key in lab.wiki}


def find(name: str) -> Lab:
    """A lab by ``section/slug``, or by ``slug`` alone when that is unambiguous."""
    matches = [lab for lab in LABS if name in (lab.key, lab.slug)]
    if len(matches) != 1:
        raise KeyError(
            f"{name!r} matches {len(matches)} labs; try one of: "
            + ", ".join(lab.key for lab in LABS)
        )
    return matches[0]


def route_labs(route: str) -> list[tuple[int, Lab]]:
    """The labs of a route in study order as (week, lab), each lab at the first week that needs it."""
    covered = by_key()
    seen: set[str] = set()
    out: list[tuple[int, Lab]] = []
    for week, pages in enumerate(ROUTES[route][1], start=1):
        for page in pages:
            lab = covered[page]
            if lab.key not in seen:
                seen.add(lab.key)
                out.append((week, lab))
    return out


def repo_root() -> Path | None:
    """The checkout this package is installed from (editable), if the labs are there."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "labs").is_dir() else None
