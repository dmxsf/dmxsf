"""24 Chinese-English sentence pairs written for this lab (CC0 / MIT, like the rest of the repo).

TRAIN is what the tokenizers may learn from; HELD_OUT is only used to measure them.
"""

TRAIN = [
    (
        "语言模型的任务是根据前面的内容预测下一个 token。",
        "A language model predicts the next token from everything that came before it.",
    ),
    (
        "训练时用交叉熵损失衡量模型给正确答案打了多少概率。",
        "Training uses cross-entropy loss to measure how much probability the model gave to the correct answer.",
    ),
    (
        "注意力机制让每个位置都可以直接读取前面所有位置的信息。",
        "Attention lets every position read information from all the positions before it.",
    ),
    (
        "因果掩码保证模型在预测时看不到未来的 token。",
        "The causal mask makes sure the model cannot see future tokens when it predicts.",
    ),
    (
        "位置编码告诉模型每个 token 在序列中的位置。",
        "Position encoding tells the model where each token sits in the sequence.",
    ),
    (
        "残差连接让梯度可以直接穿过很深的网络。",
        "Residual connections let gradients flow straight through a very deep network.",
    ),
    (
        "词表越大，同样的文字需要的 token 越少，嵌入矩阵却会变大。",
        "A larger vocabulary needs fewer tokens for the same text, but it makes the embedding matrix bigger.",
    ),
    (
        "学习率通常先预热，再按余弦曲线慢慢降低。",
        "The learning rate usually warms up first and then decays slowly along a cosine curve.",
    ),
    (
        "混合专家模型每次只激活一小部分参数，所以计算量比总参数量小得多。",
        "A mixture-of-experts model activates only a small share of its parameters, so its compute is far below what its total size suggests.",
    ),
    (
        "推理时缓存已经算好的键和值，每个新 token 只需要一步计算。",
        "At inference time the keys and values that were already computed are cached, so each new token needs only one step.",
    ),
    (
        "量化把权重从十六位浮点数压缩成八位或四位整数，显存占用随之下降。",
        "Quantization compresses the weights from sixteen-bit floats to eight-bit or four-bit integers, and memory use drops with them.",
    ),
    (
        "低秩适配只训练很少的参数，所以微调需要的显存远小于全量微调。",
        "Low-rank adaptation trains very few parameters, so fine-tuning needs far less memory than full fine-tuning.",
    ),
    (
        "偏好数据由同一个问题的好回答和差回答组成。",
        "Preference data consists of a good answer and a bad answer to the same question.",
    ),
    (
        "检索增强生成先从知识库里找出相关片段，再交给模型回答。",
        "Retrieval-augmented generation first finds relevant passages in a knowledge base and then hands them to the model to answer.",
    ),
    (
        "智能体让模型反复调用工具，直到任务完成为止。",
        "An agent lets the model call tools again and again until the task is done.",
    ),
    (
        "多模态模型把图像切成小块，再映射成和文字 token 同样的向量。",
        "A multimodal model cuts an image into patches and maps them to the same kind of vectors as text tokens.",
    ),
    (
        "评测基准的题目如果出现在训练数据里，分数就会虚高。",
        "If the questions of a benchmark appear in the training data, the score is inflated.",
    ),
    (
        "模型的上下文窗口是有限的，所以长文档需要先切成块。",
        "The context window of a model is limited, so a long document has to be cut into chunks first.",
    ),
]

HELD_OUT = [
    (
        "评估模型时不仅要看平均分，还要看这个分数有多大的不确定性。",
        "When evaluating a model, look not only at the average score but also at how uncertain that score is.",
    ),
    (
        "批大小越大，每一步的梯度噪声越小，但是需要的显存也越多。",
        "A larger batch size lowers the gradient noise of each step, but it also needs more memory.",
    ),
    (
        "推理服务既要关注首个 token 的延迟，也要关注总的吞吐量。",
        "An inference service has to watch both the latency of the first token and the overall throughput.",
    ),
    (
        "扩展定律描述了损失如何随着参数量、数据量和算力的增加而下降。",
        "Scaling laws describe how the loss falls as the number of parameters, the amount of data and the compute grow.",
    ),
    (
        "投机解码用一个小模型起草，再让大模型一次验证多个 token。",
        "Speculative decoding drafts with a small model and lets the large model verify several tokens at once.",
    ),
    (
        "提示词应该写清楚任务、格式和限制条件，而不是只给一句模糊的要求。",
        "A prompt should state the task, the format and the constraints instead of giving one vague sentence.",
    ),
]
