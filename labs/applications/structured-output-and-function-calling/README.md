# 结构化输出与函数调用

课程页面：[结构化输出与函数调用：从 JSON 模式到受约束解码](https://dmxsf.com/wiki/applications/structured-output-and-function-calling/)  
[English](README.en.md)

从自动机开始的约束解码：逐 token 复现课程页的演示；逐步掩码为什么不等于模型在合法输出上的分布；在见过与没见过该格式的字节模型上加掩码；Schema、校验与重试。

## 运行

```bash
uv run python labs/applications/structured-output-and-function-calling/run.py            # 完整运行，约 2 分钟
uv run python labs/applications/structured-output-and-function-calling/run.py --quick    # 缩小规模的冒烟测试，约 10 秒
```

也可以写成 `uv run dmxsf run structured-output-and-function-calling`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/applications/structured-output-and-function-calling/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/applications/constrained.py` | 约束解码、结构化输出与函数调用循环 |

单元测试：`uv run pytest tests/test_applications_constrained.py`

## 看什么

- 第 1 节：页面里的自动机演示逐 token 复现：24 个状态 × 30 个 token = 720 个（状态, token）对，真正能写的只有 49 个；对一个什么都不懂的模型（随机 logits）生成 2,000 次，不加掩码时 1,928 次非法、72 次是合法前缀（其中 68 次一上来就结束），加掩码在 40 步内 1,987 次完成、13 次在合法前缀处用完预算，放到 80 步则 2,000 次全部完成：掩码保证每一步合法，不保证模型会写完。
- 第 2 节：逐步掩码不是“合法输出上的分布”：页面的例子里按合法序列重新归一是 10% / 90%，逐步掩码是 50% / 50%；在字节级 GPT 上，七个合法词的两种分布相差 0.203（总变差）。
- 第 3 节：掩码在真实模型上：没见过格式的模型自己写出合法 JSON 的比例是 0%，加掩码 100% 合法；训练 80 步后自己能写出 68%，300 步后 100%；掩码的“压力”从 229.62 降到 0.05：模型越懂格式，掩码要推的越少。合法的输出也不一定好：0 步时掩码下的名字是“unererec”，300 步后才是“falstaff”、“cassius”。
- 第 4 节：严格模式的模式检查报出具体问题（缺 `additionalProperties: false`、可选字段、不支持的 `allOf`）；校验-重试循环里，脚本化模型三次才成功，错误信息指出字段与问题；函数调用循环里，错误调用作为错误结果回到模型，模型修正后答出；一个总在调工具的模型会在轮数上限处被截断。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 3 节 `stages = [0, 40] if QUICK else [0, 20, 40, 80, 300]` 里加一个 150：模型自己写出合法 JSON 的比例是在哪一步从 0 跳到 100% 的？
2. 第 1 节 `toy.generate(rng, True, 40)` 里的 40 是步数预算：改成 20，三个受约束的样本会变成什么样？

## 注意

- 第 3 节的“真实模型”是 84 万参数的字节级 GPT，语法是页面里的玩具语法；第 4 节的模型用脚本回放页面上的几次调用，不是真正的大模型。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/applications/structured-output-and-function-calling/run.py`）</summary>

```text
# Structured output: constrained decoding, its cost, and the function-calling loop
  course: https://dmxsf.com/wiki/applications/structured-output-and-function-calling/
          https://dmxsf.com/en/wiki/applications/structured-output-and-function-calling/

## 1. The automaton, the index and the mask (the page's demo, token for token)
  DFA states 24, vocabulary 30, usable (state, token) pairs 49 of 720
  state  0 allows ['{', '{"']
  state  1 allows ['"']
  state 10 allows ['name', 'age', 'al', 'ice', 'alice', 'bob', 'thirty', 'json']
  state 21 allows ['30', '3', '7', '42']
  state 22 allows ['}', '30', '3', '7', '42']
  state 23 allows ['<eos>']
  [ok] 24 states x 30 tokens = 720 pairs, of which only 49 can be written
  constrained sample: '{"name": "agealage", "age": 3074273730423374233}'
  constrained sample: '{"name": "alicealicealicebobicename", "age": 42}'
  constrained sample: '{"name": "alicenameicebob", "age": 73307}'
  [ok] the three constrained samples of the page (seed 7)

  A model that knows nothing (random logits), 2,000 generations each:
  mask  max steps  complete  legal prefix  invalid
  ----  ---------  --------  ------------  -------
    no         40         0            72     1928
   yes         40      1987            13        0
   yes         80      2000             0        0
  [ok] the page's tallies: no mask 1,928 invalid and 72 prefixes; mask 1,987 complete and 13 prefixes at 40 steps; all 2,000 complete at 80
  of the 72 unconstrained prefixes, 68 were empty (the model stopped at once)
  [ok] the page says 68 of those 72 prefixes were an immediate end-of-text  (68)

  The unfinished outputs at 40 steps are the mask's guarantee ending: every step is legal, but
  nothing forces the model to finish. Raise the budget to 80 steps and they complete.

## 2. The stepwise mask is not the model's distribution over legal outputs
  how legal outputs are drawn        P(Ax)  P(By)
  ---------------------------------  -----  -----
  renormalised over legal sequences   1/10   9/10
                  step-by-step mask    1/2    1/2
  [ok] the page's example: 10% / 90% over the legal sequences, 50% / 50% with the stepwise mask

  The byte-level GPT after 'ROMEO:\nO, speak again, bright angel! My ', legal outputs = 7 words:
  word   renormalised over legal words  stepwise mask
  -----  -----------------------------  -------------
   love                          34.5%          16.8%
   lord                          46.4%          56.7%
   lady                           9.4%           9.0%
  death                           1.7%           8.2%
   dead                           0.8%           4.3%
   king                           6.6%           4.8%
   kind                           0.5%           0.3%
  the model itself gives these 7 words 5.22% of its probability together;
  total variation distance between the two columns: 0.203
  [ok] both columns are probability distributions over the 7 words
  [ok] the two ways of drawing a legal word disagree by more than 0.05 in total variation  (0.203)

## 3. The mask on a real model: how much of the format does the model know?
  400 lines such as {"name": "titania", "age": 60} are the training text of the tuned models.
  grammar {"name": "[a-z]{3,8}", "age": [0-9]{1,2}}, 100 samples per row at temperature 1, two-line few-shot prompt:
  steps on JSON  valid, no mask  valid, mask  mask pressure (nats)
  -------------  --------------  -----------  --------------------
              0              0%         100%                229.62
             20              0%         100%                 22.66
             40              0%         100%                  4.78
             80             68%         100%                  0.48
            300            100%         100%                  0.05
  masked samples after 0 steps: ['{"name": "unererec", "age": 33}', '{"name": "unaltery", "age": 33}', '{"name": "nerening", "age": 33}']
  masked samples after 300 steps: ['{"name": "falstaff", "age": 33}', '{"name": "cassius", "age": 78}', '{"name": "romeo", "age": 63}']

  mask pressure = the log-probability the model itself gave to the bytes the mask allowed,
  summed over the steps and negated: how much probability the mask had to move.
  [ok] with the mask every output is valid, whatever the model has seen
  [ok] the model that never saw the format almost never writes it unaided (5% or fewer valid)  (0.00)
  [ok] the more of the format the model knows, the less the mask has to push  (229.62 > 22.66 > 4.78 > 0.48 > 0.05)
  [ok] unaided validity does not fall with training, and the longest-trained model is valid 95%+  (0%, 0%, 0%, 68%, 100%)

  A valid output is not a good output: the mask guarantees the shape, not that the name is a
  name or the age is plausible. Compare the masked samples of the first and last row.

## 4. Schemas, strict mode, validation and retries
  schema      strict-mode problems  what                                                                                                                                      
  ----------  --------------------  ------------------------------------------------------------------------------------------------------------------------------------------
      triage                     0                                                                                                                                           -
       loose                     2                                                    $: additionalProperties must be false; $.note: every property must be listed in required
       fixed                     0                                                                                                                                           -
  nested_bad                     3  $: keyword 'allOf' is not supported; $.items[]: additionalProperties must be false; $.items[].a: every property must be listed in required
  [ok] triage and fixed pass; the loose schema (no additionalProperties, optional field) does not
  [ok] an unsupported keyword and a nested object without additionalProperties are reported  (3 problems)
  attempt 1 rejected: ['<whole>: invalid JSON: Expecting value at column 1']
  attempt 2 rejected: ['reason: field required', "priority: input should be one of 'low', 'medium', 'high'"]
  accepted: {'reason': 'payment taken but order unpaid: needs a ledger check', 'priority': 'high', 'requires_human': True}
  [ok] the scripted model needs three attempts; the errors name the field and the problem
  [ok] validate catches extra fields and wrong types
  [ok] after the allowed retries it gives up instead of looping forever
  user                 What is the status of order A123?
  function_call        {"order": "A123"}
  function_call_output {"error": "TypeError: get_order_status() got an unexpected keyword argument 'order'"}
  function_call        {"order_id": "A123"}
  function_call_output {"order_id": "A123", "status": "charged, awaiting confirmation"}
  message              Order A123 is currently: charged, awaiting confirmation.
  [ok] the loop: a bad call comes back as an error result, the model repairs it, then answers
  [ok] a model that keeps calling tools is cut off at the turn limit

All checks passed. (134.8s)
```

</details>
