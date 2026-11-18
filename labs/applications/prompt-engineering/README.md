# Prompt Engineering：可测量的提示词

课程页面：[Prompt Engineering 系统指南：任务规格、消息分层、示例与评测回归](https://dmxsf.com/wiki/applications/prompt-engineering/)  
[English](README.en.md)

手工渲染对话模板；前缀缓存能复用什么；把格式错误与判断错误分开的评分器；用代码重新解十二道应用题；12 张工单的评测集有多可信。加 --hf 可在 Qwen2.5-1.5B-Instruct 上重跑页面里的实验。

## 运行

```bash
uv run python labs/applications/prompt-engineering/run.py            # 完整运行，几秒
uv run python labs/applications/prompt-engineering/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run prompt-engineering`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

可选参数：

- `--lang en|zh`：用哪个语言版本课程页的工单与提示词（默认 `en`）。
- `--hf`：再用真实模型 Qwen2.5-1.5B-Instruct 重跑课程页的实验（贪心解码，float32，CPU），需要 `uv sync --extra hf` 和已下载的模型（几 GB），在本机上约 10 分钟；各数字与课程页完全一致。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/applications/prompt-engineering/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/applications/prompting.py` | 把提示词量化：提示如何渲染、回复如何打分、一个小评测集能信到什么程度 |
| `src/dmxsf/applications/prompt_data.py` | Prompt Engineering 页的工单、提示词与应用题（每种语言一套） |
| `src/dmxsf/applications/hf.py` | 对 Hugging Face 聊天模型的薄封装，供各实验可选的 `--hf` 路径使用 |

单元测试：`uv run pytest tests/test_applications_prompting.py`

## 看什么

- 第 1 节：自己写的 ChatML 渲染器，与真实 Qwen2.5 分词器的模板打印出的文本逐字相同（有系统消息与没有两种）；没有系统消息时模板会补一条默认的。
- 第 2 节：稳定的内容放前面，前缀缓存才复用得上：规则在前时两个请求共享 668 / 740 字节（90%），可缓存 41 块；工单在前只共享模板自己的开头 115 字节（14%），7 块。系统提示里一行会变的时钟放在最上面，可缓存的块只剩 2 块，放在末尾是 41 块。页面的练习：命中率 90% 时每天要算的 token 从 128,000,000 降到 20,000,000，少约 84%。
- 第 3 节：格式失败与判断失败是两回事：围栏 JSON、字符串写的布尔值、JSON 前后多出文字，都是 valid 失败而 label 仍对；干净 JSON 但类别错了则相反。评分器也是代码，也会错：宽松读取器取最后出现的类别词，结尾的一句“this is not an escalation”会把对的答案判错。
- 第 4 节：12 道应用题的答案在代码里重算，与页面一致（12/12）；`answer_number` 会去掉千分位逗号，所以小数逗号写成的“2,5”被读成 25，这是评分器的盲点。
- 第 5 节：12 张工单的评测集能信多少：多对一张是 8.3 个点，准确率 80% 的 95% 区间是 ±22.6 个点；页面的 dev 7/12 与 test 8/12 只差 1，纯靠运气就有 83% 的可能；从 m 个提示里选开发集上最好的，真实准确率 75%、m = 10 时开发集得分 92.4%（高估 17.4 个点）；10 个真实准确率在 60% 到 85% 之间的提示，按 12 张工单选，赢家的开发集得分 93.4%，真实 79.3%，真正最好的提示只有 30% 的机会胜出。
- 第 6 节（`--hf`）：在 Qwen2.5-1.5B-Instruct 上重跑页面的实验（float32、CPU、贪心解码），与页面逐项一致：V1 合法率 0/12，V2 在 dev 与 test 上 6/12 与 10/12，V3 在 dev 上 12/12 合法、11/12 标签正确；应用题直接回答 1/12，一步步推理 12/12。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 2 节 `fixed, varying, requests_per_day, hit = 1200, 80, 100_000, 0.9` 里把 `fixed` 改成 4000：命中率 90% 时每天能省多少 token？
2. 第 5 节 `for m in (1, 3, 10, 30)` 里加一个 100：真实准确率 90% 时，100 个提示里最好的那个在开发集上被高估多少？

## 注意

- 第 1–5 节不需要模型，几乎瞬间完成；第 6 节要加 `--hf`，会下载 Qwen/Qwen2.5-1.5B-Instruct 并在 CPU 上以 float32 贪心解码，中文约 10 分钟、英文约 11 分钟（下面的输出来自这两次运行）。换机器或库版本，个别输出可能略有不同；中英文各自用自己的语言的场景与提示词完整运行了一遍，数字没有互相借用。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>离线运行的输出（`labs/applications/prompt-engineering/run.py --lang zh`）</summary>

```text
# Prompting, measured
  course: https://dmxsf.com/wiki/applications/prompt-engineering/
          https://dmxsf.com/en/wiki/applications/prompt-engineering/

## 1. What the model reads: the chat template
  With no system message, the rendering is (written by prompting.render_chatml):
    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    订单已扣款但还是显示未支付，怎么办？<|im_end|>
    <|im_start|>assistant
  [ok] the hand-written renderer gives the text the real Qwen2.5 tokenizer printed on the page, with a system message
  [ok] and without one
  [ok] with no system message, a default one is added; with one, it is not
  [ok] the text ends with an open assistant turn: the model continues from there

  The same ticket under the three prompts of the page. Examples are just more turns:
  prompt  messages  <|im_start|>  bytes
  ------  --------  ------------  -----
      V1         2             3    293
      V2         2             3    740
      V3         8             9   1213
  [ok] V3's three examples add six turns (user, assistant), and each assistant turn is plain JSON

## 2. Stable content first: what a prefix cache can reuse
  A server can reuse the KV cache of the leading part of a prompt when it is identical, token for
  token. Here a 'token' is a UTF-8 byte and a block is 16 of them (the page counts Qwen tokens;
  section 6 does too), so the proportions, not the absolute numbers, are what carries over.
  layout        request  shared  of request  cacheable blocks
  ------------  -------  ------  ----------  ----------------
   rules first      740     668         90%                41
  ticket first      810     115         14%                 7
  [ok] rules first: the two requests share most of their text  (668 of 740 bytes)
  [ok] ticket first: they share only the template's own opening (default system message and the user marker), none of the rules  (115 bytes = the 115-byte head)

  One changing line in the system prompt (the clock), same ticket, one second apart:
  placement         request  shared  of request  cacheable blocks
  ----------------  -------  ------  ----------  ----------------
  clock at the top      778      40          5%                 2
  clock at the end      778     662         85%                41
  [ok] a changing line near the top spoils the cache for everything behind it; at the end it costs little  (2 vs 41 cacheable blocks)

  The page's exercise: a 1200-token fixed prefix, 80 varying tokens, 100,000 requests a day.
  hit rate  tokens to compute  saved
  --------  -----------------  -----
        0%        128,000,000   0.0%
       50%         68,000,000  46.9%
       90%         20,000,000  84.4%
       99%          9,200,000  92.8%
  [ok] at a 90% hit rate: 128,000,000 tokens without the cache, 20,000,000 with it, about 84% fewer

## 3. Scoring a reply: format and judgment are separate failures
  Invented replies, one for each way the page's outputs went wrong (hand-made, not model output):
  reply                 gold        valid  label  both
  --------------------  ----------  -----  -----  ----
             bare JSON         faq    yes    yes   yes
           fenced JSON         faq     no    yes    no
     boolean as string         faq     no    yes    no
   analysis, then JSON         faq     no    yes    no
           wrong label  escalation    yes     no    no
  right, then a remark         faq     no     no    no
           extra field         faq    yes    yes   yes
       invented fields         faq     no     no    no
  [ok] a fence, a string for the boolean and text around the JSON each fail 'valid' but keep 'label': a format failure, not a wrong judgment
  [ok] a clean JSON with the wrong class is the opposite: valid, but not label
  [ok] the scorers are code, and err too: the lenient reader takes the last class word, so a trailing 'this is not an escalation' turns a right answer into a wrong one
  [ok] parse checks only the two fields, so an extra field passes (a choice to make on purpose)
  [ok] invented field names pass nothing

## 4. The twelve word problems, solved again in code
  #   problem                    page  recomputed
  --  -------------------------  ----  ----------
   0  小明有 23 个苹果，送给同学 8 个，又买...    51          51
   1  一个书架有 4 层，每层放 18 本书。借出...    63          63
   2  食堂早上做了 120 个包子，上午卖出了其中...    60          60
   3  一辆车先以每小时 60 千米开了 2.5 小...   270         270
   4  一件标价 200 元的外套打八折，之后又用了...   130         130
   5  班里有 45 名学生，男生比女生多 7 人。...    19          19
   6  小红每天读 14 页书，读了 6 天后还剩 ...   143         143
   7  一箱牛奶有 24 盒，买了 5 箱，喝掉 3...    35          35
   8  水池里有 500 升水，每分钟流出 15 升...   360         360
   9  三个数的平均数是 24，其中两个数分别是 2...    21          21
  10  小华已有 150 元，每周存入 25 元并花...   330         330
  11  长方形菜地长 18 米、宽 12 米，沿周长...    20          20
  [ok] all twelve answers of the page are right  (12/12)
  [ok] the tank problem turns on the sign of the net change: -7 per minute gives 360, +7 gives 640

  How `answer_number` reads a reply (the number after the last '答案：', else the last number):
  reply                                read as
  -----------------------------------  -------
           12 + 30 = 42。 | 答案：1,280 元     1280
                         水池最后有 640 升。      640
  2 * 60 = 120 | 答案：约 3.5 千米（2 + 1.5）      3.5
                        每分钟净变化是 -7 升。       -7
                               我算不出来。     none
  [ok] it reads those replies as intended
  [ok] but it strips commas as thousands separators, so a decimal comma ('Answer: 2,5') is read as 25: a scorer's blind spot  (read as 25)

## 5. How far a 12-ticket eval set can be trusted
  One more right answer, and the 95% interval of an accuracy of 80% (normal approximation):
  tickets  one item (points)  interval +- (points)
  -------  -----------------  --------------------
       12                8.3                  22.6
       24                4.2                  16.0
      100                1.0                   7.8
      500                0.2                   3.5
     2000                0.1                   1.8
  [ok] on 12 tickets one ticket is 8.3 points and the interval is about +-23 points

  The page's V2 prompt got 6/12 right end to end on dev and 10/12 on test: 4 apart. If its true accuracy
  were 67% on both splits, the chance that two splits of 12 differ by at least g items by luck alone is:
  g  chance
  -  ------
  1     83%
  2     52%
  3     28%
  4     13%
  5      5%
  [ok] the page's own dev/test gap (4) is within what luck produces (more than 5% likely)  (13%)

  Pick the best of m prompts on a 12-ticket dev set when every prompt is really p accurate (exact, independent items):
  true accuracy  m prompts  best prompt's dev score  overstated by
  -------------  ---------  -----------------------  -------------
            60%          1                    60.0%          +0.0%
            60%          3                    71.8%         +11.8%
            60%         10                    80.8%         +20.8%
            60%         30                    87.0%         +27.0%
            75%          1                    75.0%          +0.0%
            75%          3                    85.2%         +10.2%
            75%         10                    92.4%         +17.4%
            75%         30                    96.8%         +21.8%
            90%          1                    90.0%          +0.0%
            90%          3                    96.6%          +6.6%
            90%         10                    99.7%          +9.7%
            90%         30                   100.0%         +10.0%
  [ok] with one prompt the dev score is unbiased; with more, the best one looks better than it is
  [ok] a simulation agrees with the exact figure (40,000 rounds, m=10, p=75%)  (0.924 vs 0.924)

  Ten prompts whose true accuracies run from 60% to 85%; choose by the 12-ticket dev score (40,000 rounds):
  quantity                            value
  ----------------------------------  -----
                  winner's dev score  93.4%
              winner's true accuracy  79.3%
  winner's score on a fresh test set  79.3%
     the best prompt's true accuracy  85.0%
          the truly best prompt wins    30%
  [ok] the winner's dev score overstates it; a fresh test set sees its true accuracy; and the truly best prompt wins well under half the time

  Tickets needed to tell 11/12 from 10/12 (the page's exercise), 80% power, 5% level:
  correlation of the two prompts  tickets
  ------------------------------  -------
                             0.0      244
                             0.5      127
                             0.7       81
  [ok] a one-ticket improvement on 12 is far below what 12 tickets can resolve  (81 to 244 tickets)

(6. Rerun the page's experiments on a real model: add --hf. It downloads Qwen/Qwen2.5-1.5B-Instruct and takes a while.)

All checks passed. (0.1s)
```

</details>

<details>
<summary>加上真实模型的输出（`labs/applications/prompt-engineering/run.py --lang zh --hf`）</summary>

```text
# Prompting, measured
  course: https://dmxsf.com/wiki/applications/prompt-engineering/
          https://dmxsf.com/en/wiki/applications/prompt-engineering/

## 1. What the model reads: the chat template
  With no system message, the rendering is (written by prompting.render_chatml):
    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    订单已扣款但还是显示未支付，怎么办？<|im_end|>
    <|im_start|>assistant
  [ok] the hand-written renderer gives the text the real Qwen2.5 tokenizer printed on the page, with a system message
  [ok] and without one
  [ok] with no system message, a default one is added; with one, it is not
  [ok] the text ends with an open assistant turn: the model continues from there

  The same ticket under the three prompts of the page. Examples are just more turns:
  prompt  messages  <|im_start|>  bytes
  ------  --------  ------------  -----
      V1         2             3    293
      V2         2             3    740
      V3         8             9   1213
  [ok] V3's three examples add six turns (user, assistant), and each assistant turn is plain JSON

## 2. Stable content first: what a prefix cache can reuse
  A server can reuse the KV cache of the leading part of a prompt when it is identical, token for
  token. Here a 'token' is a UTF-8 byte and a block is 16 of them (the page counts Qwen tokens;
  section 6 does too), so the proportions, not the absolute numbers, are what carries over.
  layout        request  shared  of request  cacheable blocks
  ------------  -------  ------  ----------  ----------------
   rules first      740     668         90%                41
  ticket first      810     115         14%                 7
  [ok] rules first: the two requests share most of their text  (668 of 740 bytes)
  [ok] ticket first: they share only the template's own opening (default system message and the user marker), none of the rules  (115 bytes = the 115-byte head)

  One changing line in the system prompt (the clock), same ticket, one second apart:
  placement         request  shared  of request  cacheable blocks
  ----------------  -------  ------  ----------  ----------------
  clock at the top      778      40          5%                 2
  clock at the end      778     662         85%                41
  [ok] a changing line near the top spoils the cache for everything behind it; at the end it costs little  (2 vs 41 cacheable blocks)

  The page's exercise: a 1200-token fixed prefix, 80 varying tokens, 100,000 requests a day.
  hit rate  tokens to compute  saved
  --------  -----------------  -----
        0%        128,000,000   0.0%
       50%         68,000,000  46.9%
       90%         20,000,000  84.4%
       99%          9,200,000  92.8%
  [ok] at a 90% hit rate: 128,000,000 tokens without the cache, 20,000,000 with it, about 84% fewer

## 3. Scoring a reply: format and judgment are separate failures
  Invented replies, one for each way the page's outputs went wrong (hand-made, not model output):
  reply                 gold        valid  label  both
  --------------------  ----------  -----  -----  ----
             bare JSON         faq    yes    yes   yes
           fenced JSON         faq     no    yes    no
     boolean as string         faq     no    yes    no
   analysis, then JSON         faq     no    yes    no
           wrong label  escalation    yes     no    no
  right, then a remark         faq     no     no    no
           extra field         faq    yes    yes   yes
       invented fields         faq     no     no    no
  [ok] a fence, a string for the boolean and text around the JSON each fail 'valid' but keep 'label': a format failure, not a wrong judgment
  [ok] a clean JSON with the wrong class is the opposite: valid, but not label
  [ok] the scorers are code, and err too: the lenient reader takes the last class word, so a trailing 'this is not an escalation' turns a right answer into a wrong one
  [ok] parse checks only the two fields, so an extra field passes (a choice to make on purpose)
  [ok] invented field names pass nothing

## 4. The twelve word problems, solved again in code
  #   problem                    page  recomputed
  --  -------------------------  ----  ----------
   0  小明有 23 个苹果，送给同学 8 个，又买...    51          51
   1  一个书架有 4 层，每层放 18 本书。借出...    63          63
   2  食堂早上做了 120 个包子，上午卖出了其中...    60          60
   3  一辆车先以每小时 60 千米开了 2.5 小...   270         270
   4  一件标价 200 元的外套打八折，之后又用了...   130         130
   5  班里有 45 名学生，男生比女生多 7 人。...    19          19
   6  小红每天读 14 页书，读了 6 天后还剩 ...   143         143
   7  一箱牛奶有 24 盒，买了 5 箱，喝掉 3...    35          35
   8  水池里有 500 升水，每分钟流出 15 升...   360         360
   9  三个数的平均数是 24，其中两个数分别是 2...    21          21
  10  小华已有 150 元，每周存入 25 元并花...   330         330
  11  长方形菜地长 18 米、宽 12 米，沿周长...    20          20
  [ok] all twelve answers of the page are right  (12/12)
  [ok] the tank problem turns on the sign of the net change: -7 per minute gives 360, +7 gives 640

  How `answer_number` reads a reply (the number after the last '答案：', else the last number):
  reply                                read as
  -----------------------------------  -------
           12 + 30 = 42。 | 答案：1,280 元     1280
                         水池最后有 640 升。      640
  2 * 60 = 120 | 答案：约 3.5 千米（2 + 1.5）      3.5
                        每分钟净变化是 -7 升。       -7
                               我算不出来。     none
  [ok] it reads those replies as intended
  [ok] but it strips commas as thousands separators, so a decimal comma ('Answer: 2,5') is read as 25: a scorer's blind spot  (read as 25)

## 5. How far a 12-ticket eval set can be trusted
  One more right answer, and the 95% interval of an accuracy of 80% (normal approximation):
  tickets  one item (points)  interval +- (points)
  -------  -----------------  --------------------
       12                8.3                  22.6
       24                4.2                  16.0
      100                1.0                   7.8
      500                0.2                   3.5
     2000                0.1                   1.8
  [ok] on 12 tickets one ticket is 8.3 points and the interval is about +-23 points

  The page's V2 prompt got 6/12 right end to end on dev and 10/12 on test: 4 apart. If its true accuracy
  were 67% on both splits, the chance that two splits of 12 differ by at least g items by luck alone is:
  g  chance
  -  ------
  1     83%
  2     52%
  3     28%
  4     13%
  5      5%
  [ok] the page's own dev/test gap (4) is within what luck produces (more than 5% likely)  (13%)

  Pick the best of m prompts on a 12-ticket dev set when every prompt is really p accurate (exact, independent items):
  true accuracy  m prompts  best prompt's dev score  overstated by
  -------------  ---------  -----------------------  -------------
            60%          1                    60.0%          +0.0%
            60%          3                    71.8%         +11.8%
            60%         10                    80.8%         +20.8%
            60%         30                    87.0%         +27.0%
            75%          1                    75.0%          +0.0%
            75%          3                    85.2%         +10.2%
            75%         10                    92.4%         +17.4%
            75%         30                    96.8%         +21.8%
            90%          1                    90.0%          +0.0%
            90%          3                    96.6%          +6.6%
            90%         10                    99.7%          +9.7%
            90%         30                   100.0%         +10.0%
  [ok] with one prompt the dev score is unbiased; with more, the best one looks better than it is
  [ok] a simulation agrees with the exact figure (40,000 rounds, m=10, p=75%)  (0.924 vs 0.924)

  Ten prompts whose true accuracies run from 60% to 85%; choose by the 12-ticket dev score (40,000 rounds):
  quantity                            value
  ----------------------------------  -----
                  winner's dev score  93.4%
              winner's true accuracy  79.3%
  winner's score on a fresh test set  79.3%
     the best prompt's true accuracy  85.0%
          the truly best prompt wins    30%
  [ok] the winner's dev score overstates it; a fresh test set sees its true accuracy; and the truly best prompt wins well under half the time

  Tickets needed to tell 11/12 from 10/12 (the page's exercise), 80% power, 5% level:
  correlation of the two prompts  tickets
  ------------------------------  -------
                             0.0      244
                             0.5      127
                             0.7       81
  [ok] a one-ticket improvement on 12 is far below what 12 tickets can resolve  (81 to 244 tickets)

## 6. The page's experiments, rerun on Qwen/Qwen2.5-1.5B-Instruct (float32, CPU, greedy)
  loaded in 9s
  [ok] the tokenizer's template gives the text of section 1
  [ok] token counts with and without a system message  ((34, 41), page (34, 41))
  prefix, in Qwen tokens: rules first: request 162, shared 148; ticket first: request 178, shared 24
  [ok] the prefix experiment of the page  (page {'rules first': (162, 148), 'ticket first': (178, 24)})
  asked for 'JSON only' with no schema: fields ['问题', '解决方案'], fence False, parse accepts False
  [ok] the schema-free probe of the page  (page (['问题', '解决方案'], False, False))
  name  split  valid  label  both   tokens
  ----  -----  -----  -----  -----  ------
    V1    dev   0/12  10/12   0/12      23
    V1   test   0/12  11/12   0/12      22
    V2    dev   6/12  11/12   6/12      20
    V2   test  10/12  12/12  10/12      19
    V3    dev  12/12  11/12  11/12      15
    V3   test  12/12  12/12  12/12      15
  [ok] V1-V3 reproduce the page's table  ([])
  (fenced, boolean as string): {'V1': (19, 19), 'V2': (6, 4), 'V3': (0, 0)}
  [ok] format slips per prompt  (page {'V1': (19, 19), 'V2': (6, 4), 'V3': (0, 0)})
  [ok] the judgments that stay wrong in V2 and V3  ({'V2': [('escalation', 'faq', '订单已扣款但还是显示未支付，怎么办？')], 'V3': [('order_lookup', 'faq', '我昨天下的单，什么时候能发货？')]})
  [ok] V2's first format error is a fenced JSON
  direct: 1/12 right, 19 tokens on average
  [ok] word problems, direct  (page (1, 19))
  step: 12/12 right, 118 tokens on average
  [ok] word problems, step  (page (12, 118))
  step by step, missed: []
  CoT dev: valid 0/12, label 10/12, tokens 40
  [ok] CoT triage on dev  (page (0, 10, 0, 40))
  CoT test: valid 0/12, label 11/12, tokens 38
  [ok] CoT triage on test  (page (0, 11, 0, 38))

All checks passed. (624.0s)
```

</details>
