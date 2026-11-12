# dmxsf 配套代码：Agent 工作手册

这是 [DMXSF](https://dmxsf.com)（大模型算法教程，站点仓库 `xwzy/dmxsf.com`）的配套代码仓库：用 Python（PyTorch + NumPy）把课程里的算法从零实现一遍，每篇 Wiki 一个实验，实验在 CPU 上复现页面里的数字并逐条检查。开始工作前读完本文件。

## 1. 工作准则

1. **先想清楚再动手**，改动小而准，不过度抽象。简单可读优先：这些代码是教材，读者会逐行读。
2. **不要主动写文档。** 除本文件、顶层 `README.md` / `README.en.md` 和各实验的 `README.md` / `README.en.md`（它们是产品的一部分）之外，不新增说明文档。
3. **临时脚本与中间结果放在 `tmp/`**（已被 `.gitignore` 忽略），任务结束前删除；不要提交 checkpoints、`runs/`、缓存。
4. **包管理只用 uv。** 不提交 `uv.lock`（已忽略，锁文件会带上本机镜像的 URL）。不要使用 `git stash`；除非用户要求，不要提交。
5. **依赖只有 `numpy` 与 `torch`。** `transformers`、`datasets`、`pillow` 只在可选的 `hf` extra 里，只能在 `--hf` / `--bge` 路径里懒加载，不装也要让其余实验照常运行。
6. **内容准确第一。** 实验打印的每个数字，要么来自完整运行，要么能由公式验证；README 里写的数字必须与最近一次完整运行的输出一致；不确定的事实宁可不写。
7. **检查也是代码，会在两个方向上出错。** 阈值太松会放过错误，太紧会在别的机器上误报；第一次出现意外的数字，先读中间结果再下结论，不要直接改阈值。随机性固定，抽样、加噪声、洗牌各用独立的种子（相同种子的两个生成器吐出同一串数）。样本少就明说，小于噪声的差别不解读，实验里没有出现的现象不假装演示。
8. **离线、确定、CPU、几分钟。** 实验不能联网、不能依赖 GPU；完整运行一般在几分钟内，更长的要有理由。计时类数字取决于机器，检查只看比值与方向，阈值放宽。

## 2. 目录结构

```
src/dmxsf/
  course.py         课程地图：实验与 Wiki 页面的对应、标题与摘要、运行时间、三条学习路线（唯一来源）
  cli.py            dmxsf list | run | roadmap | doctor
  lab.py            实验脚本的公共部分：--quick / --seed，Report（section、say、check、observe、skip、table、finish）
  utils.py, data.py, models.py   小工具；字节语料与批处理；缓存的小模型（base 与 draft）
  foundations/ architecture/ pretraining/ post_training/ inference/ applications/ multimodal/ evaluation/
                    与 Wiki 章节同名的库代码，一个模块对应一到两篇页面
  data_files/       随包携带的 TinyShakespeare
labs/<section>/<slug>/run.py       实验脚本；目录是主页面的 key（post-training 目录带连字符，包名是 post_training）
labs/<section>/<slug>/README.md    中文说明，README.en.md 是英文说明
tests/              test_<section>_<module>.py 测库模块；test_course.py 保证对照表、实验与 README 一致
scripts/export_site_data.py        生成站点的 src/data/code-labs.ts
.github/workflows/ci.yml
```

## 3. 课程与实验的对应

- `src/dmxsf/course.py` 是唯一来源：`LABS`（每个实验覆盖的 Wiki key、用到的库模块、运行时间、可选的 extra）、`TEXTS`（中英文标题与摘要）、`ROUTES`（三条学习路线逐周该做的页面）。站点的 `src/data/code-labs.ts` 是它的镜像，由 `scripts/export_site_data.py` 生成，**不要手改**。
- **每篇 Wiki 恰好有一个实验**；实验目录名是它的主页面 key（`start` 章节的两页共用一个实验）。`tests/test_course.py` 里的 `PAGES` 是从站点抄来的页面清单：站点新增页面时先更新它，测试会失败，直到有对应的实验。
- 实验的 `run.py` 里 `lab.start(title, wiki=[...])` 声明的标题与页面必须与 `course.py` 一致（测试检查）。README 链接指向每个覆盖的页面（中文无前缀，英文 `/en/`）。
- `seconds` 是完整运行的耗时（笔记本 CPU，模型已缓存），改动实验后以实测为准，站点显示成“约 N 分钟”（不到 45 秒显示“不到 1 分钟”）。

## 4. 写一个实验

1. 库代码写在 `src/dmxsf/<section>/<module>.py`：模块文档字符串第一行说清楚它做什么，注明对应的课程页面；配 `tests/test_<section>_<module>.py`。
2. `labs/<section>/<slug>/run.py`：`args, report = lab.start(TITLE, wiki=[...])`，分节 `report.section("1. ...")`，每个结论一条 `report.check(name, ok, detail)`；`--quick` 缩小规模，只有完整运行才成立的检查用 `report.skip` 跳过；关于一次训练里效应大小的说法（转变发生在第几步、记住了多少、差别是否超过噪声）会随种子和机器漂移，用 `report.observe`（没成立打印 `[differs]`，不使实验失败），并在 `[differs]` 时用 `report.say` 解释为什么不能当结论；最后 `report.finish()`（有失败就以 1 退出）。
3. 能复现课程页数字的地方，把页面的数字写进检查里（精确比较或给出容差），不要只打印。
4. 在 `course.py` 登记（`LABS`、`TEXTS`，必要时 `ROUTES`），写两份 README：课程页链接、运行命令、代码位置、看什么、练习，以及一次完整运行的输出。
5. 重新生成站点数据并复制到站点仓库，在那边跑 `pnpm test`（约定见站点 `AGENTS.md` 的 5.10 节）。

README 的输出必须来自最近一次完整运行（代码改了就重跑、重贴）；练习里提到的变量名与代码行要与脚本一致。

## 5. 质量门禁

```bash
uv sync --extra hf --group dev
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not slow"     # 库模块的单元测试与对照表检查
uv run pytest                   # 另含每个实验的 --quick 冒烟运行（几分钟）
```

CI（`.github/workflows/ci.yml`）在推送与 PR 时跑 Python 3.10 与 3.12。要兼容 3.10：不要在 f 字符串里复用外层引号，不要用 3.11 之后才有的标准库。改了实验的行为之后，还要跑一遍完整运行（`uv run python labs/<section>/<slug>/run.py`），确认没有 `[FAIL]`，并看一眼有没有 `[differs]`。

## 6. 同步到站点

```bash
uv run python scripts/export_site_data.py > /path/to/dmxsf.com/src/data/code-labs.ts
```

站点（`xwzy/dmxsf.com`）据此在每篇 Wiki 页尾显示“配套代码”卡片，并生成 `/code/` 页。改了 `course.py` 的标题、摘要、页面对应、运行时间，都要重新生成；站点那边的 `pnpm validate` 会对没有实验的 Wiki 页面报警告。反过来，实验算出与页面不符的数字时，以代码为准去改站点页面（并更新那页的 `updatedAt`）。

## 7. 常见坑

- **缓存的小模型。** `models.pretrained("base")` 第一次运行要在 TinyShakespeare 上训练约 3 分钟，之后读 `~/.cache/dmxsf`（`DMXSF_CACHE` 可改）。改了 `models.py` 的 `SPECS`、训练配置、`GPT` 的结构或初始化，要清缓存再跑，否则 README 里的数字对不上。
- **`--seed N` 要平移实验里的所有种子**：`models.pretrained(…, seed=args.seed)`（缓存文件名里带种子）、数据集抽样、初始化、洗牌和噪声都要跟着动，不能留一个写死的 `0`；实验内部用多个种子量噪声时，用 `N`、`N + 1`……。抽样、加噪声与洗牌各用不同的偏移：同样的种子喂给两个 `Random`，吐出的是同一串数字。README 里的输出来自 `--seed 0`；新实验写完，确认 `--seed 1` 的输出确实与 `--seed 0` 不同。
- **Python 标准库当语料。** `ByteCorpus.python_code()` 读的是当前解释器的标准库，换 Python 版本数字会略有不同；用到它的实验在 README 里写了这一点，检查阈值要留余量。
- **计时类检查。** KV Cache、连续批处理、投机解码的耗时随机器和负载波动：只检查比值与方向，逐步的耗时取中位数而不是均值（一次系统卡顿不会带偏一整段），并且不要在跑它们的同时做别的吃 CPU 的事（构建、第二个训练）。
- **`--quick` 的检查不能依赖训练好的模型，也不能依赖版本。** 训练 60 步的小模型不会有完整运行里的现象，这类结论一律 `skip(…, "needs the full run")`。换 Python 或 PyTorch 版本结果也要一样：浮点求和后再取整用 `math.fsum`（3.12 的 `sum()` 做了补偿求和，3.10 没有，0.475 这种平局会取整成不同的数）；PyTorch 2.14 的 `FlopCounterMode` 不统计 CPU 上融合的注意力核，相关检查要先判断能否统计，不能就 `skip`。
- **训练出来的现象会随种子和机器漂移。** 全部实验在 GitHub 的 Linux 运行器（Python 3.10 与 3.12）上完整跑过：induction head 形成的步数、小模型记住多少泄漏的题、SFT 在 300 步上的起飞点、一次训练里“某效应大于某值”的幅度，都与 M2 Max 上的不同。所以检查分三层：公式、算术和同一次运行内的因果关系（逐比特相同、方向）用 `check`；依赖大小的阈值要留足余量（看过几台机器上的范围再定），并在注释里写下见过的范围；效应大小本身（“改写的泄漏传到基准措辞上超过 5 个点”）用 `observe`。需要训练到某个现象出现的实验不要写死步数：训练到现象出现、再多训练若干步、设上限，到上限仍没出现就换一个种子重来并在输出里写明（见 `architecture/attention-basics`：每轮最多 3,000 步，最多三轮）。新实验写完先用 `--seed 1`、`2`、`3` 各跑一遍，再推到分支上让 GitHub 的 Linux 运行器完整跑一遍。
- **`--hf` / `--bge` 路径** 需要 `uv sync --extra hf` 和已下载的模型；离线运行用 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`。模型换了版本，页面数字可能变，先读轨迹再改 README。
- **`ruff format` 会重排长语句**，所以用脚本打补丁时不要依赖旧文本的换行；改完跑 `ruff check --fix` 与 `ruff format`。
- **zsh 里** 变量不要叫 `path`；后台长任务用 `python -u` 并把输出重定向到文件。
- 站点页面里的真实模型数字（Agent 页的 Qwen3-4B、多模态页的 Qwen3-VL 延迟与 OCR）在这里没有复现，对应实验用脚本化的策略验证评测框架本身；不要把别处的数字搬进检查里。
