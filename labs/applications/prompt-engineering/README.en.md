# Prompting, measured

Course pages: [Prompt Engineering: A Systematic Guide to Specs, Layers, Examples and Evaluation](https://dmxsf.com/en/wiki/applications/prompt-engineering/)  
[中文](README.md)

The chat template rendered by hand, what a prefix cache can reuse, scorers that separate format failures from wrong judgments, the twelve word problems re-solved in code, and how far a 12-ticket eval set can be trusted. With --hf the page's experiments are rerun on Qwen2.5-1.5B-Instruct.

## Run

```bash
uv run python labs/applications/prompt-engineering/run.py            # full run, a few seconds
uv run python labs/applications/prompt-engineering/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run prompt-engineering` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

Optional flags:

- `--lang en|zh`: which language's tickets and prompts of the course page to use (default `en`).
- `--hf`: also rerun the course page's experiments on the real model Qwen2.5-1.5B-Instruct (greedy decoding, float32, CPU). Needs `uv sync --extra hf` and the downloaded model (several GB); about 10 minutes on the reference machine, and every number equals the course page's.

## Code

| File | What it holds |
| --- | --- |
| `labs/applications/prompt-engineering/run.py` | the lab script, numbered sections |
| `src/dmxsf/applications/prompting.py` | Prompting, measured: how a prompt is rendered, how a reply is scored, and how far a small eval set |
| `src/dmxsf/applications/prompt_data.py` | The tickets, prompts and word problems of the prompt-engineering page (one set per language) |
| `src/dmxsf/applications/hf.py` | A thin wrapper around a Hugging Face chat model, for the optional ``--hf`` paths of the labs |

Unit tests: `uv run pytest tests/test_applications_prompting.py`

## What to look for

- Section 1: the hand-written ChatML renderer gives, character for character, the text the real Qwen2.5 tokenizer's template prints (with a system message and without); with none the template adds a default one.
- Section 2: stable content first, so a prefix cache can reuse it: with the rules first two requests share 700 of 790 bytes (89%), 43 cacheable blocks; ticket first they share only the template's own 115-byte opening (13%), 7 blocks. A changing clock line at the top of the system prompt leaves 2 cacheable blocks, at the end 43. The page's exercise: at a 90% hit rate the tokens to compute per day fall from 128,000,000 to 20,000,000, about 84% fewer.
- Section 3: a format failure and a wrong judgment are different things: a fenced JSON, a string for the boolean and text around the JSON each fail 'valid' while 'label' stays right; a clean JSON with the wrong class is the opposite. The scorers are code and err too: the lenient reader takes the last class word, so a trailing 'this is not an escalation' turns a right answer into a wrong one.
- Section 4: the twelve word problems are solved again in code and agree with the page (12/12); `answer_number` strips commas as thousands separators, so a decimal comma ('2,5') is read as 25: a scorer's blind spot.
- Section 5: how far a 12-ticket eval set can be trusted: one more right answer is 8.3 points and the 95% interval at 80% accuracy is +-22.6 points; the page's dev 7/12 and test 8/12 are 1 apart, which luck alone produces 83% of the time; choosing the best of m prompts on the dev set, at a true accuracy of 75% and m = 10 the dev score is 92.4% (17.4 points too high); with ten prompts whose true accuracies run from 60% to 85%, chosen on 12 tickets, the winner's dev score is 93.4%, its true accuracy 79.3%, and the truly best prompt wins only 30% of the time.
- Section 6 (`--hf`): the page's experiments rerun on Qwen2.5-1.5B-Instruct (float32, CPU, greedy decoding) agree with the page throughout: V1 is valid 0/12 times, V2 7/12 on dev and 9/12 on test, V3 12/12 valid; the word problems answered directly 0/12, step by step 11/12 (the tank problem is missed: 640 instead of 360).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. In section 2 change `fixed` in `fixed, varying, requests_per_day, hit = 1200, 80, 100_000, 0.9` to 4000: how many tokens a day does a 90% hit rate save?
2. Add 100 to `for m in (1, 3, 10, 30)` in section 5: at a true accuracy of 90%, how much does the best of 100 prompts overstate its dev score?

## Notes

- Sections 1-5 need no model and finish at once; section 6 needs `--hf`, downloads Qwen/Qwen2.5-1.5B-Instruct and decodes greedily in float32 on the CPU, about 11 minutes in English and 10 in Chinese (the outputs below are from those runs). On another machine or library version individual outputs can differ slightly; the English and the Chinese runs each used their own language's scenarios and prompts, and no number is borrowed from the other.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the offline run (`labs/applications/prompt-engineering/run.py --lang en`)</summary>

```text
# Prompting, measured
  course: https://dmxsf.com/wiki/applications/prompt-engineering/
          https://dmxsf.com/en/wiki/applications/prompt-engineering/

## 1. What the model reads: the chat template
  With no system message, the rendering is (written by prompting.render_chatml):
    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    My card was charged but the order still shows unpaid.<|im_end|>
    <|im_start|>assistant
  [ok] the hand-written renderer gives the text the real Qwen2.5 tokenizer printed on the page, with a system message
  [ok] and without one
  [ok] with no system message, a default one is added; with one, it is not
  [ok] the text ends with an open assistant turn: the model continues from there

  The same ticket under the three prompts of the page. Examples are just more turns:
  prompt  messages  <|im_start|>  bytes
  ------  --------  ------------  -----
      V1         2             3    308
      V2         2             3    790
      V3         8             9   1278
  [ok] V3's three examples add six turns (user, assistant), and each assistant turn is plain JSON

## 2. Stable content first: what a prefix cache can reuse
  A server can reuse the KV cache of the leading part of a prompt when it is identical, token for
  token. Here a 'token' is a UTF-8 byte and a block is 16 of them (the page counts Qwen tokens;
  section 6 does too), so the proportions, not the absolute numbers, are what carries over.
  layout        request  shared  of request  cacheable blocks
  ------------  -------  ------  ----------  ----------------
   rules first      790     700         89%                43
  ticket first      860     115         13%                 7
  [ok] rules first: the two requests share most of their text  (700 of 790 bytes)
  [ok] ticket first: they share only the template's own opening (default system message and the user marker), none of the rules  (115 bytes = the 115-byte head)

  One changing line in the system prompt (the clock), same ticket, one second apart:
  placement         request  shared  of request  cacheable blocks
  ----------------  -------  ------  ----------  ----------------
  clock at the top      809      40          5%                 2
  clock at the end      809     694         86%                43
  [ok] a changing line near the top spoils the cache for everything behind it; at the end it costs little  (2 vs 43 cacheable blocks)

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
  #   problem                                              page  recomputed
  --  ---------------------------------------------------  ----  ----------
   0  Ben has 23 apples, gives 8 to a classmate, then ...    51          51
   1  A bookshelf has 4 shelves with 18 books on each....    63          63
   2  The canteen made 120 buns in the morning and sol...    60          60
   3  A car drives at 60 km/h for 2.5 hours, rests, th...   270         270
   4  A jacket is priced at 200 dollars with a 20 perc...   130         130
   5  A class has 45 students, and there are 7 more bo...    19          19
   6  Amy reads 14 pages a day. After 6 days, 59 pages...   143         143
   7  A case of milk holds 24 cartons. Sam buys 5 case...    35          35
   8  A tank holds 500 liters. Every minute 15 liters ...   360         360
   9  The average of three numbers is 24, and two of t...    21          21
  10  Leo has 150 dollars. Each week he saves 25 dolla...   330         330
  11  A rectangular garden is 18 m long and 12 m wide....    20          20
  [ok] all twelve answers of the page are right  (12/12)
  [ok] the tank problem turns on the sign of the net change: -7 per minute gives 360, +7 gives 640

  How `answer_number` reads a reply (the number after the last 'Answer:', else the last number):
  reply                                          read as
  ---------------------------------------------  -------
          12 + 30 = 42. | Answer: 1,280 dollars     1280
                   The tank ends at 640 liters.      640
  2 * 60 = 120 | Answer: about 3.5 km (2 + 1.5)      3.5
                   Net change is -7 per minute.       -7
                                 I cannot tell.     none
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

  The page's V2 prompt got 7/12 right end to end on dev and 8/12 on test: 1 apart. If its true accuracy
  were 62% on both splits, the chance that two splits of 12 differ by at least g items by luck alone is:
  g  chance
  -  ------
  1     83%
  2     53%
  3     29%
  4     14%
  5      6%
  [ok] the page's own dev/test gap (1) is within what luck produces (more than 5% likely)  (83%)

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
<summary>Output with the real models (`labs/applications/prompt-engineering/run.py --lang en --hf`)</summary>

```text
# Prompting, measured
  course: https://dmxsf.com/wiki/applications/prompt-engineering/
          https://dmxsf.com/en/wiki/applications/prompt-engineering/

## 1. What the model reads: the chat template
  With no system message, the rendering is (written by prompting.render_chatml):
    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    My card was charged but the order still shows unpaid.<|im_end|>
    <|im_start|>assistant
  [ok] the hand-written renderer gives the text the real Qwen2.5 tokenizer printed on the page, with a system message
  [ok] and without one
  [ok] with no system message, a default one is added; with one, it is not
  [ok] the text ends with an open assistant turn: the model continues from there

  The same ticket under the three prompts of the page. Examples are just more turns:
  prompt  messages  <|im_start|>  bytes
  ------  --------  ------------  -----
      V1         2             3    308
      V2         2             3    790
      V3         8             9   1278
  [ok] V3's three examples add six turns (user, assistant), and each assistant turn is plain JSON

## 2. Stable content first: what a prefix cache can reuse
  A server can reuse the KV cache of the leading part of a prompt when it is identical, token for
  token. Here a 'token' is a UTF-8 byte and a block is 16 of them (the page counts Qwen tokens;
  section 6 does too), so the proportions, not the absolute numbers, are what carries over.
  layout        request  shared  of request  cacheable blocks
  ------------  -------  ------  ----------  ----------------
   rules first      790     700         89%                43
  ticket first      860     115         13%                 7
  [ok] rules first: the two requests share most of their text  (700 of 790 bytes)
  [ok] ticket first: they share only the template's own opening (default system message and the user marker), none of the rules  (115 bytes = the 115-byte head)

  One changing line in the system prompt (the clock), same ticket, one second apart:
  placement         request  shared  of request  cacheable blocks
  ----------------  -------  ------  ----------  ----------------
  clock at the top      809      40          5%                 2
  clock at the end      809     694         86%                43
  [ok] a changing line near the top spoils the cache for everything behind it; at the end it costs little  (2 vs 43 cacheable blocks)

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
  #   problem                                              page  recomputed
  --  ---------------------------------------------------  ----  ----------
   0  Ben has 23 apples, gives 8 to a classmate, then ...    51          51
   1  A bookshelf has 4 shelves with 18 books on each....    63          63
   2  The canteen made 120 buns in the morning and sol...    60          60
   3  A car drives at 60 km/h for 2.5 hours, rests, th...   270         270
   4  A jacket is priced at 200 dollars with a 20 perc...   130         130
   5  A class has 45 students, and there are 7 more bo...    19          19
   6  Amy reads 14 pages a day. After 6 days, 59 pages...   143         143
   7  A case of milk holds 24 cartons. Sam buys 5 case...    35          35
   8  A tank holds 500 liters. Every minute 15 liters ...   360         360
   9  The average of three numbers is 24, and two of t...    21          21
  10  Leo has 150 dollars. Each week he saves 25 dolla...   330         330
  11  A rectangular garden is 18 m long and 12 m wide....    20          20
  [ok] all twelve answers of the page are right  (12/12)
  [ok] the tank problem turns on the sign of the net change: -7 per minute gives 360, +7 gives 640

  How `answer_number` reads a reply (the number after the last 'Answer:', else the last number):
  reply                                          read as
  ---------------------------------------------  -------
          12 + 30 = 42. | Answer: 1,280 dollars     1280
                   The tank ends at 640 liters.      640
  2 * 60 = 120 | Answer: about 3.5 km (2 + 1.5)      3.5
                   Net change is -7 per minute.       -7
                                 I cannot tell.     none
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

  The page's V2 prompt got 7/12 right end to end on dev and 8/12 on test: 1 apart. If its true accuracy
  were 62% on both splits, the chance that two splits of 12 differ by at least g items by luck alone is:
  g  chance
  -  ------
  1     83%
  2     53%
  3     29%
  4     14%
  5      6%
  [ok] the page's own dev/test gap (1) is within what luck produces (more than 5% likely)  (83%)

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
W1112 03:38:04.502000 51407 torch/distributed/elastic/multiprocessing/redirects.py:35] NOTE: Redirects are currently not supported in MacOs.
  loaded in 2s
  [ok] the tokenizer's template gives the text of section 1
  [ok] token counts with and without a system message  ((33, 40), page (33, 40))
  prefix, in Qwen tokens: rules first: request 163, shared 147; ticket first: request 179, shared 24
  [ok] the prefix experiment of the page  (page {'rules first': (163, 147), 'ticket first': (179, 24)})
  asked for 'JSON only' with no schema: fields ['status', 'description'], fence True, parse accepts False
  [ok] the schema-free probe of the page  (page (['status', 'description'], True, False))
  name  split  valid  label  both   tokens
  ----  -----  -----  -----  -----  ------
    V1    dev   0/12  11/12   0/12      23
    V1   test   0/12  10/12   0/12      22
    V2    dev   7/12  12/12   7/12      19
    V2   test   9/12  10/12   8/12      19
    V3    dev  12/12  12/12  12/12      15
    V3   test  12/12  12/12  12/12      15
  [ok] V1-V3 reproduce the page's table  ([])
  (fenced, boolean as string): {'V1': (24, 0), 'V2': (3, 5), 'V3': (0, 0)}
  [ok] format slips per prompt  (page {'V1': (24, 0), 'V2': (3, 5), 'V3': (0, 0)})
  [ok] the judgments that stay wrong in V2 and V3  ({'V2': [('order_lookup', 'faq', 'Has the return request for order E5005 been approved?'), ('escalation', 'faq', 'I never placed an order but got a payment text message. Was my account stolen?')], 'V3': []})
  [ok] V2's first format error is a fenced JSON
  direct: 0/12 right, 10 tokens on average
  [ok] word problems, direct  (page (0, 10))
  step: 11/12 right, 164 tokens on average
  [ok] word problems, step  (page (11, 164))
  step by step, missed: [(8, 640.0)]
  CoT dev: valid 0/12, label 11/12, tokens 51
  [ok] CoT triage on dev  (page (0, 11, 0, 51))
  CoT test: valid 0/12, label 12/12, tokens 52
  [ok] CoT triage on test  (page (0, 12, 0, 52))

All checks passed. (672.8s)
```

</details>
