# What is an LLM? The whole loop, small enough to read

Course pages: [What Is a Large Language Model? From Next-Word Prediction to General Assistants](https://dmxsf.com/en/wiki/start/what-is-an-llm/) · [LLM Learning Roadmap: Application, Engineering and Research Paths](https://dmxsf.com/en/wiki/start/learning-roadmap/)  
[中文](README.md)

Predict one byte, append it, repeat: the whole loop on an 842,496-parameter GPT you can open, from tokens and embeddings to the softmax, the training objective and the three numbers (parameters, tokens, context). It ends with the map from every page of the course to its lab.

## Run

```bash
uv run python labs/start/what-is-an-llm/run.py            # full run, a few seconds
uv run python labs/start/what-is-an-llm/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run what-is-an-llm` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/start/what-is-an-llm/run.py` | the lab script, numbered sections |
| `src/dmxsf/models.py` | Small pretrained byte-level models: trained once, cached, shared by the labs after pretraining |
| `src/dmxsf/inference/generate.py` | Autoregressive generation with a KV cache: the loop every chat product runs |
| `src/dmxsf/foundations/tokenizer.py` | Byte-level BPE, written from scratch (course page: tokenizer and embeddings) |
| `src/dmxsf/foundations/information.py` | Entropy, cross-entropy and KL divergence (course page: information theory) |

Unit tests: `uv run pytest tests/test_models.py tests/test_inference_generate.py tests/test_foundations_tokenizer.py tests/test_foundations_information.py`

## What to look for

- Section 1: the hand-written loop (read everything, predict one byte, append, repeat) writes the same 60 bytes as the library's KV-cached `generate`; writing 128 bytes without a cache re-reads 10,048 bytes, with one 142, a 70.8x difference.
- Section 2: the same sentence is three different id sequences under three tokenizers; each BPE covers more bytes per token on the domain it was trained on (Shakespeare 1.84, Python 2.05); five Chinese characters are 15 bytes and 15 tokens under both BPEs, since text a tokenizer never saw falls back to bytes.
- Section 4: the same scores have entropy 0.55, 3.90 and 4.71 bits at temperatures 0.3, 1 and 1.5; sampling gives different text, greedy decoding the same text every time.
- Section 5: knowing nothing costs 8 bits per byte; a few minutes of Shakespeare bring it to 2.308 on unseen text; a byte inside a word costs 1.50 bits, the first letter of a word 4.13: the hard part is what comes next.
- Section 7: the 30 labs cover the course's 35 pages, and the three study routes (application, engineer, researcher) list which labs to run week by week.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Change `PROMPT` in `run.py` to `"JULIET:\nO "` or another beginning and see how section 4 changes.
2. The texts in section 2 are written in the script; swap in a paragraph of your own Chinese or code and compare how many tokens each BPE needs.
3. Run with `--seed 1` and `--seed 2`: which numbers change and which do not? (Hint: the model is cached, so only the sampling changes.)

## Notes

- This lab uses the source of the running interpreter's standard library as its code corpus, so the numbers that involve it differ a little between Python versions.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/start/what-is-an-llm/run.py`)</summary>

```text
# What is an LLM? The whole loop, small enough to read
  course: https://dmxsf.com/wiki/start/what-is-an-llm/
          https://dmxsf.com/en/wiki/start/what-is-an-llm/
  course: https://dmxsf.com/wiki/start/learning-roadmap/
          https://dmxsf.com/en/wiki/start/learning-roadmap/
  model: 4 layers, width 128, 4 heads, 842,496 parameters, context 128 bytes; read 9,830,400 bytes of Shakespeare

## 1. The loop: read everything so far, predict one token, append, repeat
  'ROMEO:\nWhat is the state? what say the state?\n\nPOMPEY:\nWhat is the state?\n\n'
  [ok] the hand-written loop and the library's KV-cached `generate` produce the same 60 bytes
  [ok] round i feeds the model the prompt plus everything generated: the bytes read add up to 15 x 60 + 60 x 59 / 2  (2,670 bytes)

  Bytes the model must read to write n bytes after a 15-byte prompt:
  n    re-read every round  with a KV cache  ratio
  ---  -------------------  ---------------  -----
   16                  360               30  12.0x
   32                  976               46  21.2x
   64                2,976               78  38.2x
  128               10,048              142  70.8x
  (A KV cache keeps the keys and values of what was already read; see labs/inference/kv-cache.)

## 2. Stop 1: text becomes tokens
  Tokens needed by two tokenizers (vocabulary 600, trained on 100,000 bytes each):
  text                   bytes  BPE Shakespeare  BPE code
  ---------------------  -----  ---------------  --------
    'unbelievably good'     17               10        12
  'To be, or not to be'     19                7         8
          'def main():'     11                8         4
         'Hello, world'     12                6         9
                '你好，世界'     15               15        15

  'unbelievably good', piece by piece:
    bytes only           u | n | b | e | l | i | e | v | a | b | l | y |   | g | o | o | d
                         ids [117, 110, 98, 101, 108, 105, 101, 118, 97, 98, 108, 121, 32, 103, 111, 111, 100]
    BPE on Shakespeare   un | b | e | l | ie | v | a | b | ly |  good
                         ids [384, 98, 101, 108, 500, 118, 97, 98, 359, 479]
    BPE on Python code   un | b | e | li | e | v | ab | l | y |  g | o | od
                         ids [367, 98, 101, 347, 101, 118, 324, 108, 121, 555, 111, 350]
  [ok] ids belong to a tokenizer: the same text becomes different ids under each of the three

  Bytes per token on held-out text (more is better: fewer tokens for the same text):
  tokenizer           Shakespeare  Python code
  ------------------  -----------  -----------
  BPE on Shakespeare         1.84         1.41
  BPE on Python code         1.55         2.05
  [ok] a tokenizer is best on the text it was trained on: each BPE beats the other one on its own domain
  [ok] text the tokenizer never saw falls back to bytes: 5 Chinese characters are 15 UTF-8 bytes and 15 tokens under both BPEs, while English gets shorter  (Chinese [15, 15] tokens; English [6, 9] tokens for 12 bytes)

## 3. Stop 2: ids become vectors, and the Transformer reads them
  part                                   parameters
  -------------------------------------  ----------
              embedding table (256 x d)      32,768
               position table (128 x d)      16,384
                   4 Transformer blocks     793,088
                             final norm         256
  output matrix (shared with the table)           0
  [ok] the parts add up to the model's size; the output matrix is the embedding table used backwards (tied), so it is counted once  (842,496 = 842,496)
  [ok] an id is a row of the embedding table
  [ok] the model only looks backwards: changing bytes 40-63 changes no prediction at positions 0-39, and does change those after

## 4. Stop 3: the output is a probability distribution
  After 'ROMEO:\nWhat is ' the six most likely next bytes:
  byte  probability
  ----  -----------
   't'        0.284
   'n'        0.093
   'm'        0.088
   'a'        0.072
   'h'        0.068
   's'        0.061
  [ok] softmax of the 256 scores is a distribution: exp(z_v) / sum of exp(z_u), and it sums to 1
  [ok] greedy decoding takes the most likely byte  ('t' with probability 0.284)

  The same scores at different temperatures (decoding does not change the model):
  temperature  entropy (bits)  top probability
  -----------  --------------  ---------------
          0.3            0.55            0.929
          0.7            2.96            0.464
            1            3.90            0.284
          1.5            4.71            0.158
  [ok] a higher temperature flattens the distribution: entropy up, top probability down
  Sampling at temperature 1 with three seeds, then greedy twice:
    seed 0: "some a God's sweet? Had off yo"
    seed 1: 'such not me soundly the counte'
    seed 2: 'thy praise to stands crue?\n\nEX'
    greedy : 'the state? what say the state?'
  [ok] the same question gets different answers when sampled, and the same answer when greedy

## 5. Why predicting the next byte is the whole training objective
  [ok] the probability of a text is the product of the next-byte probabilities: one forward pass over the whole text gives the same log-probability as predicting byte by byte  (log P = -183.529 nats (-264.8 bits for 127 bytes))
  [ok] the training loss is the average of -log p over positions  (1.445 nats = 2.085 bits per byte)
  held-out Shakespeare: 2.308 bits per byte; knowing nothing, 256 bytes equally likely: 8.000
  [ok] reading Shakespeare for a few minutes cuts the cost of a byte from 8 bits to under 2.5  (2.308)

  Where the bits go: the average cost of a byte, by the kind of byte (held-out text):
  byte                    count  bits
  ----------------------  -----  ----
           inside a word   7332  1.50
  first letter of a word   1885  4.13
    first byte of a line    530  3.43
             punctuation    712  2.97
  [ok] the model has learned spelling: a byte inside a word is far cheaper than the first letter of the next word, where what is said next has to be guessed  (1.50 against 4.13 bits)

## 6. Parameters, tokens, context: the three numbers
  Memory for the weights alone (parameters x bytes per parameter):
  model       FP32     BF16     INT8     4-bit  
  ----------  -------  -------  -------  -------
  this model  3.37 MB  1.68 MB  0.84 MB  0.42 MB
  a 7B model  28.0 GB  14.0 GB   7.0 GB   3.5 GB
  [ok] weights take parameters x bytes per parameter: measured on this model's own tensors, FP32 is 4 bytes each and BF16 is 2  (3,369,984 and 1,684,992 bytes)
  [ok] the same formula gives the page's numbers for a 7B model: 14 GB in BF16 and 3.5 GB at 4 bits (the KV cache and activations come on top)

  Training tokens: 2400 steps x 32 windows x 128 bytes = 9,830,400 bytes read, 9.3 passes over
  the 1,059,624-byte training text, 11.7 bytes per parameter.
  [ok] the context window is a hard limit: this model's positions stop at 128, and a 129-byte input is refused

## 7. The stages of the page, and where each lives in this repository
  stage                  sections                                labs  course pages
  ---------------------  --------------------------------------  ----  ------------
           Pre-training  foundations, architecture, pretraining    12            15
          Post-training                           post-training     4             4
  Inference and serving                               inference     5             6
        Using the model    applications, multimodal, evaluation     8             8
  [ok] the 30 labs cover all 35 pages of the course, each page at least once

  AI application developer (8 weeks): 10 labs, in the order of the weeks
    week  1: what-is-an-llm, prompt-engineering
    week  2: decoding-and-sampling, llm-evaluation, eval-statistics
    week  4: rag
    week  5: agents-and-tool-calling
    week  6: continuous-batching, kv-cache
    week  7: quantization

  LLM engineer (12 weeks): 17 labs, in the order of the weeks
    week  1: next-token-prediction, tokenizer-and-embeddings
    week  2: attention-basics
    week  3: transformer-block, positional-encoding
    week  5: training-pipeline, scaling-laws
    week  6: distributed-training
    week  7: sft-data, lora-finetuning
    week  8: preference-optimization
    week  9: kv-cache, efficient-attention
    week 10: quantization, continuous-batching
    week 11: llm-evaluation, eval-statistics

  LLM researcher (16 weeks): 12 labs, in the order of the weeks
    week  2: attention-basics
    week  3: scaling-laws
    week  4: transformer-block
    week  5: mixture-of-experts
    week  7: efficient-attention, continuous-batching
    week  8: positional-encoding
    week  9: sft-data
    week 10: preference-optimization
    week 11: reasoning-rl
    week 12: llm-evaluation
    week 13: eval-statistics
  [ok] every lab on the map exists in this checkout  ([])

  List them all: dmxsf list. Run one: dmxsf run kv-cache --quick. A route: dmxsf roadmap engineer.

All checks passed. (3.2s)
```

</details>
