# Tokenizers and embeddings

Course pages: [Tokenizers and Embeddings: BPE, Vocabulary Size and the Embedding Table](https://dmxsf.com/en/wiki/foundations/tokenizer-and-embeddings/)  
[中文](README.md)

BPE by hand on the page's toy corpus, a byte-level BPE that never says <unk>, what the training text decides about a language's cost, an embedding table as a matrix multiply, and how much of a model the embedding table is.

## Run

```bash
uv run python labs/foundations/tokenizer-and-embeddings/run.py            # full run, a few seconds
uv run python labs/foundations/tokenizer-and-embeddings/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run tokenizer-and-embeddings` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/foundations/tokenizer-and-embeddings/run.py` | the lab script, numbered sections |
| `labs/foundations/tokenizer-and-embeddings/parallel_text.py` | the 24 Chinese-English sentence pairs of section 3 |
| `src/dmxsf/foundations/tokenizer.py` | Byte-level BPE, written from scratch (course page: tokenizer and embeddings) |

Unit tests: `uv run pytest tests/test_foundations_tokenizer.py`

## What to look for

- Section 1: the BPE merges on the page's toy corpus (`e+s -> es` 9 times, `es+t -> est` 9 times, ...) and the segmentations match the page.
- Section 2: byte-level BPE round-trips every string (emoji and CJK included); growing the vocabulary from 256 to 1024 raises bytes per token from 1.00 to 2.37 and cuts the attention cost to 0.18, while the embedding parameters grow from 32,768 to 131,072.
- Section 3: a tokenizer trained on English only pays about 2.75 tokens per Chinese character at any vocabulary size; adding Chinese data makes Chinese cheaper and English more expensive: the vocabulary is a shared budget.
- Section 5: the embedding table's share of a model: GPT-2 small 31.0%, Qwen2.5-0.5B 27.6%, Llama 3 8B 13.1%, as on the page.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Raise `max_vocab` and the vocabulary sizes of section 2 to 2048: does bytes per token keep rising? And the embedding parameters?
2. The 24 sentence pairs of section 3 are in `parallel_text.py` next to the script; replace them with your own language (Japanese, Arabic, ...) to see the tokenizer's bias.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/foundations/tokenizer-and-embeddings/run.py`)</summary>

```text
# Tokenizers and embeddings
  course: https://dmxsf.com/wiki/foundations/tokenizer-and-embeddings/
          https://dmxsf.com/en/wiki/foundations/tokenizer-and-embeddings/

## 1. BPE by hand: the course page's toy corpus
   1  e + s -> es  (9 times)
   2  es + t -> est  (9 times)
   3  ▁ + l -> ▁l  (7 times)
   4  ▁l + o -> ▁lo  (7 times)
   5  ▁lo + w -> ▁low  (7 times)
   6  ▁ + n -> ▁n  (6 times)
   7  ▁n + e -> ▁ne  (6 times)
   8  ▁ne + w -> ▁new  (6 times)
   9  ▁new + est -> ▁newest  (6 times)
  10  ▁ + w -> ▁w  (3 times)
  ['▁low', 'est'] ['▁new', 'e', 'r'] ['▁w', 'i', 'd', 'e', 'r']
  [ok] same merge table and segmentations as the course page

## 2. Byte-level BPE: pre-tokenization and a tokenizer that never says <unk>
  chunks: ['Hello', ',', ' world', '!', ' It', "'s", ' 12345678', '.']
  trained 768 merges on 300,000 characters in 5.8s
  first merges: [' t', 'he', 'ou', ' a', ' s', ' w', 'in', ' m']
  longest tokens: [' Buckingham', ' Servingman', ' ELIZABETH', 'BUCKINGHAM', 'CORIOLANUS']
  [ok] encode -> decode is lossless for any string, including emoji and CJK
  'Hello' -> [b'H', b'ell', b'o']     ' Hello' -> [b' H', b'ell', b'o']
  '你好' -> 6 tokens for 6 bytes

  A bigger vocabulary means shorter sequences (held-out Shakespeare, 20,000 characters):
  vocab  tokens  bytes/token  attention cost (x bytes)  embedding params (d=128)
  -----  ------  -----------  ------------------------  ------------------------
    256   20000        1.000                     1.000                     32768
    320   13458        1.486                    0.4528                     40960
    384   11706        1.709                    0.3426                     49152
    512   10381        1.927                    0.2694                     65536
    768    9065        2.206                    0.2054                     98304
   1024    8434        2.371                    0.1778                    131072
  [ok] each larger vocabulary compresses better
  [ok] a byte-level vocabulary of 256 is one token per byte

## 3. What the tokenizer saw decides what a language costs
  held-out text: 186 Chinese characters (522 bytes) and 107 English words (604 bytes)
  E = trained on English only, E+Z = English and Chinese, Z = Chinese only; same vocabulary size within a row
  vocab  zh tok/char E  E+Z    Z      en bytes/tok E  E+Z    Z    
  -----  -------------  -----  -----  --------------  -----  -----
    320          2.753  2.253  2.022           1.646  1.462  1.049
    384          2.753  2.022  1.801           1.893  1.692  1.049
    512          2.753  1.839  1.608           2.127  1.936  1.049
  [ok] an English-only tokenizer pays about 2.75 tokens per Chinese character at any vocab size
  [ok] putting Chinese in the training data makes Chinese cheaper ...
  [ok] ... and English more expensive: the vocabulary is a shared budget
  (24 sentence pairs written for this lab; the direction matters, not the exact ratio.
   On the course page GPT-2 needs 50 tokens for a 24-character sentence, Qwen2.5 only 16.)

## 4. From token id to vector: an embedding table is a matrix multiply
  [ok] lookup == one-hot @ weight matrix
  one backward pass touched 7 of 1000 rows: only tokens that appeared are trained
  [ok] only the rows of the tokens that appeared get gradient

## 5. How much of a model is its embedding table?
  model               total params   table V x d  copies  share
  ------------------  -------------  -----------  ------  -----
         GPT-2 small    124,439,808   38,597,376       1  31.0%
        Qwen2.5-0.5B    494,032,768  136,134,656       1  27.6%
          Llama 3 8B  8,030,261,248  525,336,576       2  13.1%
  our byte-level GPT        858,880       32,768       1   3.8%
  [ok] matches the course page: 31.0%, 27.6%, 13.1%

All checks passed. (7.2s)
```

</details>
