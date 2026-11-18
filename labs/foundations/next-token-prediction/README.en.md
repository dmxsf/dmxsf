# Next-token prediction and information theory

Course pages: [From Language Modeling to Next-Token Prediction](https://dmxsf.com/en/wiki/foundations/next-token-prediction/) · [Information Theory Refresher: Entropy, Cross-Entropy and KL Divergence](https://dmxsf.com/en/wiki/foundations/information-theory/)  
[中文](README.md)

Bigram counts on the page's toy corpus, entropy, cross-entropy and KL divergence checked against each other, character n-grams on Shakespeare until the counts run out, and a neural bigram that learns the count table by gradient descent.

## Run

```bash
uv run python labs/foundations/next-token-prediction/run.py            # full run, a few seconds
uv run python labs/foundations/next-token-prediction/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run next-token-prediction` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/foundations/next-token-prediction/run.py` | the lab script, numbered sections |
| `src/dmxsf/foundations/information.py` | Entropy, cross-entropy and KL divergence (course page: information theory) |
| `src/dmxsf/foundations/language_model.py` | Language models in their simplest forms (course page: next-token prediction) |

Unit tests: `uv run pytest tests/test_foundations_information.py tests/test_foundations_language_model.py`

## What to look for

- Section 1: the bigram probabilities on the page's toy corpus (P(sat | cat) = 0.379) and the perplexities (2.37 in natural order, 31.23 with the words shuffled) match the page one by one.
- Section 2: H(p, q) = H(p) + KL(p || q) and KL is not symmetric; two models with different tokenizers have perplexities of 7.39 and 2.72 and the same bits per byte, 0.721: compare models in bits per byte.
- Section 3: the character n-grams' validation bits per char fall from 4.849 at order 1 to 2.590 at order 5 and rise again to 2.935 at order 6: the longer the context, the sparser the counts and the worse the overfitting.
- Section 4: a neural bigram starts at ln 256 = 5.5452 and gradient descent reaches the table's optimum, H(next | prev) = 2.4522.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Change the 7 in the section 3 loop `for n in range(1, 5 if args.quick else 7):` to 10 and watch the train and validation bits per char separately.
2. In section 2 replace `q = [0.02, 0.02, 0.02, 0.02, 0.92]` with another distribution and see whether KL(p || q) or KL(q || p) reacts more to a confident mistake.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/foundations/next-token-prediction/run.py`)</summary>

```text
# Next-token prediction and information theory
  course: https://dmxsf.com/wiki/foundations/next-token-prediction/
          https://dmxsf.com/en/wiki/foundations/next-token-prediction/
  course: https://dmxsf.com/wiki/foundations/information-theory/
          https://dmxsf.com/en/wiki/foundations/information-theory/

## 1. A bigram model on the course page's toy corpus
  V = 9
  P(sat | cat) = 0.379
  P(mat | the) = 0.159
  P(cat | the) = 0.304
  perplexity, natural order  = 2.37
  perplexity, shuffled words = 31.23
  [ok] numbers match the course page

## 2. Entropy, cross-entropy and KL divergence (nats)
  quantity                 value
  -----------------------  -----
                     H(p)  1.303
                  H(p, q)  3.721
               KL(p || q)  2.417
               KL(q || p)  2.500
  perplexity exp(H(p, q))  41.29
  [ok] H(p, q) = H(p) + KL(p || q)
  [ok] KL is not symmetric
  [ok] with a one-hot target, cross-entropy is -ln(probability of the true token)  (0.511 nats)
  Same perplexity is not same quality when tokenizers differ; bits per byte is:
    model A: perplexity 7.39, bits per byte 0.721
    model B: perplexity 2.72, bits per byte 0.721

## 3. Character n-grams on Shakespeare: context helps until the counts run out
  n  contexts seen  train bits/char  val bits/char
  -  -------------  ---------------  -------------
  1              1            4.751          4.849
  2             65            3.583          3.585
  3           1391            2.821          3.002
  4          11412            2.213          2.613
  5          49692            1.923          2.590
  6         137272            1.804          2.935
  [ok] each of the first orders beats the one before
  [ok] long contexts overfit: order 6 is worse than the best order  (best 2.59, order 6 2.94)
  order 2 samples: 'Sheny,\nOKISTh witero wes, sy:\nCEd erowiveroneey y bl g medars fonoouchead To yfr owof ulmemphels\n\nug\n'
  order 5 samples: "Seconduit,\nthough bears doleful shall alread looking\nperfect\nMadam,' quondard.\n\nProduce forge the Dukes "

## 4. A neural bigram, trained by gradient descent, learns the count table
  loss at step 0     : 5.5452   (ln 256 = 5.5452)
  loss at the end    : 2.4523
  H(next | prev)     : 2.4522   (the best any bigram can do)
  [ok] starts at ln V: the uniform guess
  [ok] ends at the maximum-likelihood optimum
  after 'q' the model expects  'u': 1.00
  after 't' the model expects  'h': 0.34, ' ': 0.25, 'o': 0.09

All checks passed. (3.1s)
```

</details>
