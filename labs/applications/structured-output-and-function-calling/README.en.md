# Structured output: constrained decoding, its cost, and the function-calling loop

Course pages: [Structured Outputs and Function Calling: From JSON Mode to Constrained Decoding](https://dmxsf.com/en/wiki/applications/structured-output-and-function-calling/)  
[中文](README.md)

Constrained decoding from the automaton up: the page's demo token for token, why the stepwise mask is not the model's distribution over legal outputs, the mask on a byte-level GPT that has and has not seen the format, schemas, validation and retries.

## Run

```bash
uv run python labs/applications/structured-output-and-function-calling/run.py            # full run, about 2 minutes
uv run python labs/applications/structured-output-and-function-calling/run.py --quick    # a shrunk smoke test, about 10 seconds
```

You can also write `uv run dmxsf run structured-output-and-function-calling` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/applications/structured-output-and-function-calling/run.py` | the lab script, numbered sections |
| `src/dmxsf/applications/constrained.py` | Constrained decoding, structured output and the function-calling loop |

Unit tests: `uv run pytest tests/test_applications_constrained.py`

## What to look for

- Section 1: the page's automaton demo token for token: 24 states x 30 tokens = 720 (state, token) pairs, of which only 49 can be written; a model that knows nothing (random logits) generating 2,000 times: without the mask 1,928 invalid and 72 legal prefixes (68 of them an immediate stop), with the mask 1,987 complete within 40 steps and 13 that ran out of budget at a legal prefix, all 2,000 complete at 80 steps: the mask guarantees every step is legal, not that the model finishes.
- Section 2: the stepwise mask is not 'the distribution over legal outputs': in the page's example renormalising over the legal sequences gives 10% / 90% and the stepwise mask 50% / 50%; on the byte-level GPT the two distributions over seven legal words differ by 0.203 (total variation).
- Section 3: the mask on a real model: a model that never saw the format writes valid JSON unaided 0% of the time, with the mask 100% are valid; after 80 steps of training it manages 68% on its own, after 300 steps 100%; the mask's 'pressure' falls from 229.62 to 0.05: the more the model knows of the format, the less the mask has to push. A valid output is not a good one: with no training the masked name is 'unererec', after 300 steps it is 'falstaff' or 'cassius'.
- Section 4: strict-mode schema checks name the problem (a missing `additionalProperties: false`, an optional field, an unsupported `allOf`); in the validate-and-retry loop the scripted model needs three attempts and the errors name the field and the problem; in the function-calling loop a bad call comes back as an error result, the model repairs it and answers; a model that keeps calling tools is cut off at the turn limit.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Add 150 to `stages = [0, 40] if QUICK else [0, 20, 40, 80, 300]` in section 3: at which step does the share of valid JSON written unaided jump from 0 to 100%?
2. In section 1 the 40 in `toy.generate(rng, True, 40)` is the step budget: change it to 20 and see what the three constrained samples become.

## Notes

- The 'real model' of section 3 is the byte-level GPT of 842,496 parameters with the page's toy grammar; the models of section 4 are scripts that replay the calls on the page, not real large models.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/applications/structured-output-and-function-calling/run.py`)</summary>

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

All checks passed. (118.8s)
```

</details>
