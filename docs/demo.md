# Demo runbook (row 13.8)

Live casual prompt -> rewrite -> side-by-side target outputs + token counter, with a cached
fallback when no GPU is available. Code: [`src/terser/demo.py`](../src/terser/demo.py), test:
[`tests/test_demo.py`](../tests/test_demo.py).

## What the audience sees

For one casual prompt, in four numbered blocks:

1. **Fast path (always live, CPU).** The serving stack of
   [`configs/serving/fast_onnx.yaml`](../configs/serving/fast_onnx.yaml): ONNX span tagger
   proposes a candidate, the ONNX gate scores P(degraded) against the row 3.2.3 threshold
   (0.265), and the demo shows both the candidate and what the service would actually serve.
   Without `models/fast/*.onnx` (`dvc pull models/fast`) it uses the rule-based operators of
   `default.yaml`.
2. **Quality path.** The generative rewriter. *Live* when a vLLM endpoint is configured and
   answers `GET /models`; otherwise the *cached* rung-2 (Qwen2.5-1.5B LoRA, v1) val rewrite. The
   same gate's score is shown ("would pass" / "would be REJECTED").
3. **Token counter.** Prompt tokens on the gpt-4o, Qwen2.5 and Phi-3.5 tokenizers (ground
   truth, `terser.tokenizers.count`) for the original, the fast candidate and the quality
   rewrite, with the change vs the original.
4. **Side-by-side answers.** Each target's answer to the original and to the rewrite, with
   output-token counts and the 32B judge's label (cached), or live answers from target endpoints.

## Before the session

| When | Step | Command |
| :--- | :--- | :--- |
| Day before | Environment and models | `pip install -e ".[serve]"`, `dvc pull models/fast` |
| Day before | Tokenizers cached (the demo machine may be offline) | `python -m terser.tokenizers.count "warm up"` |
| Day before | Cache present (16 examples, ~120 KB) | `python -m terser.demo --list` |
| Day before | Rehearse cached mode (below) | `python -m terser.demo --mode cached --example dolly:11784:texting` |
| 1 h before, GPU only | Start the rewriter and targets on the GPU worker | `vllm serve <rewriter> --port 8000`, `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001`, `vllm serve microsoft/Phi-3.5-mini-instruct --port 8002` (one at a time on a 24 GB card unless they fit) |
| 1 h before, GPU only | Point the demo at them | `export TERSER_DEMO_QUALITY_URL=http://worker-1:8000/v1 TERSER_DEMO_TARGET_URLS=qwen2.5=http://worker-1:8001/v1,phi3.5=http://worker-1:8002/v1` |

The cache is rebuilt from the v1 validation files in `data/interim/rewrites/` (rung-2 rewrites,
both targets' greedy sample-0 answers, `val_labels.jsonl`) with `python -m terser.demo
--build-cache`: per category, two examples preserved on both targets and one degraded on at
least one, chosen deterministically (sha256 of the id), prompts <= 900 characters, only prompts
the rewriter actually changed. Transform has only one eligible example, so the cache holds 16.

## During the session

```bash
python -m terser.demo --list                           # pick examples
python -m terser.demo --example dolly:11784:texting    # cached example (any mode)
python -m terser.demo "<audience prompt>"              # live fast path; quality path live if configured
python -m terser.demo "<audience prompt>" --nearest    # no GPU: points at the closest cached example
python -m terser.demo ... --mode cached                # never contact a GPU endpoint
```

**Fallback rule.** With no endpoint configured, or none answering within 1.5 s, the demo does
not wait: the fast path and the token counter run live on CPU, and the quality path and
answers come from the cache when the prompt is one of the cached examples. For an audience
prompt that is not cached, say so, show the live fast path and token counter, then switch to
the nearest cached example (`--nearest` prints its id). No endpoint is ever probed unless it is
configured, so the demo cannot load a shared GPU by accident.

Talking points the rehearsal surfaced:

* The gate is strict on purpose (sensitivity >= 0.95, specificity 0.38 at 0.265): in all three
  rehearsal prompts it rejected the fast candidate and served the original. Prompt 2 shows why:
  the candidate dropped "the" and "than" from a word problem.
* Prompt 2's quality rewrite saves ~55% of tokens but Qwen's answer degraded
  (`task_metric_worse`) while Phi's was preserved: the label is per target.
* Savings are the same on gpt-4o and Qwen2.5 but smaller on Phi-3.5 (-50.3% vs -55.0%): the
  tokenizer matters.

## Rehearsal transcript (CPU, cached mode, 2026-10-17)

Rehearsed on the WSL2 workstation with no GPU use (`--mode cached`; the GPU was owned by another
job). Each run took 2.3-2.5 s wall including process start and tokenizer/ONNX loading; the fast
path itself took 0-6 ms after warm-up. Answers are cut at `--max-lines` lines per cell.

```text
$ python -m terser.demo --mode cached --example dolly:11784:texting --width 100
====================================================================================================
TERSER DEMO  mode=cached  cached example=dolly:11784:texting
====================================================================================================
[1] FAST PATH (CPU, live; fast_onnx.yaml, 0 ms): gate P(degraded) 0.924 vs threshold 0.265 -> REJECTED, original served
ORIGINAL (casual)                                | FAST CANDIDATE
-------------------------------------------------+-------------------------------------------------
sorry for the basic q but what is a planet?      | planet?

[2] QUALITY PATH (CACHED, models/rewriter/sft-qwen1.5b-lora (v1 val rewrite)): gate P(degraded) 0.787 -> would be REJECTED
ORIGINAL (casual)                                | QUALITY REWRITE
-------------------------------------------------+-------------------------------------------------
sorry for the basic q but what is a planet?      | what is a planet?

[3] TOKEN COUNTER (prompt tokens per target tokenizer; change vs original)
tokens                      gpt-4o     qwen2.5      phi3.5
----------------------------------------------------------
original                        11          11          11
fast candidate          2 (-81.8%)  2 (-81.8%)  2 (-81.8%)
quality rewrite         5 (-54.5%)  5 (-54.5%)  5 (-54.5%)

[4] TARGET qwen2.5 (CACHED, Qwen/Qwen2.5-7B-Instruct greedy): judge label = preserved (preserved; judge Qwen/Qwen2.5-32B-Instruct-AWQ)
ANSWER TO ORIGINAL (202 out tok)                 | ANSWER TO REWRITE (279 out tok)
-------------------------------------------------+-------------------------------------------------
No problem at all! A planet is a large celestial | A planet is a celestial body that meets the
body that orbits a star, like our Sun. Here are  | following criteria:
some key characteristics of planets:             |
                                                 | 1. **Orbits a Star**: It orbits around a star,
1. **Orbiting a Star**: Planets orbit around     | typically a star like our Sun.
stars, typically in elliptical paths known as    | 2. **Has Enough Mass to Assume Hydrostatic
orbits.                                          | Equilibrium (a Nearly Round Shape)**: This means
                                                 | it has enough mass and gravity to pull itself
2. **Mass and Gravity**: Planets have enough     | into a nearly round shape, although some planets
mass to assume a nearly round shape due to their | can be slightly oblate or irregular due to their
own gravity.                                     | rotation or other factors.
                                                 | 3. **Cleared the Neighborhood Around Its
3. **Not a Star**: Unlike stars, planets do not  | Orbit**: It has cleared other objects out of its
... (+17 more lines)                             | ... (+23 more lines)

[4] TARGET phi3.5 (CACHED, microsoft/Phi-3.5-mini-instruct greedy): judge label = preserved (preserved; judge Qwen/Qwen2.5-32B-Instruct-AWQ)
ANSWER TO ORIGINAL (394 out tok)                 | ANSWER TO REWRITE (380 out tok)
-------------------------------------------------+-------------------------------------------------
 No problem at all! A planet is a celestial body |  A planet is a celestial body that orbits a star
that meets certain criteria set by the           | or stellar remnant, is massive enough for its
International Astronomical Union (IAU). Here are | gravity to make it nearly round in shape, and
the key characteristics that define a planet:    | has cleared its orbital path of other debris.
                                                 | Planets are distinct from stars, which produce
1. **Orbiting the Sun**: A planet must orbit a   | their own light and heat through nuclear fusion,
star, which in our solar system is the Sun.      | and from smaller objects like asteroids and
                                                 | comets.
2. **Spherical Shape**: Due to its own gravity,  |
a planet must have enough mass for its self-     | In our solar system, there are eight recognized
gravity to overcome rigid body forces, leading   | planets:
to a nearly round shape.                         |
                                                 | 1. Mercury
... (+38 more lines)                             | ... (+35 more lines)

====================================================================================================

$ python -m terser.demo --mode cached --example gsm8k:train:6814:repetitive --width 100 --max-lines 10
====================================================================================================
TERSER DEMO  mode=cached  cached example=gsm8k:train:6814:repetitive
====================================================================================================
[1] FAST PATH (CPU, live; fast_onnx.yaml, 6 ms): gate P(degraded) 0.438 vs threshold 0.265 -> REJECTED, original served
ORIGINAL (casual)                                | FAST CANDIDATE
-------------------------------------------------+-------------------------------------------------
I need to figure out how many people are going   | figure how people museum because totalschool
to the museum because my kid is part of the trip | organizing museum 4 buses hired children
and I want to know the total. So, the school is  | teachers . second bus has twice number people
organizing a trip to the museum and 4 buses were | first , which has 12 people. The third bus has 6
hired to take the children and teachers there.   | fewer people second . The fourth bus has 9 more
The second bus has twice the number of people on | people first . calculate total number of people
it as the first bus, which has 12 people. The    | museum? Again, the first bus has 12 people,
third bus has 6 fewer people than the second     | figure the total
bus. The fourth bus has 9 more people than the   |
... (+4 more lines)                              |

[2] QUALITY PATH (CACHED, models/rewriter/sft-qwen1.5b-lora (v1 val rewrite)): gate P(degraded) 0.333 -> would be REJECTED
ORIGINAL (casual)                                | QUALITY REWRITE
-------------------------------------------------+-------------------------------------------------
I need to figure out how many people are going   | Calc total ppl going to museum. School orgs
to the museum because my kid is part of the trip | trip; 4 buses hired. 2nd bus has twice 1st bus's
and I want to know the total. So, the school is  | ppl (12). 3rd bus has 6 fewer ppl than 2nd bus.
organizing a trip to the museum and 4 buses were | 4th bus has 9 more ppl than 1st bus.
hired to take the children and teachers there.   |
The second bus has twice the number of people on |
it as the first bus, which has 12 people. The    |
third bus has 6 fewer people than the second     |
bus. The fourth bus has 9 more people than the   |
... (+4 more lines)                              |

[3] TOKEN COUNTER (prompt tokens per target tokenizer; change vs original)
tokens                      gpt-4o     qwen2.5      phi3.5
----------------------------------------------------------
original                       138         140         143
fast candidate         70 (-49.3%) 72 (-48.6%) 75 (-47.6%)
quality rewrite        62 (-55.1%) 63 (-55.0%) 71 (-50.3%)

[4] TARGET qwen2.5 (CACHED, Qwen/Qwen2.5-7B-Instruct greedy): judge label = degraded (task_metric_worse; judge Qwen/Qwen2.5-32B-Instruct-AWQ)
ANSWER TO ORIGINAL (261 out tok)                 | ANSWER TO REWRITE (361 out tok)
-------------------------------------------------+-------------------------------------------------
Certainly! Let's break down the information      | Let's break down the problem step by step.
given and calculate the number of people on each |
bus:                                             | 1. **First Bus:**
                                                 |    - Let the number of people on the first bus
1. **First Bus**: Has 12 people.                 | be \( x \).
2. **Second Bus**: Has twice the number of       |    - We don't know \( x \) yet, so we'll keep it
people as the first bus. Since the first bus has | as \( x \).
12 people, the second bus will have \(2 \times   |
12 = 24\) people.                                | 2. **Second Bus:**
... (+19 more lines)                             | ... (+36 more lines)

[4] TARGET phi3.5 (CACHED, microsoft/Phi-3.5-mini-instruct greedy): judge label = preserved (preserved; judge Qwen/Qwen2.5-32B-Instruct-AWQ)
ANSWER TO ORIGINAL (201 out tok)                 | ANSWER TO REWRITE (198 out tok)
-------------------------------------------------+-------------------------------------------------
 Sure, let's calculate the number of people on   |  Let's calculate the number of people on each
each bus and then add them up to find the total  | bus:
number of people going to the museum.            |
                                                 | 1st bus: 12 people
1. First bus: 12 people                          | 2nd bus: 2 times the number of people on the 1st
2. Second bus: Twice the number of people as the | bus, so 2 * 12 = 24 people
first bus, so 2 * 12 = 24 people                 | 3rd bus: 6 fewer people than the 2nd bus, so 24
3. Third bus: 6 fewer people than the second     | - 6 = 18 people
bus, so 24 - 6 = 18 people                       | 4th bus: 9 more people than the 1st bus, so 12 +
... (+11 more lines)                             | ... (+11 more lines)

====================================================================================================

$ python -m terser.demo --mode cached "hey so um, could you maybe write a Python function that checks if a string is a palindrome? It should ignore case. Thanks so much!!" --width 100
====================================================================================================
TERSER DEMO  mode=fast-only
====================================================================================================
[1] FAST PATH (CPU, live; fast_onnx.yaml, 1 ms): gate P(degraded) 0.544 vs threshold 0.265 -> REJECTED, original served
ORIGINAL (casual)                                | FAST CANDIDATE
-------------------------------------------------+-------------------------------------------------
hey so um, could you maybe write a Python        | could write a Python function checks string
function that checks if a string is a            | palindrome? It should ignore case. !!
palindrome? It should ignore case. Thanks so     |
much!!                                           |

[2] QUALITY PATH: unavailable (no GPU endpoint reachable and prompt not in the cache)
    16 cached examples: python -m terser.demo --list

[3] TOKEN COUNTER (prompt tokens per target tokenizer; change vs original)
tokens                      gpt-4o     qwen2.5      phi3.5
----------------------------------------------------------
original                        29          29          32
fast candidate         15 (-48.3%) 15 (-48.3%) 17 (-46.9%)

====================================================================================================
```
