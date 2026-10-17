# Error taxonomy, test split (row 12.8)

> **Annotator caveat.** Every case below was categorised by an LLM assistant (Claude), not by a human annotator. The categories and notes have **not been human-reviewed**. Treat the shares as a first-pass triage, not as gold labels.

Annotator field: `claude (LLM-assisted, not human)`. Rung(s): `sft3bv2`. Cases: 200 degraded (prompt, target) pairs on 194 unique test prompts.

Population: 1079 non-fallback degraded pairs (739 prompts); sample of 200, stratified by prompt category, seed 0. Annotations: [data/annotations/p12_errors.jsonl](../data/annotations/p12_errors.jsonl).

## Counts

| Failure category | n | Share | Median rewrite savings |
| :--- | ---: | ---: | ---: |
| dropped constraint | 27 | 14% | 48% |
| lost tone | 8 | 4% | 48% |
| dropped example | 17 | 8% | 46% |
| changed meaning | 32 | 16% | 36% |
| other | 116 | 58% | 29% |
| total | 200 | 100% | |

Secondary tags (a case can carry several):

| Tag | Failure category | n |
| :--- | :--- | ---: |
| `dropped_context` | changed meaning | 9 |
| `dropped_context` | dropped constraint | 3 |
| `dropped_context` | dropped example | 16 |
| `dropped_context` | lost tone | 2 |
| `format_change` | changed meaning | 1 |
| `format_change` | dropped constraint | 2 |
| `format_change` | other | 4 |
| `judge_noise` | other | 79 |
| `line_case` | changed meaning | 1 |
| `line_case` | other | 23 |
| `no_visible_loss` | other | 116 |
| `rewrite_answers_task` | changed meaning | 3 |
| `rewrite_answers_task` | dropped constraint | 4 |
| `rewrite_answers_task` | dropped example | 3 |
| `truncated` | dropped constraint | 1 |
| `truncated` | dropped example | 1 |
| `truncated` | other | 1 |

## By prompt category

| Failure | code | creative | explain | fact | reason | transform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| dropped constraint | 3 (9%) | 5 (17%) | 3 (10%) | 0 (0%) | 6 (16%) | 10 (23%) |
| lost tone | 0 (0%) | 2 (7%) | 3 (10%) | 2 (9%) | 1 (3%) | 0 (0%) |
| dropped example | 4 (11%) | 0 (0%) | 1 (3%) | 0 (0%) | 5 (13%) | 7 (16%) |
| changed meaning | 8 (23%) | 0 (0%) | 2 (6%) | 7 (30%) | 10 (26%) | 5 (11%) |
| other | 20 (57%) | 22 (76%) | 22 (71%) | 14 (61%) | 16 (42%) | 22 (50%) |
| n | 35 | 29 | 31 | 23 | 38 | 44 |

## By target

| Failure | Qwen/Qwen2.5-7B-Instruct | microsoft/Phi-3.5-mini-instruct |
| :--- | ---: | ---: |
| dropped constraint | 9 (12%) | 18 (15%) |
| lost tone | 3 (4%) | 5 (4%) |
| dropped example | 12 (16%) | 5 (4%) |
| changed meaning | 12 (16%) | 20 (16%) |
| other | 40 (53%) | 76 (61%) |
| n | 76 | 124 |

## By judge reason

| Failure | judge_prefers_original | task_metric_worse |
| :--- | ---: | ---: |
| dropped constraint | 27 (15%) | 0 (0%) |
| lost tone | 8 (4%) | 0 (0%) |
| dropped example | 15 (8%) | 2 (9%) |
| changed meaning | 27 (15%) | 5 (23%) |
| other | 101 (57%) | 15 (68%) |
| n | 178 | 22 |

## Examples

### dropped constraint

- `dolly:8735:hedging` (sha `4a43b6a18d`), Phi-3.5-mini-instruct: Dropped that the 3-step model is the one current when ZMOT came out; contrast kept.
  orig: "3-step model of marketing that was around when ZMOT came out" / rewrite: "3-step marketing model"
- `dolly:8735:texting` (sha `411dd87ac2`), Phi-3.5-mini-instruct: Dropped the 'model popular when ZMOT came out' framing; bare 'Zmot' lost context and target then misdefined ZMOT.
  orig: "how it's diff from the 3-step marketing model that was big when zmot came out" / rewrite: "Explain Zmot vs 3-step marketing model:"
- `dolly:1104:texting` (sha `c6d5d2b29f`), Phi-3.5-mini-instruct: Requested '{Name}' bullet template dropped; output adds a non-person entity (variance).
  orig: "list them with bullets like {Name}" / rewrite: "list them with bullets:"

### lost tone

- `dolly:11079:texting` (sha `137ec5dcd5`), Qwen2.5-7B-Instruct: Personal framing ('why do I feel') became an impersonal question; content otherwise kept.
  orig: "why do i feel like meeting ppl in person makes way stronger bonds" / rewrite: "Why do face-to-face meetings create stronger bonds"
- `dolly:11079:texting` (sha `137ec5dcd5`), Phi-3.5-mini-instruct: Same rewrite as previous case on other target; personal 'why do I feel' framing lost.
  orig: "why do i feel like..." / rewrite: "Why do face-to-face meetings create stronger bonds..."
- `dolly:13790:hedging` (sha `eef408f2a6`), Phi-3.5-mini-instruct: Beginner framing removed; answer lost the friendly beginner tone. Content otherwise same.
  orig: "I hope this isn't too basic..." / rewrite: "Explain credit score:"

### dropped example

- `gsm8k:test:356:over_polite` (sha `c22bc2e1de`), Qwen2.5-7B-Instruct: Initial $28 from his mother was dropped, so the remaining amount cannot be computed.
  orig: "gave him $28 to buy some toys" / rewrite: (no starting amount)
- `gsm8k:train:1605:hedging` (sha `3f80fd2b9a`), Qwen2.5-7B-Instruct: Total of 92 students was dropped; the output says more info is needed.
  orig: "there are 92 students in total" / rewrite: (no total)
- `dolly:3127:rambling` (sha `6e5bda39c0`), Qwen2.5-7B-Instruct: Rewrite keeps only the first sentence of the rules passage; output hallucinates rules (lead jammer crossing lines etc.).
  orig: full rules passage / rewrite: "Roller derby is played in two periods of 30 minutes."

### changed meaning

- `mbpp:350:texting` (sha `703e73c180`), Qwen2.5-7B-Instruct: Function name minimum_Length garbled to 'minimum Length' and asserts broken; target renamed function and invented a different problem.
  orig: "minimum_Length(\"mnm\") == 1" / rewrite: "minimum Length \"mnm\" == 1"
- `dolly:5916:repetitive` (sha `bf378dc482`), Qwen2.5-7B-Instruct: Rewrite garbled the bio passage: Elson put at IU in 1996, fired as IU coach, wrong timeline; output then answers the wrong facts.
  orig: "joined the Hilltoppers staff ... 1996" / rewrite: "Elson became a defensive backs coach at IU in 1996"
- `dolly:9798:over_polite` (sha `833d735129`), Phi-3.5-mini-instruct: 'Berkeley's ... process' became ambiguous; the target read it as philosopher George Berkeley.
  rewrite: "Explain Berkeley's public middle school choice process"

### other

- `humaneval:HumanEval/31:repetitive` (sha `198a4a74bb`), Phi-3.5-mini-instruct: Rewrite capitalised 'Def' and stripped docstring indentation; spec and doctests intact. Output still correct; metric drop looks like output-format variance.
  orig: "def is_prime(n):" / rewrite: "Def is_prime(n):"
- `dolly:10595:over_polite` (sha `e0fe20cfc5`), Qwen2.5-7B-Instruct: Politeness removed; core request (list of methods) intact. Outputs comparable.
  orig: "I was wondering if you might be able to assist..." / rewrite: "List methods to inform sister about recent vacation:"
- `dolly:12016:texting` (sha `66431bf22a`), Qwen2.5-7B-Instruct: Rewrite is as ambiguous as the original; target refused/asked for context on the rewrite (output variance).
  orig: "latest iceskates i haven't seen yet pls" / rewrite: "list latest iceskates not seen:"

<!-- manual -->

## Discussion

The rewriter shown is `sft3bv2` without the gate, the generative component of the full system. With
the gate on, the full system serves only about 65 degraded pairs, too few for a 200-case sample.

- **Most degraded labels show no visible loss.** 116 of 200 cases (58%) keep every number,
  constraint and example, and the two outputs are equivalent; 79 of them read as judge noise. The
  Phase 8 val analysis found the same thing at 72%
  ([p8_failures.md](p8_failures.md)). A large part of the measured degradation is label noise from
  the pairwise judge (? 7.4 of the report), not rewriter error.
- **Visible losses split two ways by slice.** On the 111 main-split cases, 25 (23%) show a loss:
  mostly changed meaning (12), concentrated in `reason` word problems where a number or relation is
  garbled. On the 89 real-slice (WildChat) cases, 59 (66%) show a loss: dropped constraints (24),
  changed meaning (20) and dropped context (12). In 10 of them the rewriter performed the task
  (summarised, edited or translated the text) instead of rewriting the request. Real prompts often
  are the task material; a compressor trained on synthetic casual prompts does not separate the
  instruction from the payload.
- **Upper bound.** If every visible loss were fixed, `sft3bv2`'s pooled test preservation would rise
  from 69.8% to roughly 82% (69.8% + 30.2% ? 42%), still below the 92% main gate. This matches the
  Phase 8 extrapolation.

Caveat repeated: categories and notes are LLM-assisted, not human-reviewed. Real-slice notes are
written abstractly and carry no excerpts, because those prompts have not had the manual PII review
(row 2.1.3).
