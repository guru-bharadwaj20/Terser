# Phase 8 rewriter failure analysis (row 8.3.10)

> **Annotator caveat.** Every case below was categorised by an LLM assistant (Claude), not by a human annotator. The categories and notes have **not been human-reviewed**. Treat the shares as a first-pass triage, not as gold labels.

Annotator field: `claude (LLM-assisted, not human)`. Rung(s): `sft3b`. Cases: 100 degraded (prompt, target) pairs on 83 unique val prompts.

Population: 144 non-fallback degraded pairs (108 prompts); sample of 100, stratified by prompt category, seed 0. Annotations: [data/annotations/p8_failures.jsonl](../data/annotations/p8_failures.jsonl).

## Counts

| Failure category | n | Share | Median rewrite savings |
| :--- | ---: | ---: | ---: |
| dropped constraint | 5 | 5% | 65% |
| lost tone | 1 | 1% | 69% |
| dropped example | 12 | 12% | 53% |
| changed meaning | 10 | 10% | 47% |
| other | 72 | 72% | 25% |
| total | 100 | 100% | |

Secondary tags (a case can carry several):

| Tag | Failure category | n |
| :--- | :--- | ---: |
| `dropped_context` | other | 2 |
| `line_case` | changed meaning | 2 |
| `line_case` | dropped constraint | 1 |
| `line_case` | dropped example | 2 |
| `line_case` | other | 7 |
| `no_visible_loss` | other | 63 |
| `truncated` | dropped example | 4 |

## By prompt category

| Failure | code | creative | explain | fact | reason | transform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| dropped constraint | 1 (3%) | 2 (12%) | 0 (0%) | 0 (0%) | 1 (8%) | 1 (5%) |
| lost tone | 0 (0%) | 0 (0%) | 1 (14%) | 0 (0%) | 0 (0%) | 0 (0%) |
| dropped example | 2 (7%) | 0 (0%) | 0 (0%) | 4 (27%) | 0 (0%) | 6 (30%) |
| changed meaning | 3 (10%) | 0 (0%) | 0 (0%) | 0 (0%) | 5 (42%) | 2 (10%) |
| other | 24 (80%) | 14 (88%) | 6 (86%) | 11 (73%) | 6 (50%) | 11 (55%) |
| n | 30 | 16 | 7 | 15 | 12 | 20 |

## By target

| Failure | Qwen/Qwen2.5-7B-Instruct | microsoft/Phi-3.5-mini-instruct |
| :--- | ---: | ---: |
| dropped constraint | 1 (2%) | 4 (7%) |
| lost tone | 0 (0%) | 1 (2%) |
| dropped example | 5 (12%) | 7 (12%) |
| changed meaning | 6 (15%) | 4 (7%) |
| other | 28 (70%) | 44 (73%) |
| n | 40 | 60 |

## By judge reason

| Failure | judge_prefers_original | task_metric_worse |
| :--- | ---: | ---: |
| dropped constraint | 4 (5%) | 1 (6%) |
| lost tone | 1 (1%) | 0 (0%) |
| dropped example | 9 (11%) | 3 (18%) |
| changed meaning | 8 (10%) | 2 (12%) |
| other | 61 (73%) | 11 (65%) |
| n | 83 | 17 |

## Examples

### dropped constraint

- `humaneval:HumanEval/78:rambling` (sha `a7bfd2bf7e`), Phi-3.5-mini-instruct: Dropped the note that B (=11) and D (=13) are the prime hex digits and the uppercase / empty-input assumptions; digit list garbled.
  orig: "So you have to determine a number of the following digits: 2, 3, 5, 7, B (=decimal 11), D (=decimal 13)." / rewrite: "Hex digits: 0-9,A-B,C-D,E-F; primes: 2,3,5,7,11,13,..."
- `gsm8k:train:15:repetitive` (sha `62bb1ca80d`), Phi-3.5-mini-instruct: Dropped the selling price (2.5 x the production cost), so profit cannot be computed; quantity expression garbled.
  orig: "...He sells each DVD for 2.5 times the production cost..." / rewrite: "Calculate James' profit over 20 weeks: movie cost=$2000; production cost/DVD=$6; sold/week=500*5 DVDs/day*20 weeks."
- `dolly:97:hedging` (sha `c67fdf143a`), Phi-3.5-mini-instruct: Dropped the second request, 'give me a list of its uses'.
  orig: "...could you maybe explain what it is? And if it's not too much trouble, perhaps you could also give me a list of its uses?" / rewrite: "...could you explain what it is?" (uses request gone)

### lost tone

- `dolly:1050:repetitive` (sha `111adbacd7`), Phi-3.5-mini-instruct: Dropped 'I just want to understand what it means', the only cue for a plain, lay-level explanation; leaves 'What is buoyant force? Explain.'
  orig: "Hey, what is buoyant force? Can you explain what buoyant force is? I just want to understand what it means." / rewrite: "What is buoyant force? Explain."

### dropped example

- `mbpp:539:rambling` (sha `2503565e00`), Qwen2.5-7B-Instruct: Rambling text copied verbatim, but the output was cut off mid-number in the second assert and the third assert is missing.
  rewrite ends: "...48873677980689257489322752273774603865660850176, 1435036016098684342856030" (third assert absent)
- `dolly:10269:hedging` (sha `8c5c327a5b`), Phi-3.5-mini-instruct: Context passage summarised; the year production halted (1939) and the incendiary-bomb detail were cut, so the answer is no longer stated in the prompt.
  orig: "On the outbreak of World War II in 1939, coach production halted..." / rewrite: "Coach production halted during WWII..."
- `dolly:5264:rambling` (sha `cf7b550036`), Qwen2.5-7B-Instruct: Passage replaced by a bullet list that omits his current job (Jacksonville Jaguars head coach) and the Kansas City name.
  orig: "...is the head coach for the Jacksonville Jaguars of the National Football League (NFL)..." / rewrite: bullets list Packers, Dolphins, Eagles, Browns, Chiefs; no Jaguars

### changed meaning

- `mbpp:282:over_polite` (sha `772d526846`), Qwen2.5-7B-Instruct: Two of three asserts lost their list brackets, changing the expected values.
  orig: "assert sub_list([1, 2, 3],[4,5,6])==[-3,-3,-3]" / rewrite: "assert sub_list([1, 2, 3],[4,5,6])==-3,-3,-3"
- `gsm8k:train:6101:repetitive` (sha `c1b2e3afa7`), Qwen2.5-7B-Instruct: No longer says the $30 is already saved toward the $120 goal; 'save $120 ... Save $30' can be read as $120 still needed.
  orig: "...shoes that cost $120. I've already saved $30..." / rewrite: "Calculate additional weekends needed to save $120 for shoes costing $120. Save $30; ..."
- `gsm8k:train:6427:repetitive` (sha `934389fe83`), Phi-3.5-mini-instruct: '6 days of deliberation (16 hours/day)' drops 'full days' and the requested conversion (6 x 24 h at 16 h/day), so it reads as plain 6 days.
  orig: "...6 full days, but they work 16 hours a day, so I need to calculate how many actual days that turns into" / rewrite: "...+ 6 days of deliberation (16 hours/day)."

### other

- `mbpp:324:repetitive` (sha `7141b87393`), Phi-3.5-mini-instruct: Instruction faithful; only `assert`->`Assert` on every test line.
  orig: "assert sum_of_alternates((5, 6, 3, 6, 10, 34)) == (46, 18)" / rewrite: "Assert sum_of_alternates((5, 6, 3, 6, 10, 34)) == (46, 18)"
- `dolly:1493:rambling` (sha `b139fac20e`), Phi-3.5-mini-instruct: Reduced to 'Write haiku about cats.'; the user's imagery (graceful cats, 'little furry ninjas') is gone. Not a stated requirement, but the judge rewarded responses that used it.
  orig: "...how gracefully they move around the house, like little furry ninjas... could you maybe write a haiku about cats" / rewrite: "Write haiku about cats."
- `dolly:5382:repetitive` (sha `7a23db311b`), Qwen2.5-7B-Instruct: Only the leading 'Um,' removed; otherwise identical.
  orig: "Um, I need some ideas to keep my kids occupied during Spring break, you know?..." / rewrite: "I need some ideas to keep my kids occupied during Spring break, you know?..."

<!-- manual -->

## Method

- **Population.** Rung `sft3b` (`terser-rewriter / 6`, Staging, `models/rewriter/sft-qwen3b-lora`),
  val split. A degraded case is a (prompt, target) pair where the rewrite did not fall back under the
  8.3.7 safeguard and the `label_v1` judge label in `val_labels_all_v2.jsonl` (candidate id
  `<prompt_id>:sft3b`) is `degraded`. sft3b has 144 such pairs (61 Qwen2.5-7B, 83 Phi-3.5) on 108
  prompts, so no other rung was pooled. sft3b falls back on 130 of 285 prompts, and those never
  enter the population.
- **Sample.** 100 pairs, allocated to prompt categories in proportion to the pool (largest
  remainder): code 30 / 43, transform 20 / 29, creative 16 / 23, fact 15 / 22, reason 12 / 17,
  explain 7 / 10; `random.Random(0)` within categories. 17 prompts were drawn for both targets.
  Reproduce with `python -m terser.models.rewriter.failures sample` (module docstring).
- **What was read.** For each case: the original val prompt, the rewrite, and the judge's
  one-line reasons for both presentation orders. Target responses were not read case by case.
- **One primary category per case**, with these rules:
  - *dropped constraint*: an explicit requirement, sub-request, scope qualifier or problem
    condition is missing from the rewrite;
  - *lost tone*: a cue about the register or depth expected of the response is missing;
  - *dropped example*: material the prompt supplies for the model to work from is removed or
    truncated. This covers examples and test cases, and also context passages in `fact` and
    `transform` prompts, which widens "example" beyond its literal sense;
  - *changed meaning*: the rewrite says something different (altered values, renamed
    identifiers, an ambiguous restatement of a quantity, a paraphrase that contradicts the passage);
  - *other*: none of the above. Tags say why: `no_visible_loss` (no task-relevant difference
    visible), `line_case` (only Python keywords upper-cased at line start, e.g. `Assert`),
    `dropped_context` (non-required user context removed).
  - Ties go to the more severe category. Line-case edits that rename an identifier the target
    must use (`Count_up_to(5)`, `Fibfib(n)`) count as *changed meaning*; keyword-only edits
    (`Assert`, `Def`) count as *other* unless something else was lost.
  - Both targets of a prompt get the same category, because the rewrite text is the same.

## Findings

1. **Most degraded cases show no visible loss.** 72 of 100 are *other*. 63 of them carry
   `no_visible_loss`: the rewrite removes filler, hedging or politeness and keeps the task. 16 of
   those 63 save 5% or less (e.g. only a leading "Um," removed). The median rewrite saving in
   *other* is 25%, against 47-69% in the four loss categories. For these cases the `degraded` label
   more likely reflects variation in the target's response or in the judge than information lost
   in the rewrite. This was not tested here. It is consistent with the judge noise already
   recorded for this project (40.9% raw A/B disagreement on the pilot, C2.6 fail).
2. **Of the 28 cases with a visible loss, dropped material leads.** *Dropped example* (12)
   concentrates on context passages in `fact` and `transform` prompts (10 of 12). Two patterns
   recur: the rewriter turns a passage into abbreviated bullets and omits facts (e.g. the 1939
   date, a current head-coach job), and long prompts are cut off. All 4 truncated cases reached
   the 512-token generation cap.
3. **Changed meaning (10) is concentrated in `reason` (5 of 12 sampled reason cases).** Compressing
   a word problem into telegraphic notation makes a quantity ambiguous ("Save $30" without "already
   saved") or drops a unit conversion ("6 full days" at 16 h/day becomes "6 days").
4. **Dropped constraints (5) and lost tone (1) are rare in this sample.** The project risk
   register rates silent constraint drops as high risk. For sft3b on val they are 5% of the sampled
   degraded cases, plus 2 *other* cases where user context was dropped. The share may be low partly
   because the 130 fallbacks remove the hardest prompts from the population.
5. **Systematic code corruption.** 12 of the 100 cases (all `code`) carry `line_case`: line-initial
   `Assert` / `Def`, sometimes renamed identifiers, and stripped docstring indentation. The likely
   source is the SFT targets: `terser.operators.base.tidy` strips each line and upper-cases its
   first letter, and 33 of 220 `code` training targets in `data/processed/sft/train.jsonl` contain
   a line-initial `Assert`, `Def`, `Return`, `For` or `If` that their prompt does not. On val, 16 of
   the 41 non-fallback `code` rewrites carry the pattern.

## Implications for C8.3 and the rewriter

| Safeguard | Status | Would catch (in this sample) |
| :--- | :--- | :--- |
| 8.3.7 length safeguard | on | none: every case is shorter than its prompt by construction |
| Fall back when generation reaches `max_new_tokens` (512) | not implemented | the 4 truncated *dropped example* cases (all at the cap) |
| Hard "no number lost" check (regex proxy for the gate's `numbers_removed`) | not implemented as a rule | 11 / 12 *dropped example*, 2 / 5 *dropped constraint*, 2 / 10 *changed meaning*, 0 / 1 *lost tone*; also fires on 1 / 72 *other* |
| Leave code lines untouched in `tidy` (or reject rewrites that change code lines), then rebuild SFT targets | not implemented | the 12 `line_case` cases, including 2 *changed meaning* |
| Quality gate (`terser-gate / 2`, features incl. `numbers_removed`, `constraints_removed`) | not applied to SFT rewrites; not validated on them | unknown; not measured here |
| Nothing cheap | — | dropped sub-requests and scope qualifiers (`dolly:97`, `dolly:7405`), lost tone, ambiguous quantities in `reason` prompts |

- The number check is computed with a regex over digit strings, not with the gate's feature
  code. It shows that a number-retention rule targets most visible losses here with almost no
  hits on *other*. It does not show how the gate would score these cases.
- **C8.3 headroom from fixing visible losses is limited.** sft3b preserves 74.7% of pairs (144 of
  570 degraded). If the 28% visible-loss share held for all 144 pairs (about 40) and every one were
  fixed, preservation would be about 82%, still below the 92% bar. Savings (2.1% median against
  25%) remain the larger gap. Both figures are extrapolations from a sample of 100.
- Since about three quarters of the degraded labels have no visible cause, re-judging these pairs
  with more samples, or human review of a subset, should come before rewriter changes aimed at
  preservation. Otherwise label noise and rewriter errors cannot be told apart.

## Limitations

- One annotator, and it is an LLM (Claude). There is no human review, no second annotator and no
  agreement statistic. The categories are a first-pass triage.
- The annotator read the judge's reasons, which can bias categories toward the judge's view.
  The judge is also an LLM, so annotator and judge errors may be correlated.
- Each case gets one primary category, and "dropped example" is widened to cover context passages.
- The sample covers one rung (sft3b) and 100 of its 144 non-fallback degraded pairs. Fallback
  prompts are excluded by definition.
