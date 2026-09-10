# Task Categories

Plan row: **1.1.2**. Code: [`src/terser/data/categories.py`](../src/terser/data/categories.py)
(`Category`, `CATEGORY_METRICS`, `parse_category`). Framing: [ADR-0002](decisions/ADR-0002-problem-framing.md).

Every casual prompt in Terser carries exactly one of six task categories. The category decides how
the quality of a target model's answer is measured, and so decides what a *degraded* rewrite is.
Categories are used for balanced sampling (1.3.6, 1.4.1, C2.1), per-category rubrics (2.3.2),
per-category reporting (every performance table), the per-category version spaces (1.3.7), the
per-category preservation gates (🚧 C8.3, S3) and as a feature for the gate (3.1.3).

---

## 1. Summary

| Category | One-line definition | Quality measure | Degradation in one sentence |
| :--- | :--- | :--- | :--- |
| `code` | Produce or modify executable code | Unit-test pass rate | The rewrite's answer passes fewer tests |
| `fact` | Retrieve a short factual answer | Exact match (EM) + token F1 | The answer is wrong, or its F1 against the reference drops |
| `reason` | Solve a problem with one checkable final answer | Final-answer accuracy | The final answer is wrong or no longer in the asked format |
| `explain` | Explain, teach or compare in prose | Rubric judge | Coverage, correctness or depth drops, or an audience cue is lost |
| `creative` | Open-ended writing where style matters | Pairwise judge | The output loses tone, form, voice or content the user asked for |
| `transform` | Rewrite, summarise, translate, reformat or extract from given text | Rubric judge + constraint checklist | Any explicit constraint is violated, or payload content is lost or altered |

Label rule for all categories (row 2.3.4): a rewrite is `preserved` iff its category's task metric is
**not worse** than the original prompt's **and** the position-swapped pairwise judge says
*tie or rewrite better* in **both** A/B orders. For `creative` and `explain`, which have no automatic
metric, the judge (with the category rubric) is the whole decision.

---

## 2. Quality measures

| Category | `CATEGORY_METRICS` | How it is computed | "Not worse" means |
| :--- | :--- | :--- | :--- |
| `code` | `unit_test_pass_rate` | Extract the code block from the answer and run the source's unit tests (HumanEval / MBPP tests, or hand-written tests for pilot seeds) in a sandbox with a timeout | Pass rate of the rewrite's answer ≥ pass rate of the original's answer |
| `fact` | `exact_match`, `token_f1` | SQuAD-style normalisation (lowercase, strip punctuation and articles); EM and token-level F1 against the reference answer(s), max over references | EM not lower **and** F1 not lower by more than 0.05 |
| `reason` | `final_answer_accuracy` | Parse the final answer (last number / option letter / boxed value) and compare to the reference (GSM8K) | Correct whenever the original's answer was correct |
| `explain` | `rubric_judge` | Judge scores both answers 1–5 on the rubric dimensions below, then gives a pairwise verdict | Pairwise verdict tie-or-better in both orders, and no rubric dimension lower by ≥ 2 |
| `creative` | `pairwise_judge` | Target samples n = 3 answers per prompt (2.2.3); judge compares original vs rewrite outputs pairwise, rubric as guidance | Tie-or-better in both orders, majority over the 3 sample pairs |
| `transform` | `rubric_judge`, `constraint_checklist` | Checklist of explicit constraints extracted from the **original** prompt (length, format, language, "keep the names", "no bullet points"), each checked on the answer (programmatically where possible, else by the judge); then rubric judge | Every checklist item satisfied by the original's answer is also satisfied by the rewrite's, and the pairwise verdict is tie-or-better |

The metric names are fixed strings; `terser.judge.metrics` (2.3.1) and the rubric YAML files (2.3.2)
must use exactly these names.

---

## 3. Category specifications

### 3.1 `code`

**Definition.** The user wants code as the main deliverable: write a function, script, query or
regex; fix, refactor, complete, optimise or translate code between languages; write tests.

**Include.** Any prompt whose best answer is mainly a code block that could be executed or
compiled, including SQL, shell commands and regexes. Debugging requests ("why does this crash, fix
it") belong here when the expected output is the fixed code.

**Exclude.**

| Borderline prompt | Category | Why |
| :--- | :--- | :--- |
| "can u explain what this python code does line by line" | `explain` | The deliverable is prose about code, not code |
| "what's the time complexity of quicksort" | `fact` | Short factual answer, no code expected |
| "convert this JSON to YAML" | `transform` | Data reformatting of given payload, not a program |
| "fix this bug and tell me what was wrong" | `code` | Code is the main deliverable; the explanation is secondary |
| "write a haiku about recursion in Python comments" | `creative` | Style is the point; nothing to test |

**Casual examples.**

1. "hey so i've got a list of dicts in python and i kinda need to sort them by the 'date' key, newest first, could you write a function for that? thanks!!"
2. "ok this is probably dumb but my js function keeps returning undefined, can you fix it? `function add(a,b){ a+b }`"

**Rubric dimensions** (used only for the judge's tie-break verdict; the test pass rate dominates):
correctness, completeness of requested features, adherence to stated language / library / signature,
code quality.

**Degradations.** Dropping or changing the function name, signature, language, library, version,
input/output format, edge cases the user named ("handle empty lists"), or any code block in the
payload (operator `D` never touches code blocks, 1.2.3). Dropped politeness or filler is **not** a
degradation.

### 3.2 `fact`

**Definition.** The user wants a short, verifiable answer: a name, date, number, definition,
yes/no, or a short list of facts.

**Include.** Knowledge lookups with a reference answer of a few words to one or two sentences.

**Exclude.**

| Borderline prompt | Category | Why |
| :--- | :--- | :--- |
| "who won the 2018 world cup and why were they so good" | `explain` | The "why" asks for multi-sentence reasoning prose |
| "how many days are there between march 3 and june 17" | `reason` | Requires computation to a checkable final answer |
| "what's the capital of australia" | `fact` | Single-entity answer |
| "what does HTTP 418 mean" | `fact` | Short definition |

**Casual examples.**

1. "umm quick question, what's the boiling point of water in fahrenheit? i always forget lol"
2. "sorry if this is a silly one but who actually wrote pride and prejudice?"

**Rubric dimensions** (tie-break only): correctness, completeness of all asked facts, concision.

**Degradations.** Dropping any entity, date, unit, qualifier or scope ("in Fahrenheit", "in 2019",
"in the UK") that changes the reference answer; merging two questions into one. Dropped tone cues,
hedging and apologies are **not** degradations: tone does not change a factual answer.

### 3.3 `reason`

**Definition.** Multi-step problems with a single checkable final answer: arithmetic word problems,
logic puzzles, unit conversions with several steps, multiple-choice questions requiring reasoning.

**Include.** Anything with a reference final answer where the model must derive, not recall, it
(GSM8K style).

**Exclude.**

| Borderline prompt | Category | Why |
| :--- | :--- | :--- |
| "explain how compound interest works" | `explain` | No single final answer |
| "what's 15% of 80" | `reason` | One step, but computed and checkable; `reason` wins over `fact` for any computation |
| "solve this and write the solution in python" | `code` | Deliverable is code with testable behaviour |
| "is it true that 0.999… = 1? prove it" | `explain` | Proof prose; no reference answer to compare beyond "yes" |

**Casual examples.**

1. "ok so if a train leaves at 3pm going 60 mph and another one leaves at 4pm going 80 mph from the same place, when does the second one catch up? pls show working"
2. "i'm trying to figure this out for my kid's homework: sarah has 3 times as many apples as tom, together they have 24, how many does tom have??"

**Rubric dimensions** (tie-break only): final-answer correctness, validity of steps, requested format
of working ("show working", "just the number").

**Degradations.** Dropping or changing any number, unit, quantity relation ("3 times as many"),
the question actually asked, or an answer-format instruction ("show working", "answer as a
fraction"). Dropping "for my kid's homework" is **not** a degradation unless the user also asked for
the answer pitched at that level.

### 3.4 `explain`

**Definition.** The user wants an explanation, tutorial, comparison, advice or analysis in prose,
where depth and correctness matter but there is no single reference answer.

**Include.** "How does X work", "why", "what's the difference between", "teach me", "pros and cons",
"explain this code / error / paper".

**Exclude.**

| Borderline prompt | Category | Why |
| :--- | :--- | :--- |
| "explain this code" + code block | `explain` | Prose about code; the code is payload to be explained, not rewritten |
| "summarise this article in 3 bullets" | `transform` | Operates on given text under explicit constraints |
| "write a blog post explaining black holes in a fun voice" | `creative` | Voice and style are an explicit requirement |
| "what is a black hole" | `explain` if an explanation is wanted, `fact` only if a one-sentence definition clearly is | See tie-break rule 2 (§4) |

**Casual examples.**

1. "so i keep hearing about vector databases at work and honestly i have no clue what they are, could you explain it like i'm not a programmer?"
2. "hey, what's the actual difference between TCP and UDP? like when would i use one over the other"

**Rubric dimensions:** coverage (all asked sub-questions answered), correctness, depth (appropriate
level of detail, not truncated), constraint adherence (length, format, audience), tone / audience fit.

**Degradations.** Dropping an audience or level cue ("like i'm not a programmer", "for a beginner"),
a sub-question, a comparison axis ("when would i use one over the other"), a length or format
constraint, or payload (code, error message) being explained. Removing pure filler and politeness is
**not** a degradation.

### 3.5 `creative`

**Definition.** Open-ended generation where style, tone, voice or form is part of the quality: stories,
poems, jokes, slogans, character dialogue, persuasive or emotional messages, brainstorming names.

**Include.** Prompts where two good answers can be very different and the user's tone and form
cues matter.

**Exclude.**

| Borderline prompt | Category | Why |
| :--- | :--- | :--- |
| "rewrite this email to sound more polite" + email | `transform` | Given payload with a style constraint |
| "write a cover letter for this job ad" | `creative` | Generated from scratch; tone and persuasion matter |
| "give me 10 names for a bakery" | `creative` | Open-ended ideation |
| "write a function that prints a random poem" | `code` | Executable deliverable |

**Casual examples.**

1. "could you write me a short, kinda bittersweet poem about moving out of my childhood home? nothing too cheesy please 🙏"
2. "my best friend's birthday is tomorrow and she loves cats and bad puns, can you write a funny card message? keep it short"

**Rubric dimensions:** adherence to requested tone / mood, form constraints (length, rhyme, format),
content requirements (named subjects, details), originality, fluency.

**Degradations.** Dropping any tone, mood or style cue ("bittersweet", "nothing too cheesy", "funny",
"bad puns"), audience, form or length constraint, or named content. **Tone cues are content in this
category**: removing "kinda bittersweet" is a degradation here even though the equivalent hedge-like
wording would be filler in `fact`. Emoji used as tone signals count as tone cues. Pure politeness
("please", "thanks") is not a degradation.

### 3.6 `transform`

**Definition.** The user supplies a payload (text, data, code comment, email, table) and wants it
changed under instructions: summarise, paraphrase, translate, change tone, reformat, extract fields,
correct grammar, shorten.

**Include.** Any prompt whose answer is mainly a function of a provided payload plus explicit
constraints.

**Exclude.**

| Borderline prompt | Category | Why |
| :--- | :--- | :--- |
| "translate this python to rust" | `code` | Output is executable code checked by tests |
| "explain this paragraph" | `explain` | Output is prose about the payload, not a transformed payload |
| "fix the grammar in this paragraph" | `transform` | Payload in, corrected payload out |
| "turn these notes into a story" | `creative` | Payload is only inspiration; the output is open-ended creative writing |

**Casual examples.**

1. "hi! can you please summarise the text below in like 3 bullet points max, and keep the names exactly as they are? thank you so much: <text>"
2. "so my boss wants this in a table, can you turn this into markdown with columns name, role and start date: <notes>"

**Rubric dimensions:** constraint adherence (checklist), content fidelity (nothing lost, added or
altered from the payload beyond what was asked), format correctness, fluency.

**Degradations.** Violating or dropping any explicit constraint (count, length, format, columns,
language, "keep the names exactly"), and **any change to the payload itself** (operators must treat
the payload as opaque, like code blocks). Dropped meta-talk ("I'm going to give you some text and I
want you to", operator `C`) is not a degradation as long as the instruction survives.

---

## 4. Labelling rule for prompts that fit two categories

Each prompt gets **exactly one** category. Annotators and the casualisation / labelling stages apply
these rules in order and stop at the first that decides:

1. **Main deliverable.** Choose the category of what the user must receive to be satisfied. If the
   answer is mainly executable code → `code`; mainly a transformed version of a given payload →
   `transform`; mainly a single checkable answer → `fact` (recalled) or `reason` (derived).
2. **Strongest applicable metric.** If still ambiguous, choose the category whose quality measure is
   the most objective and still applies to the answer, in this order:
   `code` (tests) > `reason` (final answer) > `fact` (EM/F1) > `transform` (checklist) > `explain`
   (rubric) > `creative` (pairwise). A prompt that is labelled with a stricter metric its answer cannot
   actually be scored by is mislabelled.
3. **Style cue override.** If the user explicitly asks for a voice, tone or literary form as a
   requirement (not as politeness), and rule 1 did not choose `code` or `transform`, choose
   `creative`.
4. **Record the runner-up.** Data records keep an optional `secondary_category`; it is used for
   error analysis (8.3.10, 12.8) and never for metrics or splits.

Worked examples:

| Prompt | Rule | Category (secondary) |
| :--- | :--- | :--- |
| "explain this code" + code block | 1: prose deliverable | `explain` (`code`) |
| "fix this function and explain the bug" | 1: fixed code is the deliverable | `code` (`explain`) |
| "how many seconds are in a leap year, explain your reasoning" | 2: final answer checkable | `reason` (`explain`) |
| "summarise this article in a funny pirate voice" | 1: payload transformed | `transform` (`creative`) |
| "tell me about the french revolution as a bedtime story" | 3: style requirement | `creative` (`explain`) |
| "what year did the berlin wall fall and why" | 1: "why" needs prose | `explain` (`fact`) |

Disagreements between the two annotators on category (2.3.5) are resolved by discussion against
these rules; if a rule was insufficient, update this document in the same PR.

---

## 5. What is never a degradation, in any category

Removing filler ("um", "basically"), pure politeness and hedging with no semantic effect
("could you maybe please", "thanks in advance"), meta-talk that restates the task, or exact
duplicates of a requirement, as long as every requirement survives once. These are exactly the
redundancies operators `F`, `H`, `C`, `D` and `I` target (1.2.1–1.2.6). The exception is §3.5:
in `creative`, a phrase that carries tone is content, not hedging.
