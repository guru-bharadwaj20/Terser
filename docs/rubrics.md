# Judging Rubrics

Plan row: **2.3.2**. Machine-readable source: [`src/terser/judge/rubrics/<category>.yaml`](../src/terser/judge/rubrics/),
loaded and validated by [`src/terser/judge/rubrics.py`](../src/terser/judge/rubrics.py). Category
definitions: [categories.md](categories.md). Annotation procedure: [annotation_guideline.md](annotation_guideline.md).

This page is for annotators and reviewers. It says what "better" means for each category when you
compare two answers to the same request, and which prompt changes count as a *degradation*. The
LLM judge (2.3.3) receives the same rubric, rendered from the YAML, inside its prompt. If this page
and a YAML file disagree, the YAML file is authoritative and this page must be fixed in the same PR.

---

## 1. How to use a rubric

1. Read the user's **original** request and note every explicit constraint (format, length,
   language, count, audience, tone for `creative`, "keep X unchanged").
2. For each response, check the **hard fails** first. A response with a hard fail loses unless
   both responses have one.
3. Score each response 1–5 on every dimension using the anchors. The weights say how much a
   dimension matters when the two responses differ on several dimensions.
4. Give the pairwise verdict: A, B or tie. Prefer *tie* when the weighted difference is small and
   no hard fail separates them. Length alone, position and writing style alone are never reasons.

**Role.** For `code`, `fact` and `reason` an automatic task metric (tests, EM/F1, final answer;
row 2.3.1) is checked first, and the rubric is only a *tie-break*. For `explain`, `creative` and
`transform` the rubric is the *primary* basis of the judgement.

**Never a degradation in any category:** removing filler, pure politeness, hedging with no
semantic effect, meta-talk that restates the task, or exact duplicate requirements, as long as
every requirement survives once. The single exception is tone in `creative` (section 2.5).

**Always a hard fail:** *the response violates an explicit constraint in the user request.*

### Tone cues by category

| Category | Dropping a tone / mood cue ("kinda bittersweet", "funny", 🙏) is a degradation? |
| :--- | :--- |
| `code` | No |
| `fact` | No: tone does not change a factual answer |
| `reason` | No |
| `explain` | No, **but** an audience or level cue ("like I'm not a programmer") is |
| `creative` | **Yes**: tone cues are content |
| `transform` | No, unless the tone change *is* the requested transformation ("make it more polite"), which is an explicit constraint |

---

## 2. Rubrics

Scores: 1 = poor … 5 = excellent. Weights in brackets.

### 2.1 `code` (tie-break)

| Dimension | 1 | 3 | 5 |
| :--- | :--- | :--- | :--- |
| correctness [0.40] | Does not run or does something else | Correct on typical inputs; wrong on a named or obvious edge case | Correct on all reasonable inputs incl. every named edge case |
| completeness [0.25] | Main requested code missing | Main function present; a requested feature/part missing | Every requested feature and part present |
| constraint_adherence [0.25] | Wrong language or ignores signature/format | One explicit constraint only partly followed | Language, library, name, signature, I/O format all as asked |
| code_quality [0.10] | Unreadable or grossly inefficient | Acceptable but clumsy | Clean, idiomatic, appropriately efficient |

Degradations: changed/dropped function name, signature, language, library, version or I/O format;
a dropped named edge case; any change to a user-supplied code block.
Hard fails: violates an explicit constraint; code does not run while the other's does.

### 2.2 `fact` (tie-break)

| Dimension | 1 | 3 | 5 |
| :--- | :--- | :--- | :--- |
| correctness [0.55] | Main fact wrong or missing | Main fact right; a secondary detail wrong | All facts right and precise |
| completeness [0.30] | No asked fact given | Main fact given; another asked fact or the requested unit/scope missing | All asked facts, in the requested unit and scope |
| concision [0.15] | Answer not findable or contradictory | Present with noticeable padding | Direct answer, at most a brief supporting sentence |

Degradations: dropped entity, date, unit, qualifier or scope that changes the answer ("in
Fahrenheit", "in 2019"); merged or dropped questions. **Not** degradations: tone, politeness,
hedging.
Hard fails: violates an explicit constraint; main fact wrong while the other's is right.

### 2.3 `reason` (tie-break)

| Dimension | 1 | 3 | 5 |
| :--- | :--- | :--- | :--- |
| final_answer_correctness [0.50] | No final answer, or wrong | Right but hard to find or ambiguous | Right, clearly stated, in the requested unit and form |
| step_validity [0.30] | Steps wrong or unrelated | Mostly valid; one unjustified step | Every step valid and easy to follow |
| format_adherence [0.20] | Requested format ignored | Partly followed | Followed exactly (or none requested and the answer is clear) |

Degradations: changed number, unit or quantity relation; dropped answer-format instruction ("show
working", "as a fraction"); a changed question. Context like "for my kid's homework" is **not** a
degradation unless the user asked for the answer pitched at that level.
Hard fails: violates an explicit constraint; wrong final answer while the other's is right.

### 2.4 `explain` (primary)

| Dimension | 1 | 3 | 5 |
| :--- | :--- | :--- | :--- |
| coverage [0.25] | Main question not answered | One sub-question or comparison axis missing | Every sub-question and axis answered |
| correctness [0.30] | Central claims wrong | Mostly accurate, a notable imprecision | Fully accurate |
| depth [0.20] | Superficial or cut off | Adequate but uneven | Right depth throughout |
| constraint_adherence [0.15] | Explicit constraints ignored | One constraint partly followed | Every constraint followed |
| audience_fit [0.10] | Wrong audience entirely | Mixed | Consistently pitched at the requested audience |

Degradations: dropped audience/level cue, sub-question, comparison axis, or explained payload.
Hard fails: violates an explicit constraint; any dimension 2 or more points below the other
response (the "not worse" rule of categories.md section 2).

### 2.5 `creative` (primary)

| Dimension | 1 | 3 | 5 |
| :--- | :--- | :--- | :--- |
| tone_adherence [0.30] | Opposite or unrelated tone | Tone present, one cue ignored | Every tone, mood and style cue followed |
| form_constraints [0.25] | Requested form ignored | One form constraint partly followed | Every form constraint followed |
| content_requirements [0.25] | Requested content missing | One requested detail missing | Every requested subject and detail used well |
| originality [0.10] | Generic boilerplate | Serviceable, some clichés | Distinctive and memorable |
| fluency [0.10] | Hard to read | Readable, some awkwardness | Polished |

Degradations: dropped tone, mood or style cue (emoji used as tone signals count), audience, form
or length constraint, or named content. Pure politeness is not.
Hard fail: violates an explicit constraint.

### 2.6 `transform` (primary)

| Dimension | 1 | 3 | 5 |
| :--- | :--- | :--- | :--- |
| constraint_adherence [0.40] | Most constraints violated | One constraint partly violated (e.g. 4 bullets for "3 max") | Every explicit constraint satisfied |
| content_fidelity [0.35] | Content invented or meaning changed | A secondary detail lost or altered | Fully faithful to the payload |
| format_correctness [0.15] | Not in the requested format | Valid but inconsistent | Clean and fully correct |
| fluency [0.10] | Hard to read | Readable with awkwardness | Polished |

Degradations: violated or dropped explicit constraint; any change to the payload itself. Dropped
meta-talk is not a degradation as long as the instruction survives.
Hard fails: violates an explicit constraint the other response satisfies; content invented that
is not in the payload.

Anchors 2 and 4 are in the YAML files and in the judge prompt; they sit between the neighbouring
anchors shown above.

---

## 3. Changing a rubric

Edit the YAML, bump its `version`, update this page, and run `pytest tests/test_judge_rubrics.py`.
The loader rejects files whose anchors are not exactly 1–5, whose weights do not sum to 1, or that
lack the shared rules (`dropped_tone_cue`, `dropped_politeness`, `dropped_filler`) or the
`violates_explicit_constraint` hard fail. A rubric change alters the judge prompt and therefore the
LLM cache key, so judge outputs produced with the old version must be regenerated.
