# Human Annotation Guideline (pairwise, blind)

Plan row: **2.3.5**. Used by: row 2.3.6 and checkpoints C2.3–C2.6.
Code: [`src/terser/judge/annotation.py`](../src/terser/judge/annotation.py) (batch export and
sheet import), [`src/terser/phase2/judge_validation.py`](../src/terser/phase2/judge_validation.py)
(agreement report). Categories: [categories.md](categories.md). Rubrics: [rubrics.md](rubrics.md).

Two annotators label **300 pairs** (50 per category). Each pair is one user request and two model
responses to it. One response came from the user's original prompt and the other from a shortened
rewrite of that prompt. Annotators are **not told which is which**. The human labels are the
reference used to check the LLM judge, so a careful "tie" or "cannot judge" is worth more than a
quick guess.

---

## 1. What you see and what you do

Each row of your sheet is one item:

| Column | Meaning | You edit it? |
| :--- | :--- | :--- |
| `item_id` | Opaque id such as `item042`. It says nothing about the pair. | **No** |
| `category` | Task category: `code`, `fact`, `reason`, `explain`, `creative`, `transform` | **No** |
| `request` | The user's request, exactly as the user wrote it | No |
| `response_a` | First response | No |
| `response_b` | Second response | No |
| `verdict` | `A`, `B`, `tie` or `cannot_judge` | **Yes, required** |
| `confidence` | `1`, `2`, `3`, or empty | Yes, optional |
| `notes` | Free text | Yes, optional |

For every item, answer one question:

> **Which response better does what the user asked for?**

Always judge against the `request` column. Never guess which response came from the rewrite, and
don't look for clues to it. A shorter response is not more likely to be the rewrite, and a longer
one is not more likely to be the original.

### 1.1 Blinding and randomised A/B order

* The order of rows is shuffled separately for each annotator.
* For every item, whether the original's output is shown as Response A or Response B is decided at
  random, **separately for each annotator**. Your A may be your co-annotator's B, so don't discuss
  items as "A was better". Refer to the `item_id` and the content.
* The batch directory holds a `key.jsonl` that shows which response is which. Only the study
  coordinator opens it. Annotators never receive it, and nobody should mention its contents to
  annotators until the sheets are returned.
* Do not look at judge outputs, pilot labels or the other annotator's sheet before you return
  yours.

---

## 2. Decision rules

Work through the rules in order. The first rule that decides the item wins.

1. **Read the request first.** List what it asks for: the task, every explicit constraint (language,
   format, length, audience, names to keep, things to avoid) and the implied deliverable (code, one
   fact, a final answer, prose, a transformed text).
2. **Check hard failures.** A response that is wrong, doesn't do the task, breaks an explicit
   constraint, or leaves out a part of the request that was asked for is worse than one that
   doesn't. When only one response fails, choose the other (`A` or `B`).
3. **Use the category rubric.** When both pass or both fail step 2, compare the two on the
   category's rubric dimensions (see [rubrics.md](rubrics.md) and the per-category
   "Rubric dimensions" in [categories.md](categories.md)). In short:

   | Category | What matters most | Pointer |
   | :--- | :--- | :--- |
   | `code` | Would the code run and do what was asked? Correct signature, language and libraries; named edge cases handled | rubrics.md, `code` |
   | `fact` | Is the answer correct and does it answer the exact question? Extra padding is not a plus | rubrics.md, `fact` |
   | `reason` | Is the final answer correct and in the requested format? Clear working breaks ties only | rubrics.md, `reason` |
   | `explain` | Coverage, correctness, depth, and fit to the audience the user named | rubrics.md, `explain` |
   | `creative` | Form, tone, voice and content the user asked for | rubrics.md, `creative` |
   | `transform` | Every constraint followed; payload content kept, not added to or changed | rubrics.md, `transform` |

4. **Choose `tie`** when the responses are of essentially equal quality for this request, even if
   they differ in wording, order, formatting or length. `tie` is a real answer, not a fallback.
   Don't pick a winner because of small style preferences.
5. **Choose `cannot_judge`** only when you cannot tell which response is better even after
   reasonable effort (see section 3). Write the reason in `notes`.

**Never** decide based on:

* length alone ("B is longer so it is more thorough"): more text wins only if the extra content
  was needed for the request;
* position (A vs B);
* politeness, greetings or sign-offs, unless the user asked for a tone;
* how confident the response sounds;
* your personal view on the topic, when both responses do what was asked.

### 2.1 Borderline examples

| Request (abridged) | Response A | Response B | Verdict | Why |
| :--- | :--- | :--- | :--- | :--- |
| "sort a list of dicts by 'date', newest first, python pls" | Correct `sorted(..., reverse=True)` function | Same function plus a 3-line explanation | `tie` | Both correct; the explanation was not requested and does no harm |
| "…and handle an empty list" | Correct code, no empty-list handling | Correct code with an empty-list guard | `B` | A named edge case is a requirement (step 2) |
| "capital of Australia? one word" | "Canberra" | "The capital of Australia is Canberra, not Sydney as many think." | `A` | B breaks the explicit "one word" constraint |
| "what's 15% of 80, just the number" | "12" | "15% of 80 = 0.15 × 80 = 12" | `A` | Same answer, but B ignores "just the number"; if the request had no format cue this is `tie` |
| "explain recursion to a 10-year-old" | Accurate, uses a mirrors analogy, simple words | Accurate, formal, mentions call stacks and base cases | `A` | Both correct; the audience cue decides (rubric: audience fit) |
| "write a limerick about my cat Milo" | Limerick (AABBA) naming Milo | Good rhyming poem naming Milo, 6 lines, not a limerick | `A` | The requested form is a constraint |
| "summarise this email in 2 bullets" | 2 bullets, all key points | 2 bullets, drops the meeting date | `A` | Payload content lost (transform) |
| "summarise this email in 2 bullets" | 2 bullets, key points, different wording | 2 bullets, key points | `tie` | Wording differences only |
| "write a short story about a lighthouse" | Competent 300-word story | Competent 280-word story, different plot | `tie` | Open-ended request, both do it; don't pick by plot taste. If one is clearly stronger on the creative rubric, choose it and set confidence 1–2 |
| "fix my SQL query" (query pasted) | Fixed query | Fixed query with a subtle wrong JOIN condition | `A` | Correctness failure, even if you had to read closely |
| Both responses wrong in different ways | Wrong answer, right format | Wrong answer, wrong format | `A` | Both fail step 2; A fails less. Use `tie` only if the failures are equally bad |

### 2.2 Confidence

| Value | Meaning |
| :--- | :--- |
| `3` | Clear-cut; you would bet the other annotator agrees |
| `2` | You'd defend the call, but it took some thought |
| `1` | Close call; you'd accept the opposite verdict |
| empty | Not given (allowed, but please fill it in when you can) |

Confidence doesn't change the label. It helps with adjudication and error analysis.

---

## 3. "Cannot judge"

Use `cannot_judge` when a fair comparison isn't possible, for example:

* checking correctness needs expertise or tools you don't have within reasonable effort (you can't
  tell whether either piece of code works, or the fact can't be checked), and the responses differ
  on exactly that point;
* the request is unintelligible, or both responses are cut off or garbled to the same degree;
* the request uses a language you can't read well enough.

Don't use it for close calls (use `tie`, or pick one with confidence `1`) or to skip long items.
Always explain it in `notes`. Items marked `cannot_judge` get **no human label**. They're left out
of the agreement statistics and counted as dropped in the report, so a high rate weakens C2.3–C2.5.
Aim for **under 5%**.

---

## 4. Calibration round

Before the main batch:

1. The coordinator builds a separate **calibration batch** of about 3 pairs per category (18
   items) from records that are **not** in the 300-pair sample. Drop the sampled ids from the
   records file first, then build with a different seed and output directory, for example
   `--per-category 3 --out data/gold/calibration`.
2. Both annotators label it independently, following this guideline, and return their sheets as
   described in section 5.
3. The coordinator imports the sheets and reviews every disagreement with both annotators in one
   session. Only then do they open the key. Where this guideline was unclear, add a rule or a
   borderline example to section 2.1 and note the change in the guideline's history.
4. Repeat once with new items if agreement is clearly poor (roughly under 14 of 18 raw
   agreements, or if one annotator uses `tie` much more often than the other).

Calibration items and labels are **never** part of `data/gold/human_300.jsonl`. After calibration,
annotators must not discuss main-batch items with each other until both sheets are returned.

---

## 5. Filling in and returning sheets

### 5.1 What you receive

The coordinator runs:

```bash
python -m terser.judge.annotation build --records <pairs.jsonl> --out data/gold/batch \
    --annotators ann1 ann2 --seed <seed>
```

This writes to `data/gold/batch/`:

* `annotator_<name>.csv` and `annotator_<name>.jsonl`: your sheet, in two equivalent formats.
  You get only your own sheet, in one of the two formats.
* `key.jsonl` and `manifest.json`: kept by the coordinator. **Not sent to annotators.**

The CSV is UTF-8 with the header
`item_id,category,request,response_a,response_b,verdict,confidence,notes` and LF line endings.
Response cells can span several lines inside quotes.

### 5.2 Filling it in

* Fill `verdict` for **every** row with exactly one of `A`, `B`, `tie`, `cannot_judge`.
  Upper or lower case both work, and `cannot judge` / `cannot-judge` are also accepted. Anything
  else, including an empty cell, is rejected.
* `confidence`: `1`, `2`, `3` or empty.
* `notes`: any text. Required for `cannot_judge`.
* **Don't** change `item_id` or `category`. Don't delete rows, duplicate rows or add rows. You can
  sort or filter while working (items are matched by `item_id`), and extra columns you add are
  ignored.
* Spreadsheet tools: if you use Excel, save with **"CSV UTF-8 (Comma delimited)"**. Other CSV
  options can break non-ASCII text. A byte-order mark is fine. Check that long responses weren't
  cut off (Excel limits a cell to 32,767 characters) and that `tie` wasn't auto-corrected. For the
  JSONL sheet, edit only the `verdict`, `confidence` and `notes` values and keep one JSON object
  per line.

### 5.3 Returning it

Rename the file to `filled_<name>.csv` (or `.jsonl`) and send it to the coordinator. The
coordinator imports both sheets:

```bash
python -m terser.judge.annotation import --batch data/gold/batch \
    --sheet ann1=filled_ann1.csv --sheet ann2=filled_ann2.csv \
    --out data/gold/human_300.jsonl
```

The import checks every row against `key.jsonl` and lists **all** problems at once: missing or
repeated `item_id`s, unknown items, a changed `category`, an invalid `verdict` or `confidence`.
A sheet with any problem is rejected as a whole. The coordinator sends the list back and you
return a corrected sheet.

### 5.4 What happens next (coordinator)

* Each verdict is mapped back from position to side (`original` / `rewrite` / `tie`). A side of
  `original` gives the human label `degraded`. `rewrite` or `tie` gives `preserved`.
  `cannot_judge` gives no label.
* `human_300.jsonl` has one record per pair with both annotations. `gold_label` is the shared label
  when both annotators agree. Otherwise it is empty until adjudicated.
* **Adjudication** happens only **after** both sheets are imported, so human–human κ (C2.3) is
  computed on the independent labels. A third person, or both annotators together, decides each
  disagreement without seeing judge outputs. The decisions go in a JSONL file with one
  `{"id": "<source id from human_300.jsonl>", "gold_label": "preserved" | "degraded"}` per line.
  Re-run the import with `--adjudications <file>`.
* Agreement report (row 2.3.6):

  ```bash
  python -m terser.phase2.judge_validation --gold data/gold/human_300.jsonl \
      --judge <label-stage labels.jsonl> --out reports/p2_judge_validation.md
  ```
