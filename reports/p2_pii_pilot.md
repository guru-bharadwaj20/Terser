# PII scrub report

- Source: `data/pilot/prompts.jsonl` (field `prompt`)
- Backend: `presidio`
- Records scanned: 500
- Actionable detections (redact / drop_row / flag_for_review): 8
- Recorded with action `ignore` (not PII under the policy): 67
- Allowlisted / suppressed candidates: 117
- Manual review sample: `reports/p2_pii_pilot_sample.csv` (500 rows, seed 0)

## Rows by outcome

| Status | Rows |
| :--- | ---: |
| clean | 493 |
| redacted | 0 |
| flagged | 7 |
| dropped | 0 |

## Active detections by entity type and action

| Entity type | Action | Count |
| :--- | :--- | ---: |
| LOCATION | ignore | 67 |
| PERSON | flag_for_review | 8 |

## Suppressed candidates (allowlist / context heuristic)

| Entity type | Backend | Reason | Count |
| :--- | :--- | :--- | ---: |
| EMAIL | regex | reserved_example_domain | 1 |
| PERSON | presidio | no_private_context | 113 |
| PHONE | regex | fictional_555_number | 3 |

## Actionable detections

| Record | Type | Action | Score | Backend | Context (entity masked) |
| :--- | :--- | :--- | ---: | :--- | :--- |
| creative-023 | PERSON | flag_for_review | 0.85 | presidio | one of my coworkers, <PERSON>, is retiring after 30 years at the comp |
| creative-068 | PERSON | flag_for_review | 0.85 | presidio | k you message for my kid's teacher, Mr. <PERSON>. a thank you message for the end of the |
| transform-012 | PERSON | flag_for_review | 0.85 | presidio | ep it about the same length.  "hey!! im <PERSON>, i love dogs sooo much and i have like  |
| transform-015 | PERSON | flag_for_review | 0.85 | presidio | it informal since it's my grandma.  "Hi <PERSON>! Thank you for the recipe, I made the a |
| transform-033 | PERSON | flag_for_review | 0.85 | presidio | length without dropping the dates?  "Hi <PERSON>, I just wanted to reach out and let you |
| transform-033 | PERSON | flag_for_review | 0.85 | presidio | lease feel free to contact my colleague <PERSON>, who has kindly agreed to cover for me. |
| transform-044 | PERSON | flag_for_review | 0.85 | presidio | yo turn this into json pls  name: <PERSON> age: 27 fav_color: teal pets: 2 |
| transform-049 | PERSON | flag_for_review | 0.85 | presidio | keep it casual, its for my penpal  "hey <PERSON>! school is so boring rn, but the weeken |

## Manual inspection of every detection (reviewer: implementing agent, 2026-09-11)

This section is written by hand and is lost if the scan is re-run; re-append it after any rerun.
Run: `python -m terser.data.pii scan --in data/pilot/prompts.jsonl --field prompt --out <scratch>
--report reports/p2_pii_pilot.md --sample reports/p2_pii_pilot_sample.csv --sample-size 500
--seed 0 --backend presidio` (presidio-analyzer 2.2.364, spaCy 3.8.16, en_core_web_sm 3.8.0).

The pilot set is authored/seed text (`origin` = `seed` or `authored`), so it contains no real
person's data. Every verdict below is therefore about whether the detector behaved as the policy
in `docs/pii_policy.md` intends, with "true PII" meaning "would be PII if the text were real".

### Actionable detections (8 spans, 7 rows, all `flag_for_review`)

| Record | Span | Cue | Verdict |
| :--- | :--- | :--- | :--- |
| creative-023 | first name of a coworker | "one of my coworkers," | Policy-correct flag (private individual); actually fictional, so FP against real PII |
| creative-068 | surname of the child's teacher | "my kid's teacher, Mr." | Policy-correct flag; fictional |
| transform-012 | author's first name in a pasted bio | "im" | Policy-correct flag; fictional |
| transform-015 | grandmother's name in a pasted note | salutation "Hi" | Policy-correct flag; fictional |
| transform-033 | recipient's name in an out-of-office e-mail | salutation "Hi" | Policy-correct flag; fictional |
| transform-033 | colleague's first name | "my colleague" | Policy-correct flag; fictional |
| transform-044 | full name in a personal profile | "name:" | Policy-correct flag; fictional |
| transform-049 | pen pal's name | salutation "hey" | Policy-correct flag; fictional |

Against real PII: 0 true positives, 8 false positives (all fictional names). Against the policy's
intent: 8/8 correct. Nothing was redacted or dropped, which is correct: the pilot contains no
e-mail, phone, card, IBAN, IP, credential, SSN or street address of a real person.

### Suppressed candidates (117)

- EMAIL `reserved_example_domain` (1, transform-037, `@example.com`): correct suppression.
- PHONE `fictional_555_number` (3: code-061 `555-123-4567` as a regex example; transform-037
  `555-0142` and `555-0199`, NANP numbers reserved for fiction): correct suppressions.
- PERSON `no_private_context` (113): no true PII suppressed. Breakdown by inspection:
  - spaCy mislabels of code and ordinary words (Java, JSON, Scanner, ggplot2, String, Map,
    `order_id`, `VARCHAR(255`, Jest, Mocha, Convert, Summarise, max, mug, Celsius, Kelvin,
    Spanish/Portuguese sentence fragments in translation tasks, ...).
  - public figures and works in factual questions (Adam Smith, John Maynard Keynes, Napoleon,
    Bernoulli, Norman, Frankenstein, Don Quixote, Spirited Away).
  - fictional characters, puzzle names and product names (Alice/Bob/Dan, tom/jess/max/sara,
    Snorgle, Pippa, Breezeline, Aurora, Moonlit Harbor, Kestrel Bay).
  - authored names of private-looking individuals in pasted documents with no cue:
    transform-004, -014, -018, -021, -022, -028, -035, -037, -042, -045, -052, -066, -079.
    These are fictional, so suppressing them leaks nothing here, but on real data they are the
    heuristic's known recall gap (see "Known limitations" in the policy); the human review
    sample is the backstop.

### Recorded with action `ignore` (67 LOCATION spans)

All 67 are non-personal. 46 are real place names at country/city/landform level (Australia,
Nile, Lisbon, Toronto, ...), which the policy does not treat as PII. 21 are spaCy mislabels:
code-003, -008, -010, -012, -021, -024 (x3), -028, -037, -038, -053, -069, -072 (C++, Python,
bash, Fibonacci, Fahrenheit, `customer_id`, `NUMERIC(10,2`, MM), explain-003 (UDP), explain-055
(Newton), transform-004 (Maria, Kenji), transform-022 (Dish), transform-035 (Ysolde),
transform-057 (Hindi).

### Residual PII in the 500-row sample

The sample CSV covers all 500 pilot rows. The DoD check ("0 detected PII in a 500-row manual
sample") must be done by a human reviewer filling `reviewer_pii_found` / `reviewer_notes` as
described in `docs/pii_policy.md`; this agent's inspection found no real-person PII in any row.
