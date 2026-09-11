# PII policy

Plan row **2.1.3**. Implementation: `src/terser/data/pii.py`. Tests: `tests/test_data_pii.py`.
Pilot report: `reports/p2_pii_pilot.md`.

Every text that enters `data/interim/` must first pass through the scrubber. Scrubbing
happens before casualisation, deduplication and splitting, so no downstream artefact (candidates,
target outputs, judge labels, the frozen test split) can carry PII that was present in a source.

## What counts as PII in Terser

Terser works on prompts: short instructions, questions, code and pasted documents. Something is
PII when it identifies, contacts or gives access to a **real, private** person or account:

- contact data: personal e-mail addresses, phone numbers, street addresses;
- financial and government identifiers: payment card numbers, IBANs, US SSNs;
- network identifiers: routable IP addresses;
- credentials: passwords in URLs, API keys, tokens, private keys, secret assignments;
- names of private individuals when the text ties them to the author ("my coworker Dana").

The following are **not** PII and must survive scrubbing, because deleting them would corrupt the
task and bias the dataset against technical and factual prompts:

- code: identifiers, loopback / private IPs in configs, version strings, hex digests, file paths,
  decorators and `@` operators;
- documentation placeholders: `example.com` addresses, `YOUR_API_KEY`, public test cards,
  standards-document IBANs, fictional 555 numbers;
- public figures, authors, works and places in factual questions ("who wrote Frankenstein?");
- fictional characters and invented names in creative and puzzle prompts ("a dragon named Ember",
  "Alice is taller than Bob");
- place names at country, city or landform level.

## Backends

| Backend | Always available | Covers | Dependencies |
| :--- | :---: | :--- | :--- |
| `regex` | yes | every entity type below except PERSON and LOCATION | standard library only |
| `presidio` | when installed | regex layer + spaCy NER for PERSON and LOCATION | `presidio-analyzer`, `spacy`, `en_core_web_sm` |

`--backend auto` (the default) uses Presidio when `presidio_analyzer` and `en_core_web_sm` are
importable and falls back to regex otherwise. The backend that actually ran is recorded in the
report header, in every scrubbed record (`pii.backend`) and on every detection. Presidio tests are
skipped when it is not installed. Versions used for the pilot: presidio-analyzer 2.2.364,
spaCy 3.8.16, en_core_web_sm 3.8.0.

## Entity policy

Actions: `redact` replaces the span with a typed placeholder (`<EMAIL>`); `drop_row` removes the
record from the output; `flag_for_review` keeps the text unchanged but marks the record `flagged`
so it is held until a human clears it; `ignore` records the span in the report only. When spans
overlap, the most severe action wins, then the higher score, then the longer span. Overrides can
be passed with `--policy file.{json,yaml}` (entity type to action).

| Entity | Detection (regex layer unless noted) | Default action | Why this action |
| :--- | :--- | :--- | :--- |
| `EMAIL` | RFC 5322-ish address with a real TLD | redact | Contact data; the placeholder keeps the prompt usable |
| `PHONE` | `+`-prefixed international (8-15 digits); NANP 10-digit; 7-digit only after a cue word (call, phone, at, on, ...) | redact | Contact data |
| `CREDIT_CARD` | 13-19 digits, consistent separators, leading 2-6, Luhn-valid | drop_row | Financial data; a prompt containing a live card is not worth keeping |
| `IBAN` | country code + check digits, ISO 13616 length, mod-97 valid | redact | Account identifier |
| `IP_ADDRESS` | IPv4 / IPv6 parsed by `ipaddress` | redact | Can identify a household |
| `URL_CREDENTIAL` | `scheme://user:password@`, or `?token=`, `api_key=`, `password=` ... query values (8+ chars) | redact | Credential |
| `API_KEY` | vendor formats: AWS access key IDs, GitHub, Slack, OpenAI/Anthropic `sk-`, Google `AIza`, Stripe live keys, JWTs, PEM private-key headers | redact | Credential |
| `SECRET` | `password=`, `api_key:`, `token = "..."` assignments with a value of 8+ chars | redact | Credential |
| `HIGH_ENTROPY_TOKEN` | 32+ chars mixing upper, lower and digits, entropy >= 4 bits/char | flag_for_review | Often a secret, but also base64 test data; needs a human |
| `US_SSN` | `AAA-GG-SSSS` with SSA-invalid ranges excluded (000, 666, 9xx, 00, 0000) | drop_row | Government identifier |
| `STREET_ADDRESS` | number + 1-3 capitalised words + street suffix, optional unit | flag_for_review | Conservative pattern; many hits are fictional or landmark addresses |
| `PERSON` (presidio) | spaCy PERSON, kept only with a private-individual cue | flag_for_review | Names are the hardest call; a human decides |
| `LOCATION` (presidio) | spaCy GPE/LOC | ignore | Places at the granularity NER finds are not personal; recorded for audit |

## Allowlists and heuristics

A candidate that hits an allowlist is kept as a *suppressed* detection with a reason. Suppressed
candidates are counted in the report (by type, backend and reason) so the allowlists can be
audited, but they do not change the text.

| Reason | Applies to | Rule | Rationale |
| :--- | :--- | :--- | :--- |
| `reserved_example_domain` | EMAIL | `example.com/.net/.org`, `*.example`, `*.test`, `*.invalid`, `*.localhost` | RFC 2606 / RFC 6761 reserve these; no real mailbox can exist |
| `placeholder_email` | EMAIL | `your.email@yourdomain.com`, `foo@bar.com`, `john.doe@company.com` shapes | Tutorial placeholders |
| `git_ssh_remote` | EMAIL | `git@github.com` and other forges | SSH remote, not a mailbox |
| (not emitted) | EMAIL | domain ending in a code attribute (`a@b.shape`, `@pytest.fixture`), or the `password@host` part of a URL | Code; URL passwords are handled as `URL_CREDENTIAL` |
| `fictional_555_number` | PHONE | NANP exchange or area code 555 | Reserved for fiction and documentation |
| `ofcom_drama_number` | PHONE | UK +44 20 7946 0xxx, +44 7700 900xxx, +44 1632 960xxx | Ofcom drama ranges |
| `invalid_nanp_area_code` | PHONE | area code starting 0 or 1 | Not diallable |
| `public_test_card` | CREDIT_CARD | Stripe / Braintree / PayPal documented test numbers | Published for testing |
| (not emitted) | CREDIT_CARD | Luhn failure; digits inside an IBAN | Not a card |
| `documented_example_iban` | IBAN | IBANs printed in ISO 13616 and bank documentation | Published examples |
| `loopback_ip`, `unspecified_ip`, `link_local_ip`, `multicast_ip`, `reserved_ip` | IP | `ipaddress` properties | Not routable; typical in code (`127.0.0.1`, `0.0.0.0`) |
| `private_or_documentation_ip` | IP | RFC 1918, RFC 5737 (`192.0.2/24`, `198.51.100/24`, `203.0.113/24`), `2001:db8::/32` | Private networks and documentation ranges |
| `public_dns_resolver` | IP | `8.8.8.8`, `1.1.1.1`, `9.9.9.9`, OpenDNS and their IPv6 forms | Identify a service, not a person |
| `version_string` | IP | dotted quad preceded by `v`, `version`, `==`, `>=`, `@` | `pip install pkg==1.2.3.4` |
| (not emitted) | IP | invalid octets, times `12:30:45`, MAC addresses, C++ scopes `a1::b2` | Not IPs |
| `placeholder_credential` | URL_CREDENTIAL, API_KEY, SECRET, HIGH_ENTROPY_TOKEN | contains `example`, `your_`, `changeme`, `xxxx`, `<...>`, `${...}`, `test`, `demo`, `fake`; trivial passwords (`password`, `hunter2`, ...); <= 3 distinct characters | Documentation placeholders, including AWS's `...EXAMPLE` key |
| `identifier_not_secret` | SECRET | unquoted value that is a bare identifier (`token = tokenizer`), a dotted name (`os.environ.get`) or followed by `(` / `[` | Code reads a secret from somewhere; it does not contain one |
| (not emitted) | HIGH_ENTROPY_TOKEN | pure hex (git SHAs, checksums), paths with 2+ `/` | Not secrets |
| `example_ssn` | US_SSN | `123-45-6789`, `078-05-1120` (voided Woolworth card), `111-11-1111` | Never issued or publicly voided |
| `famous_or_fictional_address` | STREET_ADDRESS | 221B Baker St, 742 Evergreen Terrace, 1600 Pennsylvania Ave, 10 Downing St, 4 Privet Drive, 42 Wallaby Way, 123 Main/Fake St, company HQs, ... | Landmarks and fiction |
| `no_private_context` | PERSON | no private-individual cue (below) | Public figures, fictional characters, NER mislabels of code |
| `overlaps_regex_detection` | PERSON, LOCATION | NER span overlapping an active regex detection | Avoid double counting |

### Private-individual cues for PERSON

A PERSON span is flagged only if one of these holds (`private_person_context`):

1. a relationship word before the name, optionally plural and with an honorific: "my colleague
   Arvid", "one of my coworkers, Delphine", "my kid's teacher, Mr. Okafor";
2. a relationship or personal attribute after it: "Priya, my roommate", "Sam's phone number";
3. a self-introduction or sign-off: "my name is", "I'm", "im", "regards,", "thanks,";
4. a salutation at the start of a line or quotation: "Dear ...", `"Hi ...`;
5. a `name:` field ("name: Wrenna Holt");
6. "named" / "called" before the name, unless it introduces a character, animal or object
   ("a dragon named Ember" stays suppressed);
7. any other active non-name PII in the same record (a name next to an e-mail is contact data).

Everything else is suppressed, which is what keeps factual questions ("is it Adam Smith or John
Maynard Keynes?"), puzzles and code (spaCy tags `Java`, `JSON`, `String` as PERSON) intact.

### Known limitations

- Names of private-looking people in pasted documents with no cue (meeting notes, thank-you
  lists, rosters) are suppressed. On the pilot these are all authored fictional names; on real
  data they are the main recall gap, and the human review sample is the backstop.
- PERSON and LOCATION need the Presidio backend; with `regex` only, no names are detected.
- Local phone numbers without a cue word, non-US national IDs, dates of birth, licence plates and
  postal addresses without a street suffix are not detected.
- Treating every 555 exchange as fictional is slightly generous (only 555-0100..0199 is formally
  reserved); accepted because 555 numbers in prompts are overwhelmingly examples.
- Detection is English-centric; names in other languages are detected only as well as
  `en_core_web_sm` manages.

## Running a scan

```bash
python -m terser.data.pii scan --in data/raw/<source>.jsonl --field prompt \
    --out data/interim/<source>.jsonl --report reports/pii_<source>.md \
    --sample data/interim/pii_review/<source>_sample.csv --sample-size 500 --seed 0
```

Outputs:

- scrubbed JSONL: dropped rows omitted, redactions applied, a `pii` object per record with
  `status` (`clean` / `redacted` / `flagged`), `backend` and the detections (offsets, type,
  score, action; never the matched text);
- markdown report: rows by outcome, detections by type and action, suppressed candidates by
  reason, and a table of actionable detections with the entity masked;
- review CSV: `sample_size` rows drawn with `random.Random(seed).sample`, sorted by index, with
  columns `id, original, scrubbed, detections, reviewer_pii_found, reviewer_notes`, the last two
  blank. The same input, seed and size always produce a byte-identical CSV.

Records with status `flagged` must not be used downstream until a reviewer clears them (keep,
redact by hand, or drop).

## Human review procedure (DoD: 0 detected PII in a 500-row manual sample)

The DoD cannot be automated: its purpose is to measure what the detector misses. For each source
before its output is accepted into `data/interim/`:

1. Run the scan with `--sample-size 500 --seed 0` (a different seed only if 0 was already used for
   a failed review of the same file; record the seed in the report).
2. A reviewer who did not write or tune the allowlists opens the CSV and reads the **scrubbed**
   column of every row (for dropped rows it is empty; check `original` to confirm the drop was
   justified).
3. For each row set `reviewer_pii_found` to `0` or `1`. `1` means the scrubbed text still contains
   PII as defined above (including a flagged row that should have been redacted). Add the entity
   type and a short, **non-identifying** description in `reviewer_notes` (e.g. "PERSON, coworker's
   surname in pasted e-mail"), never the PII itself.
4. Also note false positives (redactions or drops that destroyed non-PII content) in
   `reviewer_notes` with the prefix `FP:`. They do not fail the DoD but feed allowlist tuning.
5. Pass: all 500 rows reviewed and the sum of `reviewer_pii_found` is 0. Record reviewer, date,
   seed, backend and the FP count in the source's report and data card.
6. Fail: fix the detector or policy (adding a test for the miss), rerun the scan on the whole
   source, and review a fresh 500-row sample with a new seed. Do not patch only the sampled rows.

If a source has fewer than 500 rows, all rows are reviewed.

## Retention

- `data/raw/` holds unscrubbed text and is git-ignored. Raw files are kept only as long as the DVC
  pipeline needs them to rebuild and are never copied into reports, issues, notebooks, logs or
  model inputs.
- Scrubbed outputs in `data/interim/` and later stages carry no matched PII text; detections store
  offsets and types only.
- Reports list detections with the entity replaced by its placeholder and may be committed.
- Review CSVs contain the **original** text and therefore can contain PII. For any source that is
  not authored/synthetic they live under `data/interim/pii_review/` (git-ignored), are shared only
  with the reviewer, and are deleted within 30 days of the review being recorded; only the
  aggregate outcome goes into the report. The pilot CSV (`reports/p2_pii_pilot_sample.csv`) is
  committed only because the pilot set is authored text with no real personal data.
- Rows dropped for `CREDIT_CARD` or `US_SSN` are not retained anywhere outside `data/raw/`.
- Real user prompts are out of scope without explicit consent (see CONTRIBUTING, "Data, privacy
  and LLM usage"); production stores hashes and metrics by default, and opted-in raw text is
  retained at most 30 days.
- Secrets found by the scrubber in a source dataset are reported to the dataset owner where
  practical; they are never tested for validity.
