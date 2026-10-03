# PII scrub report

- Source: `data/pilot/prompts.jsonl` (field `prompt`)
- Backend: `regex`
- Seed: 0
- Git SHA: `7bfce9f364d02ac0ff52eebf3e507b6fdfa5812d`
- Git dirty: true
- Input SHA256: `66c473e3884407f8244ed5cb90ef513cbe2fb235040a910f0e5ab7cb080cf077`
- Rows touched: 2
- Human review: pending; no 500-row review has been performed by this command.
- Records scanned: 500
- Actionable detections (redact / drop_row / flag_for_review): 5
- Recorded with action `ignore` (not PII under the policy): 0
- Allowlisted / suppressed candidates: 0

## Rows by outcome

| Status | Rows |
| :--- | ---: |
| clean | 498 |
| redacted | 2 |
| flagged | 0 |
| dropped | 0 |

## Active detections by entity type and action

| Entity type | Action | Count |
| :--- | :--- | ---: |
| EMAIL | redact | 1 |
| PHONE | redact | 4 |
| CREDIT_CARD | redact | 0 |
| IBAN | redact | 0 |
| IP_ADDRESS | redact | 0 |
| URL_CREDENTIAL | redact | 0 |
| API_KEY | redact | 0 |
| SECRET | redact | 0 |
| HIGH_ENTROPY_TOKEN | redact | 0 |
| US_SSN | redact | 0 |
| STREET_ADDRESS | redact | 0 |
| PERSON | redact | 0 |
| LOCATION | redact | 0 |

## Manual review

Sample the scrubbed output with the command below (substitute local paths):

```sh
python -m terser.data.pii sample --in <scrubbed.jsonl> --out <review.csv> --text-field <field> --sample-size 500 --seed 0
```

The sampler requires at least 500 rows and leaves reviewer columns blank.
Read every scrubbed field, including row_json, for all 500 rows; mark residual PII
as 0 or 1 and record reviewer, date, false positives and total residual PII here.
Pass requires 500 reviewed rows and zero residual PII. Keep 2.1.3 pending until then.

## Suppressed candidates (allowlist / context heuristic)

None.

## Actionable detections

See aggregate counts above; raw text and record identifiers are omitted.

## Reproduction commands and actual output

This is a CPU smoke run on authored pilot text. Counts cover all string values, so the same
illustrative phone in a prompt and a constraint is counted twice. Nothing was written into
`data/interim/`. The CSV has blank reviewer columns; no human review is claimed.

```sh
PYTHONPATH=src python -m terser.data.pii --in data/pilot/prompts.jsonl --out /tmp/terser-pii-pilot.jsonl --report reports/p2_pii_scrub_smoke.md --text-field prompt --seed 0
```

```text
backend=regex records=500 rows_touched=2 seed=0
counts={"API_KEY": 0, "CREDIT_CARD": 0, "EMAIL": 1, "HIGH_ENTROPY_TOKEN": 0, "IBAN": 0, "IP_ADDRESS": 0, "LOCATION": 0, "PERSON": 0, "PHONE": 4, "SECRET": 0, "STREET_ADDRESS": 0, "URL_CREDENTIAL": 0, "US_SSN": 0}
input_sha256=66c473e3884407f8244ed5cb90ef513cbe2fb235040a910f0e5ab7cb080cf077
git_sha=7bfce9f364d02ac0ff52eebf3e507b6fdfa5812d git_dirty=true
human_review=pending
```

```sh
PYTHONPATH=src python -m terser.data.pii sample --in /tmp/terser-pii-pilot.jsonl --out /tmp/terser-pii-pilot-review.csv --text-field prompt --sample-size 500 --seed 0
```

```text
sample_rows=500 population=500 seed=0 human_review=pending
```

## Review record

- Reviewer: pending
- Review date: pending
- Rows manually read: pending (required: 500)
- Residual PII count: pending (required: 0)
- False positives: pending
- Decision: pending; CONTRIBUTING.md row 2.1.3 remains incomplete

The report is an aggregate CPU validation artifact; no tracking service was used. The git SHA
above identifies the base revision and `git_dirty=true` explicitly records uncommitted code.

## Files changed

- `src/terser/data/pii.py`: strict redaction, per-type counts, direct CLI, safe reports and sampler.
- `src/terser/data/collect.py`: scrub every emitted string, including references; count entities.
- `src/terser/data/real_slice.py`: strict redaction before evaluation export.
- `tests/test_data_pii.py`: optional backend fallback expectation.
- `tests/test_data_pii_boundary.py`: detector, ingestion, CLI, provenance and sampling regressions.
- `docs/pii_policy.md`: strict ingestion contract, coverage limits and manual review procedure.
- `docs/plans/2.1.3-pii.md`: implementation sequence.
- `reports/p2_pii_scrub_smoke.md`: measured smoke, reproduction and verification evidence.

## Environment and test preparation

The initial system test invocation could not collect because configuration and tracking
dependencies were missing. Dependencies were installed only under `/tmp/terser-pii-venv`;
no project dependency or runtime configuration was changed. Small tokenizer vocabularies
were cached under `/tmp` for existing tests; no weights or NER resources were downloaded.
The scrubber itself uses the standard library and makes no network calls.

Preparation commands:

```sh
python -m venv --system-site-packages /tmp/terser-pii-venv
/tmp/terser-pii-venv/bin/python -m pip install hydra-core mlflow
/tmp/terser-pii-venv/bin/python -m pip install tiktoken cvxopt lightgbm tabulate
/tmp/terser-pii-venv/bin/python -m pip install 'hydra-core==1.3.2'
```

Python 3.14's argparse rejects the dependency's `LazyCompletionHelp` object. Both installed
configuration-library versions reproduced the failure in
`tests/test_config.py::test_hello_subprocess`; the temporary dependency copy uses
`help=str(LazyCompletionHelp())` instead of `help=LazyCompletionHelp()`. This fixes help rendering
only. Repository code and tests do not contain an environment workaround. An absolute source
path lets subprocess tests import the project from their temporary working directories.

Exact temporary compatibility adjustment:

```sh
/tmp/terser-pii-venv/bin/python - <<'PYTHON'
from pathlib import Path
import hydra
path = Path(hydra.__file__).parent / '_internal' / 'utils.py'
source = path.read_text()
old = 'help=LazyCompletionHelp(),  # type: ignore'
assert source.count(old) == 1
path.write_text(source.replace(old, 'help=str(LazyCompletionHelp()),'))
print('temporary dependency adjustment: render completion help as text')
PYTHON
```

Actual output:

```text
temporary dependency adjustment: render completion help as text
```

The isolated configuration subprocess check subsequently passed: `1 passed in 0.48s`.
The full selection before these environment corrections had one failure,
`tests/test_config.py::test_hello_subprocess`, with 2711 passed, 3 skipped and 29 deselected.
That run used a relative source path and preceded two additional hex regression tests.

## Commit blocker

File writes succeeded; the active filesystem profile marks `.git` read-only. Both staging
attempts failed before any task commit could be created:

```sh
git add tests/test_data_pii_boundary.py docs/plans/2.1.3-pii.md
```

```text
fatal: Unable to create '/home/viviansobers/Documents/Terser/.git/index.lock': Read-only file system
```

Task commits: **none**. Local branch: `main`. No push was attempted. The existing HEAD is:

```sh
git log -1 --oneline
```

```text
7bfce9f merge: reconcile phase-owner rewrite with C7 queue VRAM fix
```

The requested small local commits require write access to the repository metadata.
No SSH connection or accelerator workload was run. CONTRIBUTING.md was not edited.

## Final verification

Initial red phase, before strict implementation:

```sh
PYTHONPATH=src pytest tests/test_data_pii_boundary.py -q
```

```text
36 failed in 0.72s
```

Additional failing detector regressions were recorded before their fixes, covering optional NER,
URL userinfo, address false positives, short IPv6 and hexadecimal literals. The red logs remain
under `/tmp/terser-pii-*-red.log`.

Targeted final check:

```sh
PYTHONPATH=src pytest tests/test_data_pii.py tests/test_data_pii_boundary.py tests/test_data_casualise.py tests/test_data_real_slice.py -q -m 'not slow'
```

```text
183 passed, 3 deselected in 0.58s
```

Full requested CPU selection, in the temporary environment described above:

```sh
PYTHONPATH=/home/viviansobers/Documents/Terser/src HF_HOME=/tmp/terser-pii-hf-cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /tmp/terser-pii-venv/bin/python -m pytest -m 'not gpu and not api and not network and not slow'
```

Actual summary (full output: `/tmp/terser-pii-final-tests.log`):

```text
2714 passed, 3 skipped, 29 deselected, 2 warnings in 99.22s (0:01:39)
```

Both warnings are existing tracking-library uses of deprecated `codecs.open`. The result
is qualified by the temporary dependency help-rendering adjustment, not a repository fix.

```sh
ruff check src/terser/data/pii.py src/terser/data/collect.py src/terser/data/real_slice.py tests/test_data_pii.py tests/test_data_pii_boundary.py
git diff --check
git diff --quiet -- CONTRIBUTING.md
```

```text
All checks passed!
```

Both git checks exited 0 without output. The final code was smoke-tested again with:

```sh
PYTHONPATH=src python -m terser.data.pii --in data/pilot/prompts.jsonl --out /tmp/terser-pii-pilot.jsonl --report /tmp/terser-pii-pilot.md --text-field prompt --seed 0
PYTHONPATH=src python -m terser.data.pii sample --in /tmp/terser-pii-pilot.jsonl --out /tmp/terser-pii-pilot-review.csv --text-field prompt --sample-size 500 --seed 0
```

These produced the identical smoke and sampler output shown above. The definition-of-done
human review and the requested local commits remain outstanding.
