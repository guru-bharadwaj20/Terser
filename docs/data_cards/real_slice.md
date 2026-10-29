# Data Card: `real_slice` (real-prompt evaluation slice)

This card covers 1,000 first-turn prompts written by real users, drawn from WildChat-1M. They are
**used for evaluation only**: never casualised and never used for training or tuning.

- Builder: `src/terser/data/real_slice.py`
- Tests: `tests/test_data_real_slice.py`

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Plan row(s) | `2.1.8` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-09-30` |
| DVC path | `data/processed/real_slice/` (raw: `data/raw/wildchat/`) |
| DVC hash | `5b1034c3caa360ba193b82ad72bb973f.dir` (raw: `147701f011325ec3e8f3e8c71280e695.dir`) |
| Test lock | `test.lock.json`, sha256 of `test.jsonl` = `2662cdae1ef0845aba760fd61c2d53226a3010c71d00f7d5aa9471bd722733b4` |

---

## 1. Source

| Upstream | Version / revision | URL | Retrieved | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- | :--- |
| `allenai/WildChat-1M` | `7d6490e462285cf85d91eabea0f9a954fbddcd1f` | https://huggingface.co/datasets/allenai/WildChat-1M | 2026-09-30T17:31:01Z | `wildchat` (evaluation-only section) |

Only one of the 14 shards is used: `data/train-00000-of-00014.parquet`. It is 230,786,095 bytes,
with sha256 `abec2a13…a6fdf6551fbcb34430e51c` (full value in `MANIFEST.json`). That one shard
yields far more candidates than the slice needs.

WildChat is not in the training-source registry (`terser.data.sources.SOURCES`). This means
`python -m terser.data.sources download` can never pull it into the training pool.

## 2. Licence and redistribution

| Source | Licence | Research use | Redistribution | Notes / audit link |
| :--- | :--- | :---: | :---: | :--- |
| `wildchat` | `ODC-By-1.0` | yes | yes | Attribution required. The authors relicensed from AI2 ImpACT to ODC-BY on 2024-06-26, retroactively. Decision: include (evaluation only). `reports/license_audit.md#wildchat` |

We use only user turns. No ChatGPT output from WildChat enters the slice.

## 3. Size

| Stage | Rows | Median tokens (`qwen2.5`) | p95 tokens |
| :--- | ---: | ---: | ---: |
| First user turns passing the filters (section 4) | 21,270 | — | — |
| After dedupe (2,587 duplicates removed) | 18,683 | — | — |
| After PII policy (3 rows dropped, 16 scrubbed) | 18,680 | — | — |
| Category-tagged pool (13,105 untagged removed) | 5,575 | — | — |
| **Final slice** | **1,000** | **40** | **349** |

## 4. Collection method

Build commands:

```bash
python -m terser.data.real_slice download
python -m terser.data.real_slice build      # seed 0, size 1000
python -m terser.data.real_slice verify
```

Filters are applied to the first turn of each conversation:

- Role `user` and `language == "English"`.
- Not `toxic`. The authors have already removed conversations flagged by OpenAI Moderation or
  Detoxify from this release.
- Not `redacted`. We drop rows where the authors' Presidio pass fired, so no `<PRESIDIO_…>`
  placeholders appear.
- 5–300 words.
- At least 95% ASCII characters and at least 60% letters.
- Not a pasted log, timestamp or forum dump.

After filtering:

- **Deduplication:** on case- and whitespace-normalised text.
- **Sampling:** category-balanced. Each category gets an equal quota, and a quota a category
  cannot fill is redistributed to the others. Within a category, prompts are taken in
  `sha256("terser-real-slice:0:<source_id>")` order, so the output is deterministic.

## 5. Casualisation process

Not applicable. The whole point of the slice is that it is never casualised (`casualised: false`
on every row).

## 6. Splits

| Split | Rows | Source prompts | Share | Hash |
| :--- | ---: | ---: | ---: | :--- |
| test (frozen 2026-09-30) | 1,000 | 1,000 | 100% | sha256 above |

- There is no train or val split, and every row carries `use: "eval_only"`.
- `terser.data.splits.assert_no_test_access` guards the file. Tuning code must not read it.
- `verify_frozen` fails if a single byte changes.

## 7. PII handling

| Item | Value |
| :--- | :--- |
| Detector | Upstream: Microsoft Presidio plus hand-written rules (WildChat authors); rows they redacted are excluded. Ours: `terser.data.pii.scrub`, regex layer (row 2.1.3 policy) |
| Action on detection | Per `DEFAULT_POLICY`: replace with a typed placeholder, or drop the row |
| Rows affected | 3 dropped and 16 scrubbed in the candidate pool; 1 scrubbed row in the final slice (`pii_scrubbed: true`) |
| Real user prompts included | **Yes.** Public, de-identified research release under ODC-BY. User metadata (hashed IP, country, state, headers) is dropped at parse time and never stored |

## 8. Known biases

| Bias | Evidence | Likely effect on results |
| :--- | :--- | :--- |
| **Category tags are heuristic** | Ordered keyword rules (`CATEGORY_RULES`). A spot check of 20 prompts per category put tag precision at about 80% for code, fact, creative and transform, about 60% for explain and about 50% for reason | Per-category numbers on this slice are noisy, especially for reason. Pooled numbers are unaffected |
| Only rule-matching prompts are kept | 13,105 of 18,680 candidates matched no rule and were dropped | Under-represents open-ended chit-chat and multi-intent prompts |
| 2023 ChatGPT users | WildChat was collected via a free chatbot in 2023 | Heavy on homework, coding help and role-play; light on enterprise use |
| English only | Language filter | No claims about non-English prompts |
| Long pasted payloads | p95 is 349 tokens | Savings on payload-heavy prompts are bounded, because payloads must be preserved |

## 9. Task-category balance

| Category | test | Candidate pool |
| :--- | ---: | ---: |
| code | 169 | 820 |
| fact | 159 | 159 |
| reason | 169 | 207 |
| explain | 169 | 2,222 |
| creative | 167 | 1,193 |
| transform | 167 | 974 |

## 10. Intended use

- **Intended:**
  - The headroom report on real prompts (row 2.4.2).
  - The final evaluation (row 12.3): S1–S3 reported separately on real prompts (ADR-0004).
  - The external-validity check against synthetic-prompt circularity.
- **Out of scope:**
  - Training, tuning, threshold selection or early stopping.
  - Per-category claims beyond what the tag precision in section 8 supports.
  - Anything that tries to identify WildChat users.

## 11. Maintenance and version

| Version | Date | DVC hash | Change |
| :--- | :--- | :--- | :--- |
| 1.0 | 2026-09-30 | `5b1034c3caa360ba193b82ad72bb973f.dir` | Initial release |

A new version needs a deliberate `build --force` and a new lock hash recorded here. The trigger
would be improving the category tagger, for example by using the Phase 2 judge model once it is
on the GPU.
