# Risk Register

Plan row: **0.3.5**. Expands the [Risk Register in CONTRIBUTING.md](../CONTRIBUTING.md#risk-register)
with measurable early-warning triggers, contingencies and owners. Rows R01–R10 correspond to the
CONTRIBUTING.md register; R11–R16 are additional risks.

## Scales

| Likelihood | Meaning |
| :--- | :--- |
| Low | Not expected; would surprise the team |
| Medium | Plausible; similar projects often hit it |
| High | Expected to occur in some form |

| Impact | Meaning |
| :--- | :--- |
| Low | Local rework within one row |
| Medium | Delays one phase or weakens one result |
| High | Delays several phases or invalidates a headline number |
| Critical | Blocks the critical path or invalidates the project's main claim |

Likelihood and impact values are initial judgements as of 2026-09-10 and are revised at each review.

## Review cadence

- The register is reviewed **at every checkpoint (C0–C11) and before Phase 12**, as part of the
  checkpoint report in `docs/checkpoints/C<N>/README.md`. The report lists each risk whose
  "watched by" entry falls in that checkpoint, the current trigger value, and any status change.
- A risk may also be updated at any time when its trigger fires. A fired trigger is recorded as an issue
  labelled `risk` that links the row id here.
- Status values: `Open` (monitored), `Triggered` (trigger fired, mitigation or contingency in progress),
  `Closed` (can no longer occur, with the reason). New risks get the next `R` number; ids are never reused.
- Changes to likelihood, impact or status are made in the checkpoint's commit so the history is in git.

## Register

| ID | Risk | Likelihood | Impact | Early-warning trigger (measurable) | Mitigation | Contingency | Watched by | Status |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- | :--- | :---: |
| R01 | Judge disagrees with humans | Medium | Critical | Judge–human κ on the first 100 annotated pairs < 0.60 overall or < 0.50 in any category; position-swap disagreement > 10% on pilot labels (1.3.6) | Task metrics before judge where available (2.3.1); position swap (2.3.3); per-category rubrics (2.3.2) iterated on pilot; GMM sanity check of score deltas (5.2.6) | Change judge model or rubric and re-label; restrict judge-only categories to those meeting κ; if C2.4 still fails, narrow scope to task-metric categories (`code`, `fact`, `reason`) via ADR | C2.3–C2.5, C5.5, S6 | Open |
| R02 | Too little compression headroom | Low–Medium | Critical | Median best-single-operator savings on pilot < 10% (C1.4); median oracle savings on the first 1,000 labelled prompts < 25% | Pilot tokenizer study before full data build (1.4); multiple candidate generators incl. aggressive LLM rewrites (2.2.1) | Go/no-go at C2.7: negative-result report, narrow scope to targets/categories with headroom, record in ADR | C1.4, C2.7 | Open |
| R03 | GPU arrives late | Medium | High | Card not connected by end of week 10; Phase 7 row 7.2 not ✅ by week 13 | CPU-first ordering; all GPU code written with CPU smoke tests and `gpu_queue/` jobs; `hardware=auto` | Week-13 fallback in [hardware.md § 7](hardware.md#7-fallback-gpu-still-not-connected-by-week-13): pull forward CPU halves of Phases 9–11, shift neural parts of 8–9; schedule slips, targets unchanged | C7, weekly status | Open |
| R04 | 24 GB VRAM insufficient | Low | Medium | Measured peak VRAM in 7.6 or 7.7 > 22 GB for a planned configuration; OOM in any queued job | Model ladder 0.5B full FT → 1.5B/3B LoRA → 7B QLoRA; grad checkpointing; seq ≤ 1024; no colocated serving and training | Reduce micro-batch with grad accumulation; drop to a smaller rung; hosted inference for open targets | C7.2, C8.3 | Open |
| R05 | Silent over-compression (dropped constraints) | High | High | Share of "dropped constraint" in the 8.3.10 failure analysis > 30% of degraded cases; any category preservation < 85% on val; production rolling preservation < 92% | Constraint and negation features (3.1.2–3.1.3); gate threshold at sensitivity ≥ 0.95; fallback; error taxonomy; shadow judging | Raise gate threshold per category; disable rewriting for the failing category (return original); retrain with hard examples | C8.3, C11.6, S2, S3 | Open |
| R06 | RL reward hacking | Medium | High | Gate precision on RL rewrites more than 5 pp below its precision on Phase 2 data; degenerate-output rate ≥ 1%; KL to SFT reference rising while judge-audited preservation falls | Judge audits every N steps (9.3.3); KL penalty; degenerate-output monitor (9.3.4); tabular RL first | Stop GRPO, keep SFT as champion (C9.5); retrain the gate on RL rewrites before another attempt | C9.3, C9.4 | Open |
| R07 | Rewriter overhead wipes savings | Medium | High | Net savings negative at the 25th-percentile prompt length on any target in val; quality-path p95 > 400 ms in benchmarks | Net-savings metric per request (11.3.6); CPU fast path; skip-short-prompts rule (3.5.4); caching | Route short prompts to fast path or no rewrite; ship fast path only for affected targets | S4, C11.2, C11.3 | Open |
| R08 | Tokenizer drift | Medium | Medium | Any change in fixture token counts after a dependency update or provider change (11.5.3); `tiktoken`/`tokenizers` version bump in lock file | Pinned tokenizer versions; exact-count fixture test (C0.2); tokenizer versions in model cards | Re-count affected data, re-run evaluation for affected targets, trigger CT; report numbers per tokenizer version | C0.2, 11.5.3, C11.6 | Open |
| R09 | API cost overrun | Medium | Medium | Phase spend > 70% of its cap in `configs/budget.yaml` before half the phase's rows are ✅; cost per labelled pair > 1.2× estimate (2.4.3) | Budget guard refuses calls past cap (0.3.6); disk cache; pilot cost estimate; local judge option | Reduce candidates per prompt or n-samples for `creative`; label a stratified subset; move judging to local judge if C8.7 passes | C2.9, C8.7, 12.10 | Open |
| R10 | Paraphrase leakage across splits | Medium | High | Automated leakage test reports > 0 shared source prompts; grouped-CV vs naive-KFold AUC gap near 0 in 3.2.5 (suggesting groups are ineffective); near-duplicate rate across splits > 1% | Source-grouped splits (2.1.6); MinHash with paraphrase groups kept whole (2.1.5); test frozen and hash-locked | Rebuild splits with coarser grouping (e.g. embedding clusters), re-run affected evaluations, document in checkpoint report | C2.2, C3.2 | Open |
| R11 | Provider terms of service restrict training on model outputs | Medium | High | ADR-0003 review finds a clause prohibiting use of outputs to train models that compete with the provider or any model; provider changes terms during the project | Check terms of each provider before generating SFT data; record in ADR-0003 before Phase 8; tag every generated row with provider and model | Train the rewriter only on candidates from compliant sources (operator stacks, open-weight models served locally, permitted providers); keep API outputs for evaluation only | ADR-0003, C2 review, before 8.3.1 | Open |
| R12 | Source dataset licence prevents redistribution | Medium | Medium | Licence audit (2.1.2) marks a source `unknown` or non-redistributable; a source contains text under a different licence than the dataset as a whole | Only permissively licensed sources; per-source data cards; unknown ⇒ non-redistributable | Release benchmark without that source (ids + reconstruction script instead of text), or replace the source; state it in the release data card | C2.1, 2.1.2, 12.11 | Open |
| R13 | Judge model version change or deprecation | Medium | High | Provider announces deprecation of the judge model; judge responses for cached-hash prompts differ on re-query; κ on a 100-pair re-validation sample drops > 0.05 | Pin exact judge model version string; disk-cache all judge responses; log judge version in MLflow; S6 re-validation on a fresh sample | Re-validate a replacement judge on `data/gold/human_300.jsonl` (must meet C2.4–C2.5) before relabelling; never mix judge versions within one reported table | C2.4, C2.9, C11 review, S6 | Open |
| R14 | Python 3.13 (laptop) vs 3.11 (CI / GPU machine) environment drift | Medium | Medium | A test passes locally and fails in CI or on the GPU machine; a GPU extra (vLLM, bitsandbytes) has no wheel for the local interpreter; differing resolved versions between environments | `requires-python >= 3.11`; CI matrix covering the lowest supported and the local version; pinned lock/constraints file; GPU environment verified by 7.3 import test | Standardise on the version supported by all GPU extras for experiments that are reported; run reported CPU experiments in CI-matching environment | C0.1, C0.4, C7.1 | Open |
| R15 | Annotator availability or low human–human agreement | Medium | Critical | Fewer than 2 annotators confirmed by end of week 4; κ on the first 50 pairs < 0.60 | Written annotation guideline with examples (2.3.5); blind presentation; calibration round before the 300 pairs | Revise guideline and re-annotate; recruit an adjudicator; reduce to categories with adequate agreement via ADR | C2.3 | Open |
| R16 | Stand-in open targets not representative | Medium | Medium | Preservation or savings on the 1.5B stand-in differ by > 5 pp from hosted 8B/7B results on an overlap sample; outputs tagged `stand_in: true` feed a reported number | Prefer hosted inference for the same models (2.2.4); tag `stand_in`; re-run with vLLM post-GPU (2.2.5) | Report stand-in results separately and exclude them from headline numbers until 2.2.5 completes | C2 review, 2.2.5, C8.3 | Open |
