# Proposed Phase 2 full-scale run

Rows 2.2.1–2.2.3 and 2.3.3. Proposal only: no full-scale run is authorized by this handoff or has been started.

Use the configured label subset in `params.yaml`: 300 prompts per category across six categories, hence 1,800 prompts. These are planned counts derived from configuration, not observed dataset counts.

Before launching:

- Recover the canonical, PII-reviewed Phase 2 dataset and its DVC version. Preserve the frozen test split and its configured checksum; pilot smoke prompts are not a substitute for the canonical dataset.
- Retain source IDs and reference answers/tests for code, fact and reasoning metrics. The seed pilot smoke cannot validate those reference-based metrics.
- Complete the current PII boundary work and commit the source used for production. Record the actual commit, input DVC hashes, model revisions, seed, dependency versions and hardware/host tags.
- Add production DVC stages for both target response files and position-swapped judging: the inspected `dvc.yaml` currently stops at token counts. Include Phi's real tokenizer in production counting; the inspected default tokenizer registry does not include it. The smoke wrapper explicitly counts both served target tokenizers.
- Resolve the original 32B queue startup failure and measured 22.388672 GiB net peak against the 22 GiB ceiling. Evaluate a smaller activation footprint (for example eager execution or a reduced batch limit), then measure again; no such tuning was run after this smoke. Review the judge rubric and position inconsistency (147/274 parsed comparisons), low-margin/review flags and response cap hits. Repeat a quality-focused smoke with references before accepting labels.
- Reserve exclusive slots, check current compute processes and disk space, then verify the judge configuration on worker-2. Worker-1's measurement does not verify worker-2's different driver/runtime combination.

Planned work before rejection, identical-text caching and identical-output shortcuts:

- Six candidates per prompt: 10,800 candidate records, comprising 5,400 32B rewrites, 3,600 operator-stack outputs and 1,800 CPU compression outputs.
- Original plus six candidates: 12,600 texts; counting with two served target tokenizers yields 25,200 cells, including rejected candidates for completeness.
- Each target produces at most 16,800 response records: 12,600 base records plus 4,200 extra creative samples (300 creative prompts × seven texts × two additional samples). Both targets total 33,600 records.
- Candidate labels total at most 21,600 records (1,800 × six × two targets).
- Before identical-output shortcuts, judging covers at most 28,800 sample pairs and 57,600 position-swapped calls: two targets × six candidates × (1,500 noncreative prompts + 300 creative prompts × three samples), then two A/B orders. Retries may add calls.

Host assignments:

- Worker-1: Qwen2.5-7B-Instruct bf16, then Phi-3.5-mini-instruct bf16. Keep raw responses and caches resumable and write intermediate outputs only after PII scanning.
- Worker-2: Qwen2.5-32B-Instruct-AWQ for candidate generation, then position-swapped judging. Keep the judge resident while target outputs arrive, if the reserved slot allows it. Run operator stacks and the compression baseline on CPU.
- Local workspace: validate coverage, stage DVC metadata, retain command transcripts and tracking exports, and inspect review flags. Do not promote smoke labels into production tables.

Use seed 42 for dataset selection and serving. Preserve the response stage's creative common-random-number seeds 1000, 1001 and 1002 for originals and rewrites. Use run names such as `p2-respond-qwen7b-full-s42` and `p2-judge-qwen32b-full-s42`, with the actual producing host, hardware, row, input hash and committed code SHA. Log generated tokens and GPU-hours.

Timing must be calculated from the completed hardware smoke. Download, installation and model startup durations are separate from inference throughput. Do not use bootstrap transfer rates as serving rates. The 20-prompt seed sample is a small pilot with a different category mix from a balanced production subset; a linear projection is a planning estimate, not a runtime guarantee.

## Timing projection from the completed smoke

These are **projections**, computed as observed stage wall seconds × (1,800 / 20) / 3,600. [Measured stage stats and tracking IDs](p2_gpu_smoke.md#measured-stage-results) provide the inputs. The comparison does not account for a balanced production category mix, changed candidate rejection/cache rates, longer prompts, truncation remediation, judge retries, or the necessary memory configuration change.

| Work | Observed seconds / 20 prompts | Projected hours / 1,800 prompts | Proposed host |
| :--- | ---: | ---: | :--- |
| Candidates | 19.200 | 0.480 | worker-2 |
| Qwen target | 60.839 | 1.521 | worker-1 |
| Phi target | 45.319 | 1.133 | worker-1 |
| Judge | 264.625 | 6.616 | worker-2 |

Projected aggregate stage GPU allocation is 9.750 hours: worker-1 targets 2.654 hours and worker-2 candidate/judge work 7.096 hours. CPU token counting projects to 0.142 hours separately. A conservative whole-stage schedule is candidates → both targets → judge, approximately 9.750 hours plus counting, startup and validation. With batched handoff, an ideal overlap bound is candidates + max(targets, judge) = 7.096 hours; this is not measured on two hosts and is not a guarantee.

Worker-2 deployment, downloads and serving startup are excluded and unmeasured. Worker-1's cold candidate startup was 75.122 seconds; the final cached judge startup was 30.045 seconds. Do not transfer those startup figures to worker-2 without verification. Its inventoried driver is 580.178.04 (advertised CUDA capability 13.0), whereas worker-1 uses 595.84; choose and validate its environment separately. Worker-3 has no assignment. No full-scale process, scheduler or queue job was launched.
