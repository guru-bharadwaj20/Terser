# ADR-0006: Raise the GPU queue's single-job VRAM ceiling from 22 GB to 23 GB

| Field | Value |
| :--- | :--- |
| Status | Accepted (2026-10-25, by the Phase 7 owner without a second contributor's review; see the note below) |
| Date | 2026-10-25 |
| Plan row(s) | C7.4, 7.6, 7.8; risk R04 |
| Deciders | Vivian Sobers |
| Reviewers | None. A second-contributor review was not done before acceptance |

> Acceptance note: the conventions ask for review by at least one other contributor before an ADR is
> Accepted. The phase owner accepted this one directly on 2026-10-25. A later review that disagrees
> should supersede it with a new ADR.

## Context

C7.4 requires every queued job's `est_vram_gb` to be at most `max_vram_gb`, set to 22 in
`configs/hardware/rtx4500ada.yaml` and `configs/hardware/rtx4090.yaml`. The 22 was a planning
margin: about 2 GB below the 24 GB card, chosen before anything was measured.

On 2026-10-24/25 the queued 32B judge job (`gpu_queue/p2-candidates-qwen32b.yaml`,
Qwen2.5-32B-Instruct-AWQ on vLLM 0.30.0) was measured on both RTX 4090 workers
([serving.md](../checkpoints/C7/serving.md#rtx-4090-re-measurement-2026-10-2425-rows-76-c72-c74)):

- Peak 22.39 GiB net (23,024 MiB device-wide), the same on both hosts and drivers (595.84,
  580.178.04).
- The peak is a transient of about 8 s between vLLM's "Loading weights took" and "Model loading
  took 18.14 GiB" log lines. Changing the kernel (`awq_marlin` vs `awq`), eager mode vs CUDA graphs,
  halving the batch limits and lowering `--gpu-memory-utilization` to 0.85 all left it within 5 MiB.
- Steady serving under a judge-shaped load is 19.9–20.3 GiB net, and every load run completed
  (48/48 requests) with no OOM.
- The card reports 23.99 GiB to CUDA, so the 22.39 GiB transient leaves about 1.6 GiB free.

The 22 GB figure was wrong for this workload. It assumed a job's peak tracks its serving
footprint. For the 32B AWQ judge the peak is a fixed load-time transient that no serving setting
controls, so the margin left no room for the project's main judge, which runs reliably on the card.

## Decision

Set `max_vram_gb: 23` in both 24 GB GPU profiles (`rtx4500ada.yaml`, `rtx4090.yaml`) and set the
queue's fallback `DEFAULT_MAX_VRAM_GB` to 23. C7.4's target becomes "every declared `est_vram_gb`
≤ 23 GB". Jobs still declare their measured peak (the 32B judge declares 22.4).

23 GB is the largest round value that keeps about 1 GiB of the card's 23.99 GiB for the CUDA
context, driver and allocator fragmentation above the queue's budget. The rest of the queue rules
do not change: exclusive jobs only, no serving and training at once.

## Alternatives considered

| Alternative | Pros | Cons | Why not chosen |
| :--- | :--- | :--- | :--- |
| Keep 22 and move the job to Qwen2.5-14B-AWQ (20.22 GiB at util 0.90) | No config change | Changes the judge used for every Phase 2 label already produced with the 32B; ADR-0005's protocol and all downstream results would need re-running | Swaps a model to satisfy a planning number, not a hardware limit |
| Keep 22 and record C7.4 as a permanent fail | No decision needed | The queue rejects the project's own judge job, so the queue cannot schedule Phase 2 | Leaves the queue unusable for its first job |
| Raise to 24 | Maximum room | No margin for the CUDA context or fragmentation; an overshoot would OOM | Too tight |
| Per-job override field | Keeps 22 for everything else | New schema field and validation logic for one job | More mechanism than one measured exception needs |

## Consequences

- Positive: the 32B judge job validates on measured numbers, and C7.4 is decided by measurements
  on both workers.
- Negative: about 1 GiB of headroom instead of about 2 GiB. A job estimated between 22 and 23 GB
  that underestimates its peak is likelier to OOM. Jobs must declare measured peaks that include
  load-time transients.
- Follow-up: risk R04 records the trigger and this mitigation. If a vLLM upgrade changes the
  load transient, re-measure with `docs/checkpoints/C7/raw_2026-10-25/measure.sh`.
- Verification: `python -m terser.gpu_queue validate --dir gpu_queue` passes with the 32B job at
  22.4, and `tests/test_hardware.py` asserts the new profile value.
