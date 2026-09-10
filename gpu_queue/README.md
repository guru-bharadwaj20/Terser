# GPU job queue

GPU work is never a blocker. Until the RTX 4500 Ada (24 GB, 22 GB usable budget) is connected, every
GPU job is written, CPU-smoke-tested with a tiny config, and queued here as one YAML spec per job.
See CONTRIBUTING.md, "Compute Allocation".

## Spec format

One file per job, `gpu_queue/<id>.yaml`. All fields are required; unknown fields are errors.

| Field | Type | Rule |
| :--- | :--- | :--- |
| `id` | str | Must equal the filename stem; unique |
| `phase` | int | Roadmap phase |
| `depends_on` | list[str] | Other job ids, or checkpoint gates matching `^C\d+(\.\d+)?$` (e.g. `C7.1`) |
| `command` | str | Shell command, run from the repo root |
| `est_vram_gb` | number > 0 | Must be ≤ `max_vram_gb` in `configs/hardware/rtx4500ada.yaml` (default 22) |
| `est_hours` | number > 0 | Planning estimate |
| `exclusive` | bool | Nothing else on the GPU while it runs |
| `priority` | int ≥ 1 | 1 = highest |
| `cpu_smoke_test` | path | Must exist (relative to repo root) |
| `outputs` | list[str] | Artifacts the job produces |

```yaml
# gpu_queue/sft-qwen0.5b-full.yaml
id: sft-qwen0.5b-full
phase: 8
depends_on: [C7.1, C2.3]
command: python -m terser.models.rewriter.train experiment=sft_qwen05b_full hardware=rtx4500ada
est_vram_gb: 14
est_hours: 6
exclusive: true
priority: 2
cpu_smoke_test: tests/rewriter/test_train_tiny.py
outputs: [models/rewriter/sft-qwen0.5b-full/, mlruns/]
```

`gpu_queue/gates.yaml` (optional, not a job spec) lists checkpoints that have passed:

```yaml
- C2.3
- C7.1
```

## Rules

- A job is queued only if its `cpu_smoke_test` passes in CI.
- `exclusive` jobs never share the card.
- vLLM serving and training never run at the same time on 24 GB.
- The queue drains by `priority`, then `depends_on`: a job never starts before its job dependencies;
  among jobs that are ready, the lowest `priority` number goes first (ties broken by id).

The runner executes jobs **strictly one at a time**, which is how the exclusivity and
serving-vs-training rules are enforced.

## CLI

```bash
python -m terser.gpu_queue validate [--dir gpu_queue]   # schema, duplicates, unknown deps, cycles, VRAM, smoke tests
python -m terser.gpu_queue dry-run  [--dir gpu_queue]   # execution order table + total GPU-hours
python -m terser.gpu_queue run      [--dir gpu_queue]   # needs a visible CUDA device
```

`validate` exits non-zero on any error. `run` refuses without CUDA ("GPU not connected — jobs remain
queued"). With CUDA it runs jobs in dry-run order and records `pending/running/done/failed/blocked`
in `.cache/gpu_queue_state.json`. Jobs whose gates aren't in `gates.yaml` are `blocked`, and so is
anything depending on a blocked or failed job. Jobs already `done` are skipped on re-run.
