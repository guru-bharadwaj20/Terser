# Architecture decision records

One file per decision, `ADR-NNNN-<slug>.md`, from [_template.md](_template.md).

| ADR | Decision |
| :--- | :--- |
| [0001](ADR-0001-ml-lifecycle.md) | ML lifecycle |
| [0002](ADR-0002-problem-framing.md) | Problem framing |
| [0003](ADR-0003-training-on-model-outputs.md) | Training on model outputs (licences) |
| [0004](ADR-0004-plan-revision.md) | Plan revision |
| [0005](ADR-0005-judge-protocol.md) | Judge protocol (`pairwise_avg`) |
| [0006](ADR-0006-gpu-queue-ceiling-23gb.md) | GPU queue ceiling 23 GB |
| 0007–0009 | Not used (see below) |
| [0010](ADR-0010-mlops.md) | MLOps |
| [0011](ADR-0011-second-family-phi35.md) | Phi-3.5 as the second model family |
| [0012](ADR-0012-feasible-checkpoint-targets.md) | Feasible targets for the 24 open checkpoints |

**Numbers 0007–0009.** The plan reserved ADR-0010 for the Phase 11 MLOps overview (row 11.1.1)
when it was written, so that number was taken early; the numbers in between were never assigned
to a decision. They are left unused rather than renumbering ADRs that other documents already
link to. The next new ADR takes 0013.
