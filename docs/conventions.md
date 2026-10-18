# Conventions

Plan row: **0.3.3**. These conventions apply to every contributor and every phase. Where a rule is
also stated in [CONTRIBUTING.md](../CONTRIBUTING.md), CONTRIBUTING.md is authoritative.

General rules for identifiers: lowercase, ASCII, `-` as the word separator inside a field, no spaces.
Model sizes keep their decimal point (`qwen0.5b`), except in Hydra config names, which use `_` and no dots
(`sft_qwen05b_full`) so they are valid Python-style identifiers.

---

## 1. MLflow run names

Format: `<phase>-<task>-<model>-<variant>-<seed>`

| Field | Rule | Examples |
| :--- | :--- | :--- |
| `phase` | `p` + phase number | `p3`, `p8` |
| `task` | Sub-problem or study, short | `gate`, `tagger`, `savings`, `cluster`, `rewriter`, `rl`, `hpo`, `optim`, `act` |
| `model` | Model family and size | `id3`, `lr`, `knn`, `svm-poly`, `mlp`, `lgbm`, `hmm`, `textcnn`, `qwen0.5b`, `qwen1.5b` |
| `variant` | What distinguishes this run from siblings | `depth8`, `l2-c1`, `k5-cosine`, `full`, `lora-r16`, `grpo-g8`, `tpe` |
| `seed` | `s` + integer seed | `s0`, `s1`, `s42` |

Examples:

```
p3-gate-id3-rep-pruned-s0
p3-savings-knn-k15-distance-s1
p4-optim-mlp-adam-lr3e-4-s4
p5-gate-lgbm-lr0.05-s2
p8-rewriter-qwen0.5b-full-s0
p8-rewriter-qwen1.5b-lora-r16-s1
p9-rl-qwen1.5b-grpo-g8-s0
p10-hpo-lgbm-tpe-s3
```

HPO studies use a parent run named without the seed (e.g. `p10-hpo-lgbm-tpe`) and child runs suffixed
`-t<trial>` before the seed: `p10-hpo-lgbm-tpe-t017-s3`.

## 2. MLflow tags

Every run sets all of these tags. A run missing any of them fails the tracking audit (row 11.2.1).

| Tag | Value | Example |
| :--- | :--- | :--- |
| `phase` | Phase number | `3` |
| `row` | Plan row id(s), comma-separated | `3.3.4` |
| `task` | Same value as the run-name `task` field | `gate` |
| `hardware` | `cpu`, `rtx4500ada` or `rtx4090` (from the `hardware` config) | `cpu` |
| `host` | Machine that produced the run. Required on `rtx4090` runs, since `worker-1` and `worker-2` carry different driver and torch versions | `worker-2` |
| `data_hash` | DVC hash of the input dataset version | `3f2a9c…` |
| `git_sha` | Full commit SHA of the code; runs from a dirty tree also set `git_dirty=true` | `40c00d9…` |

Also log generated tokens and GPU-hours per run (Experiment standards in CONTRIBUTING.md). All LLM calls
go to local open-source models, so there is no $ cost to log.

## 3. Branches

Format: `phase<N>/<row-id>-<topic>`

```
phase0/0.3.3-conventions
phase3/3.3.1-id3
phase8/8.3.4-sft-qwen1.5b-lora
```

For checkpoint work: `phase<N>/C<N>-checkpoint`, e.g. `phase2/C2-checkpoint`.

## 4. Commits

[Conventional Commits](https://www.conventionalcommits.org/) with the plan row id as the first token
of the description. **One commit per coherent change**, finer than a plan subtask where the subtask
has separable parts. Rebase-merge PRs into `main` so every commit is preserved on the branch history.


Format: `<type>(<scope>): <row-id> <imperative summary>`

| Type | Use |
| :--- | :--- |
| `feat` | New functionality in `src/` |
| `fix` | Bug fix |
| `exp` | Experiment run and its report |
| `docs` | Documentation, derivations, cards, ADRs |
| `test` | Tests only |
| `refactor` | No behaviour change |
| `chore` | Tooling, configs, `gpu_queue/` jobs |
| `ci` | CI workflows |

Examples:

```
feat(judge): 2.3.3 position-swapped pairwise judging
exp(phase4): 4.3.6 optimizer comparison at equal gradient budget
docs(derivations): 4.4.2 SVM Lagrangian dual
chore(gpu_queue): 8.3.4 add sft-qwen1.5b-lora job
docs(conventions): 0.3.3 naming and tracking conventions
```

## 5. Status-flip rule

A row changes from ❌ to ✅ **only** when its Definition of Done is met and the artifact exists in the
repository (code + test + logged experiment or report). The flip happens **in the same commit** that
satisfies the Definition of Done, together with the Overall Progress table and, if applicable, the
Syllabus Coverage Matrix. `GPU` rows stay ❌ until their `gpu_queue` job has run on the RTX 4500 Ada.

## 6. Reports and figures

| Artifact | Path pattern | Example |
| :--- | :--- | :--- |
| Phase report | `reports/p<N>_<topic>.md` | `reports/p3_cv_leakage.md` |
| Tabular result | `reports/p<N>_<topic>.csv` | `reports/p1_tokenizer_study.csv` |
| Figure | `reports/figures/p<N>_<topic>.png` | `reports/figures/p3_tree_overfit.png` |
| Final evaluation | `reports/final_<topic>.md` | `reports/final_results.md` |
| Cross-phase report | `reports/<topic>.md` | `reports/license_audit.md` |

`<topic>` is `snake_case`. Every number in a report links to its MLflow run.

## 7. Checkpoint reports

Location: `docs/checkpoints/C<N>/README.md` (e.g. `docs/checkpoints/C2/README.md`). Supporting files sit
beside it (e.g. `docs/checkpoints/C7/hardware.md`). Content follows the checkpoint protocol in
CONTRIBUTING.md: measured value per check, pass/fail, MLflow links, DVC hash, git SHA, seeds, figures,
deviations, reviewer, decision (`PASS`, `PASS WITH CONDITIONS`, `FAIL`).

## 8. Architecture Decision Records

| Rule | Value |
| :--- | :--- |
| Location | `docs/decisions/` |
| File name | `ADR-<NNNN>-<kebab-topic>.md`, four-digit zero-padded, e.g. `ADR-0001-ml-lifecycle.md` |
| Numbering | Sequential, never reused. Numbers named in the plan are reserved: `ADR-0002` problem framing, `ADR-0003` training on model outputs, `ADR-0010` MLOps |
| Template | [`docs/decisions/_template.md`](decisions/_template.md) |
| Status values | `Proposed`, `Accepted`, `Rejected`, `Deprecated`, `Superseded by ADR-<NNNN>` |
| Changes | Accepted ADRs are not rewritten; a new ADR supersedes them. Changing a checkpoint target always requires an ADR |

## 9. Configs

Hydra config groups live in `configs/<group>/<name>.yaml`. Names are `snake_case`.

| Group | Examples |
| :--- | :--- |
| `hardware` | `cpu.yaml`, `rtx4500ada.yaml`, `rtx4090.yaml`, `auto.yaml` |
| `model` | `p3_knn.yaml`, `p3_lr.yaml`, `p3_tree.yaml` |
| `experiment` | `sft_qwen05b_full.yaml`, `grpo_tiny.yaml`, `p1_pilot_labels.yaml` |
| `serving` | `default.yaml`, `fast_onnx.yaml` |
| (root) | `budget.yaml` |

Overrides are passed on the command line; no hard-coded paths or hyperparameters in code:

```powershell
python -m terser.models.rewriter.train experiment=sft_qwen05b_full hardware=rtx4500ada
python -m terser.models.rl.grpo experiment=grpo_tiny hardware=cpu
```

## 10. GPU queue job ids

File: `gpu_queue/<id>.yaml`; the `id` field equals the file name without extension.

Format: `<task>-<model>-<variant>` (no seed; seeds are listed inside the job's command or experiment config).

```
sft-qwen0.5b-full
sft-qwen1.5b-lora
grpo-qwen1.5b-g8
respond-open-targets
hpo-textcnn-tpe
```

Job spec fields and queue rules: [hardware.md § 4](hardware.md#4-gpu-job-queue).

## 11. Data paths

| Path | Contents | Tracked by |
| :--- | :--- | :--- |
| `data/raw/` | Unmodified downloads and raw LLM responses from local OpenAI-compatible servers | DVC |
| `data/interim/` | PII-scrubbed and intermediate stage outputs; nothing enters here unscrubbed | DVC |
| `data/processed/` | Final tables and splits (`train`, `val`, `test`); `test*` is readable only through `terser.phase12.unlock.read_locked` | DVC |
| `data/SOURCES.md` | Upstream source list | git |

Data cards: `docs/data_cards/<dataset-name>.md`. Model cards: `docs/model_cards/<model-name>.md`.

## 12. Pytest markers

| Marker | Meaning | Run in default CI |
| :--- | :--- | :---: |
| `gpu` | Requires a CUDA device | no |
| `api` | Calls a paid hosted LLM API. No such keys exist in this project, so these tests are always skipped; the marker stays for optional paths such as Claude token counting | no |
| `network` | Needs network access (e.g. model or dataset download) | no |
| `slow` | Takes more than ~30 s on the dev laptop | no |

Default local and CI command:

```powershell
pytest -m "not gpu and not api and not network and not slow"
```

CONTRIBUTING.md's minimum `pytest -m "not gpu and not api"` must also stay green. Tests of `terser.llm`
against the `local` provider use a fake in-process OpenAI-compatible server, so the default run needs no
Ollama, llama.cpp or vLLM process.
