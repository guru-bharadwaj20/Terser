# Phase 2 worker environment — 2026-10-03

Row 2.2.3. Worker-1, environment `~/terser-env`, editable project `~/terser`.

The system conda directory had no executable; uv created a Python 3.11 environment. All package, model, compiler and temporary caches are scoped under the project directory.

Verified installed versions:

- python: `3.11.17`
- vllm: `0.30.0`
- torch: `2.13.0`
- cuda_runtime: `13.0`
- transformers: `5.18.0`
- mlflow: `3.16.1`
- huggingface-hub: `1.33.0`
- llmlingua: `0.2.2`
- dvc: `3.67.1`
- terser: `0.1.0`
- nvidia-cuda-nvcc / nvidia-cuda-crt / nvidia-nvvm: `13.0.88`
- uv: `0.12.22`

The Torch build reports `2.13.0+cu130`. Driver 595.84 reports CUDA capability 13.2. A CUDA tensor reduction returned 32.0 for 32 ones on the RTX 4090, and bf16 support returned true. `uv pip check` checked 328 packages and reported all compatible.

Evidence: [versions](../data/raw/p2_smoke_20261003/versions.json), [installation transcript](../data/raw/p2_smoke_20261003/commands/23_retry_install.log), [dependency check](../data/raw/p2_smoke_20261003/commands/26_dependency_check.log). The first installer was interrupted after repeated large-wheel restarts; its completed cache was retained. The retry used a 600-second timeout and temporarily paused only the project model downloader, which resumed afterward.

Environment verification does not establish serving throughput, judge memory use, or a Phase 2 checkpoint pass. Those require the hardware smoke evidence.

Serving required `CUDA_HOME=~/terser-env/lib/python3.11/site-packages/nvidia/cu13` and its `bin` on PATH. The system compiler was 11.5.119; resolved 13.4.92 compiler components mismatched CUDA 13.0 headers. Pinning the three compiler packages to 13.0.88 produced NVCC release 13.0, V13.0.88. [Compiler alignment](../data/raw/p2_smoke_20261003/commands/42_align_cuda_compiler.log), [final compiler inventory](../data/raw/p2_smoke_20261003/commands/53_final_inventory.log), and [full dependency freeze](../data/raw/p2_smoke_20261003/pip_freeze.txt) retain evidence. The environment uses uv rather than an installed pip executable.
