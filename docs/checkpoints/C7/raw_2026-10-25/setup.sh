#!/usr/bin/env bash
# C7.2/C7.4 judge environment: mirrors reports/p2_worker_environment.md.
set -euo pipefail
ROOT=$HOME/terser-c7
mkdir -p "$ROOT"/{logs,cache,tmp}
export UV_CACHE_DIR=$ROOT/cache/uv HF_HOME=$ROOT/cache/hf TMPDIR=$ROOT/tmp
if [ ! -x "$HOME/.local/bin/uv" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
UV=$HOME/.local/bin/uv
[ -x "$ROOT/env/bin/python" ] || $UV venv --python 3.11 "$ROOT/env"
export VIRTUAL_ENV=$ROOT/env UV_HTTP_TIMEOUT=600
$UV pip install "vllm==0.30.0" "huggingface-hub" \
  "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88"
"$ROOT/env/bin/hf" download Qwen/Qwen2.5-32B-Instruct-AWQ \
  --revision 5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c
"$ROOT/env/bin/python" - <<'PY' | tee "$ROOT/logs/versions.txt"
import platform, torch, vllm, transformers
print("python", platform.python_version()); print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__)
print("device", torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed 's/^/driver /' | tee -a "$ROOT/logs/versions.txt"
echo SETUP_DONE
