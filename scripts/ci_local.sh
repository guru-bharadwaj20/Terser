#!/usr/bin/env bash
# Run the CI checks of .github/workflows/ci.yml locally, in the same order.
#
# Use it when GitHub Actions cannot run (for example, the account's Actions minutes are blocked)
# or before pushing. It uses the active environment (or .venv if present) and does not install
# anything; set up with `pip install -e ".[dev,hpo,serve,mlops]"` first.
#
#   bash scripts/ci_local.sh            # every step, stop at the first failure
#   bash scripts/ci_local.sh --keep-going
set -uo pipefail

cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUTF8=1

keep_going=0
[[ "${1:-}" == "--keep-going" ]] && keep_going=1

failed=()
step() {
  local name=$1
  shift
  echo "::: $name"
  local t0=$SECONDS
  if "$@"; then
    echo "::: $name ok ($((SECONDS - t0)) s)"
  else
    echo "::: $name FAILED"
    failed+=("$name")
    ((keep_going)) || exit 1
  fi
}

step "Ruff" ruff check .
step "Black" black --check .
step "Mypy" mypy src
step "DVC pipeline dry run" dvc repro --dry --allow-missing
step "Test-set isolation audit (row 10.8)" python -m terser.hpo.audit --scan src/terser
step "Feature parity (row 11.1.3)" pytest tests/features -q
step "Tokenizer drift (row 11.5.3)" python -m terser.monitoring.tokenizer_drift
step "Tests (CPU)" pytest -m "not gpu and not api" --cov=terser --cov-report=term

if ((${#failed[@]})); then
  echo "failed: ${failed[*]}"
  exit 1
fi
echo "all CI steps passed"
