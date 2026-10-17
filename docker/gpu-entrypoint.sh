#!/bin/sh
# terser-gpu entrypoint (row 11.3.7): vLLM OpenAI-compatible server for the quality-path rewriter.
#
# Two layouts, chosen by TERSER_LORA_PATH:
#   * LoRA rung (sft-qwen1.5b-lora, sft-qwen3b-lora, ...): TERSER_BASE_MODEL is the HF base model
#     named in the adapter's adapter_config.json; the adapter is served under TERSER_SERVED_NAME,
#     which is the `model` terser-cpu requests (docker/serving.compose.yaml, quality.model).
#   * Full checkpoint (sft-qwen0.5b-full): TERSER_LORA_PATH empty, TERSER_REWRITER_MODEL is the
#     checkpoint directory (or an HF repo id), served under TERSER_SERVED_NAME.
# Extra vLLM flags can be appended as container arguments.
set -eu

: "${TERSER_SERVED_NAME:=terser-rewriter}"
: "${TERSER_MAX_MODEL_LEN:=2048}"
: "${TERSER_GPU_MEM_UTIL:=0.85}"
: "${TERSER_MAX_LORA_RANK:=16}"
: "${TERSER_LORA_PATH:=}"

common="--host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len ${TERSER_MAX_MODEL_LEN}"
common="${common} --gpu-memory-utilization ${TERSER_GPU_MEM_UTIL}"

if [ -n "${TERSER_LORA_PATH}" ]; then
    : "${TERSER_BASE_MODEL:?TERSER_BASE_MODEL must name the base model of the adapter}"
    if [ ! -f "${TERSER_LORA_PATH}/adapter_config.json" ]; then
        echo "terser-gpu: no adapter_config.json under ${TERSER_LORA_PATH} (dvc pull models/rewriter?)" >&2
        exit 2
    fi
    echo "terser-gpu: base ${TERSER_BASE_MODEL} + LoRA ${TERSER_LORA_PATH} as ${TERSER_SERVED_NAME}"
    # shellcheck disable=SC2086
    exec vllm serve "${TERSER_BASE_MODEL}" ${common} \
        --enable-lora --max-lora-rank "${TERSER_MAX_LORA_RANK}" \
        --lora-modules "${TERSER_SERVED_NAME}=${TERSER_LORA_PATH}" "$@"
fi

: "${TERSER_REWRITER_MODEL:?set TERSER_REWRITER_MODEL (full checkpoint) or TERSER_LORA_PATH}"
echo "terser-gpu: full checkpoint ${TERSER_REWRITER_MODEL} as ${TERSER_SERVED_NAME}"
# shellcheck disable=SC2086
exec vllm serve "${TERSER_REWRITER_MODEL}" ${common} --served-model-name "${TERSER_SERVED_NAME}" "$@"
