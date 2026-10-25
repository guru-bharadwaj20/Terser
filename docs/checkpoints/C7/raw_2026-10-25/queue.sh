#!/usr/bin/env bash
# C7.2 measurements: each waits for an idle card; foreign GPU processes during a run mark it CONTAMINATED.
set -uo pipefail
R=$HOME/terser-c7; export HF_HOME=$R/cache/hf
foreign() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do u=$(ps -o user= -p $p 2>/dev/null); [ -n "$u" ] && [ "$u" != "$USER" ] && echo "$p:$u"; done; }
wait_idle() { while [ -n "$(foreign)" ] || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done; }
guard() { local f=$1; : > "$f"; while true; do foreign >> "$f"; sleep 5; done; }
verdict() { [ -s "$1" ] && echo "CONTAMINATED($(sort -u "$1" | tr '\n' ' '))" || echo CLEAN; }
serve() { local name=$1 model=$2 conc=$3; shift 3
  local rev; rev=$($R/env/bin/python -c "from huggingface_hub import HfApi;print(HfApi().model_info('$model').sha)")
  $R/env/bin/hf download "$model" --revision "$rev" > $R/logs/dl_$name.log 2>&1 || { echo "$name DOWNLOAD_FAILED"; return; }
  wait_idle; guard $R/logs/$name.foreign & local G=$!
  MODEL=$model REV=$rev $R/measure.sh "$name" "$conc" "$@" > $R/logs/$name.out 2>&1
  kill $G; echo "$name $(verdict $R/logs/$name.foreign) $(grep -E 'startup_s|load |peak_' $R/logs/$name/result.txt | tr '\n' ' ')"
  rm -rf "$HF_HOME/hub/models--${model//\//--}"; }
train() { local name=$1; shift
  wait_idle; guard $R/logs/$name.foreign & local G=$!
  (cd $R/repo && $R/env/bin/python -m terser.hardware.train_bench "$@" --output $R/logs/$name.json --no-mlflow) > $R/logs/$name.out 2>&1; local rc=$?
  kill $G; echo "$name rc=$rc $(verdict $R/logs/$name.foreign) $($R/env/bin/python -c "import json;d=json.load(open('$R/logs/$name.json'));print('peak_alloc',d['peak_allocated_gib'],'peak_reserved',d['peak_reserved_gib'],'tok_s',d['tokens_per_s'])" 2>&1 | tail -1)"
  rm -rf "$HF_HOME/hub"; }
for job in "$@"; do
  case $job in
    judge14b) serve judge14b_awq Qwen/Qwen2.5-14B-Instruct-AWQ 16 --quantization awq_marlin --max-num-seqs 16 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90 ;;
    judge8b) serve judge8b_llama_awq hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 16 --quantization awq_marlin --max-num-seqs 16 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90 ;;
    lora3b) train lora3b --model Qwen/Qwen2.5-3B-Instruct --mode lora --weights-dtype bf16 --lora-r 16 ;;
    qlora7b) train qlora7b --model Qwen/Qwen2.5-7B-Instruct --mode qlora --lora-r 16 ;;
  esac
done
echo QUEUE_DONE
