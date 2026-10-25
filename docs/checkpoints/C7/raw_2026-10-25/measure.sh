#!/usr/bin/env bash
# Usage: measure.sh <name> <concurrency> <vllm serve flags...>
# Samples device memory through startup and a judge-shaped load; prints peak.
set -uo pipefail
NAME=$1; CONC=$2; shift 2
ROOT=$HOME/terser-c7; L=$ROOT/logs/$NAME; mkdir -p "$L"
export HF_HOME=$ROOT/cache/hf TMPDIR=$ROOT/tmp CUDA_HOME=$ROOT/env/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$ROOT/env/bin:$PATH
PORT=8013
BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "$*" > "$L/flags.txt"; echo "$BASE" > "$L/baseline_mib.txt"
( while true; do echo "$(date +%s.%N),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"; sleep 0.2; done ) > "$L/vram.csv" &
SMP=$!
T0=$(date +%s)
MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}; REV=${REV:-5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c}
echo "$MODEL@$REV" > "$L/model.txt"
vllm serve "$MODEL" --revision "$REV" \
  --host 127.0.0.1 --port $PORT --seed 42 --max-model-len 4096 --max-logprobs 20 "$@" > "$L/server.log" 2>&1 &
SRV=$!
READY=0
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; then READY=1; break; fi
  kill -0 $SRV 2>/dev/null || break; sleep 5
done
echo "startup_s $(( $(date +%s) - T0 )) ready $READY" | tee "$L/result.txt"
if [ $READY = 1 ]; then
  MODEL=$MODEL python - "$PORT" "$CONC" <<'PY' 2>&1 | tee -a "$L/result.txt"
import sys, time, json, random, concurrent.futures as cf, urllib.request
port, conc = sys.argv[1], int(sys.argv[2])
random.seed(42)
words = "the quick brown fox jumps over a lazy dog while users compare two answers carefully".split()
def body(i):
    filler = " ".join(random.choice(words) for _ in range(2600))
    msg = (f"Request {i}. Compare answer A and answer B for the same prompt.\nA: {filler[:6000]}\n"
           f"B: {filler[6000:12000]}\nWhich answer is better? Reply with A or B and one sentence.")
    return json.dumps({"model": __import__("os").environ["MODEL"], "temperature": 0, "max_tokens": 256,
        "logprobs": True, "top_logprobs": 20, "messages": [{"role": "user", "content": msg}]}).encode()
def call(i):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body(i), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f:
        u = json.load(f)["usage"]; return u["prompt_tokens"], u["completion_tokens"]
n = conc * 3; t = time.time(); ok = fail = pt = ct = 0
with cf.ThreadPoolExecutor(conc) as ex:
    for fut in cf.as_completed([ex.submit(call, i) for i in range(n)]):
        try: p, c = fut.result(); ok += 1; pt += p; ct += c
        except Exception as e: fail += 1; print("request_error", repr(e)[:200])
dt = time.time() - t
print(f"load requests {n} ok {ok} fail {fail} mean_prompt_tokens {pt/max(ok,1):.0f} completion_tokens {ct} seconds {dt:.1f} gen_tok_s {ct/dt:.2f}")
PY
fi
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 3; kill $SMP
PEAK=$(cut -d, -f2 "$L/vram.csv" | sort -n | tail -1)
python3 -c "b,p=$BASE,$PEAK; print(f'peak_device_mib {p} baseline_mib {b} peak_net_gib {(p-b)/1024:.6f} samples $(wc -l < "$L/vram.csv")')" | tee -a "$L/result.txt"
grep -E "KV cache|Maximum concurrency|out of memory|Error" "$L/server.log" | tail -5 | tee -a "$L/result.txt"
echo MEASURE_DONE
