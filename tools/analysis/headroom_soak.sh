#!/usr/bin/env bash
# headroom_soak.sh <img> <tag> <minutes> [imp-server args]: 32-stream long-context soak that drives the
# growable KV pool to its cap, sampling device memory and KV blocks every second (#2482).
# MIXED=1 draws a prompt length per wave (4k..60k chars) so new prefill shapes keep allocating.
# Output: $OUT/<tag>/{samples.csv,waves.txt,server.log}; prints peak device used and the device
# growth after the pool reached its maximum.
set -uo pipefail
img="$1" tag="$2" mins="$3"; shift 3
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model="${MODEL:-/models/Qwen3.8-27B-NVFP4-vllm}"
port="${PORT:-8091}"
out="${OUT:-$HOME/.cache/imp/headroom_soak}/$tag"
mkdir -p "$out"
docker rm -f imp-headroom-soak > /dev/null 2>&1
docker run -d --name imp-headroom-soak --gpus all -v "$HOME/models:/models:ro" -p "$port:$port" "$img" \
    imp-server --model "$model" --port "$port" --host 0.0.0.0 --max-concurrent 32 "$@" > /dev/null
for _ in $(seq 1 180); do sleep 2; curl -sf "http://127.0.0.1:$port/health" > /dev/null && break; done
curl -sf "http://127.0.0.1:$port/health" > /dev/null || {
    echo "server never healthy"; docker logs imp-headroom-soak 2>&1 | tail -20; docker rm -f imp-headroom-soak > /dev/null; exit 1
}
(
    while docker ps -q --filter name=imp-headroom-soak | grep -q .; do
        m=$(curl -sf "http://127.0.0.1:$port/metrics" | grep -E '^imp_kv_blocks_(total|used) ' | awk '{print $2}' | tr '\n' ',')
        s=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,clocks.mem --format=csv,noheader,nounits)
        echo "$(date +%s),$s,$m"
        sleep 1
    done
) > "$out/samples.csv" &
sampler=$!
end=$(($(date +%s) + mins * 60))
w=0
while [ "$(date +%s)" -lt "$end" ]; do
    w=$((w + 1))
    tc=62000
    [ -n "${MIXED:-}" ] && tc=$(((RANDOM % 15 + 1) * 4000 + RANDOM % 997))
    echo "wave $w chars $tc" >> "$out/waves.txt"
    docker run --rm --network host -v "$repo/tools:/tools:ro" -e TARGET_CHARS=$tc python:3.13-slim \
        python /tools/analysis/longctx_conc_client.py "$port" 32 256 "w$w" >> "$out/waves.txt" 2>&1
done
docker logs imp-headroom-soak > "$out/server.log" 2>&1
docker rm -f imp-headroom-soak > /dev/null
kill $sampler 2>/dev/null
wait $sampler 2>/dev/null
echo "waves: $w"
grep -E '^WAVE' "$out/waves.txt" | awk '{print $5, $6}' | tr '\n' ';'
echo
awk -F, '$2>max{max=$2} {tot=$3} END{print "peak used MiB", max, "of", tot, "-> min free MiB", tot-max}' "$out/samples.csv"
awk -F, 'NR==FNR{if($6>k)k=$6; next} $6>=k && !t{t=1; u=$2} t{if($2>m)m=$2} END{print "max kv_blocks_total", k, "| device used at that point", u, "MiB, peak after", m, "MiB, growth", m-u, "MiB"}' \
    "$out/samples.csv" "$out/samples.csv"
