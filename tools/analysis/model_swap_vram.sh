#!/usr/bin/env bash
# VRAM left after in-process model swaps: one imp-server, requests alternate MODEL_A / MODEL_B CYCLES times,
# prints every "VRAM budget ... available=" line per load. Usage: model_swap_vram.sh IMAGE TAG [CYCLES].
set -euo pipefail
IMG=$1; TAG=$2; CYCLES=${3:-4}
A=${MODEL_A:-flint-nvfp4-official}
B=${MODEL_B:-Qwen3.8-27B-NVFP4-vllm}
PORT=${PORT:-8096}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/swaptest}
mkdir -p "$OUT_DIR"
name="swaptest_$TAG"
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all -p "127.0.0.1:$PORT:8080" -v "$HOME/models:/models:ro" "$IMG" \
    imp-server --model "/models/$A" --models-dir /models --host 0.0.0.0 --port 8080 >/dev/null
wait_up() {
    for _ in $(seq 1 150); do
        curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}
ask() {
    curl -s -m 600 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8}" \
        | grep -o '"content":"[^"]*"' | head -1
}
wait_up
for c in $(seq 1 "$CYCLES"); do
    echo "cycle $c B: $(ask "$B")"
    echo "cycle $c A: $(ask "$A")"
done
docker logs "$name" > "$OUT_DIR/$TAG.log" 2>&1
docker rm -f "$name" >/dev/null
grep -E 'SafeTensors model loaded successfully|vram_budget.cpp:[0-9]+: VRAM budget: strategy' "$OUT_DIR/$TAG.log" \
    | sed -E 's/.*loaded successfully from \/models\/(.*)/load \1/; s/.*available=([0-9.]+) MiB, kv=([0-9]+) blocks.*/  available=\1 MiB kv=\2/'
grep -cE 'imp_gpu_release: CUDA primary context reset' "$OUT_DIR/$TAG.log" | sed 's/^/context resets: /'
