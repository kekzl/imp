#!/usr/bin/env bash
# Phase-0 VRAM-audit run orchestrator.
#
# Boots imp-server inside the imp:test container with the MemAccount audit
# harness enabled, drives the fixed 8-concurrent workload, and collects:
#   - the init-time per-component VRAM table (server stdout, "init_complete")
#   - the device-used PEAK + final table on shutdown ("shutdown")
#   - the append-only dump file (mounted to host)
#
# Reproducible: same model, ctx, concurrency every run. Results land in
# $OUT_DIR (default /tmp/vram_audit). NOT committed — the AUDIT table in
# docs/vram_audit.md is the curated, append-only record.
set -euo pipefail

MODEL=${MODEL:-/models/Qwen3-Coder-30B-A3B-Instruct-FP4}
IMG=${IMG:-imp:test}
PORT=${PORT:-8080}
CTX=${CTX:-4096}
CONC=${CONC:-8}
ROUNDS=${ROUNDS:-3}
OUT_DIR=${OUT_DIR:-/tmp/vram_audit}
NAME=imp-vram-audit

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/dump.txt"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "== starting imp-server (model=$MODEL, max-batch=$CONC, ctx=$CTX) =="
docker run -d --name "$NAME" --gpus all \
  -p "$PORT:$PORT" \
  -v $HOME/models:/models \
  -v "$OUT_DIR:/audit" \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$IMG" imp-server \
    --model "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --max-batch "$CONC" \
    --max-tokens "$CTX" \
    --set runtime.max_seq_len="$CTX" \
    --set runtime.warmup=true \
    --set diagnostics.vram_audit=true \
    --set diagnostics.vram_audit_dump=/audit/dump.txt >/dev/null

echo "== waiting for server ready =="
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || \
     curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "  ready."
    break
  fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    echo "!! server container exited early; logs:"; docker logs "$NAME" | tail -40; exit 1
  fi
  sleep 2
done

echo "== driving load ($CONC concurrent x $ROUNDS rounds, ctx=$CTX) =="
python3 "$(dirname "$0")/vram_audit_load.py" \
  --url "http://127.0.0.1:$PORT" --concurrency "$CONC" --ctx "$CTX" --rounds "$ROUNDS" \
  | tee "$OUT_DIR/load.log"

echo "== peak sampled during load; stopping server to emit shutdown table =="
docker stop -t 30 "$NAME" >/dev/null
docker logs "$NAME" > "$OUT_DIR/server.log" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo
echo "================ INIT TABLE (init_complete) ================"
sed -n '/VRAM AUDIT \[init_complete\]/,/END VRAM AUDIT \[init_complete\]/p' "$OUT_DIR/server.log" || true
echo
echo "================ SHUTDOWN TABLE (peak) ===================="
sed -n '/VRAM AUDIT \[shutdown\]/,/END VRAM AUDIT \[shutdown\]/p' "$OUT_DIR/server.log" || true
echo
echo "Full logs: $OUT_DIR/server.log  |  dump: $OUT_DIR/dump.txt  |  load: $OUT_DIR/load.log"
