#!/bin/bash
# Run API tests against every single-file GGUF model
# Checks VRAM before each model to avoid shared memory allocation
set -uo pipefail
cd "$(dirname "$0")/../.."

# The list is aspirational: 6 of these are not on this host right now and the
# loop below reports them as SKIP with a count, so a thin run is visible in
# the summary rather than silent. Two entries were not merely absent but
# dead names (Qwen3.5-4B-Q8_0, Qwen3.5-9B-Q8_0); those are resolved.
# This script hangs off no CI job and no make target, and it sets no exit
# code - read the SUMMARY line, do not trust $?.
MODELS=(
  Llama-3.2-3B-Instruct-Q8_0.gguf
  Qwen3-4B-Instruct-2507-Q8_0.gguf
  Qwen3.5-4B-mxfp4.gguf
  Qwen3-8B-Q8_0.gguf
  gemma-3-12b-it-Q8_0.gguf
  Devstral-Small-2507-Q4_K_M.gguf
  DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf
  Qwen3-32B-Q4_K_M.gguf
  Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf
  Nemotron-3-Nano-30B-A3B-Q6_K.gguf
)

# Max model file size in GB that fits in VRAM (weights + KV + workspace)
# RTX 5090 = 32GB, ~4GB overhead → ~28GB usable for weights
MAX_FILE_GB=28

PASS=0
FAIL=0
SKIP=0
RESULTS=""

check_vram() {
  # Returns free VRAM in MiB
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "============================================"
  echo "MODEL: $MODEL"
  echo "============================================"

  # Check file exists
  if [ ! -f "models/$MODEL" ]; then
    echo "SKIP: file not found"
    SKIP=$((SKIP + 1))
    RESULTS+="SKIP  $MODEL (not found)\n"
    continue
  fi

  # Check file size — skip models too large for VRAM
  FILE_SIZE_GB=$(du -BG "models/$MODEL" | cut -f1 | tr -d 'G')
  if [ "$FILE_SIZE_GB" -gt "$MAX_FILE_GB" ]; then
    echo "SKIP: ${FILE_SIZE_GB}GB > ${MAX_FILE_GB}GB VRAM limit"
    SKIP=$((SKIP + 1))
    RESULTS+="SKIP  $MODEL (${FILE_SIZE_GB}GB too large)\n"
    continue
  fi

  # Stop previous server and wait for VRAM to fully free
  docker compose down 2>/dev/null || true
  for w in $(seq 1 10); do
    FREE_VRAM=$(check_vram)
    if [ -n "$FREE_VRAM" ] && [ "$FREE_VRAM" -gt 30000 ]; then
      break
    fi
    sleep 2
  done

  FREE_VRAM=$(check_vram)
  echo "VRAM free: ${FREE_VRAM} MiB (model: ${FILE_SIZE_GB}GB)"
  # Need at least 2x model size free (weights + KV + workspace)
  NEED_MIB=$((FILE_SIZE_GB * 2 * 1024))
  if [ -n "$FREE_VRAM" ] && [ "$FREE_VRAM" -lt "$NEED_MIB" ]; then
    echo "SKIP: ${FREE_VRAM} MiB free < ${NEED_MIB} MiB needed"
    SKIP=$((SKIP + 1))
    RESULTS+="SKIP  $MODEL (VRAM: ${FREE_VRAM} < ${NEED_MIB} MiB)\n"
    continue
  fi

  # Start server
  IMP_MODEL="/models/$MODEL" docker compose up imp-server -d 2>/dev/null

  # Wait for health + model loaded (max 180s for large models)
  READY=0
  for i in $(seq 1 60); do
    if curl -sf http://localhost:8080/health 2>/dev/null | grep -q '"model_loaded":true'; then
      READY=1
      break
    fi
    # Check if container crashed
    if ! docker compose ps imp-server 2>/dev/null | grep -q "Up"; then
      LOGS=$(docker compose logs imp-server --tail 3 2>/dev/null)
      if echo "$LOGS" | grep -qi "out of memory\|cudaMalloc failed"; then
        echo "SKIP: OOM during model load"
        SKIP=$((SKIP + 1))
        RESULTS+="SKIP  $MODEL (OOM)\n"
        docker compose down 2>/dev/null || true
        READY=-1
        break
      fi
    fi
    sleep 3
  done

  if [ "$READY" -eq -1 ]; then
    continue
  fi

  if [ "$READY" -eq 0 ]; then
    echo "SKIP: server not ready in 180s"
    SKIP=$((SKIP + 1))
    RESULTS+="SKIP  $MODEL (timeout)\n"
    docker compose down 2>/dev/null || true
    continue
  fi

  # Check VRAM after load — detect shared memory usage
  FREE_AFTER=$(check_vram)
  echo "VRAM after load: ${FREE_AFTER} MiB free"

  # Run tests
  OUTPUT=$(docker run --rm --network host \
    -v "$(pwd)/tests/api:/tests" -w /tests \
    python:3.13-slim sh -c \
    "pip install -q httpx pytest 2>/dev/null && IMP_TEST_MODEL=$MODEL python -m pytest test_errors.py test_chat.py test_streaming.py -v --tb=line 2>&1" \
  )

  # Parse result
  if echo "$OUTPUT" | grep -q "passed"; then
    PASSED=$(echo "$OUTPUT" | grep -oP '\d+ passed' | head -1)
    FAILED=$(echo "$OUTPUT" | grep -oP '\d+ failed' | head -1)
    FAILNAMES=$(echo "$OUTPUT" | grep "FAILED " | sed 's/FAILED /  /')
    if echo "$OUTPUT" | grep -q "failed"; then
      echo "FAIL: $PASSED, $FAILED"
      [ -n "$FAILNAMES" ] && echo "$FAILNAMES"
      FAIL=$((FAIL + 1))
      RESULTS+="FAIL  $MODEL ($PASSED, $FAILED)\n"
      [ -n "$FAILNAMES" ] && RESULTS+="$FAILNAMES\n"
    else
      echo "PASS: $PASSED"
      PASS=$((PASS + 1))
      RESULTS+="PASS  $MODEL ($PASSED)\n"
    fi
  else
    echo "ERROR: unexpected output"
    echo "$OUTPUT" | tail -5
    FAIL=$((FAIL + 1))
    RESULTS+="ERROR $MODEL\n"
  fi
done

docker compose down 2>/dev/null || true

echo ""
echo "============================================"
echo "SUMMARY: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================"
echo -e "$RESULTS"
