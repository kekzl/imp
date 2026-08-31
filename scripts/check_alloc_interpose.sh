#!/usr/bin/env bash
# Invariant I2 (docs/internals/MEMORY.md): nothing allocates device memory while
# the engine is serving. `steady_state_allocations()` only ever sees what routes
# through Backend, so the counter reads zero in every shipping build no matter
# what the 469 direct allocation sites outside src/memory/ do. The --wrap
# interposer is what closes that gap, and until this target existed it compiled
# in no make target and no CI job (DEBT_LEDGER_2026_08_21, item 3).
#
# Driven at a config that reaches the known serving-phase allocators:
#   batch > 1 + NVFP4 residual KV -> engine_scheduler.cpp cudaMallocAsync (B9)
#   MoE + MTP chain               -> the speculative path's per-step buffers
#
# Run via `make check-alloc-interpose`, which builds the binary this needs.
set -uo pipefail

BIN=${BIN:-build-interpose/imp-server}
IMG=${DEV_IMG:-imp:toolchain}
MODEL=${INTERPOSE_MODEL:-/models/Qwen3.8-27B-NVFP4-vllm}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
PORT=8099
LOG=$(mktemp /tmp/interpose.XXXXXX.log)

[ -x "$BIN" ] || { echo "FATAL: $BIN not built. Run: make check-alloc-interpose" >&2; exit 1; }
[ -d "$MODELS_DIR/$(basename "$MODEL")" ] || {
    echo "FATAL: model not readable: $MODELS_DIR/$(basename "$MODEL")" >&2; exit 1; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -le 3000 ] || { echo "FATAL: $used MiB already held on the GPU" >&2; exit 1; }

docker rm -f interpose >/dev/null 2>&1
docker run -d --name interpose --gpus all -p $PORT:8080 \
    -v "$PWD":/src -w /src -v "$MODELS_DIR":/models "$IMG" \
    "$BIN" --host 0.0.0.0 --port 8080 --model "$MODEL" --think-budget 0 \
    --set runtime.max_batch_size=4 \
    --set kv_cache.dtype=nvfp4 \
    --set kv_cache.bitdecoding_residual_tokens=128 \
    --set speculative.mtp_k=1 >/dev/null
trap 'docker rm -f interpose >/dev/null 2>&1' EXIT

for _ in $(seq 1 200); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    docker ps --format '{{.Names}}' | grep -q '^interpose$' || break
    sleep 3
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
    echo "FATAL: server never became healthy" >&2; docker logs interpose 2>&1 | tail -20 >&2; exit 1; }

# 18 requests, four at a time: batch > 1 is what reaches the residual allocator.
for round in 1 2 3 4 5; do
    for slot in 1 2 3 4; do
        curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$(basename "$MODEL")\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain topic $round.$slot in two paragraphs.\"}],\"max_tokens\":96,\"temperature\":0}" \
            >/dev/null &
    done
    wait
done

# Clean shutdown: the report is a static destructor, so SIGKILL loses it.
docker stop -t 60 interpose >/dev/null 2>&1
docker logs interpose > "$LOG" 2>&1

# Liveness, before any verdict. This gate claims to exercise the residual KV
# path at batch > 1; if the pool was declined (wrong kv dtype, a model with
# non-uniform head_dim) the run measures a config that cannot reach the
# allocator it is looking for, and a clean result means nothing.
if ! grep -q 'residual buffer enabled' "$LOG"; then
    echo "FATAL: the residual KV pool was never enabled, so this run did not"   >&2
    echo "       exercise the path it claims to. Check kv_cache.dtype=nvfp4 and" >&2
    echo "       kv_cache.bitdecoding_residual_tokens against this model."       >&2
    grep -i 'residual' "$LOG" | head -5                                         >&2
    exit 1
fi

CLEAN=$(grep -c 'alloc-interpose\] steady state clean' "$LOG")
VIOL=$(grep -c 'alloc-interpose\] I2 VIOLATIONS' "$LOG")

# Known residue, pinned by CALL COUNT and named by site. This number may only
# ever go DOWN: the gate fails on a rise (a new serving allocation) and on a
# fall (someone fixed one and left the pin stale), the same two-way shape as
# tools/alloc_allowlist.txt. A pin that can be raised is an exemption with
# extra steps.
#
# The five sites behind the 19, from addr2line on a debug-level run. Each is a
# work item, not a blessing; docs/audit/DEBT_LEDGER_2026_08_21.md tracks them.
#
#  15 calls, ~0 MiB   Engine::try_launch_async_graph_loop
#                     src/runtime/engine_graph_decode.cpp:408-412, :434
#                     d_bt / d_token / d_pos / d_ctx / d_banned, allocated and
#                     torn down PER REQUEST. The real violation of the five:
#                     making cpipe_ persistent needs it sized from the KV plan
#                     at init, and the teardown path reworked with it.
#   2 calls, 128 MiB  GraphExecutor::run_attention -> chunk_eager_k_ / _v_
#                     src/exec/executor_attention_prefill.cu:176-177
#                     Grow-only gather scratch for the eager chunked prefill
#                     path, sized from the live context on first use. Fixing it
#                     means pre-sizing from ctx_capacity the way its sibling
#                     chunk_capture_k_/_v_ already is.
#   1 call, ~0 MiB    Engine::banned_tokens_device_
#                     src/runtime/engine_graph_decode.cpp:29
#                     Lazy first-use upload of a list known at engine init.
#                     The cheapest of the five to hoist.
#   1 call, 0.001 MiB VRAMAllocator::allocate
#                     The engine arena growing after the phase flip. Whether
#                     that is a defect or a plan that was one slab short is its
#                     own question, and it is the smallest of the five.
#
PINNED_CALLS=19

if [ "$CLEAN" -eq 0 ] && [ "$VIOL" -eq 0 ]; then
    echo "FATAL: neither report line appeared. The binary was not built with"        >&2
    echo "       -DIMP_ALLOC_INTERPOSE=ON, or it did not shut down cleanly."         >&2
    echo "       Passing here would be passing for the wrong reason."                >&2
    tail -20 "$LOG" >&2
    exit 1
fi
if [ "$VIOL" -eq 0 ]; then
    [ "$PINNED_CALLS" -eq 0 ] && { echo "PASS: no device allocation while serving"; exit 0; }
    echo "FAIL: the pin says $PINNED_CALLS serving allocation(s) and there are none."
    echo "      Someone fixed them. Set PINNED_CALLS=0 in this script and delete"
    echo "      the site list above. The pin only ever goes down, so a stale one"
    echo "      is a gate that would not notice the next regression."
    exit 1
fi

# Match on "<class> <n> calls" anywhere in the line, NOT anchored at the start
# of one. The first class used to be glued to the banner, so an anchored reader
# skipped it and this gate reported 2 allocations when there were 19. The
# format is fixed, and this pattern no longer depends on it being fixed.
# awk, not bc: bc is not installed on this host, and a missing binary would
# read as "0 allocations" rather than as a broken gate.
CALLS=$(sed -n '/alloc-interpose\] I2 VIOLATIONS/,/pinned host/p' "$LOG" \
        | grep -oP '(cudaMalloc|cudaMallocAsync|pinned host)\s+\K[0-9]+(?=\s+calls)' \
        | awk '{s+=$1} END {print s+0}')
if [ "${CALLS:-0}" -eq 0 ]; then
    echo "FATAL: the violation banner is present but no per-class call counts"  >&2
    echo "       parsed out of it. The report format changed; fix this parser." >&2
    sed -n '/alloc-interpose\] I2 VIOLATIONS/,/pinned host/p' "$LOG"           >&2
    exit 1
fi

if [ "$CALLS" -gt "$PINNED_CALLS" ]; then
    echo "FAIL: $CALLS device allocations while serving, pin is $PINNED_CALLS (invariant I2)."
    echo "      A new serving-phase allocation. Find it with:"
    echo "        --set diagnostics.log_level=debug, then addr2line -e $BIN -f -C <offset>"
    sed -n '/alloc-interpose\] I2 VIOLATIONS/,/pinned host/p' "$LOG"
    exit 1
fi
if [ "$CALLS" -lt "$PINNED_CALLS" ]; then
    echo "FAIL: $CALLS device allocations while serving, pin is $PINNED_CALLS."
    echo "      Fewer than pinned: lower PINNED_CALLS in this script to $CALLS and"
    echo "      remove the fixed site from the list above."
    exit 1
fi
echo "PASS: $CALLS serving allocation(s), exactly the pinned residue, over 20"
echo "      requests at batch 4 (NVFP4 residual KV + MTP chain; log: $LOG)"
exit 0
