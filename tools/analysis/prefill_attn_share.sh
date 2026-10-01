#!/usr/bin/env bash
# Attention-kernel share of the prefill kernel time at one context (roadmap row 3, prefill
# sparsity). One nsys run: imp-cli --bench at pp=N, 8 decode tokens, graphs traced per node.
# Prints per-class kernel ms + launch counts and share = attention / all kernels in the
# prefill window = kernels starting inside the NVTX range "bench:pp" (tools/imp-cli/mode_bench.cpp).
# Usage: IMG=imp:r3 tools/analysis/prefill_attn_share.sh <model-path-in-/models> <pp> <outdir>
set -uo pipefail
MODEL=$1
PP=$2
OUT=$3
IMG=${IMG:-imp:test}
NSYS=${NSYS:-/opt/nvidia/nsight-systems/2026.1.3}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$OUT" && chmod 777 "$OUT"
TAG=$(basename "$MODEL")_pp$PP
docker run --rm --gpus all -v "$HOME/models:/models" -v "$OUT:/out" -v "$NSYS:/nsys" \
    --entrypoint /nsys/target-linux-x64/nsys "$IMG" profile --sample=none --cpuctxsw=none \
    --backtrace=none -t cuda,nvtx --cuda-graph-trace=node -o "/out/$TAG" --force-overwrite=true \
    imp-cli --model "/models/$MODEL" --bench --bench-pp "$PP" --bench-reps 1 --max-tokens 8 \
    --set speculative.ngram=false --set speculative.mtp_k=0 > "$OUT/$TAG.log" 2>&1
echo "nsys rc=$?"
docker run --rm -v "$OUT:/out" -v "$NSYS:/nsys" --entrypoint /nsys/target-linux-x64/nsys "$IMG" \
    export --type sqlite --force-overwrite=true -o "/out/$TAG.sqlite" "/out/$TAG.nsys-rep" >/dev/null 2>&1
echo "export rc=$?"
grep -E "^pp|^tg|prefill|tok/s|KV cache dtype|attn_decode=|Resolved dispatch" "$OUT/$TAG.log" | head -12
docker run --rm -v "$OUT:/out" -v "$REPO/tools/analysis:/a:ro" python:3.13-slim \
    python3 /a/prefill_attn_share.py "/out/$TAG.sqlite"
echo "share rc=$?"
