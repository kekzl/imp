#!/usr/bin/env bash
# KV dtype quality at long context: one long text scored by imp-cli --perplexity in prefill chunks, so every
# chunk after the first attends to cached K/V in the arm's dtype; per-token NLL dumped for a paired comparison.
# Not deterministic: runtime.deterministic ignores kv_cache.swa_sizing (74k tokens need it); a repeated arm
# ("kv-fp16:b") is the noise floor. Usage: flint_kv_dtype_ppl.sh IMAGE TEXT_FILE [ARMS...]
# ARMS default: kv-fp16 kv-fp16:b kv-fp8; the part before ':' is the imp-cli flag.
# Writes $OUT_DIR/nll_<arm>.txt ("<row> <nll>") and $OUT_DIR/<arm>.log; compare with flint_kv_dtype_cmp.sh.
set -euo pipefail
IMG=$1; TEXT=$2; shift 2
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(kv-fp16 kv-fp16:b kv-fp8)
MODEL=${MODEL:-/models/flint-nvfp4-official}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/flint_kv}
CHUNK=${CHUNK:-2048}
MAX_SEQ=${MAX_SEQ:-81920}
mkdir -p "$OUT_DIR"
cp "$TEXT" "$OUT_DIR/text.txt"
chmod 777 "$OUT_DIR"
for arm in "${ARMS[@]}"; do
    tag=${arm/:/_}
    docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v "$HOME/models:/models:ro" \
        -v "$OUT_DIR:/data" --entrypoint imp-cli "$IMG" --model "$MODEL" --perplexity /data/text.txt \
        --"${arm%%:*}" --prefill-chunk-size "$CHUNK" --max-seq-len "$MAX_SEQ" \
        --set runtime.max_batch_size=1 --set kv_cache.swa_sizing=on --set speculative.mtp_k=0 \
        --set "diagnostics.ppl_dump=/data/nll_${tag}.txt" > "$OUT_DIR/${tag}.log" 2>&1 || echo "rc=$? $arm"
    printf '%s\t%s\n' "$arm" "$(grep -E '^perplexity:' "$OUT_DIR/${tag}.log" | tail -1)"
done
