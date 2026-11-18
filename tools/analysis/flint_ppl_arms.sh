#!/usr/bin/env bash
# imp --perplexity on one text per arm, per-token NLL dumped (compare: flint_kv_dtype_cmp.sh). An arm is
# "TAG extra imp-cli args...", e.g. "decode --prefill-chunk-size 1" (every row through the decode path) or
# "smallm16 --prefill-chunk-size 16 --set gemm.nvfp4_smallm_a4=false".
# Usage: tools/analysis/flint_ppl_arms.sh IMAGE TEXT_FILE ARM...   -> $OUT_DIR/nll_<TAG>.txt, <TAG>.log
set -euo pipefail
IMG=$1; TEXT=$2; shift 2
MODEL=${MODEL:-/models/flint-nvfp4-official}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/flint_ppl_arms}
mkdir -p "$OUT_DIR"
cp "$TEXT" "$OUT_DIR/text.txt"
chmod 777 "$OUT_DIR"
for arm in "$@"; do
    read -r -a a <<< "$arm"
    tag=${a[0]}
    docker run --rm --init --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v "$HOME/models:/models:ro" \
        -v "$OUT_DIR:/data" --entrypoint imp-cli "$IMG" --model "$MODEL" --perplexity /data/text.txt \
        --set runtime.deterministic=true --set speculative.mtp_k=0 \
        --set "diagnostics.ppl_dump=/data/nll_${tag}.txt" "${a[@]:1}" > "$OUT_DIR/${tag}.log" 2>&1 || echo "rc=$? $tag"
    printf '%s\t%s\n' "$tag" "$(grep -E '^perplexity:' "$OUT_DIR/${tag}.log" | tail -1)"
done
