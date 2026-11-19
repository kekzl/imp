#!/usr/bin/env bash
# Decode-path PPL against prefill PPL, one row per model. `--perplexity` runs prefill only; the arm
# `--prefill-chunk-size 1` sends every row through the M=1 decode path. A decode-only defect shows as
# decode PPL far above prefill PPL (#3172: Ornith-1.5 +38.04 % before the fix, +0.64 % after).
# Usage: tools/analysis/decode_path_parity.sh IMAGE MODEL_DIR_NAME...   (names under $MODELS_DIR)
#   TEXT=<file>      corpus, default the first $BYTES bytes of tools/analysis/ppl_corpus_45k.txt
#   THRESHOLD=3      FAIL when decode PPL exceeds prefill PPL by more than this many percent
#   EXTRA="..."      extra imp-cli args for both arms (e.g. "--set runtime.max_batch_size=1")
# GPU job: take the lock first (scripts/gpu_lock.sh). Exit 1 on any FAIL or missing PPL line.
set -uo pipefail
IMG=$1; shift
MODELS_DIR=${MODELS_DIR:-$HOME/models}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/decode_path_parity}
BYTES=${BYTES:-6000}
THRESHOLD=${THRESHOLD:-3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT_DIR"
head -c "$BYTES" "${TEXT:-$REPO/tools/analysis/ppl_corpus_45k.txt}" > "$OUT_DIR/text.txt" || exit 2

ppl() {  # model tag extra args...
    local model=$1 tag=$2; shift 2
    docker run --rm --init --gpus all -v "$MODELS_DIR:/models:ro" -v "$OUT_DIR:/data:ro" \
        --entrypoint imp-cli "$IMG" --model "/models/$model" --perplexity /data/text.txt \
        --set runtime.deterministic=true --set speculative.mtp_k=0 ${EXTRA:-} "$@" > "$OUT_DIR/$model.$tag.log" 2>&1
    grep -oP '^perplexity: \K[0-9.]+' "$OUT_DIR/$model.$tag.log" | tail -1
}

rc=0
printf '%-44s %10s %10s %9s  %s\n' model prefill decode delta_pct verdict
for m in "$@"; do
    p=$(ppl "$m" prefill)
    d=$(ppl "$m" decode --prefill-chunk-size 1)
    if [ -z "$p" ] || [ -z "$d" ]; then
        printf '%-44s %10s %10s %9s  %s\n' "$m" "${p:--}" "${d:--}" - "FAIL (no perplexity line, $OUT_DIR/$m.*.log)"
        rc=1
        continue
    fi
    delta=$(awk -v p="$p" -v d="$d" 'BEGIN { printf "%.2f", (d / p - 1) * 100 }')
    verdict=ok
    if awk -v x="$delta" -v t="$THRESHOLD" 'BEGIN { exit !(x > t) }'; then
        verdict=FAIL
        rc=1
    fi
    printf '%-44s %10s %10s %9s  %s\n' "$m" "$p" "$d" "$delta" "$verdict"
done
exit $rc
