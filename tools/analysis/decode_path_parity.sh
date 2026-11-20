#!/usr/bin/env bash
# Decode-path PPL against prefill PPL, one row per model, the same scored rows ($PREFILL .. n-2) per arm.
# `--perplexity` runs prefill only. Arm chunk1 (`--prefill-chunk-size 1`) sends every row through the M=1
# kernels but forwards with is_prefill; arm step (`--perplexity-decode`) is the per-step decode route a
# chat takes (is_prefill off). #3172 showed in chunk1 (Ornith-1.5 +38.04 %), #3179 only in step
# (Qwen3.8-27B-NVFP4-QAD, fix reverted: chunk1 5.3669 unchanged, step 5.3812 -> 10.0269).
# Usage: tools/analysis/decode_path_parity.sh IMAGE MODEL_DIR_NAME...   (names under $MODELS_DIR)
#   TEXT=<file>      corpus, default the first $BYTES bytes of tools/analysis/ppl_corpus_45k.txt
#   PREFILL=64       rows the step arm prefills; every arm scores the rows after them
#   THRESHOLD=3      FAIL when chunk1 or step PPL exceeds prefill PPL by more than this many percent
#   EXTRA="..."      extra imp-cli args for every arm (e.g. "--set runtime.max_batch_size=1")
# GPU job: take the lock first (scripts/gpu_lock.sh). Exit 1 on any FAIL or missing PPL line.
set -uo pipefail
IMG=$1; shift
MODELS_DIR=${MODELS_DIR:-$HOME/models}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/decode_path_parity}
BYTES=${BYTES:-6000}
PREFILL=${PREFILL:-64}
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

pct() { awk -v p="$1" -v d="$2" 'BEGIN { printf "%.2f", (d / p - 1) * 100 }'; }

rc=0
printf '%-44s %9s %9s %9s %9s %9s  %s\n' model prefill chunk1 step chunk1_pct step_pct verdict
for m in "$@"; do
    p=$(ppl "$m" prefill --set "diagnostics.ppl_first=$PREFILL")
    c=$(ppl "$m" chunk1 --set "diagnostics.ppl_first=$PREFILL" --prefill-chunk-size 1)
    s=$(ppl "$m" step --perplexity-decode "$PREFILL")
    if [ -z "$p" ] || [ -z "$c" ] || [ -z "$s" ]; then
        printf '%-44s %9s %9s %9s %9s %9s  %s\n' "$m" "${p:--}" "${c:--}" "${s:--}" - - \
            "FAIL (no perplexity line, $OUT_DIR/$m.*.log)"
        rc=1
        continue
    fi
    dc=$(pct "$p" "$c")
    ds=$(pct "$p" "$s")
    verdict=ok
    if awk -v a="$dc" -v b="$ds" -v t="$THRESHOLD" 'BEGIN { exit !(a > t || b > t) }'; then
        verdict=FAIL
        rc=1
    fi
    printf '%-44s %9s %9s %9s %9s %9s  %s\n' "$m" "$p" "$c" "$s" "$dc" "$ds" "$verdict"
done
exit $rc
