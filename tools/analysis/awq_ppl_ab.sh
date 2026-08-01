#!/usr/bin/env bash
# imp-quantize quality A/B: BF16 source vs round-to-nearest NVFP4 vs
# AWQ-calibrated NVFP4, all three scored by the same imp binary on the same
# corpus.
#
# The two corpora are deliberately different. Calibration reads general
# public-domain prose (fetch_calib_corpus.sh); scoring reads imp's own
# architecture doc (ppl_corpus_45k.txt), which is the repo's established PPL
# corpus. Calibrating and scoring on one text would report a gain that exists
# only on that text.
#
# usage: MODEL_DIR=/home/user/models/Qwen3-0.6B tools/analysis/awq_ppl_ab.sh
set -euo pipefail

MODELS_HOST="${MODELS_HOST:-$HOME/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
IMG="${IMP_IMG:-imp:test}"
WORK="${WORK:-/tmp/imp_awq_ab}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The container runs as its own user; the calibration file is written from
# inside it, so the bind-mounted directory has to be writable there too.
mkdir -p "$WORK"
chmod 777 "$WORK"
CALIB_CORPUS="$WORK/calib_corpus.txt"
[ -f "$CALIB_CORPUS" ] || "$REPO/tools/analysis/fetch_calib_corpus.sh" "$CALIB_CORPUS"

run() {
    docker run --rm --gpus all \
        -v "$MODELS_HOST:/models" \
        -v "$WORK:/work" \
        -v "$REPO/tools/analysis:/corpus:ro" \
        "$IMG" "$@"
}

ppl() {  # ppl <model-path-in-container> <label>
    echo "=== PPL: $2"
    run imp-cli --model "$1" --perplexity /corpus/ppl_corpus_45k.txt \
        --set gemm.deterministic=true 2>&1 | grep -E "^perplexity:|^Perplexity:"
}

echo "### 1/4 baseline (BF16 source)"
ppl "/models/$MODEL_NAME" "BF16"

# --calibrate forces runtime.deterministic_gemm itself: without it the
# collected file is not reproducible run to run, which makes the checkpoint
# built from it not reproducible either.
echo "### 2/4 calibration pass"
run imp-cli --model "/models/$MODEL_NAME" --perplexity /work/calib_corpus.txt \
    --calibrate /work/calib.bin 2>&1 | tail -3

# Output directories are written by the container's user, so the host cannot
# remove them — clear them from inside a throwaway container instead. Leaving a
# stale directory in place would silently mix two runs' checkpoints.
clean() { docker run --rm -v "$WORK:/work" --entrypoint /bin/sh "$IMG" -c "rm -rf /work/$1"; }

echo "### 3/4 quantize round-to-nearest + score"
clean rtn
run imp-quantize --model "/models/$MODEL_NAME" --out /work/rtn | tail -3
ppl "/work/rtn" "NVFP4 round-to-nearest"

echo "### 4/4 quantize AWQ-calibrated + score"
clean awq
run imp-quantize --model "/models/$MODEL_NAME" --out /work/awq --calib /work/calib.bin | tail -5
ppl "/work/awq" "NVFP4 AWQ"
