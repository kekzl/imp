#!/bin/bash
# imp-quantize --calib-weight abs vs sq and --calib-groups ABCD vs BD on a wide-GQA model
# (roadmap row 6). Arms: RTN, ABCD abs/sq, BD abs/sq, scored on ppl_corpus_45k, deterministic.
# Calibration stats come from the RTN checkpoint: Qwen3-14B BF16 (28 GB) does not fit for
# --calibrate (upload aborts at layer 36). Resumable per arm (.done marker).
# Usage: SRC_NAME=Qwen3-14B tools/analysis/awq_wide_gqa_ab.sh > log; grep '^PPL' log
set -uo pipefail
IMG="${IMP_IMG:-ghcr.io/kekzl/imp:latest}"
MODELS_HOST="${MODELS_HOST:-$HOME/models}"
WORK="${WORK:-$HOME/imp-awq-wide-gqa}"  # not /tmp: five 14B checkpoints, ~45 GB
SRC="/models/${SRC_NAME:-Qwen3-14B}"
CALIB_CORPUS="${CALIB_CORPUS:-/models/calib_corpus.txt}"  # fetch_calib_corpus.sh output
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$WORK" && chmod 777 "$WORK"
run() {
    docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" \
        -v "$REPO/tools/analysis:/corpus:ro" "$IMG" "$@"
}
sh_in() { docker run --rm -v "$WORK:/work" --entrypoint /bin/sh "$IMG" -c "$1"; }
ppl() {
    run imp-cli --model "$1" --perplexity /corpus/ppl_corpus_45k.txt \
        --set runtime.deterministic=true 2>&1 | grep -iE "^perplexity:" | sed "s|^|PPL $2 |"
}
arm() {  # arm <name> [quantize flags...]
    local name=$1
    shift
    if [ ! -f "$WORK/$name/.done" ]; then
        sh_in "rm -rf /work/$name"
        run imp-quantize --model "$SRC" --out "/work/$name" "$@" 2>&1 | tail -4 &&
            sh_in "touch /work/$name/.done"
    fi
    ppl "/work/$name" "$name"
}
arm rtn
if [ ! -f "$WORK/calib.bin" ]; then
    echo "### calibrate"
    run imp-cli --model /work/rtn --perplexity "$CALIB_CORPUS" --calibrate /work/calib.bin 2>&1 |
        grep -v "mempool trim" | tail -3
fi
head -c 8 "$WORK/calib.bin"
echo
arm abcd_abs --calib /work/calib.bin --calib-groups ABCD --calib-weight abs
arm abcd_sq --calib /work/calib.bin --calib-groups ABCD --calib-weight sq
arm bd_abs --calib /work/calib.bin --calib-groups BD --calib-weight abs
arm bd_sq --calib /work/calib.bin --calib-groups BD --calib-weight sq
echo ALLDONE
