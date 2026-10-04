#!/bin/bash
# imp-quantize --calib on Gemma-4-26B-A4B (#2476): expert groups X (shared gate/up scale) and Y
# (per-expert down) against RTN and the reference NVFP4 export, deterministic. Judges:
# ppl_corpus_gemma4_turn.txt (704 tokens) and ppl_corpus_45k.txt in the same turn framing (14676
# tokens; Gemma-4-it scores only inside its turn framing, #2044). Calibration stats come from the
# RTN checkpoint (BF16 51.6 GB does not fit the card) over calib_corpus.txt, also turn-framed.
# Resumable per arm (.done marker).
# Usage: tools/analysis/awq_moe_gemma4_ab.sh > log; grep '^PPL' log
set -uo pipefail
IMG="${IMP_IMG:-imp:toolchain}"
BIN="${IMP_BIN:-/repo/build-dev}"  # dev build; IMP_BIN= IMP_IMG=imp:test for an image
MODELS_HOST="${MODELS_HOST:-$HOME/models}"
WORK="${WORK:-$HOME/imp-awq-gemma4}"  # not /tmp: 16 GB per arm
SRC="/models/${SRC_NAME:-gemma-4-26B-A4B-it-BF16}"
RTN="/models/${RTN_NAME:-Gemma-4-26B-A4B-it-NVFP4-imp}"
REF="/models/${REF_NAME:-Gemma-4-26B-A4B-it-NVFP4}"
CALIB_TXT="${CALIB_TXT:-$MODELS_HOST/calib_corpus.txt}"  # fetch_calib_corpus.sh output
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$WORK" && chmod 777 "$WORK"

# frame <in> <out>: 3000-char chunks, each one Gemma-4 model turn.
frame() {
    local total off
    : >"$2"
    total=$(wc -c <"$1")
    for ((off = 0; off < total; off += 3000)); do
        printf '<|turn>user\nContinue this passage.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>' >>"$2"
        dd if="$1" bs=1000 skip=$((off / 1000)) count=3 status=none >>"$2"
        printf '<turn|>\n' >>"$2"
    done
}
[ -f "$WORK/calib_turn.txt" ] || frame "$CALIB_TXT" "$WORK/calib_turn.txt"
[ -f "$WORK/ppl45k_turn.txt" ] || frame "$REPO/tools/analysis/ppl_corpus_45k.txt" "$WORK/ppl45k_turn.txt"

bin() { if [ -n "$BIN" ]; then echo "$BIN/$1"; else echo "$1"; fi; }
run() {
    docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" -v "$REPO:/repo:ro" \
        -v "$REPO/tools/analysis:/corpus:ro" "$IMG" "$@"
}
sh_in() { docker run --rm -v "$WORK:/work" --entrypoint /bin/sh "$IMG" -c "$1"; }
ppl() {
    local c
    for c in /corpus/ppl_corpus_gemma4_turn.txt /work/ppl45k_turn.txt; do
        run "$(bin imp-cli)" --model "$1" --perplexity "$c" --set runtime.deterministic=true 2>&1 |
            grep -E "^perplexity:" | sed "s|^|PPL $2 |"
    done
}
arm() {  # arm <name> [quantize flags...]
    local name=$1
    shift
    if [ ! -f "$WORK/$name/.done" ]; then
        sh_in "rm -rf /work/$name"
        run "$(bin imp-quantize)" --model "$SRC" --out "/work/$name" "$@" 2>&1 |
            grep -E "^AWQ|Error|error" | tail -6 &&
            sh_in "touch /work/$name/.done"
    fi
    ppl "/work/$name" "$name"
}
ppl "$REF" ref
ppl "$RTN" rtn
if [ ! -f "$WORK/calib.bin" ]; then
    echo "### calibrate"
    run "$(bin imp-cli)" --model "$RTN" --perplexity /work/calib_turn.txt --calibrate /work/calib.bin 2>&1 |
        grep -E "calibration|perplexity" | tail -3
fi
for a in ${ARMS:-x y xy abd abdxy}; do
    arm "$a" --calib /work/calib.bin --calib-groups "$(echo "$a" | tr a-z A-Z)"
done
echo ALLDONE
