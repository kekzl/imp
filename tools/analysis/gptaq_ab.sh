#!/usr/bin/env bash
# Roadmap row 101 step 1b: GPTAQ (arXiv 2504.02692) next to BF16, AWQ and AWQ+GPTQ(H from BF16).
# Calibration corpus from fetch_calib_corpus.sh, scored on ppl_corpus_45k.txt with deterministic GEMM.
# Pairs: the BF16 pass dumps every weight input (calibration.pair_dump, Qwen3-1.7B ~26 GB, keep WORK
# off tmpfs), the AWQ export pass reads it on the same tokens and writes calib_q.bin.gptaq (IMPHES02).
# ALPHAS: 0 = GPTQ on the export's inputs, 0.25 = reference-code default, 1 = the paper's update.
# SEQ_GROUP=N: sequential GPTAQ, the pairs of layers [lo, lo+N) come from an export whose layers < lo
# are already GPTAQ at the same damp and alpha (one export + calibration pass per group and arm; the
# dump stays until the end).
# Usage: MODEL_NAME=Qwen3-1.7B [IMP_IMG=imp:test] [WORK=/mnt/data/scratch/gptaq_<model>] tools/analysis/gptaq_ab.sh
set -euo pipefail

MODELS_HOST="${MODELS_HOST:-$HOME/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
IMG="${IMP_IMG:-imp:test}"
WORK="${WORK:-/mnt/data/scratch/gptaq_$MODEL_NAME}"
DAMPS="${DAMPS:-1 10}"
ALPHAS="${ALPHAS:-0 0.25 1}"
SEQ_GROUP="${SEQ_GROUP:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAYERS="$(grep -oE '"num_hidden_layers": *[0-9]+' "$MODELS_HOST/$MODEL_NAME/config.json" | grep -oE '[0-9]+$')"

mkdir -p "$WORK"
chmod 777 "$WORK"
CALIB_CORPUS="$WORK/calib_corpus.txt"
[ -f "$CALIB_CORPUS" ] || "$REPO/tools/analysis/fetch_calib_corpus.sh" "$CALIB_CORPUS"

run() {
    docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" "$IMG" "$@"
}
PPL_CORPUS="${PPL_CORPUS:-$REPO/tools/analysis/ppl_corpus_45k.txt}"  # untracked: make_ppl_corpus.sh
[ -f "$PPL_CORPUS" ] || { echo "missing $PPL_CORPUS (tools/analysis/make_ppl_corpus.sh)" >&2; exit 1; }
ppl() {  # ppl <model-path-in-container> <label>
    printf '%s\t' "$2"
    docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" -v "$PPL_CORPUS:/ppl.txt:ro" "$IMG" \
        imp-cli --model "$1" --perplexity /ppl.txt \
        --set runtime.deterministic_gemm=true 2>&1 | grep -E "^[Pp]erplexity:" | tail -1
}
clean() { docker run --rm -v "$WORK:/work" --entrypoint /bin/sh "$IMG" -c "rm -rf /work/$1"; }
calib() {  # calib <model-path> <out-file> <extra --set args...>
    run imp-cli --model "$1" --perplexity /work/calib_corpus.txt --calibrate "$2" "${@:3}" 2>&1 |
        grep -E "pair|GPTAQ|Hessians|calibration:|rror" | tail -6
}
quant() {  # quant <out-dir> <extra args...>
    clean "$1"
    run imp-quantize --model "/models/$MODEL_NAME" --out "/work/$1" "${@:2}" | grep -E "GPTA?Q:|rror" | tail -2
}

FIRST_PAIRS="$WORK/calib_q.bin.gptaq"
[ "$SEQ_GROUP" = 0 ] || FIRST_PAIRS="$WORK/seq_g0.bin.gptaq"
if [ "${REUSE_CALIB:-0}" = 1 ] && [ -f "$FIRST_PAIRS" ] && [ -d "$WORK/awq" ]; then
    echo "### reusing the calibration in $WORK"
else
    echo "### calibration on the BF16 source (AWQ stats, H, pair dump)"
    calib "/models/$MODEL_NAME" /work/calib_bf16.bin --set calibration.hessian=true \
        --set calibration.pair_dump=/work/x.dump
    quant awq --calib /work/calib_bf16.bin
    echo "### pairs on the AWQ export, same corpus"
    if [ "$SEQ_GROUP" = 0 ]; then
        calib /work/awq /work/calib_q.bin --set calibration.pair_read=/work/x.dump
        clean x.dump
    else  # group 0 sees the plain AWQ export in every arm
        calib /work/awq /work/seq_g0.bin --set calibration.pair_read=/work/x.dump \
            --set calibration.pair_layer_begin=0 --set calibration.pair_layer_end="$SEQ_GROUP"
    fi
    ls -la "$WORK" | grep -E "\.(hess|gptaq)$"
fi

echo "### scores"
ppl "/models/$MODEL_NAME" "BF16"
ppl /work/awq "AWQ"
quant gptq_hbf16_d10 --calib /work/calib_bf16.bin --gptq /work/calib_bf16.bin.hess --gptq-damp 10
ppl /work/gptq_hbf16_d10 "AWQ+GPTQ(H bf16) damp 10"
for d in $DAMPS; do
    for a in $ALPHAS; do
        if [ "$SEQ_GROUP" = 0 ]; then
            quant "gptaq_d${d}_a$a" --calib /work/calib_bf16.bin --gptaq /work/calib_q.bin.gptaq \
                --gptq-damp "$d" --gptaq-alpha "$a"
            ppl "/work/gptaq_d${d}_a$a" "AWQ+GPTAQ damp $d alpha $a"
            continue
        fi
        # Sequential: group [lo, lo+G) pairs come from the export whose layers < lo are GPTAQ.
        files=/work/seq_g0.bin.gptaq
        for ((lo = SEQ_GROUP; lo < LAYERS; lo += SEQ_GROUP)); do
            hi=$((lo + SEQ_GROUP < LAYERS ? lo + SEQ_GROUP : LAYERS))
            quant seq_tmp --calib /work/calib_bf16.bin --gptaq "$files" --gptq-damp "$d" --gptaq-alpha "$a"
            calib /work/seq_tmp "/work/seq_d${d}_a${a}_g$lo.bin" --set calibration.pair_read=/work/x.dump \
                --set calibration.pair_layer_begin="$lo" --set calibration.pair_layer_end="$hi"
            files="$files,/work/seq_d${d}_a${a}_g$lo.bin.gptaq"
        done
        quant "gptaq_seq_d${d}_a$a" --calib /work/calib_bf16.bin --gptaq "$files" \
            --gptq-damp "$d" --gptaq-alpha "$a"
        ppl "/work/gptaq_seq_d${d}_a$a" "AWQ+GPTAQ sequential (group $SEQ_GROUP) damp $d alpha $a"
        clean seq_tmp
        clean "gptaq_seq_d${d}_a$a"
    done
done

[ "$SEQ_GROUP" = 0 ] || clean x.dump
