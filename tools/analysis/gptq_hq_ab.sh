#!/usr/bin/env bash
# Roadmap row 101 step 1a: GPTQ with the Hessian taken on the quantized (AWQ) export's inputs
# instead of the BF16 source's, scored next to BF16, AWQ and AWQ+GPTQ(H from BF16).
# Same binary, same corpora as awq_ppl_ab.sh (calib: fetch_calib_corpus.sh, score: ppl_corpus_45k.txt).
# Usage: MODEL_NAME=Qwen3-1.7B [IMP_IMG=imp:test] [WORK=/mnt/data/scratch/gptq_hq] tools/analysis/gptq_hq_ab.sh
set -euo pipefail

MODELS_HOST="${MODELS_HOST:-$HOME/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
IMG="${IMP_IMG:-imp:test}"
WORK="${WORK:-/mnt/data/scratch/gptq_hq_$MODEL_NAME}"
DAMPS="${DAMPS:-1 10}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$WORK"
chmod 777 "$WORK"
CALIB_CORPUS="$WORK/calib_corpus.txt"
[ -f "$CALIB_CORPUS" ] || "$REPO/tools/analysis/fetch_calib_corpus.sh" "$CALIB_CORPUS"

run() {
    docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" \
        -v "$REPO/tools/analysis:/corpus:ro" "$IMG" "$@"
}
ppl() {  # ppl <model-path-in-container> <label>
    printf '%s\t' "$2"
    run imp-cli --model "$1" --perplexity /corpus/ppl_corpus_45k.txt \
        --set runtime.deterministic_gemm=true 2>&1 | grep -E "^[Pp]erplexity:" | tail -1
}
clean() { docker run --rm -v "$WORK:/work" --entrypoint /bin/sh "$IMG" -c "rm -rf /work/$1"; }
calib() {  # calib <model-path> <out-file>: AWQ stats + Hessian (F and F.hess)
    run imp-cli --model "$1" --perplexity /work/calib_corpus.txt --calibrate "$2" \
        --set calibration.hessian=true 2>&1 | tail -2
}
quant() {  # quant <out-dir> <extra args...>
    clean "$1"
    run imp-quantize --model "/models/$MODEL_NAME" --out "/work/$1" "${@:2}" | tail -2
}

echo "### calibration on the BF16 source"
calib "/models/$MODEL_NAME" /work/calib_bf16.bin
quant awq --calib /work/calib_bf16.bin
echo "### calibration on the AWQ export (quantized inputs), twice for run-to-run noise"
calib /work/awq /work/calib_q1.bin
calib /work/awq /work/calib_q2.bin
ls -la "$WORK" | grep -E "calib_.*\.hess"

echo "### scores"
ppl "/models/$MODEL_NAME" "BF16"
ppl /work/awq "AWQ"
for d in $DAMPS; do
    quant "gptq_hbf16_d$d" --calib /work/calib_bf16.bin --gptq /work/calib_bf16.bin.hess --gptq-damp "$d"
    ppl "/work/gptq_hbf16_d$d" "AWQ+GPTQ(H bf16) damp $d"
    for q in 1 2; do
        quant "gptq_hq${q}_d$d" --calib /work/calib_bf16.bin --gptq "/work/calib_q$q.bin.hess" --gptq-damp "$d"
        ppl "/work/gptq_hq${q}_d$d" "AWQ+GPTQ(H quant #$q) damp $d"
    done
done
