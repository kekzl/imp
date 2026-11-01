#!/usr/bin/env bash
# Roadmap row 101: residual-stream rotation folded into the BF16 source before the NVFP4 export.
# Per arm: rotate_residual.py (CPU image ROT_IMG), rotated BF16 PPL (must equal the source: the
# correctness gate), AWQ calibration + export on the rotated model, optional AWQ+GPTQ damp 10.
# ARMS: none = gamma fold only, had16 = block-diagonal Had16 with random signs, full = randomized
# Hadamard of the hidden size; suffix _vo adds a per-head Hadamard on v_proj rows / o_proj columns.
# Scored on ppl_corpus_45k.txt with deterministic GEMM; calibration corpus from fetch_calib_corpus.sh.
# Usage: MODEL_NAME=Qwen3-1.7B [ARMS="none had16 full full_vo"] [GPTQ=1] [SEED=0] [SOURCE=1] [DTYPE=source|f16]
#        [EXPORT=1] tools/analysis/rotate_ab.sh
set -euo pipefail

MODELS_HOST="${MODELS_HOST:-$HOME/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
IMG="${IMP_IMG:-imp:test}"
ROT_IMG="${ROT_IMG:-hfref:cpu}"  # any CPU image with torch + safetensors (Dockerfile.gptqref)
WORK="${WORK:-/mnt/data/scratch/rotate_$MODEL_NAME}"
ARMS="${ARMS:-none had16 full full_vo}"
GPTQ="${GPTQ:-1}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-source}"  # f16: rotated weights written as F16 (one rounding step less)
EXPORT="${EXPORT:-1}"     # 0: rotated BF16 PPL only (the correctness gate)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$WORK"
chmod 777 "$WORK"
CALIB_CORPUS="$WORK/calib_corpus.txt"
[ -f "$CALIB_CORPUS" ] || "$REPO/tools/analysis/fetch_calib_corpus.sh" "$CALIB_CORPUS"
PPL_CORPUS="${PPL_CORPUS:-$REPO/tools/analysis/ppl_corpus_45k.txt}"  # untracked: make_ppl_corpus.sh
[ -f "$PPL_CORPUS" ] || { echo "missing $PPL_CORPUS (tools/analysis/make_ppl_corpus.sh)" >&2; exit 1; }

run() {
    docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" "$IMG" "$@"
}
ppl() {  # ppl <model-path-in-container> <label>
    printf '%s\t' "$2"
    docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" -v "$PPL_CORPUS:/ppl.txt:ro" "$IMG" \
        imp-cli --model "$1" --perplexity /ppl.txt \
        --set runtime.deterministic_gemm=true 2>&1 | grep -E "^[Pp]erplexity:" | tail -1
}
clean() { docker run --rm -v "$WORK:/work" --entrypoint /bin/sh "$IMG" -c "rm -rf /work/$1"; }
calib() {  # calib <model-path> <out-file>
    run imp-cli --model "$1" --perplexity /work/calib_corpus.txt --calibrate "$2" \
        --set calibration.hessian=true 2>&1 | grep -E "Hessians|calibration:|rror" | tail -3
}
quant() {  # quant <src-path> <out-dir> <extra args...>
    clean "$2"
    run imp-quantize --model "$1" --out "/work/$2" "${@:3}" | grep -E "GPTQ:|rror" | tail -2
}
export_arms() {  # export_arms <src-path> <tag>
    calib "$1" "/work/calib_$2.bin"
    quant "$1" "awq_$2" --calib "/work/calib_$2.bin"
    ppl "/work/awq_$2" "$2 AWQ"
    clean "awq_$2"
    if [ "$GPTQ" = 1 ]; then
        quant "$1" "gptq_$2" --calib "/work/calib_$2.bin" --gptq "/work/calib_$2.bin.hess" --gptq-damp 10
        ppl "/work/gptq_$2" "$2 AWQ+GPTQ damp 10"
        clean "gptq_$2"
    fi
    clean "calib_$2.bin*"
}

if [ "${SOURCE:-1}" = 1 ]; then
    echo "### source"
    ppl "/models/$MODEL_NAME" "source BF16"
    export_arms "/models/$MODEL_NAME" source
fi
for arm in $ARMS; do
    kind="${arm%_vo}"
    vo=()
    [ "$kind" = "$arm" ] || vo=(--vo)
    echo "### arm $arm (seed $SEED)"
    rm -rf "${WORK:?}/rot_$arm"
    # Host uid, so the host can delete it; the script makes the 0600 safetensors readable for user imp.
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$MODELS_HOST:/models:ro" -v "$WORK:/work" \
        -v "$REPO/tools/analysis:/s:ro" --entrypoint python3 "$ROT_IMG" -I /s/rotate_residual.py --in "/models/$MODEL_NAME" \
        --out "/work/rot_$arm" --rotation "$kind" "${vo[@]}" --seed "$SEED" --dtype "$DTYPE" | tail -1
    ppl "/work/rot_$arm" "$arm BF16"
    [ "$EXPORT" = 0 ] || export_arms "/work/rot_$arm" "$arm"
    rm -rf "${WORK:?}/rot_$arm"
done
