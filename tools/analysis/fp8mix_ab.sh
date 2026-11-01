#!/usr/bin/env bash
# Roadmap row 101: mixed NVFP4 + FP8 exports (imp-quantize --fp8-keep / --fp8-budget) next to BF16 and AWQ.
# One calibration on the BF16 source writes the AWQ stats and the Hessians (calibration.hessian); the
# AWQ arm is the --fp8-budget 0 export, which also prints the output-loss proxy ranking (fp8-rank).
# SINGLE: one unit FP8 each (validates the proxy against PPL); BUDGETS: percent over pure NVFP4;
# KEEPS: explicit --fp8-keep lists (";"-separated arms). Scored on ppl_corpus_45k.txt, deterministic GEMM.
# Output per arm: label, PPL, export bytes (model*.safetensors).
# Usage: MODEL_NAME=Qwen3-1.7B [IMP_IMG=imp:test] [WORK=/mnt/data/scratch/fp8mix_<model>]
#        [SINGLE="27.down 0.qkv"] [BUDGETS="5 10 15 20"] [KEEPS="*.down;27.down,26.down"] tools/analysis/fp8mix_ab.sh
set -euo pipefail

MODELS_HOST="${MODELS_HOST:-$HOME/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
IMG="${IMP_IMG:-imp:test}"
WORK="${WORK:-/mnt/data/scratch/fp8mix_$MODEL_NAME}"
SINGLE="${SINGLE:-}"
BUDGETS="${BUDGETS:-5 10 15 20}"
KEEPS="${KEEPS:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$WORK"
chmod 777 "$WORK"
CALIB_CORPUS="$WORK/calib_corpus.txt"
[ -f "$CALIB_CORPUS" ] || "$REPO/tools/analysis/fetch_calib_corpus.sh" "$CALIB_CORPUS"
PPL_CORPUS="${PPL_CORPUS:-$REPO/tools/analysis/ppl_corpus_45k.txt}"  # untracked: make_ppl_corpus.sh
[ -f "$PPL_CORPUS" ] || { echo "missing $PPL_CORPUS (tools/analysis/make_ppl_corpus.sh)" >&2; exit 1; }

run() { docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" "$IMG" "$@"; }
clean() { docker run --rm -v "$WORK:/work" --entrypoint /bin/sh "$IMG" -c "rm -rf /work/$1"; }
bytes() { du -cb "$WORK/$1"/*.safetensors | tail -1 | cut -f1; }
ppl() {  # ppl <model-path-in-container> <label> <bytes>
    local p
    p="$(docker run --rm --gpus all -v "$MODELS_HOST:/models" -v "$WORK:/work" -v "$PPL_CORPUS:/ppl.txt:ro" "$IMG" \
        imp-cli --model "$1" --perplexity /ppl.txt --set runtime.deterministic_gemm=true 2>&1 |
        grep -E "^[Pp]erplexity:" | tail -1)"
    printf '%s\t%s\t%s\n' "$2" "$p" "$3"
}
quant() {  # quant <out-dir> <log> <extra args...>
    clean "$1"
    run imp-quantize --model "/models/$MODEL_NAME" --out "/work/$1" --calib /work/calib_bf16.bin "${@:3}" \
        > "$WORK/$2" 2>&1 || { tail -5 "$WORK/$2" >&2; return 1; }
    grep -E "^FP8 keep:|^size:" "$WORK/$2" | sed 's/^/  /'
}

if [ ! -f "$WORK/calib_bf16.bin.hess" ]; then
    echo "### calibration on the BF16 source (AWQ stats + Hessians)"
    run imp-cli --model "/models/$MODEL_NAME" --perplexity /work/calib_corpus.txt --calibrate /work/calib_bf16.bin \
        --set calibration.hessian=true 2>&1 | grep -E "Hessians|calibration:|rror" | tail -4
fi

echo "### AWQ (= --fp8-budget 0) and the proxy ranking"
quant awq rank.log --fp8-score /work/calib_bf16.bin.hess --fp8-budget 0
grep -E "fp8-rank" "$WORK/rank.log" | head -20
ppl "/models/$MODEL_NAME" "BF16" "$(du -cb "$MODELS_HOST/$MODEL_NAME"/*.safetensors | tail -1 | cut -f1)"
ppl /work/awq "AWQ" "$(bytes awq)"

for u in $SINGLE; do
    quant arm "single_$u.log" --fp8-keep "$u" > /dev/null
    ppl /work/arm "AWQ + FP8 $u" "$(bytes arm)"
done
for b in $BUDGETS; do
    quant arm "budget_$b.log" --fp8-score /work/calib_bf16.bin.hess --fp8-budget "$b"
    ppl /work/arm "AWQ + FP8 budget $b %" "$(bytes arm)"
done
IFS=';' read -r -a keep_arms <<< "$KEEPS"
for k in "${keep_arms[@]}"; do
    quant arm "keep.log" --fp8-keep "$k"
    ppl /work/arm "AWQ + FP8 keep $k" "$(bytes arm)"
done
clean arm

# BENCH_BUDGET=P: pp512 + tg128 of AWQ vs the budget-P export, BENCH_RUNS alternating pairs, n-gram off.
if [ -n "${BENCH_BUDGET:-}" ]; then
    quant mixed bench.log --fp8-score /work/calib_bf16.bin.hess --fp8-budget "$BENCH_BUDGET"
    for r in $(seq "${BENCH_RUNS:-3}"); do
        for m in awq mixed; do
            printf 'bench run %s %s\t' "$r" "$m"
            run imp-cli --model "/work/$m" --bench --bench-pp 512 --bench-reps 5 --max-tokens 128 --temperature 0 \
                --set speculative.ngram=false 2>&1 | grep -E "^(pp|tg) " | tr '\n' ' '
            echo
        done
    done
    clean mixed
fi
