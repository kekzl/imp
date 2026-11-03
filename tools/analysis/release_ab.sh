#!/bin/bash
# Release-over-release A/B: two images, SafeTensors heroes, one model per process, arms alternate.
# Per (model, image, trial): one --bench-pp 512 run (pp512 + tg128, the gate shape) and one
# --bench-pp 4096 run, speculation off (bench pins + speculative.ngram=false), 5 reps each.
# Usage: tools/analysis/release_ab.sh OLD_IMAGE NEW_IMAGE [TRIALS=3] > out.tsv. Rows: model image
# trial pp512 tg128 pp4096 (LONG_PP="4096 8192": one column per length). Qwen3.8-Flash-Next (host
# experts): tools/analysis/flashnext_tg_ab.sh.
set -uo pipefail

OLD="${1:?old image}"
NEW="${2:?new image}"
TRIALS="${3:-3}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODELS="${MODELS:-Qwen3-8B-NVFP4-cortecs Qwen3-8B-FP8 Qwen3-14B-NVFP4 Qwen3-Coder-30B-A3B-Instruct-FP4 Gemma-4-26B-A4B-it-NVFP4 Nemotron-3-Nano-30B-A3B-NVFP4 Qwen3.8-27B-NVFP4-vllm}"

bash scripts/require_free_gpu.sh "release A/B" || exit 2
echo "# harness: md5=$(md5sum "$0" | cut -d' ' -f1) old=$OLD new=$NEW trials=$TRIALS" >&2

bench() {  # image model pp -> "prefill_tps decode_tps" (FAILED on any error)
    docker run --rm --gpus all -u "$(id -u):$(id -g)" -w /tmp -v "$MODELS_DIR":/models \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 --entrypoint bash "$1" -c \
        "imp-cli --model /models/$2 --bench --bench-pp $3 --bench-reps 5 --max-tokens 128 \
         --temperature 0 --json --set speculative.ngram=false 2>/dev/null \
         | jq -er '\"\(.prefill_tps) \(.decode_tps)\"'" 2>/dev/null || echo "FAILED FAILED"
}

LONG_PP="${LONG_PP:-4096}"  # prompt-only columns after tg128, e.g. LONG_PP="4096 8192"
printf 'model\timage\ttrial\tpp512\ttg128'
for pp in $LONG_PP; do printf '\tpp%s' "$pp"; done
printf '\n'
for m in $MODELS; do
    for t in $(seq 1 "$TRIALS"); do
        # Alternate which arm goes first per trial so drift does not favour one image.
        if [ $((t % 2)) -eq 1 ]; then order="old new"; else order="new old"; fi
        for arm in $order; do
            img=$OLD; [ "$arm" = new ] && img=$NEW
            read -r pp512 tg128 <<< "$(bench "$img" "$m" 512)"
            row="$m	$arm	$t	$pp512	$tg128"
            msg="pp512=$pp512 tg128=$tg128"
            for pp in $LONG_PP; do
                read -r tps _ <<< "$(bench "$img" "$m" "$pp")"
                row="$row	$tps"
                msg="$msg pp$pp=$tps"
            done
            printf '%s\n' "$row"
            echo "$(date -u +%T) $m $arm t$t $msg" >&2
        done
    done
done
