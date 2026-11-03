#!/usr/bin/env bash
# imp vs vLLM teacher-forced PPL on one SafeTensors checkpoint (GOAL release bar 1): same token ids,
# same rows. imp-cli --perplexity scores the whole corpus as one sequence (BOS prepended when the
# tokenizer adds one), rows [FIRST, LAST] (row i = NLL of token i+1); vLLM gets imp's dumped ids
# via prompt_logprobs=0. HF tokenization (vLLM's text path) is checked against imp's ids separately.
# Usage: MODEL=Qwen3-8B-FP8 CORPORA="45k=/abs/ppl_corpus_45k.txt" \
#        IMP_ARMS="default: lmhead_off:gemm.nvfp4_lm_head=off" VLLM_ARMS="default: eager:--enforce-eager" \
#        bash tools/analysis/ppl_vs_vllm.sh      # results: $OUT/summary.tsv, $OUT/$MODEL/<tag>/cmp-*.txt
# Arm = name:item,item (imp items are --set key=value, vLLM items are ppl_vs_vllm.py vllm flags).
# GPU: the caller holds the lock; every arm re-checks gpu-busy-check.sh and aborts on BUSY.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR=${MODELS_DIR:-$HOME/models}
MODEL=${MODEL:?MODEL=<dir under $MODELS_DIR>}
CORPORA=${CORPORA:?CORPORA=\"tag=/abs/corpus ...\"}
IMP_ARMS=${IMP_ARMS:-"default: lmhead_off:gemm.nvfp4_lm_head=off"}
VLLM_ARMS=${VLLM_ARMS:-"default:"}
FIRST=${FIRST:-0}
LAST=${LAST:--1}
OUT=${OUT:-/mnt/data/scratch/pplvllm_runs}
IMG_IMP=${IMG_IMP:-imp:test}
IMG_VLLM=${IMG_VLLM:-vllm/vllm-openai:v0.31.0}
BUSY=${BUSY:-$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh}
M="$OUT/$MODEL"
mkdir -p "$M" && chmod 777 "$M"
LOG="$OUT/run.log"
SUM="$OUT/summary.tsv"
[ -f "$SUM" ] || printf 'model\tcorpus\timp_arm\tvllm_arm\tppl_imp\tppl_vllm\tdelta_pct\tbar\trows\n' > "$SUM"

py() {  # stdlib/transformers subcommands, CPU only
    docker run --rm -v "$HERE":/t:ro -v "$M":/d -v "$MODELS_DIR":/models:ro --entrypoint python3 "$IMG_VLLM" \
        -I /t/ppl_vs_vllm.py "$@"
}
gpu_free() {  # previous arm's VRAM can take seconds to return on WSL2: 6 checks, 10 s apart
    for _ in 1 2 3 4 5 6; do
        if "$BUSY" >/dev/null; then return 0; fi
        sleep 10
    done
    "$BUSY" | tee -a "$LOG"
    echo "BUSY before $1: abort" | tee -a "$LOG"
    return 1
}
{
    echo "start $(date -Is) model=$MODEL commit=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)"
    echo "harness md5 sh=$(md5sum < "$0" | cut -c1-12) py=$(md5sum < "$HERE/ppl_vs_vllm.py" | cut -c1-12)"
    for img in "$IMG_IMP" "$IMG_VLLM"; do
        echo "image $img id=$(docker image inspect --format '{{.Id}}' "$img" 2>/dev/null | cut -c8-19)" \
             "tree=$(docker image inspect --format '{{index .Config.Labels "imp.tree"}}' "$img" 2>/dev/null)"
    done
    echo "imp_arms=[$IMP_ARMS] vllm_arms=[$VLLM_ARMS] rows=[$FIRST,$LAST]"
} | tee -a "$LOG"

tags=""
for c in $CORPORA; do
    tag=${c%%=*}
    tags="$tags $tag"
    mkdir -p "$M/$tag" && chmod 777 "$M/$tag"
    cp "${c#*=}" "$M/$tag/corpus.txt"
    echo "corpus $tag $(wc -c < "$M/$tag/corpus.txt") bytes md5=$(md5sum < "$M/$tag/corpus.txt" | cut -c1-12)" | tee -a "$LOG"
    py hf-tokens "/models/$MODEL" "/d/$tag/corpus.txt" "/d/$tag/hf_tokens.txt" 2>&1 | grep '^hf_tokens:' | sed "s|^|$tag |" | tee -a "$LOG"
done

for arm in $IMP_ARMS; do
    name=${arm%%:*}
    sets=()
    IFS=, read -ra items <<< "${arm#*:}"
    for s in "${items[@]}"; do [ -n "$s" ] && sets+=(--set "$s"); done
    for tag in $tags; do
        gpu_free "imp-$name" || exit 2
        docker run --rm --gpus all -e IMP_DETERMINISTIC=1 -v "$MODELS_DIR":/models:ro -v "$M":/d "$IMG_IMP" \
            imp-cli --model "/models/$MODEL" --perplexity "/d/$tag/corpus.txt" \
            --set speculative.mtp_k=0 --set diagnostics.dump_tokens=true \
            --set "diagnostics.ppl_first=$FIRST" --set "diagnostics.ppl_last=$LAST" \
            --set "diagnostics.ppl_dump=/d/$tag/imp-$name.nll" "${sets[@]}" > "$M/$tag/imp-$name.log" 2>&1
        grep '^TOK ' "$M/$tag/imp-$name.log" > "$M/$tag/imp-$name.tok"
        if [ ! -s "$M/$tag/imp_tokens.txt" ]; then
            cp "$M/$tag/imp-$name.tok" "$M/$tag/imp_tokens.txt"
        elif ! cmp -s "$M/$tag/imp-$name.tok" "$M/$tag/imp_tokens.txt"; then
            echo "$tag imp-$name: TOKEN STREAM DIFFERS from the first imp arm" | tee -a "$LOG"
        fi
        echo "$tag imp-$name: $(grep -h -E '^perplexity:|perplexity_nll:|greedy top1' "$M/$tag/imp-$name.log" | tr '\n' ' ')" | tee -a "$LOG"
    done
done

for tag in $tags; do
    echo "$tag $(py parity "/d/$tag/imp_tokens.txt" "/d/$tag/hf_tokens.txt" 2>&1 | grep -A2 '^parity:' | tr '\n' ' ')" | tee -a "$LOG"
done

for arm in $VLLM_ARMS; do
    name=${arm%%:*}
    IFS=, read -ra flags <<< "${arm#*:}"
    pairs=()
    for tag in $tags; do pairs+=("/d/$tag/imp_tokens.txt" "/d/$tag/vllm-$name.nll"); done
    gpu_free "vllm-$name" || exit 2
    docker run --rm --gpus all --ipc=host -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 -v "$HERE":/t:ro -v "$M":/d \
        -v "$MODELS_DIR":/models:ro --entrypoint python3 "$IMG_VLLM" -I /t/ppl_vs_vllm.py vllm \
        "/models/$MODEL" "${pairs[@]}" "${flags[@]}" > "$M/vllm-$name.log" 2>&1
    echo "vllm-$name: $(grep -h -E '^vllm_ppl:|Error|error:' "$M/vllm-$name.log" | tail -4 | tr '\n' ' ')" | tee -a "$LOG"
done

for tag in $tags; do
    for ia in $IMP_ARMS; do
        for va in $VLLM_ARMS; do
            i=${ia%%:*} v=${va%%:*} c="$M/$tag/cmp-$i-vs-$v"
            [ -s "$M/$tag/imp-$i.nll" ] && [ -s "$M/$tag/vllm-$v.nll" ] || { echo "$tag $i vs $v: missing nll" | tee -a "$LOG"; continue; }
            py compare "/d/$tag/imp_tokens.txt" "/d/$tag/imp-$i.nll" "/d/$tag/vllm-$v.nll" --first "$FIRST" \
                --last "$LAST" --json "/d/$tag/cmp-$i-vs-$v.json" > "$c.txt" 2>&1
            head -2 "$c.txt" | sed "s|^|$tag $i vs $v: |" | tee -a "$LOG"
            l=$(grep -m1 '^PPL ' "$c.txt")
            [ -n "$l" ] && printf '%s\t%s\t%s\t%s\t%s\n' "$MODEL" "$tag" "$i" "$v" \
                "$(echo "$l" | sed -E 's/^PPL imp=([^ ]+) vllm=([^ ]+) delta=([^%]+)% bar[^=]*=([A-Z]+) rows=([^ ]+).*/\1\t\2\t\3\t\4\t\5/')" >> "$SUM"
        done
    done
done
echo "end $(date -Is) model=$MODEL" | tee -a "$LOG"
