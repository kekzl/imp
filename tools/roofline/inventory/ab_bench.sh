#!/usr/bin/env bash
# Alternating two-image imp-cli --bench A/B, one model per process, N pairs.
# Usage: tools/roofline/inventory/ab_bench.sh <imgA> <imgB> <model path under ~/models> <pairs> [bench args...]
# Prints per-run pp/tg tok/s and the medians per arm; refuses a busy card before each run.
set -uo pipefail
img_a="$1" img_b="$2" model="$3" pairs="$4"
shift 4
args=("$@")
[[ ${#args[@]} -eq 0 ]] && args=(--bench-pp 512 --max-tokens 128 --bench-reps 3 --set speculative.ngram=false)
declare -A pp tg
run() {
    local img="$1"
    "$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" >/dev/null || { echo "GPU busy" >&2; exit 3; }
    docker run --rm --gpus all -v "$HOME/models:/models:ro" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --entrypoint imp-cli "$img" --model "/models/$model" --bench "${args[@]}" 2>&1 \
        | awk '/^pp +[0-9]+ tokens/{gsub(/[()]/,""); p=$(NF-3)} /^tg +[0-9]+ tokens/{gsub(/[()]/,""); t=$(NF-3)} END{print p, t}'
}
for ((i = 1; i <= pairs; i++)); do
    for arm in A B; do
        img="$img_a"; [[ $arm == B ]] && img="$img_b"
        read -r p t < <(run "$img")
        echo "pair $i $arm pp $p tg $t"
        pp[$arm]+="$p "
        tg[$arm]+="$t "
    done
done
med() { tr ' ' '\n' | grep -v '^$' | sort -g | awk '{a[NR]=$1} END{print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}'; }
for arm in A B; do
    echo "median $arm pp $(med <<<"${pp[$arm]}") tg $(med <<<"${tg[$arm]}")"
done
