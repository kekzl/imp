#!/usr/bin/env bash
# Captured verify cost per chunk width vs one decode step, batch 1 (ROADMAP row 49): token recycling with
# a fixed width x depth fills every verify chunk to width * (1 + depth) rows (min_streak 0), n-gram off.
# imp-cli --bench turns speculative.capture off unless set (tools/imp-cli/args.cpp), so it is set here.
# Usage: tools/analysis/verify_width_cost.sh <model path under ~/models> [image=imp:test]
# Output (stdout, TSV): rows, width, depth, ms/verify, tok/verify, ms/verify over the decode step.
set -euo pipefail
model="$1" img="${2:-imp:test}"
if ! "$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" >/dev/null; then
    echo "GPU busy" >&2
    exit 3
fi
run() {
    docker run --rm --gpus all -v "$HOME/models:/models:ro" --entrypoint imp-cli "$img" \
        --model "/models/$model" --bench --bench-pp 512 --max-tokens 128 --bench-reps 3 \
        --set speculative.capture=true "$@" 2>&1
}
tg_ms="$(run --set speculative.ngram=false | awk '/^tg / && /avg/ {printf "%.3f", $5 / $2}' | tail -1)"
[[ -z "$tg_ms" ]] && { echo "no tg line" >&2; exit 1; }
printf 'rows\twidth\tdepth\tms_verify\ttok_verify\tx_decode\n1\t-\t-\t%s\t1.00\t1.00\n' "$tg_ms"
for wd in "1 2" "1 3" "1 7" "2 3" "1 15" "4 3"; do
    read -r w d <<<"$wd"
    line="$(run --set speculative.ngram=false --set speculative.token_recycling=true \
        --set speculative.recycle_min_streak=0 --set speculative.recycle_width="$w" \
        --set speculative.recycle_depth="$d" --set speculative.k="$d" | grep 'verify_steps=' | tail -1)"
    ms="$(sed -E 's/.* ([0-9.]+) ms\/verify.*/\1/' <<<"$line")"
    tpv="$(sed -E 's/.*\(([0-9.]+) tok\/verify.*/\1/' <<<"$line")"
    awk -v r="$((w * (1 + d)))" -v w="$w" -v d="$d" -v ms="$ms" -v t="$tpv" -v tg="$tg_ms" \
        'BEGIN {printf "%d\t%d\t%d\t%s\t%s\t%.2f\n", r, w, d, ms, t, ms / tg}'
done
