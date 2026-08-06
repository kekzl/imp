#!/usr/bin/env bash
# Long-context decode A/B across models.
#
# WHY THIS EXISTS: the perf gate measures tg128 at pp512. A change whose effect
# depends on context length is invisible there by construction — #1270 shipped a
# split-count heuristic that gained +10% at 32k on one model, cost -7.30% at 32k
# on another, and passed `verify-fast` at +0.33% because the boost is inactive at
# pp512. It was reverted in #1271.
#
# Two rules this encodes, both learned the expensive way:
#
#   1. TWO MODELS MINIMUM. Six context lengths on one checkpoint produced a clean
#      monotone curve with sub-0.5% spreads and was still wrong. Precision is not
#      coverage. The default model list has different GQA shapes on purpose
#      (n_kv_heads 8/g 4 vs n_kv_heads 4/g 8) — that pair is what caught #1270.
#
#   2. SPEC-OFF. n-gram speculation puts 14-17% spread on short-context points,
#      enough to hide a 1% effect and to make a -11% median look real. Spec-OFF
#      (what the gate measures) brings it under 0.5%.
#
# Usage:
#   scripts/bench_longctx_ab.sh <image-A> <image-B> [ctx-list] [model-list]
#   scripts/bench_longctx_ab.sh imp:base imp:test "2048 8192 32768"
#
# A and B are docker images or host paths to a build dir (mounted at /bd).
# Exits non-zero if any model/context pair regresses beyond the threshold.

set -euo pipefail

IMG_A="${1:?usage: $0 <A> <B> [ctx-list] [model-list]}"
IMG_B="${2:?usage: $0 <A> <B> [ctx-list] [model-list]}"
CTXS="${3:-2048 8192 32768}"
MODELS="${4:-/models/Qwen3-8B-Q8_0.gguf /models/Qwen3-30B-A3B-NVFP4-Modelopt}"
ROUNDS="${LONGCTX_ROUNDS:-3}"
THRESHOLD="${LONGCTX_THRESHOLD:-3.0}"   # percent; regressions worse than this fail
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"

# A busy GPU makes every number below meaningless.
if command -v nvidia-smi >/dev/null 2>&1; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -gt 2000 ]; then
        echo "REFUSING: GPU holds ${used} MiB — free it first (docker ps)." >&2
        exit 2
    fi
fi

run_one() {  # $1=image-or-builddir  $2=model  $3=ctx
    local img="$1" model="$2" ctx="$3" mount=() cli
    if [ -d "$img" ]; then
        mount=(-v "$(readlink -f "$img"):/bd"); cli=(/bd/imp-cli); img=imp:toolchain
    else
        cli=(imp-cli)
    fi
    timeout 1800 docker run --rm --gpus all -v "$MODELS_DIR:/models" "${mount[@]}" "$img" \
        "${cli[@]}" --model "$model" --bench --bench-pp "$ctx" --bench-reps 3 --max-tokens 128 \
        --set speculative.ngram=false 2>&1 \
        | grep -oE 'tg +128 tokens.*\( *[0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | tail -1
}

fail=0
printf '%-42s %8s %10s %10s %9s\n' model ctx A B delta
for model in $MODELS; do
    for ctx in $CTXS; do
        a_vals=(); b_vals=()
        # Alternating, so a drifting host hits both arms equally.
        for _ in $(seq "$ROUNDS"); do
            a_vals+=("$(run_one "$IMG_A" "$model" "$ctx")")
            b_vals+=("$(run_one "$IMG_B" "$model" "$ctx")")
        done
        read -r med_a med_b delta spread_a spread_b < <(
            python3 - "${a_vals[*]}" "${b_vals[*]}" <<'PY'
import sys, statistics as st
a = [float(x) for x in sys.argv[1].split()]
b = [float(x) for x in sys.argv[2].split()]
ma, mb = st.median(a), st.median(b)
print(f"{ma:.2f} {mb:.2f} {100*(mb-ma)/ma:+.2f} {max(a)-min(a):.2f} {max(b)-min(b):.2f}")
PY
        )
        flag=""
        # A delta smaller than either spread is not a result.
        awk -v d="$delta" -v sa="$spread_a" -v sb="$spread_b" -v ma="$med_a" \
            'BEGIN { exit !((d<0?-d:d) < 100*(sa>sb?sa:sb)/ma) }' && flag=" (within spread)"
        awk -v d="$delta" -v t="$THRESHOLD" 'BEGIN { exit !(d < -t) }' && { flag=" REGRESSION"; fail=1; }
        printf '%-42s %8s %10s %10s %8s%%%s\n' "$(basename "$model")" "$ctx" "$med_a" "$med_b" "$delta" "$flag"
    done
done

[ "$fail" -eq 0 ] || { echo "FAIL: at least one pair regressed beyond ${THRESHOLD}%." >&2; exit 1; }
echo "OK: no pair regressed beyond ${THRESHOLD}%."
