#!/usr/bin/env bash
# Measure the MoE expert LRU cache under host-resident experts.
#
# The engine only offloads experts to host when they do not fit in VRAM, which
# makes the path awkward to reach on purpose. `moe.force_host_experts=N` pins
# the last N MoE layers to host regardless of fit, so the whole range from
# "barely offloaded" to "nothing on the GPU" is reachable on a model that
# otherwise fits.
#
# Reports, per arm, the cache's own sizing and hit-rate lines plus pp/tg
# throughput, so the hit rate can be read against what it costs.
#
# Two things this harness exists to keep honest:
#   * Prefill throughput on this path is NOISY — a 15% spread between two runs
#     of the SAME arm was measured on 2026-08-11. Judge only paired, alternating
#     rounds (ROUNDS>=5), never two runs 20 minutes apart.
#   * The cache's hit rate pools prefill and decode. `--bench-pp 8` isolates
#     decode; a large `--bench-pp` with `--max-tokens 8` isolates prefill.
#
# Usage:
#   tools/analysis/expert_cache_offload_sweep.sh              # host-layer sweep
#   MODE=ab ROUNDS=5 tools/analysis/expert_cache_offload_sweep.sh   # paired A/B
set -u

MODEL=${MODEL:-/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf}
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
IMG=${IMG:-imp:test}
OUT=${OUT:-/tmp/expert_cache_sweep}
MODE=${MODE:-sweep}
ARMS=${ARMS:-"0 2 8 24 48"}
ROUNDS=${ROUNDS:-5}
PP=${PP:-512}
TOKENS=${TOKENS:-256}
REPS=${REPS:-3}

mkdir -p "$OUT"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "WARNING: another process holds the GPU — numbers will not be comparable." >&2
fi

# Run one arm; echo "<pp tok/s> <tg tok/s> <hit rate>".
run_arm() {
    local log=$1; shift
    docker run --rm --gpus all -v "$MODELS_DIR":/models "$IMG" \
        imp-cli --model "$MODEL" "$@" \
        --bench --bench-pp "$PP" --bench-reps "$REPS" \
        --max-tokens "$TOKENS" --temperature 0 \
        >"$log" 2>&1
    local tp
    tp=$(grep -E "^(pp|tg) " "$log" | sed -E 's/.*\(\s*([0-9.]+) tok\/s.*/\1/' | tr '\n' ' ')
    local hr
    hr=$(grep -oE "[0-9.]+% hit rate" "$log" | head -1)
    echo "${tp}${hr:-n/a}"
}

case "$MODE" in
sweep)
    # How the cache behaves as more of the model moves off the GPU. Note the
    # budget is 15% of FREE VRAM, so moving layers to host makes the cache
    # BIGGER — slots/layer and hit rate both climb with N.
    printf "%-6s %-10s %-10s %s\n" "hostN" "pp tok/s" "tg tok/s" "hit rate"
    for n in $ARMS; do
        log="$OUT/host${n}.log"
        extra=(--set "moe.force_host_experts=$n")
        [ "$n" = "0" ] && extra=()
        read -r pp tg hr <<<"$(run_arm "$log" "${extra[@]}")"
        printf "%-6s %-10s %-10s %s\n" "$n" "$pp" "$tg" "${hr:-—}"
        grep -E "Expert LRU cache: [0-9]+ layers" "$log" | sed 's/^/       /'
    done
    ;;
ab)
    # Paired, alternating A/B of the expert cache against the staging-buffer
    # fallback with everything host-resident. Alternating within a round is the
    # only comparison this path's variance supports.
    printf "%-7s %-24s %-10s %-10s %s\n" "round" "arm" "pp tok/s" "tg tok/s" "hit rate"
    for r in $(seq 1 "$ROUNDS"); do
        for arm in cache staging; do
            log="$OUT/ab_${arm}_r${r}.log"
            extra=(--set "moe.force_host_experts=48")
            [ "$arm" = "staging" ] && extra+=(--set "moe.no_expert_cache=true")
            read -r pp tg hr <<<"$(run_arm "$log" "${extra[@]}")"
            printf "%-7s %-24s %-10s %-10s %s\n" "$r" "$arm" "$pp" "$tg" "${hr:-—}"
        done
    done
    echo
    echo "Judge the paired deltas per round, not the means — see the header."
    ;;
*)
    echo "unknown MODE=$MODE (want: sweep | ab)" >&2
    exit 2
    ;;
esac
