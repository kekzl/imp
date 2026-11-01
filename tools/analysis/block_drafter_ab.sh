#!/usr/bin/env bash
# Block drafter A/B on prose / code / list (ROADMAP row 49, PERF_LOG 2026-10-19 / 2026-10-20 protocol):
# imp-server --max-batch 1, greedy, 256 tokens with ignore_eos, /no_think, one server start per arm and round,
# one warmup request, then prose / code / list x 2. Arms alternate order every round.
# tok/s = completion tokens / request wall; tok/verify = emitted / verify steps from the /metrics delta.
# Arms: nospec (speculative.ngram=false), dspark (block7), dflash (z-lab b16, 15 drafts),
# dsparkeX (block7, speculative.dspark_min_emit X; dsparke0 = the n-gram 15 % rule).
# Usage: ROUNDS=3 ARMS="nospec dspark dflash" tools/analysis/block_drafter_ab.sh
#   IMP_SERVER_IMG=imp:toolchain runs build-dev/imp-server from this tree; IMP_SERVER_IMG=imp:test the image's.
#   EXTRA="--set k=v ..." is appended to every arm.
# Output (stdout, TSV): round, arm, workload, rep, tokens, ms, tok_s, verify_steps, emitted, tok_verify.
set -uo pipefail
cd "$(dirname "$0")/../.."
IMG=${IMP_SERVER_IMG:-imp:toolchain}
MODEL=${MODEL:-/models/Qwen3-8B-NVFP4-cortecs}
DSPARK=${DSPARK:-/models/dflash_qwen3_8b_block7}
DFLASH=${DFLASH:-/models/Qwen3-8B-DFlash-b16}
ROUNDS=${ROUNDS:-3}
ARMS=${ARMS:-"nospec dspark dflash"}
PORT=${PORT:-8097}
NAME=imp-blockab

arm_flags() {
    case "$1" in
        nospec) echo "--set speculative.ngram=false" ;;
        dspark) echo "--set speculative.dspark_model=$DSPARK" ;;
        dflash) echo "--set speculative.dspark_model=$DFLASH" ;;
        dsparke*) echo "--set speculative.dspark_model=$DSPARK --set speculative.dspark_min_emit=${1#dsparke}" ;;
        dflashe*) echo "--set speculative.dspark_model=$DFLASH --set speculative.dspark_min_emit=${1#dflashe}" ;;
        *) echo "unknown arm $1" >&2; return 1 ;;
    esac
}

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT

start_server() {
    cleanup
    local bin=imp-server mounts=(-v "$HOME/models:/models:ro")
    if [[ "$IMG" == imp:toolchain ]]; then
        bin=/src/build-dev/imp-server
        mounts+=(-v "$PWD:/src:ro")
    fi
    # shellcheck disable=SC2046
    docker run -d --name "$NAME" --gpus all -p "$PORT:8080" "${mounts[@]}" --entrypoint "$bin" "$IMG" \
        --host 0.0.0.0 --port 8080 --model "$MODEL" --max-batch 1 $(arm_flags "$1") ${EXTRA:-} >/dev/null || return 1
    for _ in $(seq 1 200); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        docker ps --format '{{.Names}}' | grep -qx "$NAME" || { docker logs "$NAME" 2>&1 | tail -20 >&2; return 1; }
        sleep 2
    done
    return 1
}

metric() { curl -s "http://127.0.0.1:$PORT/metrics" | awk -v k="$1" '$1 == k {print $2}'; }

declare -A PROMPT=(
    [prose]="Write a short story about a lighthouse keeper who finds a message in a bottle. /no_think"
    [code]="Write a Python class implementing an LRU cache with get and put methods, with type hints and docstrings. /no_think"
    [list]="List 40 countries with their capital cities, one per line, formatted as Country: Capital. /no_think"
)

# prompt -> "tokens ms"
ask() {
    local body t0 t1 r
    body="{\"model\":\"$(basename "$MODEL")\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":256,\"temperature\":0,\"ignore_eos\":true,\"stream\":false}"
    t0=$(date +%s%N)
    r=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
    t1=$(date +%s%N)
    echo "$(grep -oP '"completion_tokens":\K[0-9]+' <<<"$r" || echo 0) $(((t1 - t0) / 1000000))"
}

printf 'round\tarm\tworkload\trep\ttokens\tms\ttok_s\tverify_steps\temitted\ttok_verify\n'
for r in $(seq 1 "$ROUNDS"); do
    order=$ARMS
    ((r % 2 == 0)) && order=$(tr ' ' '\n' <<<"$ARMS" | tac | tr '\n' ' ')
    for arm in $order; do
        start_server "$arm" || { echo "# $arm round $r: server failed" >&2; continue; }
        ask "${PROMPT[prose]}" >/dev/null  # warmup
        for w in prose code list; do
            for rep in 1 2; do
                v0=$(metric imp_spec_ngram_verify_steps_total) e0=$(metric imp_spec_ngram_emitted_total)
                read -r tk ms <<<"$(ask "${PROMPT[$w]}")"
                v1=$(metric imp_spec_ngram_verify_steps_total) e1=$(metric imp_spec_ngram_emitted_total)
                awk -v r="$r" -v a="$arm" -v w="$w" -v p="$rep" -v tk="$tk" -v ms="$ms" \
                    -v v="$(( ${v1:-0} - ${v0:-0} ))" -v e="$(( ${e1:-0} - ${e0:-0} ))" \
                    'BEGIN {ts = ms > 0 ? tk * 1000 / ms : 0; tv = v > 0 ? e / v : 0
                            printf "%s\t%s\t%s\t%s\t%d\t%d\t%.2f\t%d\t%d\t%.2f\n", r, a, w, p, tk, ms, ts, v, e, tv}'
            done
        done
        docker logs "$NAME" 2>&1 | grep -E "dspark: /models|\[dspark\] [0-9]+ drafts" | tail -2 >&2
    done
done
