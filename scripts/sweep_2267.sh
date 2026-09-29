#!/usr/bin/env bash
# Sweep for #2267: Q8_0 prefill time per arm, prompt length and model. No PASS/FAIL (scripts/accept_2267.sh).
# Arms (IMP_SWEEP_ARMS, "name[@image][=override,override]"; no @image = this tree's image), default:
#   new (default config), off (gemm.q8_imma_enabled=false), plus old@$IMP_SWEEP_OLD_IMG when set
#   (pre-#2267 kernel image, built by scripts/bench_2267.sh).
# Rows per GEMM = min(tokens, runtime.prefill_chunk_size); default chunk 2048, IMP_SWEEP_CHUNK overrides.
# Usage: make build && bash scripts/sweep_2267.sh. Exit 0 = every measurement completed.
# Env: IMP_MODELS_DIR (~/models), IMP_SWEEP_MODELS (Qwen3-8B-Q8_0.gguf Qwen3-4B-Instruct-2507-Q8_0.gguf),
#      IMP_SWEEP_TOKENS (256 512 1024 1536 2048 4096 8192), IMP_SWEEP_ROUNDS (5), IMP_SWEEP_PORT (8267),
#      IMP_SWEEP_CHUNK (unset = engine default 2048), IMP_SWEEP_TSV (write "model tokens arm median_s" rows),
#      IMP_SWEEP_OLD_IMG, IMP_TEST_IMG, IMP_GPU_BUSY_CHECK (busy-check script).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
read -r -a MODELS <<<"${IMP_SWEEP_MODELS:-Qwen3-8B-Q8_0.gguf Qwen3-4B-Instruct-2507-Q8_0.gguf}"
read -r -a TOKENS <<<"${IMP_SWEEP_TOKENS:-256 512 1024 1536 2048 4096 8192}"
DEF_ARMS="new off=gemm.q8_imma_enabled=false${IMP_SWEEP_OLD_IMG:+ old@$IMP_SWEEP_OLD_IMG}"
read -r -a ARMS <<<"${IMP_SWEEP_ARMS:-$DEF_ARMS}"
ROUNDS="${IMP_SWEEP_ROUNDS:-5}"
CHUNK="${IMP_SWEEP_CHUNK:-}"
TSV="${IMP_SWEEP_TSV:-}"

for tool in docker curl jq; do
    command -v "$tool" >/dev/null || { echo "ERROR setup: $tool not on PATH"; exit 1; }
done
for m in "${MODELS[@]}"; do
    [ -f "$MODELS_DIR/$m" ] || { echo "ERROR setup: $MODELS_DIR/$m missing"; exit 1; }
done

if [ -z "${IMP_SWEEP_2267_LOCKED:-}" ]; then
    BUSY="${IMP_GPU_BUSY_CHECK:-$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh}"
    if [ -x "$BUSY" ]; then "$BUSY" || { echo "ERROR setup: GPU busy ($BUSY)"; exit 1; }; fi
    bash scripts/require_free_gpu.sh "sweep_2267" || exit 1
    IMP_SWEEP_2267_LOCKED=1 exec bash scripts/gpu_lock.sh run "sweep_2267" -- bash "$0" "$@"
fi

IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_SWEEP_PORT:-8267}"
CTR=imp_sweep_2267
BASE="http://127.0.0.1:$PORT"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "ERROR setup: $IMG is not this tree (make build)"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sweep_2267.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

ERRORS=0
err() { echo "ERROR $*"; ERRORS=$((ERRORS + 1)); }

MAX_TOK=0
for n in "${TOKENS[@]}"; do [ "$n" -gt "$MAX_TOK" ] && MAX_TOK=$n; done
MAX_SEQ=$((MAX_TOK + 64))

# Prompt = token ids 1000 + (i * 7919) % 30000: fixed, inside every Qwen3 vocab, no tokenizer run.
for n in "${TOKENS[@]}"; do
    jq -nc --argjson n "$n" '{prompt: [range(0; $n) | 1000 + ((. * 7919) % 30000)], max_tokens: 1,
        temperature: 0}' >"$WORK/req_$n.json"
done

arm_head() { echo "${1%%=*}"; }                  # name[@image]
arm_name() { local h; h=$(arm_head "$1"); echo "${h%%@*}"; }
arm_img() {  # arm_img <spec> -> the image this arm runs
    local h; h=$(arm_head "$1")
    if [[ "$h" == *@* ]]; then echo "${h#*@}"; else echo "$IMG"; fi
}
for spec in "${ARMS[@]}"; do
    a_img=$(arm_img "$spec")
    docker image inspect "$a_img" >/dev/null 2>&1 || { echo "ERROR setup: arm $(arm_name "$spec") image $a_img missing"; exit 1; }
done

start_server() {  # start_server <model> <arm spec>
    local -a extra=()
    local ov
    if [[ "$2" == *=* ]]; then
        IFS=, read -r -a ovs <<<"${2#*=}"
        for ov in "${ovs[@]}"; do extra+=(--set "$ov"); done
    fi
    [ -n "$CHUNK" ] && extra+=(--set "runtime.prefill_chunk_size=$CHUNK")
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT":"$PORT" "$(arm_img "$2")" \
        imp-server --model "/models/$1" --host 0.0.0.0 --port "$PORT" --max-batch 1 \
        --set server.prefix_cache=false --set "runtime.max_seq_len=$MAX_SEQ" "${extra[@]}" >/dev/null || return 1
    for _ in $(seq 1 180); do
        curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && break
        docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || break
        sleep 1
    done
    curl -s "$BASE/health" | grep -q '"model_loaded":true' || { docker logs "$CTR" 2>&1 | tail -20; return 1; }
    MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '.data[] | select(.loaded == true) | .id' | head -1)
}

complete() {  # complete <json body file> <out-file> -> prints "<http_code> <seconds>"
    local body
    body=$(jq -c --arg m "$MODEL_ID" '. + {model: $m}' "$1")
    curl -s -o "$2" -w '%{http_code} %{time_total}' -H 'Content-Type: application/json' -d "$body" \
        "$BASE/v1/completions"
}

# Per round, model and arm: one server load, 1 warm-up + 1 timed request per length.
measure_arm() {  # measure_arm <model index> <arm spec> <round>
    local mi="$1" spec="$2" r="$3" arm n code t np
    arm=$(arm_name "$spec")
    start_server "${MODELS[$mi]}" "$spec" || { err "${MODELS[$mi]} round $r arm $arm: server did not load"; return; }
    for n in "${TOKENS[@]}"; do
        read -r code _ < <(complete "$WORK/req_$n.json" "$WORK/warm.json")
        [ "$code" = 200 ] || { err "${MODELS[$mi]} round $r arm $arm warm-up $n: HTTP $code"; continue; }
        read -r code t < <(complete "$WORK/req_$n.json" "$WORK/timed.json")
        np=$(jq -r '.usage.prompt_tokens // "none"' "$WORK/timed.json" 2>/dev/null)
        [ "$code" = 200 ] && [ "$np" = "$n" ] ||
            { err "${MODELS[$mi]} round $r arm $arm timed $n: HTTP $code, prompt_tokens $np"; continue; }
        echo "$t" >>"$WORK/t_${mi}_${arm}_$n"
    done
    echo "${MODELS[$mi]} round $r arm $arm done"
    docker rm -f "$CTR" >/dev/null 2>&1 || true
}

# Odd rounds run the arms forward, even rounds reversed (AB/BA for two arms).
for r in $(seq 1 "$ROUNDS"); do
    order=("${ARMS[@]}")
    if [ $((r % 2)) = 0 ]; then
        order=()
        for ((i = ${#ARMS[@]} - 1; i >= 0; i--)); do order+=("${ARMS[$i]}"); done
    fi
    for mi in "${!MODELS[@]}"; do
        for spec in "${order[@]}"; do measure_arm "$mi" "$spec" "$r"; done
    done
done

median() {  # median <file> -> median of its lines, "ERR" if fewer than ROUNDS
    [ -f "$1" ] && [ "$(wc -l <"$1")" = "$ROUNDS" ] || { echo ERR; return; }
    sort -g "$1" | sed -n "$(((ROUNDS + 1) / 2))p"
}

NAMES=()
for spec in "${ARMS[@]}"; do NAMES+=("$(arm_name "$spec")"); done
echo "== sweep_2267: image $IMG (arms without @image), median of $ROUNDS, arm order alternates per round, prefix cache off =="
echo "   arms: ${ARMS[*]}; prefill = curl time_total, max_tokens 1; ratio = off / arm (>1: arm faster than off)"
hdr=$(printf '%-34s %6s %5s' model tokens rows)
for a in "${NAMES[@]}"; do hdr+=$(printf ' %9s' "${a}_s"); done
for a in "${NAMES[@]}"; do [ "$a" = off ] || hdr+=$(printf ' %9s' "off/$a"); done
echo "$hdr"
[ -n "$TSV" ] && : >"$TSV"
for mi in "${!MODELS[@]}"; do
    for n in "${TOKENS[@]}"; do
        rows=$n
        eff_chunk="${CHUNK:-2048}"
        [ "$eff_chunk" -gt 0 ] && [ "$rows" -gt "$eff_chunk" ] && rows=$eff_chunk
        line=$(printf '%-34s %6s %5s' "${MODELS[$mi]}" "$n" "$rows")
        declare -A med=()
        for a in "${NAMES[@]}"; do
            med[$a]=$(median "$WORK/t_${mi}_${a}_$n")
            [ "${med[$a]}" = ERR ] && err "${MODELS[$mi]} $n arm $a: fewer than $ROUNDS timed requests"
            line+=$(printf ' %9s' "${med[$a]}")
            [ -n "$TSV" ] && printf '%s\t%s\t%s\t%s\n' "${MODELS[$mi]}" "$n" "$a" "${med[$a]}" >>"$TSV"
        done
        for a in "${NAMES[@]}"; do
            [ "$a" = off ] && continue
            q=ERR
            if [ -n "${med[off]:-}" ] && [ "${med[off]}" != ERR ] && [ "${med[$a]}" != ERR ]; then
                q=$(jq -rn --argjson o "${med[off]}" --argjson x "${med[$a]}" '$o / $x * 1e3 | round / 1e3')
                q=$(printf '%.3f' "$q")
            fi
            line+=$(printf ' %9s' "$q")
        done
        echo "$line"
        unset med
    done
done
[ "$ERRORS" = 0 ] && { echo "ALL MEASUREMENTS COMPLETED"; exit 0; }
echo "$ERRORS measurement errors"
exit 1
