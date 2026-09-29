#!/usr/bin/env bash
# GPU acceptance for #2257: prompt_logprobs LM head as one GEMM per VRAM chunk, single-pass top-N.
# C1 PromptLogprobsRows.* (test-compute) green on the GPU, 3 tests, 0 skipped.
# C2 Qwen3-8B-Q8_0, 2048-token prompt, median of 5 after 1 warm-up, arms interleaved:
#    prefill(prompt_logprobs=0) <= 1.30 x prefill(off); prompt_logprobs=20 reported (INFO); the
#    server log shows the one-GEMM route. C3 scripts/accept_2207.sh (#2246 correctness) exit 0.
# PASS/FAIL per criterion, exit 0 only if all pass. Usage: make build && bash scripts/accept_2257.sh
# Env: IMP_MODELS_DIR (~/models), IMP_ACCEPT_GGUF (Qwen3-8B-Q8_0.gguf), IMP_TEST_IMG,
#      IMP_ACCEPT_PORT (8257), IMP_ACCEPT_RATIO (1.30).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
GGUF="${IMP_ACCEPT_GGUF:-Qwen3-8B-Q8_0.gguf}"
RATIO="${IMP_ACCEPT_RATIO:-1.30}"

for tool in docker curl jq awk; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
[ -f "$MODELS_DIR/$GGUF" ] || { echo "FAIL setup: $MODELS_DIR/$GGUF missing"; exit 1; }

if [ -z "${IMP_ACCEPT_2257_LOCKED:-}" ]; then
    # CPU phase before the lock: the HF fp32 reference C3 needs (cached by accept_2207.sh).
    IMP_ACCEPT_REF_ONLY=1 IMP_MODELS_DIR="$MODELS_DIR" IMP_ACCEPT_GGUF="$GGUF" bash scripts/accept_2207.sh ||
        { echo "FAIL setup: accept_2207.sh HF reference phase"; exit 1; }
    # Busy check first, then hold the card for the whole run (scripts/gpu_lock.sh).
    HOST_CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
    if [ -x "$HOST_CHECK" ]; then
        "$HOST_CHECK" || { echo "FAIL setup: GPU busy ($HOST_CHECK)"; exit 1; }
    fi
    bash scripts/require_free_gpu.sh "accept_2257" || exit 1
    IMP_ACCEPT_2257_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2257" -- bash "$ROOT/scripts/accept_2257.sh" "$@"
fi

IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_ACCEPT_PORT:-8257}"
CTR=imp_accept_2257
BASE="http://127.0.0.1:$PORT"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

# Host-only scratch: no container mounts it, so mktemp's 0700 is never a uid mismatch.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/accept_2257.XXXXXX")"
echo "logs: $WORK"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL|INFO> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

# ---- C1: fused row kernel vs CPU reference on the GPU ----
docker run --rm --gpus all "$IMG" test-compute --gtest_filter='PromptLogprobsRows.*' >"$WORK/c1.log" 2>&1
rc=$?
npass=$(grep -c '^\[       OK \] PromptLogprobsRows\.' "$WORK/c1.log")
nskip=$(grep -c '^\[  SKIPPED \]' "$WORK/c1.log")
if [ "$rc" -eq 0 ] && [ "$npass" -eq 3 ] && [ "$nskip" -eq 0 ]; then
    verdict C1 PASS "test-compute PromptLogprobsRows.*: exit 0, $npass OK, 0 skipped"
else
    verdict C1 FAIL "test-compute PromptLogprobsRows.*: exit $rc, $npass OK, $nskip skipped ($WORK/c1.log)"
fi

# ---- C2: prefill cost, same server setup as accept_2207.sh C5 ----
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT":"$PORT" "$IMG" \
    imp-server --model "/models/$GGUF" --host 0.0.0.0 --port "$PORT" --max-batch 1 \
    --set server.prefix_cache=true >/dev/null || { echo "FAIL setup: docker run imp-server"; exit 1; }
for _ in $(seq 1 180); do
    curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && break
    docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || break
    sleep 1
done
curl -s "$BASE/health" | grep -q '"model_loaded":true' || {
    echo "FAIL setup: server did not load"; docker logs "$CTR" 2>&1 | tail -20; exit 1; }
MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '.data[] | select(.loaded == true) | .id' | head -1)

# 2047 fixed ordinary ids below Qwen3's special range; a unique first id per request keeps the
# prefix cache out of every arm.
BASE_IDS=$(jq -nc '[range(0; 2047) | ((. * 7919) % 150000) + 100]')
complete() {  # complete <extra-json> <first-id> <out-file> -> prints "<http_code> <seconds>"
    local body
    body=$(jq -nc --arg m "$MODEL_ID" --argjson ids "$BASE_IDS" --argjson f "$2" \
        "{model: \$m, prompt: ([\$f] + \$ids), max_tokens: 1, temperature: 0} + $1")
    curl -s -o "$3" -w '%{http_code} %{time_total}' -H 'Content-Type: application/json' -d "$body" \
        "$BASE/v1/completions"
}

declare -a T_OFF=() T_P0=() T_P20=()
c2_err=""
for i in 0 1 2 3 4 5; do
    for arm in off p0 p20; do
        case "$arm" in
            off) extra='{}' ;;
            p0) extra='{"prompt_logprobs": 0}' ;;
            *) extra='{"prompt_logprobs": 20}' ;;
        esac
        read -r code t < <(complete "$extra" "$((1000 + RANDOM))" "$WORK/c2_$arm.json")
        if [ "$code" != 200 ]; then
            c2_err="$arm HTTP $code: $(head -c 300 "$WORK/c2_$arm.json")"
            break 2
        fi
        [ "$i" = 0 ] && continue
        case "$arm" in
            off) T_OFF+=("$t") ;;
            p0) T_P0+=("$t") ;;
            *) T_P20+=("$t") ;;
        esac
    done
done
median() { printf '%s\n' "$@" | sort -n | sed -n 3p; }
route=$(docker logs "$CTR" 2>&1 | grep -m1 'prompt_logprobs: one-GEMM LM head')
docker logs "$CTR" >"$WORK/server.log" 2>&1
if [ -n "$c2_err" ]; then
    verdict C2 FAIL "a timed request failed: $c2_err"
else
    n_rows=$(jq '.choices[0].prompt_logprobs | length' "$WORK/c2_p20.json")
    t_off=$(median "${T_OFF[@]}")
    t_p0=$(median "${T_P0[@]}")
    t_p20=$(median "${T_P20[@]}")
    info=$(awk -v a="$t_off" -v b="$t_p0" -v c="$t_p20" \
        'BEGIN{printf "off %.4f s, prompt_logprobs=0 %.4f s (%.3fx), prompt_logprobs=20 %.4f s (%.3fx)", a, b, b/a, c, c/a}')
    ok=$(awk -v a="$t_off" -v b="$t_p0" -v r="$RATIO" 'BEGIN{print (b <= r * a) ? "yes" : "no"}')
    if [ "$ok" = yes ] && [ -n "$route" ] && [ "$n_rows" = 2048 ]; then
        verdict C2 PASS "2048-token prefill, median of 5: $info; bound ${RATIO}x; route log: $route"
    else
        verdict C2 FAIL "2048-token prefill, median of 5: $info; bound ${RATIO}x; rows $n_rows (want 2048); route log '${route}'"
    fi
    verdict C2-p20 INFO "prompt_logprobs=20: $t_p20 s vs off $t_off s"
fi
docker rm -f "$CTR" >/dev/null 2>&1 || true

# ---- C3: #2246 correctness battery (PPL vs imp-cli --perplexity, sampled = greedy, echo, cache) ----
IMP_ACCEPT_2207_LOCKED=1 IMP_MODELS_DIR="$MODELS_DIR" IMP_ACCEPT_GGUF="$GGUF" IMP_TEST_IMG="$IMG" \
    bash scripts/accept_2207.sh >"$WORK/c3.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
    verdict C3 PASS "accept_2207.sh exit 0: $(grep -E '^(PASS|FAIL) C2:' "$WORK/c3.log" | tail -1)"
else
    verdict C3 FAIL "accept_2207.sh exit $rc: $(grep -E '^FAIL' "$WORK/c3.log" | tr '\n' ' ' | head -c 600) ($WORK/c3.log)"
fi

echo "== accept_2257 summary ($GGUF, image $IMG) =="
printf '%s\n' "${RESULTS[@]}"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')
[ "$(printf '%s\n' "${RESULTS[@]}" | grep -cE '^(PASS|FAIL) ')" -eq 3 ] || { echo "FAIL: not every criterion reported"; exit 1; }
[ "$fails" = 0 ] && { echo "ALL PASS"; exit 0; }
echo "$fails FAIL"
exit 1
