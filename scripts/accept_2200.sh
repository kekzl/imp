#!/usr/bin/env bash
# GPU acceptance for #2200: --model hf://<org>/<repo> fetches inside the container, then serves.
# Prints PASS/FAIL per criterion, exit 0 only if every criterion passes. Needs a GPU, network,
# and the worktree image (make build). Wipes and reuses only $IMP_ACCEPT_DIR.
# Usage: bash scripts/accept_2200.sh
# Env: IMP_ACCEPT_REPO (default Qwen/Qwen3-0.6B-GGUF), IMP_ACCEPT_DIR (default
#      ~/models/.accept_2200), IMP_TEST_IMG, IMP_ACCEPT_PORT (default 8200).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Hold the card for the whole run (scripts/gpu_lock.sh); refuse a busy one.
if [ -z "${IMP_ACCEPT_2200_LOCKED:-}" ]; then
    bash scripts/require_free_gpu.sh "accept_2200" || exit 1
    IMP_ACCEPT_2200_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2200" -- bash "$0" "$@"
fi

REPO="${IMP_ACCEPT_REPO:-Qwen/Qwen3-0.6B-GGUF}"
DIR="${IMP_ACCEPT_DIR:-$HOME/models/.accept_2200}"
IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_ACCEPT_PORT:-8200}"
GATED="meta-llama/Llama-3.2-1B"
CTR=imp_accept_2200
BASE="http://127.0.0.1:$PORT"
USER_ARGS=(--user "$(id -u):$(id -g)")

for tool in docker curl jq; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

case "$DIR" in */.accept_2200) ;; *) echo "FAIL setup: IMP_ACCEPT_DIR must end in /.accept_2200"; exit 1 ;; esac
rm -rf "$DIR"
mkdir -p "$DIR"
LOGS="$(mktemp -d "${TMPDIR:-/tmp}/accept_2200.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$LOGS"; }
trap cleanup EXIT

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

start_server() {  # start_server <log file>; cold start includes the download, so 900 s
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    docker run -d --name "$CTR" --gpus all "${USER_ARGS[@]}" -v "$DIR":/models -e HF_TOKEN \
        -p "127.0.0.1:$PORT:$PORT" "$IMG" imp-server --model "hf://$REPO" --host 0.0.0.0 --port "$PORT" \
        >/dev/null || return 1
    local _
    for _ in $(seq 1 900); do
        curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && break
        docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || break
        sleep 1
    done
    docker logs "$CTR" >"$1" 2>&1
    curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && return 0
    echo "server did not load:"
    tail -20 "$1"
    return 1
}

chat() {  # chat -> "<http_code>", body in $LOGS/resp.json
    local id body
    id=$(curl -s "$BASE/v1/models" | jq -r '.data[0].id')
    body=$(jq -nc --arg m "$id" '{model: $m, messages: [{role: "user", content: "Say hi."}], max_tokens: 16, temperature: 0}')
    curl -s -o "$LOGS/resp.json" -w '%{http_code}' -H 'Content-Type: application/json' -d "$body" \
        "$BASE/v1/chat/completions"
}

rx_bytes() { docker exec "$CTR" cat /sys/class/net/eth0/statistics/rx_bytes 2>/dev/null || echo -1; }

# ---- C1: cold fetch by id, then a chat completion ----
if start_server "$LOGS/cold.log"; then
    code=$(chat)
    content=$(jq -r '.choices[0].message.content // ""' "$LOGS/resp.json" 2>/dev/null)
    fetched=$(grep -c 'hf-fetch: fetched' "$LOGS/cold.log")
    shaok=$(grep -c 'hf-fetch: sha256 ok' "$LOGS/cold.log")
    echo "cold: $(grep 'hf-fetch: fetched' "$LOGS/cold.log" | tail -1)"
    if [ "$code" = 200 ] && [ -n "$content" ] && [ "$fetched" -ge 1 ] && [ "$shaok" -ge 1 ]; then
        verdict C1-cold-fetch-serves PASS "HTTP $code, fetched=$fetched sha256ok=$shaok, reply: ${content:0:60}"
    else
        verdict C1-cold-fetch-serves FAIL "HTTP $code, fetched=$fetched sha256ok=$shaok, reply: ${content:0:60}"
    fi
else
    verdict C1-cold-fetch-serves FAIL "server did not come up"
fi

# ---- C2: second start, no download (log line + container rx bytes) ----
if start_server "$LOGS/warm.log"; then
    rx=$(rx_bytes)
    code=$(chat)
    hit=$(grep -c 'hf-fetch: cache hit' "$LOGS/warm.log")
    dl=$(grep -c 'hf-fetch: downloading' "$LOGS/warm.log")
    echo "warm: $(grep 'hf-fetch: cache hit' "$LOGS/warm.log" | tail -1)"
    # Health polls reach eth0 too; a re-download would be the whole GGUF.
    if [ "$code" = 200 ] && [ "$hit" -ge 1 ] && [ "$dl" = 0 ] && [ "$rx" -ge 0 ] && [ "$rx" -lt 1048576 ]; then
        verdict C2-second-start-no-download PASS "HTTP $code, cache-hit=$hit downloading=$dl rx_bytes=$rx"
    else
        verdict C2-second-start-no-download FAIL "HTTP $code, cache-hit=$hit downloading=$dl rx_bytes=$rx"
    fi
else
    verdict C2-second-start-no-download FAIL "server did not come up"
fi
docker rm -f "$CTR" >/dev/null 2>&1 || true

# ---- C3: offline start, no network interface at all ----
docker run --rm --gpus all --network none "${USER_ARGS[@]}" -v "$DIR":/models "$IMG" \
    imp-cli --model "hf://$REPO" --prompt "Say hi." --max-tokens 8 --temperature 0 >"$LOGS/offline.log" 2>&1
rc=$?
if [ "$rc" = 0 ] && grep -q 'hf-fetch: cache hit' "$LOGS/offline.log"; then
    verdict C3-offline-cached-run PASS "imp-cli exit $rc under --network none"
else
    verdict C3-offline-cached-run FAIL "imp-cli exit $rc; $(tail -3 "$LOGS/offline.log" | tr '\n' ' ')"
fi

# ---- C4: gated repo without HF_TOKEN: clear error, nothing written ----
docker run --rm "${USER_ARGS[@]}" -v "$DIR":/models "$IMG" \
    imp-cli --model "hf://$GATED" --prompt x --max-tokens 1 >"$LOGS/gated.log" 2>&1
rc=$?
gdir="$DIR/huggingface/hub/models--${GATED/\//--}"
if [ "$rc" != 0 ] && grep -q 'HF_TOKEN' "$LOGS/gated.log" && [ ! -d "$gdir/snapshots" ]; then
    verdict C4-gated-without-token PASS "exit $rc: $(grep -m1 'HF_TOKEN' "$LOGS/gated.log" | cut -c1-140)"
else
    verdict C4-gated-without-token FAIL "exit $rc; $(tail -3 "$LOGS/gated.log" | tr '\n' ' ')"
fi

echo
printf '%s\n' "${RESULTS[@]}"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')
[ ${#RESULTS[@]} -eq 4 ] || { echo "FAIL: ${#RESULTS[@]}/4 criteria reported"; exit 1; }
[ "$fails" = 0 ] && { echo "ALL PASS"; exit 0; }
echo "FAIL: $fails of ${#RESULTS[@]}"
exit 1
