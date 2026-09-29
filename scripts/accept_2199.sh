#!/usr/bin/env bash
# GPU acceptance for #2199 (--idle-unload-seconds, POST /admin/lora/{load,unload}) and #2217 (swap drops LoRA).
# Prints PASS/FAIL per criterion, exit 0 only if every criterion passes. Needs a GPU, the
# worktree image (make build) and a safetensors model dir with config.json under IMP_MODELS_DIR.
# Usage: bash scripts/accept_2199.sh
# Env: IMP_ACCEPT_MODEL (default Qwen3-0.6B), IMP_MODELS_DIR (default ~/models), IMP_TEST_IMG,
#      IMP_ACCEPT_PORT (default 8199), IMP_ACCEPT_BASELINE_TOL_MIB (default 256),
#      IMP_ACCEPT_LORA_RANK (default 64).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Hold the card for the whole run (scripts/gpu_lock.sh); refuse a busy one.
if [ -z "${IMP_ACCEPT_2199_LOCKED:-}" ]; then
    bash scripts/require_free_gpu.sh "accept_2199" || exit 1
    IMP_ACCEPT_2199_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2199" -- bash "$0" "$@"
fi

MODEL="${IMP_ACCEPT_MODEL:-Qwen3-0.6B}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_ACCEPT_PORT:-8199}"
TOL="${IMP_ACCEPT_BASELINE_TOL_MIB:-256}"
RANK="${IMP_ACCEPT_LORA_RANK:-64}"
TTL=30
CTR=imp_accept_2199
BASE="http://127.0.0.1:$PORT"

for tool in docker curl jq nvidia-smi; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
[ -f "$MODELS_DIR/$MODEL/config.json" ] || { echo "FAIL setup: $MODELS_DIR/$MODEL/config.json missing"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

ADIR="$(mktemp -d "${TMPDIR:-/tmp}/accept_2199.XXXXXX")"
# mktemp -d is 0700; imp-server in the container runs as uid 1001 and must read the adapter.
chmod 755 "$ADIR"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$ADIR"; }
trap cleanup EXIT

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' '; }
now_ms() { date +%s%3N; }

# ---- zero-valued PEFT adapter sized from the model config (F16, all 7 projections) ----
le64() {
    local n=$1 i
    # shellcheck disable=SC2059  # the format IS the byte escape
    for i in 0 1 2 3 4 5 6 7; do printf "\\x$(printf %02x $(((n >> (8 * i)) & 255)))"; done
}
gen_adapter() {  # gen_adapter <dir>
    local dir=$1 cfg="$MODELS_DIR/$MODEL/config.json"
    local d h kvh hd ff nl
    d=$(jq -r '.hidden_size' "$cfg")
    h=$(jq -r '.num_attention_heads' "$cfg")
    kvh=$(jq -r '.num_key_value_heads // .num_attention_heads' "$cfg")
    hd=$(jq -r --argjson d "$d" --argjson h "$h" '.head_dim // ($d / $h)' "$cfg")
    ff=$(jq -r '.intermediate_size' "$cfg")
    nl=$(jq -r '.num_hidden_layers' "$cfg")
    local q=$((h * hd)) kv=$((kvh * hd))
    # proj:K:N (A is [r, K], B is [N, r])
    local specs=("self_attn.q_proj:$d:$q" "self_attn.k_proj:$d:$kv" "self_attn.v_proj:$d:$kv"
        "self_attn.o_proj:$q:$d" "mlp.gate_proj:$d:$ff" "mlp.up_proj:$d:$ff" "mlp.down_proj:$ff:$d")
    local hdr="{" off=0 first=1 l s name k n sz
    for ((l = 0; l < nl; l++)); do
        for s in "${specs[@]}"; do
            IFS=: read -r name k n <<<"$s"
            local pre="base_model.model.model.layers.$l.$name"
            sz=$((RANK * k * 2))
            [ $first = 1 ] || hdr+=","
            first=0
            hdr+="\"$pre.lora_A.weight\":{\"dtype\":\"F16\",\"shape\":[$RANK,$k],\"data_offsets\":[$off,$((off + sz))]}"
            off=$((off + sz))
            sz=$((n * RANK * 2))
            hdr+=",\"$pre.lora_B.weight\":{\"dtype\":\"F16\",\"shape\":[$n,$RANK],\"data_offsets\":[$off,$((off + sz))]}"
            off=$((off + sz))
        done
    done
    hdr+="}"
    while [ $(((${#hdr} + 8) % 8)) -ne 0 ]; do hdr+=" "; done
    mkdir -p "$dir"
    printf '{"r": %d, "lora_alpha": %d}\n' "$RANK" "$RANK" >"$dir/adapter_config.json"
    { le64 "${#hdr}"; printf '%s' "$hdr"; head -c "$off" /dev/zero; } >"$dir/adapter_model.safetensors"
    chmod 755 "$dir" && chmod 644 "$dir/adapter_config.json" "$dir/adapter_model.safetensors"  # uid 1001 reads
    echo "adapter: $nl layers x 7 projections, rank $RANK, $((off / 1048576)) MiB payload"
}

DOCKER_EXTRA=()  # extra docker run args for the next start_server (Phase C mounts)
start_server() {  # start_server <extra imp-server args...>
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -v "$ADIR":/adapters:ro \
        "${DOCKER_EXTRA[@]}" -p "$PORT":"$PORT" "$IMG" imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" \
        --max-concurrent 8 "$@" >/dev/null || return 1
    local i
    for i in $(seq 1 120); do
        curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && return 0
        docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || break
        sleep 1
    done
    echo "server did not load within 120 s:"
    docker logs "$CTR" 2>&1 | tail -20
    return 1
}

MODEL_ID=""
chat() {  # chat [lora] -> "<http_code> <seconds>" and the body in $ADIR/resp.json
    local lora=${1:-} body
    body=$(jq -nc --arg m "$MODEL_ID" --arg l "$lora" \
        '{model: $m, messages: [{role: "user", content: "Say hi."}], max_tokens: 8, temperature: 0}
         + (if $l == "" then {} else {lora: $l} end)')
    curl -s -o "$ADIR/resp.json" -w '%{http_code} %{time_total}' -H 'Content-Type: application/json' \
        -d "$body" "$BASE/v1/chat/completions"
}
admin() {  # admin <route> <json> -> body on stdout, status in $ADIR/status
    curl -s -o "$ADIR/admin.json" -w '%{http_code}' -H 'Content-Type: application/json' -d "$2" \
        "$BASE$1" >"$ADIR/status"
    cat "$ADIR/admin.json"
}

gen_adapter "$ADIR/acc"

# ================= Phase A: --idle-unload-seconds 30 =================
BASELINE=$(vram_used)
echo "baseline VRAM used before load: $BASELINE MiB"
if ! start_server --idle-unload-seconds "$TTL"; then
    verdict C1-idle-drop FAIL "server with --idle-unload-seconds $TTL did not start"
else
    MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.loaded == true)][0].id')
    LOADED=$(vram_used)
    echo "VRAM used with $MODEL_ID loaded: $LOADED MiB"
    read -r code secs <<<"$(chat)"
    t_last=$(now_ms)
    echo "last request before idle: HTTP $code in ${secs}s"
    dropped_ms=-1
    while [ $(($(now_ms) - t_last)) -le 35000 ]; do
        used=$(vram_used)
        if [ "$used" -le $((BASELINE + TOL)) ]; then
            dropped_ms=$(($(now_ms) - t_last))
            break
        fi
        sleep 0.5
    done
    used=$(vram_used)
    health=$(curl -s "$BASE/health")
    idle_log=$(docker logs "$CTR" 2>&1 | grep -c '\[idle-unload\] suspended after')
    echo "health: $(jq -c '{suspended, idle_suspended, idle_unload_seconds}' <<<"$health")"
    if [ "$dropped_ms" -ge 0 ] && [ "$idle_log" -ge 1 ] && jq -e '.idle_suspended == true' <<<"$health" >/dev/null; then
        verdict C1-idle-drop PASS "used $used MiB (baseline $BASELINE, tol $TOL, loaded $LOADED) ${dropped_ms} ms after last request"
    else
        verdict C1-idle-drop FAIL "used $used MiB (baseline $BASELINE, tol $TOL, loaded $LOADED), drop_ms=$dropped_ms, idle log lines=$idle_log"
    fi

    read -r code secs <<<"$(chat)"
    content=$(jq -r '.choices[0].message.content // empty' "$ADIR/resp.json" 2>/dev/null)
    resumed_log=$(docker logs "$CTR" 2>&1 | grep '\[idle-unload\] resumed on request' | tail -1)
    if [ "$code" = 200 ] && [ -n "$content" ]; then
        verdict C2-resume-request PASS "HTTP 200, first request after idle took ${secs}s ($resumed_log)"
    else
        verdict C2-resume-request FAIL "HTTP $code in ${secs}s: $(head -c 300 "$ADIR/resp.json")"
    fi
    echo "VRAM used after resume: $(vram_used) MiB"
fi
docker rm -f "$CTR" >/dev/null 2>&1

# ================= Phase B: no idle flag, LoRA routes =================
if ! start_server; then
    verdict C3-lora-cycle FAIL "server without the idle flag did not start"
else
    MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.loaded == true)][0].id')
    # One full warm cycle: load, use, unload, refused.
    before_load=$(vram_used)
    load=$(admin /admin/lora/load '{"path": "/adapters/acc", "name": "acc"}')
    load_status=$(cat "$ADIR/status")
    after_load=$(vram_used)
    id=$(jq -r '.id // empty' <<<"$load")
    read -r use_code _ <<<"$(chat acc)"
    unload=$(admin /admin/lora/unload "{\"id\": ${id:-0}}")
    unload_status=$(cat "$ADIR/status")
    read -r gone_code _ <<<"$(chat acc)"
    gone_msg=$(jq -r '.error.message // empty' "$ADIR/resp.json" 2>/dev/null)
    echo "load: HTTP $load_status $load"
    echo "request with adapter: HTTP $use_code"
    echo "unload: HTTP $unload_status $unload"
    echo "request after unload: HTTP $gone_code $gone_msg"
    if [ "$load_status" = 200 ] && [ -n "$id" ] && [ "$use_code" = 200 ] && [ "$unload_status" = 200 ] &&
        { [ "$gone_code" = 400 ] || [ "$gone_code" = 404 ]; } && grep -q "not loaded" <<<"$gone_msg"; then
        verdict C3-lora-cycle PASS "load id=$id, use 200, unload 200, after unload HTTP $gone_code \"$gone_msg\""
    else
        verdict C3-lora-cycle FAIL "load $load_status id=$id, use $use_code, unload $unload_status, after unload $gone_code \"$gone_msg\""
    fi

    # Control arm: the adapter must be visible in device memory, or the cycle check proves nothing.
    load_delta=$((after_load - before_load))
    if [ "$load_delta" -ge 32 ]; then
        verdict C4-control-load-visible PASS "one adapter load moved used VRAM by $load_delta MiB"
    else
        verdict C4-control-load-visible FAIL "one adapter load moved used VRAM by $load_delta MiB (< 32): cycle check blind"
    fi

    start_used=$(vram_used)
    cycle_fail=0
    for i in $(seq 1 20); do
        admin /admin/lora/load '{"path": "/adapters/acc", "name": "acc"}' >"$ADIR/cyc.json"
        [ "$(cat "$ADIR/status")" = 200 ] || cycle_fail=$((cycle_fail + 1))
        cid=$(jq -r '.id // 0' "$ADIR/cyc.json")
        admin /admin/lora/unload "{\"id\": $cid}" >/dev/null
        [ "$(cat "$ADIR/status")" = 200 ] || cycle_fail=$((cycle_fail + 1))
    done
    end_used=$(vram_used)
    drift=$((end_used - start_used))
    abs=${drift#-}
    if [ "$cycle_fail" = 0 ] && [ "$abs" -le 64 ]; then
        verdict C5-lora-20-cycles PASS "used $start_used -> $end_used MiB (drift $drift, limit 64), 40/40 calls 200"
    else
        verdict C5-lora-20-cycles FAIL "used $start_used -> $end_used MiB (drift $drift, limit 64), $cycle_fail failed calls"
    fi

    # Default off: 35 s idle without the flag leaves the model resident.
    pre_idle=$(vram_used)
    sleep 35
    health=$(curl -s "$BASE/health")
    post_idle=$(vram_used)
    if jq -e '.idle_unload_seconds == 0 and .suspended == false and .model_loaded == true' <<<"$health" >/dev/null &&
        [ "$post_idle" -ge $((pre_idle - TOL)) ]; then
        verdict C6-default-off PASS "35 s idle without the flag: still loaded, used $pre_idle -> $post_idle MiB"
    else
        verdict C6-default-off FAIL "health $(jq -c '{idle_unload_seconds, suspended, model_loaded}' <<<"$health"), used $pre_idle -> $post_idle MiB"
    fi

    # WSL2 note (LIMITATIONS: free VRAM only decreases within a process): what suspend returns.
    sus=$(admin /admin/suspend '{}')
    sus_status=$(cat "$ADIR/status")
    sus_used=$(vram_used)
    res=$(admin /admin/resume '{}')
    echo "suspend: HTTP $sus_status $(jq -c '{vram_free_before, vram_free_after, device_reset, snapshot_bytes}' <<<"$sus" 2>/dev/null)"
    echo "resume: $(jq -c '{resume_ms, warm_hits}' <<<"$res" 2>/dev/null)"
    if [ "$sus_status" = 200 ]; then
        verdict C7-wsl2-suspend-measured PASS "nvidia-smi used after /admin/suspend: $sus_used MiB vs pre-load baseline $BASELINE MiB (delta $((sus_used - BASELINE)))"
    else
        verdict C7-wsl2-suspend-measured FAIL "/admin/suspend HTTP $sus_status: $sus"
    fi
fi
docker rm -f "$CTR" >/dev/null 2>&1

# ================= Phase C: model swap drops LoRA adapters (#2217) =================
# The same weights mounted under a second name are a real swap (full teardown + load) whose
# shapes still fit the adapter: the drop is policy, not a shape refusal.
SWAP_ID="$MODEL-swap"
SWAP_CYCLES=4  # even: ends on the start model
DOCKER_EXTRA=(-v "$MODELS_DIR/$MODEL":"/swap/$SWAP_ID":ro)
if ! start_server --models-dir /swap; then
    verdict C8-swap-drops-lora FAIL "server with --models-dir /swap did not start"
    verdict C9-swap-frees-lora-vram FAIL "server with --models-dir /swap did not start"
else
    A_ID=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.loaded == true)][0].id')
    # Warm both sides once so the baseline already carries a post-swap engine.
    MODEL_ID=$SWAP_ID; read -r warm_b _ <<<"$(chat)"
    MODEL_ID=$A_ID; read -r warm_a _ <<<"$(chat)"
    echo "swap warm-up: $SWAP_ID HTTP $warm_b, $A_ID HTTP $warm_a"
    swap_base=$(vram_used)
    load_delta=0
    swap_bad=""
    for ((c = 1; c <= SWAP_CYCLES; c++)); do
        cur=$([ $((c % 2)) = 1 ] && echo "$A_ID" || echo "$SWAP_ID")
        nxt=$([ $((c % 2)) = 1 ] && echo "$SWAP_ID" || echo "$A_ID")
        pre=$(vram_used)
        load=$(admin /admin/lora/load '{"path": "/adapters/acc", "name": "acc"}')
        lst=$(cat "$ADIR/status")
        [ "$c" = 1 ] && load_delta=$(($(vram_used) - pre))
        sid=$(jq -r '.id // 0' <<<"$load")
        MODEL_ID=$cur; read -r use_code _ <<<"$(chat acc)"
        MODEL_ID=$nxt; read -r swap_code _ <<<"$(chat acc)"
        swap_err=$(jq -r '.error.code // empty' "$ADIR/resp.json" 2>/dev/null)
        admin /admin/lora/unload "{\"id\": $sid}" >"$ADIR/gone.json"
        ust=$(cat "$ADIR/status")
        uerr=$(jq -r '.error.code // empty' "$ADIR/gone.json" 2>/dev/null)
        echo "swap cycle $c: load HTTP $lst id=$sid, use on $cur HTTP $use_code, swap to $nxt HTTP $swap_code $swap_err, unload id=$sid HTTP $ust $uerr"
        [ "$lst" = 200 ] && [ "$use_code" = 200 ] && [ "$swap_code" = 400 ] && [ "$swap_err" = lora_not_loaded ] &&
            [ "$ust" = 404 ] && [ "$uerr" = lora_not_found ] || swap_bad+=" c$c"
    done
    swap_end=$(vram_used)
    drop_logs=$(docker logs "$CTR" 2>&1 | grep -c "\[model-swap\] dropped LoRA adapter 'acc'.*: model swapped")
    echo "drop log lines: $drop_logs"
    if [ -z "$swap_bad" ] && [ "$drop_logs" = "$SWAP_CYCLES" ]; then
        verdict C8-swap-drops-lora PASS "$SWAP_CYCLES/$SWAP_CYCLES swaps: request 400 lora_not_loaded, unload 404 lora_not_found, $drop_logs drop log lines"
    else
        verdict C8-swap-drops-lora FAIL "failed cycles:${swap_bad:- none}, drop log lines $drop_logs (want $SWAP_CYCLES)"
    fi
    # A leaked adapter per swap would add load_delta each cycle; one adapter's worth is the limit.
    swap_drift=$((swap_end - swap_base))
    if [ "$load_delta" -ge 32 ] && [ "$swap_drift" -lt "$load_delta" ]; then
        verdict C9-swap-frees-lora-vram PASS "used $swap_base -> $swap_end MiB after $SWAP_CYCLES swaps with an adapter loaded (drift $swap_drift, adapter $load_delta MiB)"
    else
        verdict C9-swap-frees-lora-vram FAIL "used $swap_base -> $swap_end MiB (drift $swap_drift), adapter load delta $load_delta MiB (control needs >= 32, drift must be below it)"
    fi
fi

echo
echo "== accept_2199 summary ($MODEL, image $IMG) =="
printf '%s\n' "${RESULTS[@]}"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')
[ ${#RESULTS[@]} -eq 9 ] || { echo "FAIL: ${#RESULTS[@]}/9 criteria reported"; exit 1; }
[ "$fails" = 0 ] && { echo "ALL PASS"; exit 0; }
echo "$fails FAIL"
exit 1
