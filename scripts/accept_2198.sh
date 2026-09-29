#!/usr/bin/env bash
# GPU acceptance for #2198 on Qwen3-8B: `bash scripts/accept_2198.sh`, PASS/FAIL per check, exit 0 iff all pass.
# Runs make build (IMP_ACCEPT_SKIP_BUILD=1 skips), GPU busy check, then holds scripts/gpu_lock.sh.
# Env: IMP_ACCEPT_MODEL (default Qwen3-8B-Q8_0.gguf), IMP_ACCEPT_HYBRID (Qwen3.5-4B-mxfp4.gguf), IMP_ACCEPT_PORT.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

MODEL="${IMP_ACCEPT_MODEL:-Qwen3-8B-Q8_0.gguf}"
HYBRID="${IMP_ACCEPT_HYBRID:-Qwen3.5-4B-mxfp4.gguf}"
PORT="${IMP_ACCEPT_PORT:-18198}"
URL="http://127.0.0.1:${PORT}"
NAME="imp-accept-2198"
SYS="Answer only with the letter of the correct option."
N_ITEMS=32

gpu_free() {
    local busy="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
    if [ -x "$busy" ] && ! "$busy"; then
        echo "FAIL gpu-free: $busy reports the card busy; not started"
        return 1
    fi
    bash scripts/require_free_gpu.sh accept_2198 || { echo "FAIL gpu-free: require_free_gpu.sh refused"; return 1; }
}

if [ -z "${IMP_ACCEPT_LOCKED:-}" ]; then
    for m in "$MODEL" "$HYBRID"; do [ -f "$HOME/models/$m" ] || { echo "FAIL model: $HOME/models/$m missing"; exit 1; }; done
    gpu_free || exit 1
    if [ "${IMP_ACCEPT_SKIP_BUILD:-0}" != 1 ]; then
        make build || { echo "FAIL build: make build"; exit 1; }
    fi
    exec bash scripts/gpu_lock.sh run accept_2198 -- env IMP_ACCEPT_LOCKED=1 bash "$0"
fi

gpu_free || exit 1
IMG="$(bash scripts/image_tag.sh)"
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL image: $IMG is not built from this tree (make build)"; exit 1; }

FAILS=0
pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
post() { curl -s -m 600 -H 'Content-Type: application/json' -X POST "$URL$1" --data-binary @-; }

start_server() {  # start_server <model file in ~/models>; exits the script if it never gets healthy
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all -p "127.0.0.1:${PORT}:8080" -v "$HOME/models:/models:ro" "$IMG" \
        imp-server --host 0.0.0.0 --port 8080 --model "/models/$1" >/dev/null || { echo "FAIL server $1: docker run"; exit 1; }
    for _ in $(seq 1 180); do
        [ "$(curl -s -m 5 "$URL/health" | jq -r '.model_loaded // false' 2>/dev/null)" = true ] && return 0
        sleep 2
    done
    echo "FAIL server $1: not healthy after 360 s"; docker logs --tail 30 "$NAME"; exit 1
}

trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT
start_server "$MODEL"

# 32 records; item k asks the color (even k) or city (odd k) of record k. Options: 4 to 8
# consecutive entries of the pool, the true answer at position k % count.
WORK="$(mktemp -d)"
trap 'docker rm -f "$NAME" >/dev/null 2>&1; rm -rf "$WORK"' EXIT
jq -n --argjson n "$N_ITEMS" '
  ["red","green","blue","yellow","black","white","purple","orange"] as $col |
  ["Berlin","Paris","Madrid","Rome","Vienna","Oslo","Lisbon","Prague"] as $city |
  [range(0; $n) | {k: ., color: $col[(. * 3) % 8], city: $city[(. * 5 + 1) % 8]}] as $recs |
  {evidence: ($recs | map("Record \(.k): color \(.color), city \(.city).") | join("\n")),
   items: ($recs | map(. as $r | (4 + ($r.k % 5)) as $m |
     (if $r.k % 2 == 0 then {q: "Which color does record \($r.k) have?", pool: $col, ans: $r.color}
      else {q: "Which city is record \($r.k) in?", pool: $city, ans: $r.city} end) as $x |
     ($x.pool | index($x.ans)) as $ai | ($r.k % $m) as $pos |
     {id: "r\($r.k)", criterion: $x.q,
      options: [range(0; $m) | $x.pool[((($ai - $pos + .) % 8) + 8) % 8]]}))}' > "$WORK/set.json"

decide() {  # decide <mode> <n> <evidence-prefix> -> response on stdout
    jq -c --arg mode "$1" --argjson n "$2" --arg pre "$3" \
        '{mode: $mode, evidence: ($pre + .evidence), items: .items[0:$n]}' "$WORK/set.json" | post /v1/decide
}

# Check 1: probabilities sum to 1 per item.
decide serial "$N_ITEMS" "" > "$WORK/serial.json"
if ! jq -e '.items | length == '"$N_ITEMS" "$WORK/serial.json" >/dev/null; then
    fail "decide-serial: bad response: $(head -c 300 "$WORK/serial.json")"
else
    dev="$(jq '[.items[] | ((.probs | add) - 1) | if . < 0 then -. else . end] | max' "$WORK/serial.json")"
    if jq -e -n "$dev <= 1e-5" >/dev/null; then pass "probs-sum: max |sum-1| = $dev over $N_ITEMS items"
    else fail "probs-sum: max |sum-1| = $dev > 1e-5"; fi
fi

# Check 2: argmax equals the letter a temperature-0, max_tokens=1, regex-constrained chat produces.
match=0
# /v1/chat/completions requires "model"; the decide response names the loaded one.
MODEL_ID="$(jq -r '.model // empty' "$WORK/serial.json")"
for i in $(seq 0 $((N_ITEMS - 1))); do
    body="$(jq -c --argjson i "$i" --arg sys "$SYS" --arg model "$MODEL_ID" '
        .items[$i] as $it |
        ([range(0; $it.options | length) | [65 + .] | implode]) as $L |
        {evidence: .evidence, criterion: $it.criterion,
         options: ([range(0; $L | length)] | map({key: $L[.], value: $it.options[.]}) | from_entries)} as $u |
        {model: $model, messages: [{role: "system", content: $sys}, {role: "user", content: ($u | tojson)}],
         temperature: 0, max_tokens: 1, repetition_penalty: 1.0,
         response_format: {type: "regex", regex: ("(" + ($L | join("|")) + ")")}}' "$WORK/set.json")"
    raw="$(printf '%s' "$body" | post /v1/chat/completions)"
    chat="$(jq -r '.choices[0].message.content // "ERR"' <<<"$raw" 2>/dev/null)"
    chat="${chat:-ERR}"
    [ "$chat" = ERR ] && echo "  item $i chat error: $(head -c 200 <<<"$raw")"
    want="$(jq -r --argjson i "$i" '.items[$i].argmax // "none"' "$WORK/serial.json")"
    if [ "$chat" = "$want" ]; then match=$((match + 1)); else echo "  item $i: decide=$want chat=$chat"; fi
done
if [ "$match" -eq "$N_ITEMS" ]; then pass "argmax-vs-chat: $match/$N_ITEMS"
else fail "argmax-vs-chat: $match/$N_ITEMS"; fi
echo "  info: argmax equals the true option on $(jq '[.items[] | select(.argmax_index == ((.id | ltrimstr("r") | tonumber) % (.probs | length)))] | length' "$WORK/serial.json")/$N_ITEMS items"

# Check 3: serial items 2+ hit the cache for at least the evidence tokens.
ev_tokens="$(jq -c '{content: .evidence}' "$WORK/set.json" | post /tokenize | jq '.tokens | length')"
min_cached="$(jq '[.items[1:][] | .cached_tokens] | min' "$WORK/serial.json")"
if [ -n "$ev_tokens" ] && [ -n "$min_cached" ] && [ "$min_cached" != null ] && [ "$min_cached" -ge "$ev_tokens" ]; then
    pass "serial-cached: min cached_tokens over items 2..$N_ITEMS = $min_cached >= evidence tokens $h_ev"
else
    fail "serial-cached: min cached_tokens over items 2..$N_ITEMS = $min_cached, evidence tokens $h_ev"
fi

# Check 4: direct reports cached_tokens == 0 on every item, with the evidence already cached.
decide direct "$N_ITEMS" "" > "$WORK/direct.json"
max_cached="$(jq '[.items[] | .cached_tokens] | max' "$WORK/direct.json" 2>/dev/null)"
if [ "$(jq -r '.mode_used' "$WORK/direct.json" 2>/dev/null)" = direct ] && [ "$max_cached" = 0 ]; then
    pass "direct-uncached: max cached_tokens = 0 over $N_ITEMS items"
else
    fail "direct-uncached: max cached_tokens = $max_cached"
fi
agree="$(jq -n --slurpfile s "$WORK/serial.json" --slurpfile d "$WORK/direct.json" \
    '[range(0; $s[0].items | length) | select($s[0].items[.].argmax == $d[0].items[.].argmax)] | length')"
echo "  info: direct and serial argmax agree on $agree/$N_ITEMS items"

# Check 5: throughput, items/s. A fresh evidence prefix per run so serial starts cold.
for mode in direct serial; do
    for n in 1 8 32; do
        t0="$(date +%s.%N)"
        decide "$mode" "$n" "Run $mode-$n-$RANDOM$RANDOM."$'\n' > "$WORK/tp.json"
        t1="$(date +%s.%N)"
        if jq -e ".items | length == $n" "$WORK/tp.json" >/dev/null 2>&1; then
            pass "throughput $mode n=$n: $(jq -n "$n / ($t1 - $t0) * 100 | round / 100") items/s ($(jq -n "($t1 - $t0) * 1000 | round") ms)"
        else
            fail "throughput $mode n=$n: $(head -c 200 "$WORK/tp.json")"
        fi
    done
done

# Check 6: hybrid model (recurrent-snapshot prefix path). Serial items 2..8 must restore the shared
# evidence prefix (snapshot saved at its block floor, #2198); it also caches every prompt direct sends.
start_server "$HYBRID"
decide serial 8 "" > "$WORK/h_serial.json"
h_ctl="$(jq '[.items[1:][] | .cached_tokens] | min' "$WORK/h_serial.json" 2>/dev/null)"
echo "  info: hybrid serial cached_tokens $(jq -c '[.items[] | .cached_tokens]' "$WORK/h_serial.json" 2>/dev/null)"
h_ev="$(jq -c '{content: .evidence}' "$WORK/set.json" | post /tokenize | jq '.tokens | length')"
decide direct 8 "" > "$WORK/h_direct.json"
h_max="$(jq '[.items[] | .cached_tokens] | max' "$WORK/h_direct.json" 2>/dev/null)"
if [[ "$h_ctl" =~ ^[0-9]+$ ]] && [[ "$h_ev" =~ ^[0-9]+$ ]] && [ "$h_ctl" -ge "$h_ev" ]; then
    pass "hybrid-serial-cached ($HYBRID): items 2..8 min cached_tokens = $h_ctl >= evidence tokens $h_ev"
else
    fail "hybrid-serial-cached ($HYBRID): items 2..8 min cached_tokens = $h_ctl, evidence tokens $h_ev"
fi
if ! [[ "$h_ctl" =~ ^[0-9]+$ ]] || [ "$h_ctl" -le 0 ]; then
    fail "hybrid-direct-uncached ($HYBRID): control inactive, serial items 2..8 min cached_tokens = $h_ctl"
elif [ "$h_max" = 0 ]; then
    pass "hybrid-direct-uncached ($HYBRID): direct max cached_tokens = 0 over 8 items; serial control min = $h_ctl"
else
    fail "hybrid-direct-uncached ($HYBRID): direct max cached_tokens = $h_max (serial control min = $h_ctl)"
fi

echo "checks failed: $FAILS"
[ "$FAILS" -eq 0 ]
