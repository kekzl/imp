#!/usr/bin/env bash
# shellcheck disable=SC2046,SC2086 # job lists and word lists split on purpose
# inkling_smoke.sh <imp-server> <model-dir> [port]: Inkling-2-Mini serving smoke (audit D-4, #3272).
# 2 sequence slots at runtime.max_seq_len=81920: 2 concurrent chats whose replies run past 256
# tokens, then 9 new chats in 3 waves of 3 (more requests than slots), temperature 0. Each reply
# must be served (HTTP 200, finish_reason) and pass scripts/degen_verdict.sh on its words; a banned word
# (banned_strings, B-111 S6) never appears and the reply before it equals the unbanned one.
# Prints one line per request; exit 0 = 11 of 11 served and coherent and a logit_lens request
# refused with 400 unsupported_feature (audit C-6), the draft attached at init (B-111). verify.sh and
# `make verify-inkling` run it.
set -uo pipefail
SRV="$1" MODEL="$2" PORT="${3:-18190}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
"$SRV" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --set runtime.max_seq_len=81920 >"$TMP/log" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; rm -rf "$TMP"' EXIT
for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$PID" 2>/dev/null || { echo "FAIL server exited during load"; tail -20 "$TMP/log"; exit 1; }
    sleep 1
done
M=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id') || { echo "FAIL no /v1/models"; exit 1; }

# chat <tag> <max_tokens> <prompt>: writes $TMP/<tag>.json and $TMP/<tag>.code.
chat() {
    jq -nc --arg m "$M" --argjson n "$2" --arg p "$3" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0}' >"$TMP/$1.req"
    curl -s -m 600 -o "$TMP/$1.json" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @"$TMP/$1.req" >"$TMP/$1.code"
}

FAILS=0 SERVED=0
# B-111: the draft loads at init (after the KV pool binds), not at the first forward.
la=$(grep -n "inkling2_loop: decode lookahead" "$TMP/log" | head -1 | cut -d: -f1)
mp=$(grep -n "Serving metadata pool" "$TMP/log" | head -1 | cut -d: -f1)
if [ -n "$la" ] && [ -n "$mp" ] && [ "$la" -lt "$mp" ]; then
    echo "ok   draft: '$(sed -n "${la}p" "$TMP/log" | grep -o 'decode lookahead.*')' at init"
else
    echo "FAIL draft: lookahead line ${la:-missing}, metadata pool line ${mp:-missing}: not attached at init"
    FAILS=$((FAILS + 1))
fi
# C-6: the layer-loop features Inkling's own forward skips are refused, not served without them.
jq -nc --arg m "$M" '{model:$m,messages:[{role:"user",content:"hi"}],max_tokens:4,logit_lens:2}' >"$TMP/lens.req"
code=$(curl -s -m 120 -o "$TMP/lens.json" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$TMP/lens.req")
err=$(jq -r '"\(.error.code // "")/\(.error.param // "")"' "$TMP/lens.json" 2>/dev/null)
if [ "$code" = "400" ] && [ "$err" = "unsupported_feature/logit_lens" ]; then
    echo "ok   caps: logit_lens refused, HTTP 400 $err"
else
    echo "FAIL caps: logit_lens answered HTTP ${code:-none} '$err', want 400 unsupported_feature/logit_lens"
    FAILS=$((FAILS + 1))
fi
# judge <tag> <min_completion_tokens>: served, long enough, coherent.
judge() {
    local tag="$1" min="$2" code toks fin words verdict
    code=$(cat "$TMP/$tag.code" 2>/dev/null)
    toks=$(jq -r '.usage.completion_tokens // 0' "$TMP/$tag.json" 2>/dev/null)
    fin=$(jq -r '.choices[0].finish_reason // empty' "$TMP/$tag.json" 2>/dev/null)
    if [ "$code" != "200" ] || [ -z "$fin" ]; then
        echo "FAIL $tag: HTTP ${code:-none}, finish_reason '${fin}': $(head -c 300 "$TMP/$tag.json" 2>/dev/null)"
        FAILS=$((FAILS + 1))
        return
    fi
    SERVED=$((SERVED + 1))
    if [ "${toks:-0}" -lt "$min" ]; then
        echo "FAIL $tag: $toks completion tokens, need $min"
        FAILS=$((FAILS + 1))
        return
    fi
    # The model writes its thinking block either way: both fields are its output.
    words=$(jq -r '(.choices[0].message.reasoning_content // "") + " " + (.choices[0].message.content // "")' \
        "$TMP/$tag.json" | tr -s '[:space:]' '\n' | grep -v '^$')
    verdict=$(printf '%s\n' "$words" | bash "$ROOT/scripts/degen_verdict.sh" 8 2>&1)
    if [ "${verdict#OK}" = "$verdict" ]; then
        echo "FAIL $tag: ${verdict#FAIL } (words: $(printf '%s ' $words | head -c 300))"
        FAILS=$((FAILS + 1))
        return
    fi
    echo "ok   $tag: $toks tokens, finish $fin, ${verdict#OK }"
}

# Wave 1: both slots decode past 256 tokens side by side (the #3272 trigger).
chat long1 384 "Write a detailed story of about 600 words about a lighthouse keeper and a storm." &
chat long2 384 "Explain in detail, step by step, how a refrigerator moves heat out of its box." &
wait $(jobs -p | grep -v "^$PID$") 2>/dev/null
judge long1 257
judge long2 257
# B-111 S6: a banned word never appears, and the reply before its first place in the unbanned reply is kept.
# Both requests alone, one after the other: two decoding rows take another LM-head path.
text_of() { jq -r '(.choices[0].message.reasoning_content // "") + "\n" + (.choices[0].message.content // "")' \
    "$TMP/$1.json" 2>/dev/null; }
BANP="Describe a sunrise over the sea in four sentences."
chat ban0 160 "$BANP"
text0=$(text_of ban0)
word=$(printf '%s' "$text0" | tr -cs '[:alpha:]' '\n' | awk 'length($0) >= 6' | sed -n 8p)
jq -nc --arg m "$M" --arg p "$BANP" --arg w "$word" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:160,temperature:0,banned_strings:[$w]}' >"$TMP/ban1.req"
code=$(curl -s -m 600 -o "$TMP/ban1.json" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$TMP/ban1.req")
text1=$(text_of ban1)
pre="${text0%%"$word"*}"
if [ "$(cat "$TMP/ban0.code")" = "200" ] && [ "$code" = "200" ] && [ -n "$word" ] &&
    [[ "$text1" != *"$word"* ]] && [ "${text1:0:${#pre}}" = "$pre" ]; then
    echo "ok   ban: '$word' absent, the first ${#pre} characters equal the unbanned reply"
else
    echo "FAIL ban: HTTP ${code:-none}, word '$word', present $([[ "$text1" == *"$word"* ]] && echo yes || echo no)," \
        "kept prefix $([ "${text1:0:${#pre}}" = "$pre" ] && echo yes || echo no): $(head -c 300 "$TMP/ban1.json")"
    FAILS=$((FAILS + 1))
fi
# Waves 2-4: 9 new requests on slots whose previous requests ended.
TOPICS=(volcanoes honeybees "the Moon" glaciers "the printing press" octopuses rainbows "the Roman roads" coffee)
for w in 0 1 2; do
    for k in 0 1 2; do
        i=$((w * 3 + k))
        chat "new$((i + 1))" 96 "Name three facts about ${TOPICS[$i]}." &
    done
    wait $(jobs -p | grep -v "^$PID$") 2>/dev/null
    for k in 0 1 2; do
        judge "new$((w * 3 + k + 1))" 8
    done
done
echo "served $SERVED of 11, $FAILS failure(s)"
if [ "$FAILS" -ne 0 ]; then
    grep -E "inkling2_loop|ERROR" "$TMP/log" | tail -20
    exit 1
fi
exit 0
