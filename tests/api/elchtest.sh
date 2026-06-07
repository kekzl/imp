#!/bin/bash
# MANUAL tool — not wired into ctest/CI/verify.sh (TEST_AUDIT.md §7).
# Needs a running imp-server on :8080 with the model below loaded.
MODEL="Qwen3-8B-Q8_0.gguf"
URL="http://localhost:8080/v1/chat/completions"
PASS=0; FAIL=0; TOTAL=0; FAILS=""

t() {
  local name="$1" category="$2"
  TOTAL=$((TOTAL+1))
  shift 2
  local ok="$1"
  if [ "$ok" = "1" ]; then
    PASS=$((PASS+1))
    printf "  %-3d %-8s %-45s PASS\n" "$TOTAL" "$category" "$name"
  else
    FAIL=$((FAIL+1))
    printf "  %-3d %-8s %-45s FAIL\n" "$TOTAL" "$category" "$name"
    FAILS="$FAILS\n  #$TOTAL $name"
  fi
}

# Use -s (silent) without -f (fail) so we get the response body for non-2xx
req() { curl -s --max-time "${2:-30}" "$URL" -H "Content-Type: application/json" -d "$1" 2>/dev/null; }
req_status() { curl -s -o /dev/null -w '%{http_code}' --max-time "${2:-10}" "$URL" -H "Content-Type: application/json" -d "$1" 2>/dev/null; }
health() { curl -s --max-time 5 http://localhost:8080"$1" 2>/dev/null; }

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ELCHTEST (moose test) — imp-server stability & correctness    ║"
echo "║  Model: Qwen3-8B-Q8_0    GPU: RTX 5090                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════
echo "── 1. ENDPOINTS ──"
# ═══════════════════════════════════════════════════════════

r=$(health "/health")
t "GET /health → 200 + status ok" "endpoint" "$(echo "$r" | grep -q '"status":"ok"' && echo 1 || echo 0)"

r=$(health "/v1/models")
t "GET /v1/models → list with data[]" "endpoint" "$(echo "$r" | grep -q '"object":"list"' && echo 1 || echo 0)"

r=$(health "/metrics")
t "GET /metrics → Prometheus format" "endpoint" "$(echo "$r" | grep -q 'imp_requests_total' && echo 1 || echo 0)"

# ═══════════════════════════════════════════════════════════
echo ""
echo "── 2. ERROR HANDLING ──"
# ═══════════════════════════════════════════════════════════

sc=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$URL" -H "Content-Type: application/json" -d 'not json{{{' 2>/dev/null)
t "Malformed JSON → 400" "error" "$([ "$sc" = "400" ] && echo 1 || echo 0)"

sc=$(req_status '{"model":"nonexistent.gguf","messages":[{"role":"user","content":"hi"}]}')
t "Unknown model → 404" "error" "$([ "$sc" = "404" ] && echo 1 || echo 0)"

sc=$(req_status '{"model":"'$MODEL'","messages":"not array"}')
t "messages not an array → 400" "error" "$([ "$sc" = "400" ] && echo 1 || echo 0)"

sc=$(req_status '{"model":"'$MODEL'","messages":[]}')
t "Empty messages[] → 400" "error" "$([ "$sc" = "400" ] && echo 1 || echo 0)"

sc=$(req_status '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"temperature":5}')
t "temperature=5 → 400" "error" "$([ "$sc" = "400" ] && echo 1 || echo 0)"

sc=$(req_status '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":0}')
t "max_tokens=0 → 400" "error" "$([ "$sc" = "400" ] && echo 1 || echo 0)"

sc=$(req_status '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"n":3}')
t "n=3 → 400 (not supported)" "error" "$([ "$sc" = "400" ] && echo 1 || echo 0)"

r=$(health "/health")
t "Server alive after 7 error requests" "error" "$(echo "$r" | grep -q '"status":"ok"' && echo 1 || echo 0)"

# ═══════════════════════════════════════════════════════════
echo ""
echo "── 3. CORRECTNESS ──"
# ═══════════════════════════════════════════════════════════

r=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"temperature":0,"max_tokens":30}' 30)
c=$(echo "$r" | jq -r '.choices[0].message.content // ""')
t "Greedy 2+2 → contains '4'" "correct" "$(echo "$c" | grep -q '4' && echo 1 || echo 0)"

r=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":"Capital of France? One word."}],"temperature":0,"max_tokens":30}' 30)
c=$(echo "$r" | jq -r '.choices[0].message.content // ""')
t "Greedy capital of France → Paris" "correct" "$(echo "$c" | grep -qi 'paris' && echo 1 || echo 0)"

# Determinism
p='{"model":"'$MODEL'","messages":[{"role":"user","content":"Say hello"}],"temperature":0,"max_tokens":20,"seed":42}'
c1=$(req "$p" 30 | jq -r '.choices[0].message.content // ""')
c2=$(req "$p" 30 | jq -r '.choices[0].message.content // ""')
t "Seed=42 temp=0 → deterministic" "correct" "$([ "$c1" = "$c2" ] && echo 1 || echo 0)"

# JSON output
r=$(req '{"model":"'$MODEL'","messages":[{"role":"system","content":"Reply JSON: {\"answer\":\"...\"}"},{"role":"user","content":"Capital of Germany?"}],"temperature":0.7,"max_tokens":100}' 30)
c=$(echo "$r" | jq -r '.choices[0].message.content // ""')
t "JSON mode → valid JSON" "correct" "$(echo "$c" | jq . >/dev/null 2>&1 && echo 1 || echo 0)"

# Usage correct
r=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0}' 15)
pt=$(echo "$r" | jq -r '.usage.prompt_tokens // 0')
ct=$(echo "$r" | jq -r '.usage.completion_tokens // 0')
tt=$(echo "$r" | jq -r '.usage.total_tokens // 0')
t "Usage: total = prompt + completion" "correct" "$([ "$tt" -eq "$((pt + ct))" ] && echo 1 || echo 0)"

# max_tokens respected
r=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":"Count 1 to 1000"}],"max_tokens":3,"temperature":0}' 15)
ct=$(echo "$r" | jq -r '.usage.completion_tokens // 0')
t "max_tokens=3 respected (≤5 tok)" "correct" "$([ "$ct" -le 5 ] && echo 1 || echo 0)"

# ═══════════════════════════════════════════════════════════
echo ""
echo "── 4. STREAMING ──"
# ═══════════════════════════════════════════════════════════

raw=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10,"temperature":0,"stream":true}' 15)
t "Stream: contains chat.completion.chunk" "stream" "$(echo "$raw" | grep -q 'chat.completion.chunk' && echo 1 || echo 0)"
t "Stream: [DONE] sentinel" "stream" "$(echo "$raw" | grep -q 'data: \[DONE\]' && echo 1 || echo 0)"
t "Stream: finish_reason present" "stream" "$(echo "$raw" | grep -q 'finish_reason' && echo 1 || echo 0)"

# Stream vs non-stream: assemble stream chunks via jq, compare trimmed
p_base='{"model":"'$MODEL'","messages":[{"role":"user","content":"What is 1+1? Number only."}],"max_tokens":5,"temperature":0,"think_budget":0'
ns=$(req "${p_base}}" 15 | jq -r '.choices[0].message.content // ""' | sed 's/^[[:space:]]*//')
ss_raw=$(req "${p_base},\"stream\":true}" 15)
ss=$(echo "$ss_raw" | grep '^data: {' | sed 's/^data: //' | jq -r '.choices[0].delta.content // empty' 2>/dev/null | tr -d '\n' | sed 's/^[[:space:]]*//')
t "Stream content ≈ non-stream (trimmed)" "stream" "$([ "$ns" = "$ss" ] && echo 1 || echo 0)"

# ═══════════════════════════════════════════════════════════
echo ""
echo "── 5. CONCURRENCY ──"
# ═══════════════════════════════════════════════════════════

pids=""
for i in $(seq 1 8); do
  curl -s --max-time 60 "$URL" -H "Content-Type: application/json" \
    -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"What is '$i'*'$i'?"}],"max_tokens":10,"temperature":0}' \
    -o /tmp/conc_$i.json 2>/dev/null &
  pids="$pids $!"
done
wait $pids 2>/dev/null
all_ok=1
for i in $(seq 1 8); do
  if ! jq -e '.choices[0].message.content' /tmp/conc_$i.json >/dev/null 2>&1; then all_ok=0; fi
done
t "8 concurrent requests → all with content" "concur" "$all_ok"

r=$(health "/health")
t "Server stable after 8-concurrent" "concur" "$(echo "$r" | grep -q '"status":"ok"' && echo 1 || echo 0)"

# ═══════════════════════════════════════════════════════════
echo ""
echo "── 6. THINK BUDGET ──"
# ═══════════════════════════════════════════════════════════

r=$(req '{"model":"'$MODEL'","messages":[{"role":"system","content":"Write code."},{"role":"user","content":"Merge sort Python"}],"temperature":0.7,"max_tokens":4000,"think_budget":0.3}' 120)
rt=$(echo "$r" | jq -r '.usage.completion_tokens_details.reasoning_tokens // 0')
ct_total=$(echo "$r" | jq -r '.usage.completion_tokens // 0')
content_tok=$((ct_total - rt))
t "think_budget=0.3 → reasoning ≤1300" "think" "$([ "$rt" -le 1300 ] && echo 1 || echo 0)"
t "think_budget=0.3 → content > 100 tok" "think" "$([ "$content_tok" -gt 100 ] && echo 1 || echo 0)"

r=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":20,"think_budget":0}' 15)
rt=$(echo "$r" | jq -r '.usage.completion_tokens_details.reasoning_tokens // 0')
t "think_budget=0 → reasoning ≤5" "think" "$([ "$rt" -le 5 ] && echo 1 || echo 0)"

# ═══════════════════════════════════════════════════════════
echo ""
echo "── 7. EDGE CASES ──"
# ═══════════════════════════════════════════════════════════

r=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":""}],"max_tokens":5,"temperature":0}' 15)
t "Empty user content → no crash" "edge" "$(echo "$r" | jq -e '.choices[0]' >/dev/null 2>&1 && echo 1 || echo 0)"

longtext=$(printf 'word %.0s' $(seq 1 600))
r=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":"Summarize: '"$longtext"'"}],"max_tokens":20,"temperature":0}' 30)
t "Long prompt (600 words) → response" "edge" "$(echo "$r" | jq -e '.choices[0].message.content' >/dev/null 2>&1 && echo 1 || echo 0)"

# Prompt intentionally German ("What does 🎉🚀💻 mean?") to exercise unicode/emoji + multilingual input
r=$(req '{"model":"'$MODEL'","messages":[{"role":"user","content":"Was bedeutet 🎉🚀💻?"}],"max_tokens":30,"temperature":0}' 30)
t "Unicode/emoji in prompt → no crash" "edge" "$(echo "$r" | jq -e '.choices[0]' >/dev/null 2>&1 && echo 1 || echo 0)"

rapid_ok=1
for i in $(seq 1 20); do
  r=$(curl -s --max-time 10 "$URL" -H "Content-Type: application/json" \
    -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"'$i'"}],"max_tokens":1,"temperature":0}' 2>/dev/null)
  echo "$r" | jq -e '.choices[0]' >/dev/null 2>&1 || rapid_ok=0
done
t "20 rapid-fire requests → all OK" "edge" "$rapid_ok"

timeout 1 curl -s "$URL" -H "Content-Type: application/json" \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"Count 1 to 1000"}],"max_tokens":500,"stream":true}' >/dev/null 2>&1 || true
sleep 0.5
r=$(health "/health")
t "Client disconnect mid-stream → Server OK" "edge" "$(echo "$r" | grep -q '"status":"ok"' && echo 1 || echo 0)"

# ═══════════════════════════════════════════════════════════
echo ""
echo "── 8. PERFORMANCE ──"
# ═══════════════════════════════════════════════════════════

req '{"model":"'$MODEL'","messages":[{"role":"user","content":"warmup"}],"max_tokens":16,"temperature":0}' 15 > /dev/null

t0=$(date +%s%N)
curl -s --max-time 10 "$URL" -H "Content-Type: application/json" \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"Hi"}],"max_tokens":1,"temperature":0}' > /dev/null
t1=$(date +%s%N)
ttft_ms=$(( (t1 - t0) / 1000000 ))
t "TTFT < 500ms (got ${ttft_ms}ms)" "perf" "$([ $ttft_ms -lt 500 ] && echo 1 || echo 0)"

t0=$(date +%s%N)
req '{"model":"'$MODEL'","messages":[{"role":"user","content":"Count 1 to 500"}],"max_tokens":128,"temperature":0,"ignore_eos":true}' 30 > /dev/null
t1=$(date +%s%N)
dec_ms=$(( (t1 - t0) / 1000000 ))
tps=$(( 128000 / dec_ms ))
t "Decode ≥200 tok/s (got ${tps})" "perf" "$([ $tps -ge 200 ] && echo 1 || echo 0)"

# ═══════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
printf "║  RESULT: %d/%d PASS  (%d FAIL)                                 ║\n" "$PASS" "$TOTAL" "$FAIL"
echo "╠══════════════════════════════════════════════════════════════════╣"
if [ $FAIL -eq 0 ]; then
echo "║  ✓ ELCHTEST PASSED                                             ║"
else
echo "║  ✗ ELCHTEST FAILED                                             ║"
printf "║  Failures:%-54s║\n" ""
printf "$FAILS\n" | while read line; do [ -n "$line" ] && printf "║    %-60s║\n" "$line"; done
fi
echo "╚══════════════════════════════════════════════════════════════════╝"
