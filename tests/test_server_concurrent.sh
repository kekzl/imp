#!/bin/bash
# MANUAL tool — not wired into ctest/CI/verify.sh (TEST_AUDIT (retired) §7).
# Needs a running imp-server on :8080; fires 4 concurrent chat requests.
set -e

BASE="http://localhost:8080/v1/chat/completions"

# Fire 4 concurrent requests with different prompts
for i in 1 2 3 4; do
    curl -s "$BASE" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"imp\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from $i to $((i+5))\"}],\"max_tokens\":32,\"temperature\":0}" \
        -o "/tmp/imp_concurrent_$i.json" &
done

wait

# Verify all got valid responses
failures=0
for i in 1 2 3 4; do
    f="/tmp/imp_concurrent_$i.json"
    if [ ! -s "$f" ]; then
        echo "FAIL: request $i got empty response"
        failures=$((failures + 1))
        continue
    fi
    content=$(jq -r '.choices[0].message.content // empty' "$f" 2>/dev/null)
    if [ -z "$content" ]; then
        echo "FAIL: request $i got no content: $(cat "$f")"
        failures=$((failures + 1))
        continue
    fi
    echo "OK: request $i: ${content:0:80}..."
done

if [ $failures -gt 0 ]; then
    echo "FAIL: $failures/4 requests failed"
    exit 1
fi
echo "PASS: all 4 concurrent requests completed successfully"
