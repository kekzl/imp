#!/usr/bin/env bash
# Roadmap row 46: decode tok/s of JSON-schema, tool-call and prose turns per speculation arm, one client.
# Usage: spec_request_class.sh [model]; IMG, MODELS_DIR
set -uo pipefail
MODEL="${1:-Qwen3.8-27B-NVFP4-vllm}"
REQ_JSON='{"messages":[{"role":"user","content":"List 12 fictional employees with name, age, department, salary and a short bio."}],
 "max_tokens":700,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},
 "response_format":{"type":"json_schema","json_schema":{"name":"staff","schema":{"type":"object","properties":{"employees":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"department":{"type":"string"},"salary":{"type":"integer"},"bio":{"type":"string"}},"required":["name","age","department","salary","bio"]}}},"required":["employees"]}}}}'
REQ_TOOL='{"messages":[{"role":"user","content":"Create calendar events for every weekday next week at 9:00 titled Standup, at 13:00 titled Lunch review, and at 17:00 titled Wrap-up. Use the tool for each event."}],
 "max_tokens":700,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},
 "tools":[{"type":"function","function":{"name":"create_event","description":"Create a calendar event","parameters":{"type":"object","properties":{"title":{"type":"string"},"date":{"type":"string"},"time":{"type":"string"}},"required":["title","date","time"]}}}]}'
REQ_PROSE='{"messages":[{"role":"user","content":"Write a 500-word essay about the history of bridges."}],
 "max_tokens":700,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
for arm in "" "--set speculative.mtp_k=0" "--set speculative.mtp_k=2"; do
  bash "$(dirname "$0")/../../scripts/require_free_gpu.sh" >/dev/null || { echo BUSY; exit 3; }
  docker rm -f sp46 >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --init --name sp46 --gpus all --network host -v "${MODELS_DIR:-$HOME/models}:/models:ro" "${IMG:-imp:test}" \
    imp-server --host 127.0.0.1 --port 8106 --model "/models/$MODEL" $arm >/dev/null
  for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8106/health >/dev/null 2>&1 && break; sleep 1; done
  M=$(curl -s http://127.0.0.1:8106/v1/models | jq -r '.data[0].id')
  for kind in JSON TOOL PROSE; do
    var="REQ_$kind"; body=$(jq -c --arg m "$M" '. + {model:$m}' <<<"${!var}")
    for i in 1 2; do
      t0=$(date +%s%N)
      r=$(curl -s -m 600 http://127.0.0.1:8106/v1/chat/completions -H 'Content-Type: application/json' -d "$body")
      t1=$(date +%s%N)
      n=$(jq '.usage.completion_tokens' <<<"$r")
      echo "arm='${arm:-default}' $kind run$i: $n tok in $(( (t1 - t0) / 1000000 )) ms = $(awk -v n="$n" -v ms=$(( (t1 - t0) / 1000000 )) 'BEGIN{printf "%.1f", n*1000/ms}') tok/s finish=$(jq -r '.choices[0].finish_reason' <<<"$r")"
    done
  done
  docker logs sp46 2>&1 | grep -i -E "mtp.*(engag|declin|auto|off)|speculat.*(off|disabled|declin)" | sort -u | cut -c30-200 | head -4
  docker rm -f sp46 >/dev/null 2>&1
done
