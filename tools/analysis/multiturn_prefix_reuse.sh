#!/usr/bin/env bash
# Roadmap row 29: multi-turn chat on a hybrid, per turn prompt tokens vs cached tokens and wall.
# Usage: multiturn_prefix_reuse.sh <model> [turns]; IMG, MODELS_DIR
set -uo pipefail
MODEL="$1"; TURNS="${2:-6}"
CORPUS="$(dirname "$0")/ppl_corpus_45k.txt"
bash "$(dirname "$0")/../../scripts/require_free_gpu.sh" >/dev/null || { echo BUSY; exit 3; }
docker rm -f mt29 >/dev/null 2>&1
docker run -d --init --name mt29 --gpus all --network host -v "${MODELS_DIR:-$HOME/models}:/models:ro" "${IMG:-imp:test}" \
  imp-server --host 127.0.0.1 --port 8105 --model "/models/$MODEL" --set speculative.mtp_k=0 >/dev/null
for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8105/health >/dev/null 2>&1 && break; sleep 1; done
M=$(curl -s http://127.0.0.1:8105/v1/models | jq -r '.data[0].id')
msgs='[{"role":"system","content":"You are a concise assistant."}]'
for t in $(seq 1 "$TURNS"); do
  chunk=$(tail -c +$(( (t - 1) * 3000 + 1 )) "$CORPUS" | head -c 3000)
  msgs=$(jq -c --arg c "Read this and answer in two sentences what it is about: $chunk" '. + [{"role":"user","content":$c}]' <<<"$msgs")
  t0=$(date +%s%N)
  resp=$(curl -s -m 600 http://127.0.0.1:8105/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$(jq -n --arg m "$M" --argjson msgs "$msgs" '{model:$m,messages:$msgs,max_tokens:64,temperature:0,chat_template_kwargs:{enable_thinking:false}}')")
  t1=$(date +%s%N)
  ans=$(jq -r '.choices[0].message.content // ""' <<<"$resp")
  echo "turn $t: prompt $(jq '.usage.prompt_tokens' <<<"$resp") cached $(jq '.usage.prompt_tokens_details.cached_tokens' <<<"$resp") wall $(( (t1 - t0) / 1000000 )) ms"
  msgs=$(jq -c --arg a "$ans" '. + [{"role":"assistant","content":$a}]' <<<"$msgs")
done
docker rm -f mt29 >/dev/null 2>&1
