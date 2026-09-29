#!/usr/bin/env bash
# GPU acceptance for #2207: prompt_logprobs and echo + logprobs on /v1/completions.
# Reference: HF transformers fp32 on CPU over the SAME Qwen3-8B-Q8_0 GGUF, dequantized via
# `gguf_file=` (no 16-bit Qwen3-8B checkpoint in ~/models; method of #2166), plus imp-cli --perplexity.
# Prints PASS/FAIL per criterion (INFO for the throughput numbers), exit 0 only if all pass.
# Usage: make build && bash scripts/accept_2207.sh
# Env: IMP_MODELS_DIR (~/models), IMP_ACCEPT_GGUF (Qwen3-8B-Q8_0.gguf), IMP_ACCEPT_CORPUS (ppl_4k.txt),
#      IMP_TEST_IMG, IMP_ACCEPT_PORT (8207), IMP_ACCEPT_CACHE (HF reference cache dir),
#      IMP_ACCEPT_HF_IMG (python:3.12-slim), IMP_ACCEPT_REF_ONLY=1 (build the CPU reference, no GPU).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
GGUF="${IMP_ACCEPT_GGUF:-Qwen3-8B-Q8_0.gguf}"
CORPUS="${IMP_ACCEPT_CORPUS:-ppl_4k.txt}"
CACHE="${IMP_ACCEPT_CACHE:-${TMPDIR:-/tmp}/accept_2207_cache}"
HF_IMG="${IMP_ACCEPT_HF_IMG:-python:3.12-slim}"
HF_REF="$CACHE/hf_ref.json"
# 32-token prompt for the HF comparison: the first 32 ids of this text.
REF_TEXT="The history of the printing press begins in the fifteenth century, when Johannes Gutenberg \
combined movable metal type, oil-based ink and a wooden screw press into a system that could produce \
books far faster than scribes."

for tool in docker curl jq; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
[ -f "$MODELS_DIR/$GGUF" ] || { echo "FAIL setup: $MODELS_DIR/$GGUF missing"; exit 1; }
[ -f "$MODELS_DIR/$CORPUS" ] || { echo "FAIL setup: $MODELS_DIR/$CORPUS missing"; exit 1; }

# ---- phase 0 (CPU, before the GPU lock): HF fp32 reference, cached ----
hf_reference() {
    mkdir -p "$CACHE"
    [ -s "$HF_REF" ] && { echo "HF reference: cached $HF_REF"; return 0; }
    echo "HF reference: transformers fp32 on CPU, $GGUF dequantized (several minutes, ~40 GB RAM)"
    docker run --rm -i -v "$MODELS_DIR":/models:ro -v "$CACHE":/out -e GGUF="$GGUF" -e REF_TEXT="$REF_TEXT" \
        "$HF_IMG" bash -c '
        set -e
        pip install -q --root-user-action=ignore torch --index-url https://download.pytorch.org/whl/cpu >/dev/null
        pip install -q --root-user-action=ignore transformers accelerate gguf numpy sentencepiece >/dev/null
        python - <<"PY"
import json, os
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
gguf = os.environ["GGUF"]
tok = AutoTokenizer.from_pretrained("/models", gguf_file=gguf)
ids = tok(os.environ["REF_TEXT"], add_special_tokens=False)["input_ids"][:32]
assert len(ids) == 32, len(ids)
model = AutoModelForCausalLM.from_pretrained("/models", gguf_file=gguf, dtype=torch.float32)
model.eval()
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0].float()
lp = torch.log_softmax(logits, dim=-1)
ref = [float(lp[p, ids[p + 1]]) for p in range(len(ids) - 1)]
json.dump({"ids": ids, "logprobs": ref, "torch": torch.__version__,
           "transformers": transformers.__version__}, open("/out/hf_ref.json", "w"))
print("HF reference: %d ids, torch %s, transformers %s" % (len(ids), torch.__version__, transformers.__version__))
PY'
}

if [ -z "${IMP_ACCEPT_2207_LOCKED:-}" ]; then
    hf_reference || { echo "FAIL setup: HF reference run failed"; exit 1; }
    [ -n "${IMP_ACCEPT_REF_ONLY:-}" ] && { jq -c . "$HF_REF"; exit 0; }
    # Hold the card for the whole GPU phase (scripts/gpu_lock.sh); refuse a busy one.
    bash scripts/require_free_gpu.sh "accept_2207" || exit 1
    IMP_ACCEPT_2207_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2207" -- bash "$0" "$@"
fi

IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_ACCEPT_PORT:-8207}"
CTR=imp_accept_2207
BASE="http://127.0.0.1:$PORT"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/accept_2207.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL|INFO> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

# ---- phase 1: imp-cli --perplexity over the corpus, with its exact token stream ----
docker run --rm --gpus all -v "$MODELS_DIR":/models:ro "$IMG" imp-cli --model "/models/$GGUF" \
    --perplexity "/models/$CORPUS" --json --set diagnostics.dump_tokens=true \
    >"$WORK/ppl.json" 2>"$WORK/ppl.err"
PPL_TOOL=$(jq -r '.perplexity // empty' "$WORK/ppl.json" 2>/dev/null)
grep '^TOK ' "$WORK/ppl.err" | awk '{print $3}' | jq -sc . >"$WORK/corpus_ids.json"
N_CORPUS=$(jq length "$WORK/corpus_ids.json")
echo "imp-cli --perplexity: PPL=$PPL_TOOL over $N_CORPUS tokens"

# ---- phase 2: imp-server; prefix cache on (default, pinned) so C6 can prove prompt logprobs bypass it ----
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT":"$PORT" "$IMG" \
    imp-server --model "/models/$GGUF" --host 0.0.0.0 --port "$PORT" --max-batch 1 --set server.prefix_cache=true >/dev/null ||
    { echo "FAIL setup: docker run imp-server"; exit 1; }
for _ in $(seq 1 180); do
    curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && break
    docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || break
    sleep 1
done
curl -s "$BASE/health" | grep -q '"model_loaded":true' || {
    echo "FAIL setup: server did not load"; docker logs "$CTR" 2>&1 | tail -20; exit 1; }
MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '.data[] | select(.loaded == true) | .id' | head -1)

complete() {  # complete <jq object expression> <out-file> -> prints "<http_code> <seconds>"
    local body
    body=$(jq -nc --arg m "$MODEL_ID" "($1) + {model: \$m}")
    curl -s -o "$2" -w '%{http_code} %{time_total}' -H 'Content-Type: application/json' -d "$body" \
        "$BASE/v1/completions"
}

# C1: 32-token prompt vs HF fp32, every scored token within 0.02 nats.
REF_IDS=$(jq -c .ids "$HF_REF")
read -r code _ < <(complete "{\"prompt\": $REF_IDS, \"prompt_logprobs\": 5, \"max_tokens\": 1, \"temperature\": 0}" \
    "$WORK/c1.json")
if [ "$code" = 200 ]; then
    c1=$(jq -r --slurpfile ref "$HF_REF" '
        [.choices[0].prompt_logprobs as $p | $ref[0] as $r | range(1; $r.ids | length) as $i
         | ($p[$i][($r.ids[$i] | tostring)].logprob - $r.logprobs[$i - 1]) | if . < 0 then -. else . end]
        | "\(length) \(max) \(add / length)"' "$WORK/c1.json")
    read -r n1 max1 mean1 <<<"$c1"
    ok=$(jq -n --argjson m "$max1" --argjson n "$n1" '$n == 31 and $m <= 0.02')
    verdict C1 "$([ "$ok" = true ] && echo PASS || echo FAIL)" \
        "vs HF fp32 ($(jq -r '"torch \(.torch), transformers \(.transformers)"' "$HF_REF")): $n1 tokens, max |diff| $max1, mean $mean1 (bound 0.02)"
else
    verdict C1 FAIL "HTTP $code: $(head -c 300 "$WORK/c1.json")"
fi

# C2: PPL from prompt_logprobs over the tool's exact tokens vs imp-cli --perplexity, within 0.1 %.
read -r code _ < <(complete "{\"prompt\": $(cat "$WORK/corpus_ids.json"), \"prompt_logprobs\": 0, \"max_tokens\": 1, \"temperature\": 0}" \
    "$WORK/c2.json")
if [ "$code" = 200 ] && [ -n "$PPL_TOOL" ]; then
    jq -c --slurpfile ids "$WORK/corpus_ids.json" '[.choices[0].prompt_logprobs as $p | range(1; $p | length) as $i
        | $p[$i][($ids[0][$i] | tostring)].logprob]' "$WORK/c2.json" >"$WORK/c2_lp.json"
    ppl_srv=$(jq -r '(add / length) | -. | exp' "$WORK/c2_lp.json")
    n2=$(jq length "$WORK/c2_lp.json")
    rel=$(jq -n --argjson a "$ppl_srv" --argjson b "$PPL_TOOL" '(($a - $b) / $b) | if . < 0 then -. else . end')
    ok=$(jq -n --argjson r "$rel" --argjson n "$n2" --argjson c "$N_CORPUS" '$r <= 0.001 and $n == $c - 1')
    verdict C2 "$([ "$ok" = true ] && echo PASS || echo FAIL)" \
        "PPL prompt_logprobs $ppl_srv vs imp-cli $PPL_TOOL over $n2 scored tokens: rel diff $rel (bound 0.001)"
else
    verdict C2 FAIL "HTTP $code or no tool PPL ('$PPL_TOOL'): $(head -c 300 "$WORK/c2.json")"
fi

# C3: the sampled-prefill path (temperature 0.7, top_k 40) scores the prompt identically to greedy.
read -r code _ < <(complete "{\"prompt\": $REF_IDS, \"prompt_logprobs\": 5, \"max_tokens\": 1, \"temperature\": 0.7, \"seed\": 1}" \
    "$WORK/c3.json")
if [ "$code" = 200 ]; then
    d3=$(jq -rn --slurpfile a "$WORK/c1.json" --slurpfile b "$WORK/c3.json" --slurpfile ref "$HF_REF" '
        [range(1; $ref[0].ids | length) as $i | ($ref[0].ids[$i] | tostring) as $k
         | ($a[0].choices[0].prompt_logprobs[$i][$k].logprob - $b[0].choices[0].prompt_logprobs[$i][$k].logprob)
         | if . < 0 then -. else . end] | max')
    ok=$(jq -n --argjson d "$d3" '$d <= 1e-5')
    verdict C3 "$([ "$ok" = true ] && echo PASS || echo FAIL)" "greedy vs sampled prefill: max |diff| $d3 (bound 1e-5)"
else
    verdict C3 FAIL "HTTP $code: $(head -c 300 "$WORK/c3.json")"
fi

# C4: echo + logprobs carries the same prompt values as prompt_logprobs, first entry null.
read -r code _ < <(complete "{\"prompt\": $REF_IDS, \"echo\": true, \"logprobs\": 2, \"max_tokens\": 4, \"temperature\": 0}" \
    "$WORK/c4.json")
if [ "$code" = 200 ]; then
    c4=$(jq -rn --slurpfile a "$WORK/c1.json" --slurpfile e "$WORK/c4.json" --slurpfile ref "$HF_REF" '
        $e[0].choices[0].logprobs as $l
        | ([range(1; $ref[0].ids | length) as $i
            | ($a[0].choices[0].prompt_logprobs[$i][($ref[0].ids[$i] | tostring)].logprob - $l.token_logprobs[$i])
            | if . < 0 then -. else . end] | max) as $d
        | "\($d) \($l.token_logprobs[0] == null) \($l.tokens | length) \($e[0].usage.prompt_tokens)"')
    read -r d4 null4 ntok4 np4 <<<"$c4"
    ok=$(jq -n --argjson d "$d4" --argjson n "$ntok4" --argjson p "$np4" --argjson z "$null4" \
        '$d <= 1e-5 and $z and $n >= $p and $p == 32')
    verdict C4 "$([ "$ok" = true ] && echo PASS || echo FAIL)" \
        "echo shape: max |diff| to prompt_logprobs $d4, first null $null4, $ntok4 tokens for $np4 prompt tokens"
else
    verdict C4 FAIL "HTTP $code: $(head -c 300 "$WORK/c4.json")"
fi

# C5 (INFO): prefill cost on a 2048-token prompt, median of 5 after 1 warm-up. A unique first
# token per request keeps the prefix cache out of both arms.
BASE_IDS=$(jq -c '.[1:2048]' "$WORK/corpus_ids.json")
time_arm() {  # time_arm <extra-json> -> median seconds
    local i t first
    local -a ts=()
    for i in 0 1 2 3 4 5; do
        first=$((1000 + RANDOM))
        read -r code t < <(complete "{\"prompt\": ([$first] + $BASE_IDS), \"max_tokens\": 1, \"temperature\": 0} + $1" \
            "$WORK/c5.json")
        [ "$code" = 200 ] || { echo "ERR $code"; return; }
        [ "$i" = 0 ] || ts+=("$t")
    done
    printf '%s\n' "${ts[@]}" | sort -n | sed -n 3p
}
t_off=$(time_arm '{}')
t_p0=$(time_arm '{"prompt_logprobs": 0}')
t_p20=$(time_arm '{"prompt_logprobs": 20}')
if [[ "$t_off $t_p0 $t_p20" == *ERR* ]]; then
    verdict C5 FAIL "a timed request failed: off=$t_off p0=$t_p0 p20=$t_p20"
else
    info=$(jq -rn --argjson a "$t_off" --argjson b "$t_p0" --argjson c "$t_p20" \
        '"off \($a) s, prompt_logprobs=0 \($b) s (+\((($b / $a - 1) * 1000 | round) / 10) %), prompt_logprobs=20 \($c) s (+\((($c / $a - 1) * 1000 | round) / 10) %)"')
    verdict C5 INFO "2048-token prefill, median of 5: $info"
fi

# C6: a repeated prompt_logprobs request is not served from the prefix cache (every row forwarded).
read -r code _ < <(complete "{\"prompt\": $REF_IDS, \"prompt_logprobs\": 5, \"max_tokens\": 1, \"temperature\": 0}" \
    "$WORK/c6.json")
if [ "$code" = 200 ]; then
    c6=$(jq -rn --slurpfile a "$WORK/c1.json" --slurpfile b "$WORK/c6.json" '
        "\($b[0].usage.prompt_tokens_details.cached_tokens // 0) \($a[0].choices[0].prompt_logprobs == $b[0].choices[0].prompt_logprobs)"')
    read -r cached6 same6 <<<"$c6"
    ok=$([ "$cached6" = 0 ] && [ "$same6" = true ] && echo PASS || echo FAIL)
    verdict C6 "$ok" "repeat of the C1 request: cached_tokens $cached6, identical prompt_logprobs $same6"
else
    verdict C6 FAIL "HTTP $code: $(head -c 300 "$WORK/c6.json")"
fi

echo "== accept_2207 summary ($GGUF, image $IMG, reference HF fp32 GGUF-dequant + imp-cli --perplexity) =="
printf '%s\n' "${RESULTS[@]}"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')
[ ${#RESULTS[@]} -eq 6 ] || { echo "FAIL: ${#RESULTS[@]}/6 criteria reported"; exit 1; }
[ "$fails" = 0 ] && { echo "ALL PASS"; exit 0; }
echo "$fails FAIL"
exit 1
