#!/usr/bin/env bash
# One command per new checkpoint: download, HF BF16 reference, imp PPL, tokenizer parity, chat smoke.
# Usage: scripts/check_model.sh <hf-repo> [--nvfp4] [--bos] [--skip-hf] [--image IMG]
#   --nvfp4    imp-quantize the BF16 checkpoint to <name>-NVFP4-imp first (needed above ~28 GB BF16)
#   --bos      prepend BOS in the HF reference (as imp-cli --perplexity does for add_bos tokenizers)
#   --skip-hf  no CPU reference PPL (checkpoint larger than host RAM)
#   --image    imp image (default imp:test, `make build` tag)
# Env: MODELS_DIR (default ~/models), CHECK_OUT (default ~/imp-models-check/<name>).
# CPU steps run in Docker (python:3.12-slim, hfref:cpu); the GPU block holds scripts/gpu_lock.sh and
# refuses a busy card (scripts/require_free_gpu.sh). Exit 0 = every row ok, 1 = a row failed.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

REPO=""
NVFP4=0
BOS=0
SKIP_HF=0
IMG=imp:test
while [ $# -gt 0 ]; do
    case "$1" in
    --nvfp4) NVFP4=1 ;;
    --bos) BOS=1 ;;
    --skip-hf) SKIP_HF=1 ;;
    --image) IMG=$2; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) REPO=$1 ;;
    esac
    shift
done
[ -n "$REPO" ] || { sed -n '3,9p' "$0"; exit 2; }

MODELS_DIR=${MODELS_DIR:-$HOME/models}
NAME=$(basename "$REPO")
OUT=${CHECK_OUT:-$HOME/imp-models-check/$NAME}
CORPUS=tools/analysis/ppl_corpus_45k.txt
mkdir -p "$OUT"
SERVE=$NAME
[ $NVFP4 = 1 ] && SERVE=$NAME-NVFP4-imp
fail=0
row() {
    printf '| %s | %s | %s |\n' "$1" "$2" "$3" >> "$OUT/summary.txt"
    [ "$2" != FAIL ] || fail=1
}
# The KPI corpus is gitignored and rebuilt byte-identical from history.
[ -s "$CORPUS" ] || tools/analysis/make_ppl_corpus.sh "$CORPUS" >/dev/null
printf '| step | result | detail |\n|---|---|---|\n' > "$OUT/summary.txt"

# 1. Download (BF16 SafeTensors, no GGUF).
if [ ! -f "$MODELS_DIR/$NAME/config.json" ]; then
    docker run --rm -v "$MODELS_DIR:/models" -e HF_TOKEN="${HF_TOKEN:-}" python:3.12-slim bash -c "
        pip install -q --no-cache-dir 'huggingface_hub[hf_xet]' >/dev/null 2>&1 &&
        hf download '$REPO' --local-dir '/models/$NAME' --exclude '*.gguf' --exclude 'original/*' >/dev/null &&
        chown -R $(id -u):$(id -g) '/models/$NAME'" > "$OUT/download.log" 2>&1
fi
if [ -f "$MODELS_DIR/$NAME/config.json" ]; then
    row download ok "$(du -sh "$MODELS_DIR/$NAME" | cut -f1), $(jq -r '[.architectures[0], .model_type] | join(" / ")' "$MODELS_DIR/$NAME/config.json")"
else
    row download FAIL "$OUT/download.log"
    cat "$OUT/summary.txt"
    exit 1
fi

# 2. HF BF16 reference PPL on the CPU.
if [ $SKIP_HF = 0 ]; then
    docker image inspect hfref:cpu >/dev/null 2>&1 ||
        docker build -q -t hfref:cpu - <<< $'FROM python:3.12-slim\nRUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir transformers safetensors' >/dev/null
    docker run --rm -e HF_PPL_BOS=$BOS -v "$MODELS_DIR:/models:ro" -v "$PWD/tools/analysis:/a:ro" hfref:cpu \
        python -I /a/hf_ppl_cpu.py "/models/$NAME" /a/ppl_corpus_45k.txt > "$OUT/hf_ppl.log" 2>&1
    HF_PPL=$(sed -n 's/.* PPL \([0-9.]*\).*/\1/p' "$OUT/hf_ppl.log" | tail -1)
    [ -n "$HF_PPL" ] && row "HF BF16 PPL" ok "$HF_PPL (BOS=$BOS)" || row "HF BF16 PPL" FAIL "$OUT/hf_ppl.log"
fi

# 3. GPU block: quantize, imp PPL, server tokenize + chat. One lock for the whole block.
if ! scripts/require_free_gpu.sh "check_model $NAME"; then
    row GPU FAIL "card busy: nothing on the GPU ran"
    cat "$OUT/summary.txt"
    exit 1
fi
gpu_block() {
    if [ $NVFP4 = 1 ] && [ ! -f "$MODELS_DIR/$SERVE/config.json" ]; then
        docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$MODELS_DIR:/models" --entrypoint imp-quantize "$IMG" \
            --model "/models/$NAME" --out "/models/$SERVE" > "$OUT/quantize.log" 2>&1
    fi
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$MODELS_DIR:/models" -v "$PWD/tools/analysis:/a:ro" \
        --entrypoint imp-cli "$IMG" --model "/models/$SERVE" --perplexity /a/ppl_corpus_45k.txt > "$OUT/imp_ppl.log" 2>&1
    docker rm -f imp-check-model >/dev/null 2>&1
    docker run -d --name imp-check-model --gpus all --user "$(id -u):$(id -g)" -v "$MODELS_DIR:/models" \
        -p 127.0.0.1:18091:8080 --entrypoint imp-server "$IMG" --model "/models/$SERVE" --host 0.0.0.0 --port 8080 >/dev/null
    local url=http://127.0.0.1:18091 id=""
    for _ in $(seq 1 150); do
        id=$(curl -sf $url/v1/models 2>/dev/null | jq -r '.data[0].id' 2>/dev/null)
        [ -n "$id" ] && break
        docker inspect -f '{{.State.Running}}' imp-check-model 2>/dev/null | grep -q true || break
        sleep 2
    done
    if [ -n "$id" ]; then
        jq -Rs '{content: ., add_special: false}' "$CORPUS" |
            curl -s $url/tokenize -H 'content-type: application/json' -d @- > "$OUT/imp_tokenize.json"
        : > "$OUT/chat.jsonl"
        local p
        for p in plain:"What is 17 * 23? Answer briefly." plain:"Explain in two sentences why the sky is blue." \
                 json:"Return a JSON object with keys city and population for Berlin."; do
            jq -nc --arg id "$id" --arg k "${p%%:*}" --arg q "${p#*:}" \
                '{model:$id, messages:[{role:"user",content:$q}], max_tokens:1500, temperature:0.7}
                 + (if $k == "json" then {response_format:{type:"json_object"}} else {} end)' |
                curl -s $url/v1/chat/completions -H 'content-type: application/json' -d @- |
                jq -c --arg k "${p%%:*}" '{k:$k, c:.choices[0].message.content, r:(.choices[0].message.reasoning_content//""), f:.choices[0].finish_reason, err:.error.message}' \
                >> "$OUT/chat.jsonl"
        done
    fi
    docker logs imp-check-model > "$OUT/server.log" 2>&1
    docker rm -f imp-check-model >/dev/null 2>&1
}
export -f gpu_block
export NVFP4 MODELS_DIR SERVE NAME IMG OUT CORPUS
scripts/gpu_lock.sh run "check_model $NAME" -- bash -c gpu_block

[ $NVFP4 = 1 ] && { [ -f "$MODELS_DIR/$SERVE/config.json" ] && row quantize ok "$(du -sh "$MODELS_DIR/$SERVE" | cut -f1)" || row quantize FAIL "$OUT/quantize.log"; }
IMP_PPL=$(sed -n 's/.*perplexity: \([0-9.]*\) *(\([0-9]*\) tokens).*/\1 (\2 tokens)/p' "$OUT/imp_ppl.log" | tail -1)
[ -n "$IMP_PPL" ] && row "imp PPL" ok "$IMP_PPL" || row "imp PPL" FAIL "$OUT/imp_ppl.log"
if [ -n "${HF_PPL:-}" ] && [ -n "$IMP_PPL" ]; then
    d=$(awk -v a="${IMP_PPL%% *}" -v b="$HF_PPL" 'BEGIN { printf "%+.2f", (a - b) / b * 100 }')
    # BF16 vs BF16 sits within ~2 %; NVFP4 adds a few % on small models: the row reports, the reader judges.
    row "imp vs HF" ok "$d %"
fi

# 4. Tokenizer parity (CPU).
if [ -s "$OUT/imp_tokenize.json" ]; then
    docker run --rm -v "$MODELS_DIR:/models:ro" -v "$OUT:/o:ro" -v "$PWD/tools/analysis:/a:ro" hfref:cpu \
        python -I /a/hf_tokdiff.py "/models/$NAME" /o/imp_tokenize.json /a/ppl_corpus_45k.txt > "$OUT/tokdiff.log" 2>&1 &&
        row tokenizer ok "$(head -1 "$OUT/tokdiff.log"), ids identical" ||
        row tokenizer FAIL "$(grep -m1 'first diff' "$OUT/tokdiff.log"), $OUT/tokdiff.log"
else
    row tokenizer FAIL "server did not answer /tokenize: $OUT/server.log"
fi

# 5. Chat smoke: answers, no reasoning in content, JSON parses.
if [ -s "$OUT/chat.jsonl" ]; then
    jq -rs 'map(select(.err != null)) | length' "$OUT/chat.jsonl" | grep -qx 0 || row chat FAIL "HTTP error: $OUT/chat.jsonl"
    leak=$(jq -rs 'map(select((.c // "") | test("</?think>"))) | length' "$OUT/chat.jsonl")
    [ "$leak" = 0 ] && row "reasoning split" ok "no <think> in content" || row "reasoning split" FAIL "$leak replies carry <think> in content"
    # Model quality, not an imp defect (MiniCPM5-2B answers 161, 2, 42): WARN, never FAIL.
    jq -rs 'map(select(.k == "plain")) | all((.c // "") | test("391"))' "$OUT/chat.jsonl" | grep -qx true &&
        row "17 * 23" ok "391 in content" || row "17 * 23" WARN "wrong answer: $OUT/chat.jsonl"
    jq -rs 'map(select(.k == "json")) | all(.c as $c | try ($c | fromjson | type == "object") catch false)' "$OUT/chat.jsonl" |
        grep -qx true && row json_object ok "parses" || row json_object FAIL "$OUT/chat.jsonl"
else
    row chat FAIL "no replies: $OUT/server.log"
fi

cat "$OUT/summary.txt"
exit $fail
