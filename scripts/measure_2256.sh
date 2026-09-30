#!/usr/bin/env bash
# Measurement for #2256: INT8-activation IMMA prefill (gemm.q8_imma_enabled) vs HF fp32, no PASS/FAIL.
# A = default (IMMA on), B = --set gemm.q8_imma_enabled=false. Same token ids to HF and imp.
# Reference: tools/analysis/imma_kl.py ref (CPU, cached). Exit 0 = every measurement completed.
# Usage: make build && bash scripts/measure_2256.sh
# Env: IMP_MODELS_DIR (~/models), IMP_MEASURE_GGUF (Qwen3-8B-Q8_0.gguf), IMP_MEASURE_N (2048),
#      IMP_MEASURE_CORPUS (PPL KPI corpus, rebuilt by tools/analysis/make_ppl_corpus.sh),
#      IMP_MEASURE_CACHE (~/.cache/imp/measure_2256), IMP_MEASURE_ROUNDS (5), IMP_MEASURE_PORT (8256),
#      IMP_TEST_IMG, IMP_MEASURE_HF_IMG / IMP_MEASURE_PY_IMG (python:3.12-slim),
#      IMP_MEASURE_REF_ONLY=1 (build the CPU reference, no GPU), IMP_GPU_BUSY_CHECK (busy-check script).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
GGUF="${IMP_MEASURE_GGUF:-Qwen3-8B-Q8_0.gguf}"
N="${IMP_MEASURE_N:-2048}"
CACHE="${IMP_MEASURE_CACHE:-$HOME/.cache/imp/measure_2256}"
CORPUS="${IMP_MEASURE_CORPUS:-$CACHE/ppl_corpus_45k.txt}"
ROUNDS="${IMP_MEASURE_ROUNDS:-5}"
HF_IMG="${IMP_MEASURE_HF_IMG:-python:3.12-slim}"
PY_IMG="${IMP_MEASURE_PY_IMG:-python:3.12-slim}"
REF="$CACHE/hf_ref_${N}.json"
TOPN=20  # prompt_logprobs maximum (docs/API.md)

for tool in docker curl jq; do
    command -v "$tool" >/dev/null || { echo "ERROR setup: $tool not on PATH"; exit 1; }
done
[ -f "$MODELS_DIR/$GGUF" ] || { echo "ERROR setup: $MODELS_DIR/$GGUF missing"; exit 1; }
mkdir -p "$CACHE"

# ---- phase 0 (CPU, before the GPU lock): HF fp32 reference, cached ----
if [ ! -s "$REF" ]; then
    [ -f "$CORPUS" ] || bash tools/analysis/make_ppl_corpus.sh "$CORPUS" ||
        { echo "ERROR setup: corpus $CORPUS"; exit 1; }
    echo "HF reference: transformers fp32 on CPU, $GGUF dequantized, $N tokens (~40 GB RAM)"
    docker run --rm -v "$MODELS_DIR":/models:ro -v "$CACHE":/out -v "$(dirname "$CORPUS")":/corpus:ro \
        -v "$ROOT/tools/analysis":/t:ro "$HF_IMG" bash -c '
        set -e
        pip install -q --root-user-action=ignore torch --index-url https://download.pytorch.org/whl/cpu >/dev/null
        pip install -q --root-user-action=ignore transformers accelerate gguf numpy sentencepiece >/dev/null
        python /t/imma_kl.py ref --gguf "$0" --corpus "/corpus/$1" --n "$2" --out "/out/$3"' \
        "$GGUF" "$(basename "$CORPUS")" "$N" "$(basename "$REF")" ||
        { echo "ERROR setup: HF reference run failed"; exit 1; }
fi
[ -n "${IMP_MEASURE_REF_ONLY:-}" ] && { jq -c '{gguf, corpus_sha256, n, topk, vocab, torch, transformers,
    load_s, forward_s, first_targets: .target_logprobs[:3]}' "$REF"; exit 0; }

pyrun() { docker run --rm -v "$ROOT/tools/analysis":/t:ro -v "$CACHE":/cache:ro "$@"; }
pyrun "$PY_IMG" python /t/imma_kl.py selftest >/dev/null || { echo "ERROR setup: imma_kl.py selftest"; exit 1; }

if [ -z "${IMP_MEASURE_2256_LOCKED:-}" ]; then
    BUSY="${IMP_GPU_BUSY_CHECK:-$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh}"
    if [ -x "$BUSY" ]; then "$BUSY" || { echo "ERROR setup: GPU busy ($BUSY)"; exit 1; }; fi
    bash scripts/require_free_gpu.sh "measure_2256" || exit 1
    IMP_MEASURE_2256_LOCKED=1 exec bash scripts/gpu_lock.sh run "measure_2256" -- bash "$0" "$@"
fi

IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_MEASURE_PORT:-8256}"
CTR=imp_measure_2256
BASE="http://127.0.0.1:$PORT"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "ERROR setup: $IMG is not this tree (make build)"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/measure_2256.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

ERRORS=0
err() { echo "ERROR $*"; ERRORS=$((ERRORS + 1)); }

# Prefix cache off: warm-up and timed requests share ids; prompt_logprobs bypasses it anyway.
start_server() {  # start_server <A|B>
    local -a extra=()
    [ "$1" = B ] && extra=(--set gemm.q8_imma_enabled=false)
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT":"$PORT" "$IMG" \
        imp-server --model "/models/$GGUF" --host 0.0.0.0 --port "$PORT" --max-batch 1 \
        --set server.prefix_cache=false "${extra[@]}" >/dev/null || return 1
    for _ in $(seq 1 180); do
        curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && break
        docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || break
        sleep 1
    done
    curl -s "$BASE/health" | grep -q '"model_loaded":true' || { docker logs "$CTR" 2>&1 | tail -20; return 1; }
    MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '.data[] | select(.loaded == true) | .id' | head -1)
}

complete() {  # complete <json body file> <out-file> -> prints "<http_code> <seconds>"
    local body
    body=$(jq -c --arg m "$MODEL_ID" '. + {model: $m}' "$1")
    curl -s -o "$2" -w '%{http_code} %{time_total}' -H 'Content-Type: application/json' -d "$body" \
        "$BASE/v1/completions"
}

# Request bodies: token ids from the reference, so HF and imp see identical ids.
for len in 512 "$N"; do
    jq -c --argjson k "$len" '{prompt: .ids[:$k], max_tokens: 1, temperature: 0}' "$REF" >"$WORK/req_$len.json"
done
jq -c --argjson t "$TOPN" '{prompt: .ids, prompt_logprobs: $t, max_tokens: 1, temperature: 0}' "$REF" \
    >"$WORK/req_lp.json"

# Rounds alternate the arm order (AB, BA, ...); per round and arm: 1 warm-up per length, 1 timed.
measure_arm() {  # measure_arm <A|B> <round>
    local arm="$1" r="$2" len code t np
    start_server "$arm" || { err "round $r arm $arm: server did not load"; return; }
    for len in 512 "$N"; do
        read -r code _ < <(complete "$WORK/req_$len.json" "$WORK/warm.json")
        [ "$code" = 200 ] || { err "round $r arm $arm warm-up $len: HTTP $code"; continue; }
        read -r code t < <(complete "$WORK/req_$len.json" "$WORK/timed.json")
        [ "$code" = 200 ] || { err "round $r arm $arm timed $len: HTTP $code"; continue; }
        echo "$t" >>"$WORK/t_${arm}_$len"
    done
    if [ "$r" = 1 ]; then
        read -r code _ < <(complete "$WORK/req_lp.json" "$WORK/lp_$arm.json")
        np=$(jq -r '.usage.prompt_tokens // "none"' "$WORK/lp_$arm.json" 2>/dev/null)
        [ "$code" = 200 ] && [ "$np" = "$N" ] ||
            err "arm $arm prompt_logprobs: HTTP $code, prompt_tokens $np (want $N): $(head -c 300 "$WORK/lp_$arm.json")"
    fi
    echo "round $r arm $arm: 512 $(tail -1 "$WORK/t_${arm}_512" 2>/dev/null) s, $N $(tail -1 "$WORK/t_${arm}_$N" 2>/dev/null) s"
    docker rm -f "$CTR" >/dev/null 2>&1 || true
}

for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) = 1 ]; then order="A B"; else order="B A"; fi
    for arm in $order; do measure_arm "$arm" "$r"; done
done

median() {  # median <file> -> median of its lines, "ERR" if fewer than ROUNDS
    [ -f "$1" ] && [ "$(wc -l <"$1")" = "$ROUNDS" ] || { echo ERR; return; }
    sort -g "$1" | sed -n "$(((ROUNDS + 1) / 2))p"
}

STATS=""
if [ -s "$WORK/lp_A.json" ] && [ -s "$WORK/lp_B.json" ]; then
    cp "$WORK/lp_A.json" "$WORK/lp_B.json" "$CACHE/"
    STATS=$(pyrun -v "$WORK":/w:ro "$PY_IMG" python /t/imma_kl.py stats --ref "/cache/$(basename "$REF")" \
        --arm A=/w/lp_A.json --arm B=/w/lp_B.json) || { err "imma_kl.py stats failed"; STATS=""; }
fi

echo "== measure_2256: $GGUF, $N ids (corpus sha256 $(jq -r .corpus_sha256 "$REF" | cut -c1-12)), image $IMG =="
echo "   ref HF fp32 $(jq -r '"torch \(.torch), transformers \(.transformers), top-\(.topk) of vocab \(.vocab)"' "$REF")"
echo "   KL(HF||imp) on S = HF top-$(jq -r .topk "$REF") ids imp reports (target + top-$TOPN), both renormalized over S"
row() { printf '%-34s %-42s %s\n' "$@"; }
row metric "A (IMMA on, default)" "B (q8_imma_enabled=false)"
if [ -n "$STATS" ]; then
    TABLE=$(jq -r '
        def r($k): . * $k | round / $k;
        def four: "\(.mean | r(1e5)) / \(.p50 | r(1e5)) / \(.p99 | r(1e5)) / \(.max | r(1e5))";
        def row($m; f): [$m, (.A | f | tostring), (.B | f | tostring)];
        row("KL mean / p50 / p99 / max"; .kl | four),
        row("|dlogp target| mean/p50/p99/max"; .absdiff | four),
        row("top-1 agreement %"; .top1_pct | r(100)),
        row("PPL imp / PPL HF"; "\(.ppl_imp | r(1e4)) / \(.ppl_hf | r(1e4))"),
        row("S HF mass mean / min"; "\(.cov_mean | r(1e4)) / \(.cov_min | r(1e4))"),
        row("scored positions"; .n)
        | @tsv' <<<"$STATS") || err "summary: jq over stats failed: $STATS"
    while IFS=$'\t' read -r m a b; do row "$m" "$a" "$b"; done <<<"$TABLE"
else
    err "no logprob statistics"
fi
for len in 512 "$N"; do
    a=$(median "$WORK/t_A_$len"); b=$(median "$WORK/t_B_$len")
    [ "$a" = ERR ] || [ "$b" = ERR ] && { err "prefill $len: fewer than $ROUNDS timed requests"; continue; }
    read -r a b d < <(jq -rn --argjson a "$a" --argjson b "$b" \
        '"\($a * 1e4 | round / 1e4) \($b * 1e4 | round / 1e4) \(($b / $a - 1) * 1000 | round / 10)"')
    row "prefill $len s, median of $ROUNDS" "$a" "$b (B/A $d %)"
done
echo "   prefill = curl time_total, max_tokens 1, no prompt_logprobs, rounds alternate AB/BA"
[ "$ERRORS" = 0 ] && { echo "ALL MEASUREMENTS COMPLETED"; exit 0; }
echo "$ERRORS measurement errors"
exit 1
