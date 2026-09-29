#!/usr/bin/env bash
# GPU acceptance for #2249: a GPTQ 4-bit checkpoint (hf://) serves with PPL within 1 % of its original
# and the same greedy first token; the dequant kernel is bit-equal to the host reference; a marlin
# checkpoint_format stays refused. Needs a GPU, network, the worktree image (make build).
# Usage: bash scripts/accept_2249.sh
# Env: IMP_ACCEPT_REPO (default Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4), IMP_ACCEPT_REF (default
#      Qwen/Qwen2.5-0.5B-Instruct), IMP_ACCEPT_DIR (default ~/models/.accept_2249), IMP_TEST_IMG.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Hold the card for the whole run (scripts/gpu_lock.sh); refuse a busy one.
if [ -z "${IMP_ACCEPT_2249_LOCKED:-}" ]; then
    bash scripts/require_free_gpu.sh "accept_2249" || exit 1
    IMP_ACCEPT_2249_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2249" -- bash "$0" "$@"
fi

REPO="${IMP_ACCEPT_REPO:-Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4}"
REF="${IMP_ACCEPT_REF:-Qwen/Qwen2.5-0.5B-Instruct}"
DIR="${IMP_ACCEPT_DIR:-$HOME/models/.accept_2249}"
IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
USER_ARGS=(--user "$(id -u):$(id -g)")
# Same flags on both arms: FP16 weights, no runtime NVFP4 cache, deterministic GEMM.
FLAGS=(--no-nvfp4 --set runtime.deterministic_gemm=true)
PROMPTS=("The capital of France is" "def fibonacci(n):" "Explain photosynthesis in one sentence.")

for tool in docker jq awk; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

case "$DIR" in */.accept_2249) ;; *) echo "FAIL setup: IMP_ACCEPT_DIR must end in /.accept_2249"; exit 1 ;; esac
mkdir -p "$DIR/corpus"
LOGS="$(mktemp -d "${TMPDIR:-/tmp}/accept_2249.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { rm -rf "$LOGS" "$DIR/variant_marlin"; }
trap cleanup EXIT

# The repo's PPL KPI corpus (docs/quantization.md), rebuilt byte-identical from git history.
CORPUS="$DIR/corpus/ppl_corpus_45k.txt"
bash tools/analysis/make_ppl_corpus.sh "$CORPUS" >/dev/null || { echo "FAIL setup: make_ppl_corpus.sh"; exit 1; }

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

run_img() {  # run_img <log> <cmd...>; HF cache under /models/hub
    local log="$1"
    shift
    docker run --rm --gpus all "${USER_ARGS[@]}" -v "$DIR":/models -e HF_HOME=/models -e HF_TOKEN \
        "$IMG" "$@" >"$log" 2>&1
}

ppl_of() { grep -E '^perplexity: ' "$1" | tail -1 | awk '{print $2}'; }
first_tok() { grep -oE '\[tok=[0-9]+' "$1" | head -1 | cut -d= -f2; }

# ---- C1: dequant kernel bit-equal to the host reference (v1, v2, with and without g_idx) ----
run_img "$LOGS/gtest.log" test-quant --gtest_filter='DequantGptqGpu.*'
rc=$?
if [ "$rc" = 0 ] && grep -q '^\[  PASSED  \] 1 test' "$LOGS/gtest.log"; then
    verdict C1-kernel-equals-reference PASS "$(grep -E '^\[       OK \]' "$LOGS/gtest.log" | head -1)"
else
    verdict C1-kernel-equals-reference FAIL "exit $rc; $(grep -E 'FAILED|mismatches|Skipped' "$LOGS/gtest.log" | head -3 | tr '\n' ' ')"
fi

# ---- C2: GPTQ checkpoint loads through the dequant path and scores ----
run_img "$LOGS/gptq_ppl.log" imp-cli --model "hf://$REPO" --perplexity /models/corpus/ppl_corpus_45k.txt "${FLAGS[@]}"
rc=$?
gptq_ppl=$(ppl_of "$LOGS/gptq_ppl.log")
proj=$(grep -oE 'GPTQ 4-bit: [0-9]+ projections, [^,]*, zero offset \+[01]' "$LOGS/gptq_ppl.log" | head -1)
if [ "$rc" = 0 ] && [ -n "$gptq_ppl" ] && [ -n "$proj" ]; then
    verdict C2-gptq-loads-and-scores PASS "$REPO: $proj, PPL $gptq_ppl"
else
    verdict C2-gptq-loads-and-scores FAIL "$REPO: exit $rc, '$proj', PPL '$gptq_ppl'; $(tail -3 "$LOGS/gptq_ppl.log" | tr '\n' ' ')"
fi

# ---- C3: PPL within 1 % of the original on the same corpus, same binary, same flags ----
run_img "$LOGS/ref_ppl.log" imp-cli --model "hf://$REF" --perplexity /models/corpus/ppl_corpus_45k.txt "${FLAGS[@]}"
ref_ppl=$(ppl_of "$LOGS/ref_ppl.log")
if [ -n "$gptq_ppl" ] && [ -n "$ref_ppl" ]; then
    delta=$(awk -v a="$gptq_ppl" -v r="$ref_ppl" 'BEGIN { printf "%.3f", (a / r - 1) * 100 }')
    if awk -v d="$delta" 'BEGIN { exit !(d <= 1.0 && d >= -1.0) }'; then
        verdict C3-ppl-within-1pct PASS "GPTQ $gptq_ppl vs $REF $ref_ppl: ${delta} %"
    else
        verdict C3-ppl-within-1pct FAIL "GPTQ $gptq_ppl vs $REF $ref_ppl: ${delta} %"
    fi
else
    verdict C3-ppl-within-1pct FAIL "missing PPL: GPTQ '$gptq_ppl' ref '$ref_ppl'"
fi

# ---- C4: greedy first token (raw completion, no chat template) equals the original's ----
match=0
detail=""
for i in "${!PROMPTS[@]}"; do
    run_img "$LOGS/gptq_tok_$i.log" imp-cli --model "hf://$REPO" --prompt "${PROMPTS[$i]}" --chat-template none --max-tokens 1 --temperature 0 "${FLAGS[@]}"
    run_img "$LOGS/ref_tok_$i.log" imp-cli --model "hf://$REF" --prompt "${PROMPTS[$i]}" --chat-template none --max-tokens 1 --temperature 0 "${FLAGS[@]}"
    a=$(first_tok "$LOGS/gptq_tok_$i.log")
    r=$(first_tok "$LOGS/ref_tok_$i.log")
    detail+="p$i gptq=${a:-none} ref=${r:-none}; "
    [ -n "$a" ] && [ "$a" = "$r" ] && match=$((match + 1))
done
if [ "$match" = "${#PROMPTS[@]}" ]; then
    verdict C4-first-token-top1 PASS "$match/${#PROMPTS[@]}: $detail"
else
    verdict C4-first-token-top1 FAIL "$match/${#PROMPTS[@]}: $detail"
fi

# ---- C5: checkpoint_format marlin on the same weights is refused, not dequantized as v1 ----
snap=$(find "$DIR/hub/models--${REPO//\//--}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
vd="$DIR/variant_marlin"
rm -rf "$vd"
mkdir -p "$vd"
for f in "$snap"/*; do
    ln -s "/models/hub/models--${REPO//\//--}/snapshots/$(basename "$snap")/$(basename "$f")" "$vd/$(basename "$f")"
done
rm -f "$vd/config.json"
jq '.quantization_config.checkpoint_format = "marlin"' "$snap/config.json" >"$vd/config.json"
run_img "$LOGS/variant_marlin.log" imp-cli --model /models/variant_marlin --prompt "hi" --max-tokens 1 "${FLAGS[@]}"
rc=$?
if [ "$rc" != 0 ] && grep -q "GPTQ SafeTensors detected" "$LOGS/variant_marlin.log" &&
    grep -q "checkpoint_format=marlin" "$LOGS/variant_marlin.log"; then
    verdict C5-refused-marlin PASS "exit $rc, log names checkpoint_format=marlin"
else
    verdict C5-refused-marlin FAIL "exit $rc; $(grep -m1 GPTQ "$LOGS/variant_marlin.log" || tail -2 "$LOGS/variant_marlin.log" | tr '\n' ' ')"
fi

echo "---- summary ($REPO vs $REF, image $IMG) ----"
printf '%s\n' "${RESULTS[@]}"
for r in "${RESULTS[@]}"; do
    case "$r" in FAIL*) exit 1 ;; esac
done
exit 0
