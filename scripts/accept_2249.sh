#!/usr/bin/env bash
# GPU acceptance for #2249: a GPTQ 4-bit checkpoint (hf://) scores PPL within 1 % of an independent
# CPU reference (tools/analysis/gptq_ref_ppl.py, AutoGPTQ dequant in transformers) on the same tokens
# and the same greedy first token as its original; the dequant kernel is bit-equal to the host
# reference; a marlin checkpoint_format stays refused. Needs a GPU, network, the worktree image (make build).
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

within_1pct() { awk -v a="$1" -v r="$2" 'BEGIN { d = (a / r - 1) * 100; printf "%.3f", d; exit !(d <= 1.0 && d >= -1.0) }'; }

# ---- C2: GPTQ checkpoint loads through the dequant path and scores; token ids kept for C3 ----
run_img "$LOGS/gptq_ppl.log" imp-cli --model "hf://$REPO" --perplexity /models/corpus/ppl_corpus_45k.txt \
    "${FLAGS[@]}" --set diagnostics.dump_tokens=true
rc=$?
gptq_ppl=$(ppl_of "$LOGS/gptq_ppl.log")
proj=$(grep -oE 'GPTQ 4-bit: [0-9]+ projections, [^,]*, zero offset \+[01]' "$LOGS/gptq_ppl.log" | head -1)
grep -E '^TOK [0-9]+ [0-9]+$' "$LOGS/gptq_ppl.log" | awk '{print $3}' >"$DIR/ids_imp.txt"
n_ids=$(wc -l <"$DIR/ids_imp.txt")
if [ "$rc" = 0 ] && [ -n "$gptq_ppl" ] && [ -n "$proj" ] && [ "$n_ids" -gt 1 ]; then
    verdict C2-gptq-loads-and-scores PASS "$REPO: $proj, PPL $gptq_ppl, $n_ids tokens"
else
    verdict C2-gptq-loads-and-scores FAIL "$REPO: exit $rc, '$proj', PPL '$gptq_ppl', $n_ids tokens; $(tail -3 "$LOGS/gptq_ppl.log" | tr '\n' ' ')"
fi

# ---- C3: imp PPL within 1 % of an independent CPU reference on the same token ids ----
# tools/analysis/gptq_ref_ppl.py dequantizes as AutoGPTQ does, runs transformers FP32, and models imp's
# default LM head (per-row FP8, gemm.nvfp4_lm_head=auto). C3-harness checks that model on the original.
run_img "$LOGS/orig_ppl.log" imp-cli --model "hf://$REF" --perplexity /models/corpus/ppl_corpus_45k.txt "${FLAGS[@]}"
orig_ppl=$(ppl_of "$LOGS/orig_ppl.log")
docker build -q --target base -t imp-gptqref:base -f tools/analysis/Dockerfile.gptqref tools/analysis \
    >"$LOGS/refimg.log" 2>&1 || { echo "FAIL setup: Dockerfile.gptqref"; tail -5 "$LOGS/refimg.log"; exit 1; }
ref_py() {  # ref_py <log> <args...>: CPU only, no --gpus
    local log="$1"
    shift
    docker run --rm "${USER_ARGS[@]}" -e HOME=/tmp -v "$ROOT/tools/analysis":/t:ro -v "$DIR":/models \
        imp-gptqref:base python /t/gptq_ref_ppl.py --ids /models/ids_imp.txt --head fp8 "$@" >"$log" 2>&1
    grep -oE 'ref_ppl .* ppl=[0-9.]+' "$log" | sed 's/.*ppl=//'
}
snap_of() { find "$DIR/hub/models--${1//\//--}/snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | head -1; }
gptq_snap="/models/hub/models--${REPO//\//--}/snapshots/$(snap_of "$REPO")"
orig_snap="/models/hub/models--${REF//\//--}/snapshots/$(snap_of "$REF")"
ref_gptq=$(ref_py "$LOGS/ref_gptq.log" --arm dequant --model "$gptq_snap")
ref_orig=$(ref_py "$LOGS/ref_orig.log" --arm orig --model "$orig_snap")
for arm in gptq orig; do
    imp_v=$([ "$arm" = gptq ] && echo "$gptq_ppl" || echo "$orig_ppl")
    ref_v=$([ "$arm" = gptq ] && echo "$ref_gptq" || echo "$ref_orig")
    id=$([ "$arm" = gptq ] && echo C3-ppl-within-1pct-of-reference || echo C3-harness-original-within-1pct)
    if [ -z "$imp_v" ] || [ -z "$ref_v" ]; then
        verdict "$id" FAIL "missing PPL: imp '$imp_v' reference '$ref_v'; $(tail -2 "$LOGS/ref_$arm.log" | tr '\n' ' ')"
    elif d=$(within_1pct "$imp_v" "$ref_v"); then
        verdict "$id" PASS "imp $imp_v vs reference $ref_v: $d %"
    else
        verdict "$id" FAIL "imp $imp_v vs reference $ref_v: $d %"
    fi
done
# Int4 cost on this model, stated, not a verdict.
[ -n "$gptq_ppl" ] && [ -n "$orig_ppl" ] &&
    echo "INFO int4-cost: imp GPTQ $gptq_ppl vs imp original $orig_ppl: $(within_1pct "$gptq_ppl" "$orig_ppl") %"

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
