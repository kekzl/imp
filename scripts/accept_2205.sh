#!/usr/bin/env bash
# GPU acceptance for #2205: an AWQ 4-bit GEMM checkpoint (hf://) serves with PPL within 1 % of its
# FP16/BF16 original and the same greedy first token; GEMV and 3-bit AWQ configs stay refused.
# Needs a GPU, network, the worktree image (make build). Downloads reuse $IMP_ACCEPT_DIR.
# Usage: bash scripts/accept_2205.sh
# Env: IMP_ACCEPT_REPO (default Qwen/Qwen3-4B-AWQ), IMP_ACCEPT_REF (default Qwen/Qwen3-4B, the BF16
#      original the AWQ export was made from), IMP_ACCEPT_DIR (default ~/models/.accept_2205), IMP_TEST_IMG.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Hold the card for the whole run (scripts/gpu_lock.sh); refuse a busy one.
if [ -z "${IMP_ACCEPT_2205_LOCKED:-}" ]; then
    bash scripts/require_free_gpu.sh "accept_2205" || exit 1
    IMP_ACCEPT_2205_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2205" -- bash "$0" "$@"
fi

REPO="${IMP_ACCEPT_REPO:-Qwen/Qwen3-4B-AWQ}"
REF="${IMP_ACCEPT_REF:-Qwen/Qwen3-4B}"
DIR="${IMP_ACCEPT_DIR:-$HOME/models/.accept_2205}"
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

case "$DIR" in */.accept_2205) ;; *) echo "FAIL setup: IMP_ACCEPT_DIR must end in /.accept_2205"; exit 1 ;; esac
mkdir -p "$DIR/corpus"
LOGS="$(mktemp -d "${TMPDIR:-/tmp}/accept_2205.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { rm -rf "$LOGS" "$DIR/variant_gemv" "$DIR/variant_3bit"; }
trap cleanup EXIT

# The repo's PPL KPI corpus (docs/quantization.md), rebuilt byte-identical from git history.
CORPUS="$DIR/corpus/ppl_corpus_45k.txt"
bash tools/analysis/make_ppl_corpus.sh "$CORPUS" >/dev/null || { echo "FAIL setup: make_ppl_corpus.sh"; exit 1; }

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

cli() {  # cli <log> <imp-cli args...>; HF cache under /models/hub
    local log="$1"
    shift
    docker run --rm --gpus all "${USER_ARGS[@]}" -v "$DIR":/models -e HF_HOME=/models -e HF_TOKEN \
        "$IMG" imp-cli "$@" >"$log" 2>&1
}

ppl_of() { grep -E '^perplexity: ' "$1" | tail -1 | awk '{print $2}'; }
first_tok() { grep -oE '\[tok=[0-9]+' "$1" | head -1 | cut -d= -f2; }

# ---- C1: AWQ checkpoint loads through the dequant path and scores ----
cli "$LOGS/awq_ppl.log" --model "hf://$REPO" --perplexity /models/corpus/ppl_corpus_45k.txt "${FLAGS[@]}"
rc=$?
awq_ppl=$(ppl_of "$LOGS/awq_ppl.log")
proj=$(grep -oE 'AWQ GEMM 4-bit: [0-9]+ projections, group_size=[0-9]+' "$LOGS/awq_ppl.log" | head -1)
if [ "$rc" = 0 ] && [ -n "$awq_ppl" ] && [ -n "$proj" ]; then
    verdict C1-awq-loads-and-scores PASS "$REPO: $proj, PPL $awq_ppl"
else
    verdict C1-awq-loads-and-scores FAIL "$REPO: exit $rc, '$proj', PPL '$awq_ppl'; $(tail -3 "$LOGS/awq_ppl.log" | tr '\n' ' ')"
fi

# ---- C2: PPL within 1 % of the original on the same corpus, same binary, same flags ----
cli "$LOGS/ref_ppl.log" --model "hf://$REF" --perplexity /models/corpus/ppl_corpus_45k.txt "${FLAGS[@]}"
ref_ppl=$(ppl_of "$LOGS/ref_ppl.log")
if [ -n "$awq_ppl" ] && [ -n "$ref_ppl" ]; then
    delta=$(awk -v a="$awq_ppl" -v r="$ref_ppl" 'BEGIN { printf "%.3f", (a / r - 1) * 100 }')
    if awk -v d="$delta" 'BEGIN { exit !(d <= 1.0 && d >= -1.0) }'; then
        verdict C2-ppl-within-1pct PASS "AWQ $awq_ppl vs $REF $ref_ppl: ${delta} %"
    else
        verdict C2-ppl-within-1pct FAIL "AWQ $awq_ppl vs $REF $ref_ppl: ${delta} %"
    fi
else
    verdict C2-ppl-within-1pct FAIL "missing PPL: AWQ '$awq_ppl' ref '$ref_ppl'"
fi

# ---- C3: greedy first token (raw completion, no chat template) equals the original's ----
match=0
detail=""
for i in "${!PROMPTS[@]}"; do
    cli "$LOGS/awq_tok_$i.log" --model "hf://$REPO" --prompt "${PROMPTS[$i]}" --chat-template none --max-tokens 1 --temperature 0 "${FLAGS[@]}"
    cli "$LOGS/ref_tok_$i.log" --model "hf://$REF" --prompt "${PROMPTS[$i]}" --chat-template none --max-tokens 1 --temperature 0 "${FLAGS[@]}"
    a=$(first_tok "$LOGS/awq_tok_$i.log")
    r=$(first_tok "$LOGS/ref_tok_$i.log")
    detail+="p$i awq=${a:-none} ref=${r:-none}; "
    [ -n "$a" ] && [ "$a" = "$r" ] && match=$((match + 1))
done
if [ "$match" = "${#PROMPTS[@]}" ]; then
    verdict C3-first-token-top1 PASS "$match/${#PROMPTS[@]}: $detail"
else
    verdict C3-first-token-top1 FAIL "$match/${#PROMPTS[@]}: $detail"
fi

# ---- C4: GEMV and 3-bit configs on the same weights are refused, not loaded as wire dtype ----
snap=$(find "$DIR/hub/models--${REPO//\//--}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
for v in gemv 3bit; do
    vd="$DIR/variant_$v"
    rm -rf "$vd"
    mkdir -p "$vd"
    for f in "$snap"/*; do
        ln -s "/models/hub/models--${REPO//\//--}/snapshots/$(basename "$snap")/$(basename "$f")" "$vd/$(basename "$f")"
    done
    rm -f "$vd/config.json"
    if [ "$v" = gemv ]; then
        jq '.quantization_config.version = "gemv"' "$snap/config.json" >"$vd/config.json"
        want="bits=4 group_size=128 zero_point=true version=gemv"
    else
        jq '.quantization_config.bits = 3' "$snap/config.json" >"$vd/config.json"
        want="bits=3 group_size=128 zero_point=true version=gemm"
    fi
    cli "$LOGS/variant_$v.log" --model "/models/variant_$v" --prompt "hi" --max-tokens 1 "${FLAGS[@]}"
    rc=$?
    if [ "$rc" != 0 ] && grep -q "AWQ variant not supported" "$LOGS/variant_$v.log" &&
        grep -q "$want" "$LOGS/variant_$v.log"; then
        verdict "C4-refused-$v" PASS "exit $rc, log names '$want'"
    else
        verdict "C4-refused-$v" FAIL "exit $rc; $(grep -m1 AWQ "$LOGS/variant_$v.log" || tail -2 "$LOGS/variant_$v.log" | tr '\n' ' ')"
    fi
done

echo "---- summary ($REPO vs $REF, image $IMG) ----"
printf '%s\n' "${RESULTS[@]}"
for r in "${RESULTS[@]}"; do
    case "$r" in FAIL*) exit 1 ;; esac
done
exit 0
