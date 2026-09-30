#!/usr/bin/env bash
# GPU acceptance for #2298: --gpu-layers N < n_layers streams dense layers from host.
#   C1 Qwen3-8B-Q8_0 --gpu-layers 20 loads, logs "16/36 layers offloaded", upload consumes
#      >= 14 x 195 MiB less VRAM than --gpu-layers -1 (engine line "upload consumed N MiB").
#   C2 greedy 64 tokens identical to --gpu-layers -1 on 3 prompts. Both arms: --no-nvfp4
#      --set runtime.deterministic_gemm=true (NVFP4 decode cache is lossy and covers resident layers only).
#   C3 decode tok/s of both arms (informational, no threshold).
#   C4 GPU test modules (test-core .. test-e2e) exit 0.
# Usage: bash scripts/accept_2298.sh
# Env: IMP_MODELS_DIR (~/models), IMP_ACCEPT_GGUF (Qwen3-8B-Q8_0.gguf), IMP_TEST_IMG,
#      IMP_ACCEPT_TIMEOUT (s per run, default 900).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Hold the card for the whole run (scripts/gpu_lock.sh); refuse a busy one.
if [ -z "${IMP_ACCEPT_2298_LOCKED:-}" ]; then
    BUSY="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
    if [ -x "$BUSY" ] && ! "$BUSY"; then
        echo "FAIL setup: GPU busy ($BUSY), ask the owner before running"
        exit 1
    fi
    bash scripts/require_free_gpu.sh "accept_2298" || exit 1
    IMP_ACCEPT_2298_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2298" -- bash "$0" "$@"
fi

MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
GGUF="${IMP_ACCEPT_GGUF:-Qwen3-8B-Q8_0.gguf}"
IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
TMO="${IMP_ACCEPT_TIMEOUT:-900}"
LAYER_MIB=195
MIN_SAVED_MIB=$((14 * LAYER_MIB))
FLAGS=(--no-nvfp4 --set runtime.deterministic_gemm=true)
PROMPTS=("The capital of France is" "def fibonacci(n):" "Explain photosynthesis in one sentence.")

for tool in docker awk timeout; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
[ -f "$MODELS_DIR/$GGUF" ] || { echo "FAIL setup: $MODELS_DIR/$GGUF missing"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

LOGS="$(mktemp -d "${TMPDIR:-/tmp}/accept_2298.XXXXXX")"
echo "logs: $LOGS"
declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL|INFO> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

NRUN=0
run_img() {  # run_img <log> <cmd...>: one container, killed after $TMO s; returns the command's exit
    local log="$1" name rc
    shift
    NRUN=$((NRUN + 1))
    name="accept2298_$$_$NRUN"
    timeout "$TMO" docker run --rm --name "$name" --gpus all -v "$MODELS_DIR":/models "$IMG" "$@" >"$log" 2>&1
    rc=$?
    if [ "$rc" = 124 ]; then
        docker kill "$name" >/dev/null 2>&1
        echo "TIMEOUT after ${TMO}s" >>"$log"
    fi
    return "$rc"
}

cli() {  # cli <log> <gpu_layers> <prompt> <max_tokens> [flags...]
    local log="$1" gl="$2" prompt="$3" n="$4"
    shift 4
    run_img "$log" imp-cli --model "/models/$GGUF" --gpu-layers "$gl" --prompt "$prompt" --chat-template none \
        --max-tokens "$n" --temperature 0 --token-trace "$@"
}

consumed_mib() { grep -oE 'upload consumed [0-9]+ MiB' "$1" | tail -1 | awk '{print $3}'; }
tok_ids() { grep -oE '\[tok=[0-9]+' "$1" | cut -d= -f2 | tr '\n' ' '; }
tg_tps() { grep -E '^tg ' "$1" | tail -1 | grep -oE '\( *[0-9.]+ tok/s' | tr -d '( ' | sed 's/tok\/s//'; }

# ---- C1: loads with 16/36 layers offloaded, VRAM saved ----
cli "$LOGS/c1_gl20.log" 20 "${PROMPTS[0]}" 8
rc20=$?
cli "$LOGS/c1_all.log" -1 "${PROMPTS[0]}" 8
rcall=$?
off_line=$(grep -oE 'Layer offloading: 16/36 layers offloaded[^,]*' "$LOGS/c1_gl20.log" | head -1)
keep_line=$(grep -oE -- '--gpu-layers: [0-9]+ matmul weights \([0-9.]+ MiB\) stay on host, [0-9]+ pinned, [0-9]+ pageable' \
    "$LOGS/c1_gl20.log" | head -1)
c20=$(consumed_mib "$LOGS/c1_gl20.log")
call=$(consumed_mib "$LOGS/c1_all.log")
if [ "$rc20" = 0 ] && [ -n "$off_line" ]; then
    verdict C1-loads-16-of-36-offloaded PASS "$off_line; $keep_line"
else
    verdict C1-loads-16-of-36-offloaded FAIL "exit $rc20, line '$off_line'; $(grep -E 'ERROR|error' "$LOGS/c1_gl20.log" | head -3 | tr '\n' ' ')"
fi
if [ "$rcall" = 0 ] && [ -n "$c20" ] && [ -n "$call" ] && [ $((call - c20)) -ge "$MIN_SAVED_MIB" ]; then
    verdict C1-vram-saved PASS "upload consumed -1: $call MiB, 20: $c20 MiB, saved $((call - c20)) MiB (need >= $MIN_SAVED_MIB)"
else
    verdict C1-vram-saved FAIL "exit -1:$rcall 20:$rc20; upload consumed -1: '${call}' MiB, 20: '${c20}' MiB (need saved >= $MIN_SAVED_MIB)"
fi

# ---- C2: greedy 64 tokens identical; C3 from the same runs ----
match=0
detail=""
declare -a TPS_ALL=() TPS_20=()
for i in "${!PROMPTS[@]}"; do
    cli "$LOGS/c2_all_$i.log" -1 "${PROMPTS[$i]}" 64 "${FLAGS[@]}"
    ra=$?
    cli "$LOGS/c2_gl20_$i.log" 20 "${PROMPTS[$i]}" 64 "${FLAGS[@]}"
    rb=$?
    a=$(tok_ids "$LOGS/c2_all_$i.log")
    b=$(tok_ids "$LOGS/c2_gl20_$i.log")
    na=$(wc -w <<<"$a")
    nb=$(wc -w <<<"$b")
    TPS_ALL+=("$(tg_tps "$LOGS/c2_all_$i.log")")
    TPS_20+=("$(tg_tps "$LOGS/c2_gl20_$i.log")")
    if [ "$ra" = 0 ] && [ "$rb" = 0 ] && [ "$na" -gt 0 ] && [ "$a" = "$b" ]; then
        match=$((match + 1))
        detail+="p$i equal ($na tokens); "
    else
        first_diff=$(paste -d' ' <(tr ' ' '\n' <<<"$a") <(tr ' ' '\n' <<<"$b") | awk '$1 != $2 {print NR; exit}')
        detail+="p$i exit $ra/$rb, $na vs $nb tokens, first diff at ${first_diff:-none}; "
    fi
done
if [ "$match" = "${#PROMPTS[@]}" ]; then
    verdict C2-greedy-64-identical PASS "$match/${#PROMPTS[@]}: $detail"
else
    verdict C2-greedy-64-identical FAIL "$match/${#PROMPTS[@]}: $detail"
fi
verdict C3-decode-tps INFO "tg tok/s -1: ${TPS_ALL[*]}; --gpu-layers 20: ${TPS_20[*]}"

# ---- C4: GPU test modules ----
fails=""
for m in test-core test-text test-compute test-attention test-quant test-kv test-moe-gdn test-e2e; do
    run_img "$LOGS/c4_$m.log" "$m"
    rc=$?
    [ "$rc" = 0 ] || fails+="$m(exit $rc: $(grep -E '^\[  FAILED  \]' "$LOGS/c4_$m.log" | head -2 | tr '\n' ' ')) "
done
if [ -z "$fails" ]; then
    verdict C4-gpu-test-modules PASS "8/8 modules exit 0"
else
    verdict C4-gpu-test-modules FAIL "$fails"
fi

echo "----"
printf '%s\n' "${RESULTS[@]}"
if printf '%s\n' "${RESULTS[@]}" | grep -q '^FAIL '; then
    echo "ACCEPT_2298 FAIL (logs: $LOGS)"
    exit 1
fi
echo "ACCEPT_2298 PASS (logs: $LOGS)"
exit 0
