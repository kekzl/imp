#!/usr/bin/env bash
# GPU acceptance for #2298: --gpu-layers N < n_layers streams dense layers from host.
#   C1  Qwen3-8B-Q8_0 --gpu-layers 20 loads, logs "16/36 layers offloaded", upload consumes
#       >= 14 x 195 MiB less VRAM than --gpu-layers -1 (engine line "upload consumed N MiB").
#   C2  greedy tokens bit-identical, >= 64 on each of 3 prompts, -1 vs 20, same GEMM route per layer
#       in both arms (AB flags below): no NVFP4 decode cache, no dense FP16/FP8/fused caches
#       (gemm.dense_weight_cache=false), no n-gram speculation (off under offload), no CUDA graphs.
#   C2b teacher-forced mean NLL, default flags, -1 vs 20, over prompt + the -1 arm's 64 tokens:
#       |delta| <= 0.002 nats/token. INFO: p0 default-flags greedy first diff + top-2 logit margin there.
#   C3  decode tok/s, default flags (informational).
#   C4  GPU test modules (test-core .. test-e2e) exit 0.
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
NTOK=64
NLL_TOL=0.002
AB=(--no-nvfp4 --set runtime.deterministic_gemm=true --set gemm.dense_weight_cache=false
    --set speculative.ngram=false --set runtime.cuda_graphs=never)
PROMPTS=("The capital of France is" "def fibonacci(n):" "Explain photosynthesis in one sentence.")

for tool in docker awk timeout jq od; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
[ -f "$MODELS_DIR/$GGUF" ] || { echo "FAIL setup: $MODELS_DIR/$GGUF missing"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

LOGS="$(mktemp -d "${TMPDIR:-/tmp}/accept_2298.XXXXXX")"
chmod 777 "$LOGS"  # the container writes logit dumps here
echo "logs: $LOGS"
declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL|INFO> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

NRUN=0
run_img() {  # run_img <stdout> <stderr> <cmd...>: one container, killed after $TMO s
    local out="$1" err="$2" name rc
    shift 2
    NRUN=$((NRUN + 1))
    name="accept2298_$$_$NRUN"
    timeout "$TMO" docker run --rm --name "$name" --gpus all -v "$MODELS_DIR":/models -v "$LOGS":/logs \
        "$IMG" "$@" >"$out" 2>"$err"
    rc=$?
    if [ "$rc" = 124 ]; then
        docker kill "$name" >/dev/null 2>&1
        echo "TIMEOUT after ${TMO}s" >>"$err"
    fi
    return "$rc"
}

cli() {  # cli <tag> <gpu_layers> <prompt> <max_tokens> [flags...]: $LOGS/<tag>.out (text), .log (stderr)
    local tag="$1" gl="$2" prompt="$3" n="$4"
    shift 4
    run_img "$LOGS/$tag.out" "$LOGS/$tag.log" imp-cli --model "/models/$GGUF" --gpu-layers "$gl" \
        --prompt "$prompt" --chat-template none --max-tokens "$n" --temperature 0 --token-trace "$@"
}

consumed_mib() { grep -oE 'upload consumed [0-9]+ MiB' "$1" | tail -1 | awk '{print $3}'; }
tok_ids() { grep -oE '\[tok=[0-9]+' "$1" | cut -d= -f2 | tr '\n' ' '; }
tg_tps() { grep -E '^tg ' "$1" | tail -1 | grep -oE '\( *[0-9.]+ tok/s' | tr -d '( ' | sed 's/tok\/s//'; }
first_diff() {  # 1-based position of the first differing token id, empty when equal
    paste -d' ' <(tr ' ' '\n' <<<"$1") <(tr ' ' '\n' <<<"$2") | awk '$1 != $2 {print NR; exit}'
}
top2() {  # top2 <npy, 1 row FP32>: "argmax top1 top2 margin"
    local hl off
    hl=$(od -An -tu2 -j8 -N2 "$1" | tr -d ' ')
    off=$((10 + hl))
    od -An -v -tf4 -j"$off" "$1" | awk '{ for (f = 1; f <= NF; f++) { v = $f + 0; i++;
        if (i == 1 || v > a) { b = a; a = v; ia = i - 1 } else if (i == 2 || v > b) b = v } }
        END { printf "%d %.6f %.6f %.6f\n", ia, a, b, a - b }'
}

# ---- C1: loads with 16/36 layers offloaded, VRAM saved ----
cli c1_gl20 20 "${PROMPTS[0]}" 8
rc20=$?
cli c1_all -1 "${PROMPTS[0]}" 8
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

# ---- C2: same route per layer in both arms -> bit-identical greedy tokens ----
match=0
detail=""
for i in "${!PROMPTS[@]}"; do
    cli "c2_all_$i" -1 "${PROMPTS[$i]}" "$NTOK" "${AB[@]}"
    ra=$?
    cli "c2_gl20_$i" 20 "${PROMPTS[$i]}" "$NTOK" "${AB[@]}"
    rb=$?
    a=$(tok_ids "$LOGS/c2_all_$i.log")
    b=$(tok_ids "$LOGS/c2_gl20_$i.log")
    na=$(wc -w <<<"$a")
    nb=$(wc -w <<<"$b")
    skipped=$(grep -c 'pre_dequant phases 1-2 skipped' "$LOGS/c2_all_$i.log")
    if [ "$ra" = 0 ] && [ "$rb" = 0 ] && [ "$na" -ge "$NTOK" ] && [ "$a" = "$b" ] && [ "$skipped" -ge 1 ]; then
        match=$((match + 1))
        detail+="p$i equal ($na/$nb tokens); "
    else
        detail+="p$i exit $ra/$rb, $na vs $nb tokens, caches-skipped log $skipped, first diff at $(first_diff "$a" "$b"); "
    fi
done
if [ "$match" = "${#PROMPTS[@]}" ]; then
    verdict C2-greedy-identical-same-route PASS "$match/${#PROMPTS[@]}: $detail"
else
    verdict C2-greedy-identical-same-route FAIL "$match/${#PROMPTS[@]}: $detail"
fi

# ---- default flags: greedy runs (C3 tok/s, C2b text, p0 divergence) ----
declare -a TPS_ALL=() TPS_20=()
for i in "${!PROMPTS[@]}"; do
    cli "def_all_$i" -1 "${PROMPTS[$i]}" "$NTOK"
    cli "def_gl20_$i" 20 "${PROMPTS[$i]}" "$NTOK"
    TPS_ALL+=("$(tg_tps "$LOGS/def_all_$i.log")")
    TPS_20+=("$(tg_tps "$LOGS/def_gl20_$i.log")")
done
verdict C3-decode-tps INFO "default flags, tg tok/s -1: ${TPS_ALL[*]}; --gpu-layers 20: ${TPS_20[*]}"

# ---- C2b: teacher-forced mean NLL over prompt + the -1 arm's greedy text ----
nll_sum_all=0
nll_sum_20=0
ntok_sum=0
c2b_ok=1
c2b_detail=""
for i in "${!PROMPTS[@]}"; do
    printf '%s%s' "${PROMPTS[$i]}" "$(cat "$LOGS/def_all_$i.out")" >"$LOGS/c2b_p$i.txt"
    for arm in all gl20; do
        gl=$([ "$arm" = all ] && echo -1 || echo 20)
        run_img "$LOGS/c2b_${arm}_$i.json" "$LOGS/c2b_${arm}_$i.log" imp-cli --model "/models/$GGUF" \
            --gpu-layers "$gl" --perplexity "/logs/c2b_p$i.txt" --json
    done
    pa=$(jq -r '.perplexity // empty' "$LOGS/c2b_all_$i.json" 2>/dev/null)
    pb=$(jq -r '.perplexity // empty' "$LOGS/c2b_gl20_$i.json" 2>/dev/null)
    ta=$(jq -r '.tokens // empty' "$LOGS/c2b_all_$i.json" 2>/dev/null)
    if [ -z "$pa" ] || [ -z "$pb" ] || [ -z "$ta" ]; then
        c2b_ok=0
        c2b_detail+="p$i missing ppl ('$pa' '$pb' tokens '$ta'); "
        continue
    fi
    nll_sum_all=$(awk -v s="$nll_sum_all" -v p="$pa" -v n="$ta" 'BEGIN { printf "%.10f", s + log(p) * n }')
    nll_sum_20=$(awk -v s="$nll_sum_20" -v p="$pb" -v n="$ta" 'BEGIN { printf "%.10f", s + log(p) * n }')
    ntok_sum=$((ntok_sum + ta))
    c2b_detail+="p$i ppl $pa vs $pb ($ta tokens); "
done
if [ "$c2b_ok" = 1 ] && [ "$ntok_sum" -gt 0 ] &&
    res=$(awk -v a="$nll_sum_all" -v b="$nll_sum_20" -v n="$ntok_sum" -v tol="$NLL_TOL" 'BEGIN {
        ma = a / n; mb = b / n; d = mb - ma; ad = d < 0 ? -d : d
        printf "mean NLL -1 %.6f, 20 %.6f, delta %+.6f nats/token (tol %s)", ma, mb, d, tol; exit !(ad <= tol) }'); then
    verdict C2b-teacher-forced-nll PASS "$res; $c2b_detail"
else
    verdict C2b-teacher-forced-nll FAIL "${res:-no result}; $c2b_detail"
fi

# ---- INFO: p0 default-flags divergence and the -1 arm's top-2 logit margin there ----
a=$(tok_ids "$LOGS/def_all_0.log")
b=$(tok_ids "$LOGS/def_gl20_0.log")
k=$(first_diff "$a" "$b")
if [ -z "$k" ]; then
    verdict C2b-p0-margin INFO "default flags p0: -1 and 20 identical over $(wc -w <<<"$a") tokens"
else
    mkdir -p "$LOGS/dump_p0" && chmod 777 "$LOGS/dump_p0"
    cli margin_p0 -1 "${PROMPTS[0]}" "$k" --set speculative.ngram=false --set runtime.cuda_graphs=never \
        --set diagnostics.dump_final_logits_dir=/logs/dump_p0
    mapfile -t dumps < <(find "$LOGS/dump_p0" -name 'imp_logits_pass*.npy' | sort)
    pre=-1
    for j in "${!dumps[@]}"; do
        n_in=$(basename "${dumps[$j]}" | sed -E 's/.*_in([0-9]+)_n.*/\1/')
        [ "$n_in" -gt 1 ] && pre=$j
    done
    idx=$((pre + k - 1))
    ta=$(awk -v k="$k" '{print $k}' <<<"$a")
    tb=$(awk -v k="$k" '{print $k}' <<<"$b")
    if [ "$pre" -ge 0 ] && [ "$idx" -lt "${#dumps[@]}" ]; then
        read -r am t1 t2 mg <<<"$(top2 "${dumps[$idx]}")"
        verdict C2b-p0-margin INFO "default flags p0 first diff at token $k (-1: $ta, 20: $tb); -1 arm logits there: argmax $am, top1 $t1, top2 $t2, margin $mg"
    else
        verdict C2b-p0-margin INFO "default flags p0 first diff at token $k (-1: $ta, 20: $tb); logit dump missing (${#dumps[@]} passes, prefill index $pre)"
    fi
fi

# ---- C4: GPU test modules ----
fails=""
for m in test-core test-text test-compute test-attention test-quant test-kv test-moe-gdn test-e2e; do
    run_img "$LOGS/c4_$m.log" "$LOGS/c4_$m.err" "$m"
    rc=$?
    [ "$rc" = 0 ] || fails+="$m(exit $rc: $(grep -hE '^\[  FAILED  \]' "$LOGS/c4_$m.log" "$LOGS/c4_$m.err" | head -2 | tr '\n' ' ')) "
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
