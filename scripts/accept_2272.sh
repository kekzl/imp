#!/usr/bin/env bash
# GPU acceptance for #2272: Qwen3.8-Flash-Next MTP k=1 vs k=0 decode.
# C1 FP8GemmTest.RowscaleRowsBitEqualToPerRowGemv (test-compute) passes.
# C2 MtpGreedyIdentityTest.* (test-e2e, Flash-Next, k=1, chat prompts) passes.
# C3 4 prompts of scripts/mtp_accuracy_bench.sh x RUNS, 128 tokens greedy, raw, arm order AB/BA/AB:
#    mean k=1 tg tok/s >= mean k=0 tg tok/s, OR mean ms/verify <= 1.84 x 1000 / k0_toks
#    (break-even at the measured tok/verify; #2272 table: 29.7 ms at 61.90 tok/s).
# Usage: bash scripts/accept_2272.sh
# Env: IMP_ACCEPT_2272_MODEL (default $HOME/models/Qwen3.8-Flash-Next-NVFP4), RUNS (3),
#      IMP_ACCEPT_2272_IMG (default scripts/image_tag.sh), IMP_ACCEPT_2272_SKIP_BUILD=1,
#      IMP_ACCEPT_2272_TIMEOUT (s per container, default 1200).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Busy check first, then hold the card for the whole run (scripts/gpu_lock.sh).
if [ -z "${IMP_ACCEPT_2272_LOCKED:-}" ]; then
    HOST_CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
    if [ -x "$HOST_CHECK" ]; then
        "$HOST_CHECK" || { echo "FAIL setup: GPU busy ($HOST_CHECK)"; exit 1; }
    fi
    bash scripts/require_free_gpu.sh "accept_2272" || exit 1
    IMP_ACCEPT_2272_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2272" -- bash "$ROOT/scripts/accept_2272.sh" "$@"
fi

for tool in docker awk timeout; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
MODEL="${IMP_ACCEPT_2272_MODEL:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}"
RUNS="${RUNS:-3}"
TMO="${IMP_ACCEPT_2272_TIMEOUT:-1200}"
IMG="${IMP_ACCEPT_2272_IMG:-$(bash scripts/image_tag.sh)}"
[ -d "$MODEL" ] || { echo "FAIL setup: model dir $MODEL missing"; exit 1; }
CMODEL="/models/$(basename "$MODEL")"
ADIR="$(mktemp -d "${TMPDIR:-/tmp}/accept_2272.XXXXXX")"
echo "logs: $ADIR  image: $IMG  model: $MODEL"

if [ -z "${IMP_ACCEPT_2272_SKIP_BUILD:-}" ]; then
    make build >"$ADIR/build.log" 2>&1 || { echo "FAIL setup: make build ($ADIR/build.log)"; exit 1; }
fi

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

# run_c <log> <docker run args...>: named container, killed on timeout (a killed client leaves it running).
run_c() {
    local log="$1" name="accept2272_$$_${RANDOM}"
    shift
    timeout "$TMO" docker run --rm --name "$name" --gpus all -v "$HOME/models:/models" "$@" >"$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 124 ]; then
        docker rm -f "$name" >/dev/null 2>&1
        echo "TIMEOUT after ${TMO}s" >>"$log"
    fi
    return "$rc"
}

# gtest_verdict <id> <log> <rc>: exit 0, >= 1 test ran, 0 FAILED, 0 SKIPPED.
gtest_verdict() {
    local id="$1" log="$2" rc="$3" ran nfail nskip
    ran="$(sed -n 's/.*\[  PASSED  \] \([0-9]*\) test.*/\1/p' "$log" | tail -1)"
    nfail="$(grep -c '\[  FAILED  \]' "$log")"
    nskip="$(grep -c '\[  SKIPPED \]' "$log")"
    if [ "$rc" -eq 0 ] && [ "${ran:-0}" -ge 1 ] && [ "$nfail" -eq 0 ] && [ "$nskip" -eq 0 ]; then
        verdict "$id" PASS "exit 0, ${ran} passed, 0 failed, 0 skipped"
    else
        verdict "$id" FAIL "exit $rc, passed=${ran:-0} failed=$nfail skipped=$nskip ($log)"
    fi
}

# C1: kernel bit identity.
run_c "$ADIR/c1.log" "$IMG" test-compute --gtest_filter='FP8GemmTest.RowscaleRowsBitEqualToPerRowGemv'
gtest_verdict C1-rows-gemv-bits "$ADIR/c1.log" $?

# C2: greedy identity on the model, k=1, chat prompts.
run_c "$ADIR/c2.log" -e IMP_TEST_MODEL_MTP="$CMODEL" -e IMP_TEST_MTP_K=1 -e IMP_TEST_MTP_CHAT=1 \
    "$IMG" test-e2e --gtest_filter='MtpGreedyIdentityTest.*'
gtest_verdict C2-mtp-greedy-identity "$ADIR/c2.log" $?

# C3: decode A/B. Prompts = scripts/mtp_accuracy_bench.sh PROMPTS.
declare -A PROMPTS=(
    [factual]="What is the chemical formula for water, and what are its boiling and freezing points at standard atmospheric pressure?"
    [verbose-think]="Explain why the sky appears blue during the day but red during sunset. Walk me through the physics step by step."
    [code]="Write a Python function that computes the nth Fibonacci number using dynamic programming. Include a docstring."
    [instruction]="Compose a polite email to a colleague asking them to review a draft document by end of week. Keep it under 80 words."
)
TSV="$ADIR/c3.tsv"  # class run k tok_s tok_per_verify ms_per_verify
: >"$TSV"
cli_arm() {  # cli_arm <class> <run> <k>
    local cls="$1" run="$2" k="$3" log="$ADIR/c3_${1}_${2}_k${3}.log" tg tpv mpv
    run_c "$log" "$IMG" imp-cli --model "$CMODEL" --chat-template none --prompt "${PROMPTS[$cls]}" \
        --max-tokens 128 --temperature 0 --set "speculative.mtp_k=$k"
    local rc=$?
    tg="$(sed -n 's/^tg .*(\s*\([0-9.]*\) tok\/s).*/\1/p' "$log" | tail -1)"
    tpv="$(sed -n 's/.*\[spec-ngram\].*(\([0-9.]*\) tok\/verify, \([0-9.]*\) ms\/verify).*/\1/p' "$log" | tail -1)"
    mpv="$(sed -n 's/.*\[spec-ngram\].*(\([0-9.]*\) tok\/verify, \([0-9.]*\) ms\/verify).*/\2/p' "$log" | tail -1)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$cls" "$run" "$k" "${tg:-ERR}" "${tpv:--}" "${mpv:--}" >>"$TSV"
    [ "$rc" -eq 0 ] && [ -n "$tg" ] || echo "  arm $cls run $run k=$k: exit $rc, tg='${tg}' ($log)"
}
for run in $(seq 1 "$RUNS"); do
    for cls in factual verbose-think code instruction; do
        if [ $((run % 2)) -eq 1 ]; then
            cli_arm "$cls" "$run" 1; cli_arm "$cls" "$run" 0
        else
            cli_arm "$cls" "$run" 0; cli_arm "$cls" "$run" 1
        fi
    done
done
echo "== C3 rows (class run k tok/s tok/verify ms/verify) =="
cat "$TSV"
c3="$(awk -F'\t' '
    $4 == "ERR" { err++; next }
    $3 == 1 { s1 += $4; n1++; if ($6 != "-") { m += $6; t += $5; nv++ } }
    $3 == 0 { s0 += $4; n0++ }
    END {
        if (err || !n1 || !n0 || !nv) { printf "FAIL err=%d k1_rows=%d k0_rows=%d verify_rows=%d", err, n1, n0, nv; exit }
        a1 = s1 / n1; a0 = s0 / n0; mv = m / nv; tv = t / nv; be = tv * 1000 / a0
        ok = (a1 >= a0) || (mv <= be)
        printf "%s k1 %.2f tok/s vs k0 %.2f tok/s (%+.1f %%); %.2f ms/verify at %.2f tok/verify, break-even %.2f ms",
               ok ? "PASS" : "FAIL", a1, a0, (a1 / a0 - 1) * 100, mv, tv, be
    }' "$TSV")"
verdict C3-mtp-decode "${c3%% *}" "${c3#* }"

echo
echo "== summary =="
printf '%s\n' "${RESULTS[@]}"
for r in "${RESULTS[@]}"; do
    case "$r" in FAIL*) exit 1 ;; esac
done
exit 0
