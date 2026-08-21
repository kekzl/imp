#!/usr/bin/env bash
# Competitive decode sweep: imp against llama.cpp on the shared-quant hero GGUFs.
#
# The competitor image is pinned BY DIGEST, not by tag. `:full-cuda` moves, and
# twice now a published lead was compared against a build nobody recorded. To
# refresh deliberately: pull the tag, read `docker images --digests`, update
# LLAMA_DIGEST here, and say which build it resolved to in the PROV block.
#
# Model files are required, not optional: a path that is set but unreadable is a
# failure, never a silent skip. See docs/BENCHMARKS.md.
set -euo pipefail

# Run from a frozen copy of ourselves. This harness runs for the better part of
# an hour, and bash reads a script incrementally from the file: an edit landing
# mid-run is executed by the same process that started on the old text, and the
# result is a number produced by a mixture of two versions with nothing in the
# output to say so. It happened on 2026-08-21 and cost a re-run.
#
# The hash is printed so it can go in the PROV block beside the commit. "Which
# tree" and "which script" are different questions and a commit only answers the
# first.
if [ "${IMP_BENCH_FROZEN:-0}" != "1" ]; then
    _self=$(mktemp /tmp/bench_competitive.XXXXXX.sh)
    cp -- "$0" "$_self"
    export IMP_BENCH_FROZEN=1
    export IMP_BENCH_HARNESS_MD5
    IMP_BENCH_HARNESS_MD5=$(md5sum < "$_self" | cut -d' ' -f1)
    echo "harness: $_self  md5=$IMP_BENCH_HARNESS_MD5" >&2
    exec bash "$_self" "$@"
fi

LLAMA_IMAGE="ghcr.io/ggml-org/llama.cpp@sha256:c49f4d485fb08d3002fcbd6b43be8b18758b4a2f021243b42968f64a37b57e1d"
IMP_IMAGE="${IMP_IMAGE:-imp:test}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OUT="${OUT:-/tmp/bench_competitive.tsv}"

# name <TAB> gguf path relative to MODELS_DIR <TAB> hero|nonhero
#
# `hero` marks the rows GOAL.md's release blocker is defined over. RELEASE_BAR=1
# fails on any of them under HERO_LEAD_PCT.
read -r -d '' MATRIX <<'TSV' || true
Qwen3-8B Q8_0	Qwen3-8B-Q8_0.gguf	hero
Qwen3-14B Q6_K	Qwen3-14B-Q6_K.gguf	hero
Qwen3.6-35B-A3B UD-Q4_K_M	qwen3.6-35B-A3B-gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf	hero
Gemma-4-26B-A4B UD-Q4_K_M	gemma-4-26B-A4B-it-UD-Q4_K_M.gguf	hero
gpt-oss-20b MXFP4	gpt-oss-20b-mxfp4.gguf	hero
Qwen3-30B-A3B Q4_K_M	Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf	nonhero
TSV

# Heroes this sweep structurally cannot contest, with the reason. They are
# PRINTED, never silently omitted: five heroes went unmeasured for six weeks
# because nothing said they were unmeasured (DEBT_LEDGER section (h)).
read -r -d '' UNCONTESTED <<'TSV' || true
Qwen3-Coder-30B-A3B	NVFP4 only, no GGUF on this host and no llama.cpp NVFP4 path on sm_120
Nemotron-H	NVFP4 only, no GGUF on this host and no llama.cpp NVFP4 path on sm_120
TSV

RELEASE_BAR=${RELEASE_BAR:-0}      # 1 = evaluate GOAL.md's hero bar and exit non-zero on a breach
HERO_LEAD_PCT=${HERO_LEAD_PCT:-5}  # GOAL.md release bar 2: decode must lead by at least this much

require_readable() {
    [ -r "$1" ] || { echo "FATAL: model not readable: $1" >&2; exit 1; }
}

check_gpu() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -gt 3000 ]; then
        echo "FATAL: $used MiB already held on the GPU. On WSL2 the process list is" >&2
        echo "       blank even when memory is held, so this is the only usable guard." >&2
        exit 1
    fi
    [ "$(docker ps -q | wc -l)" -eq 0 ] || { echo "FATAL: containers running, they depress every number" >&2; exit 1; }
}

llama_tg() {  # $1 = container path
    docker run --rm --gpus all -v "$MODELS_DIR":/models "$LLAMA_IMAGE" \
        --bench -m "$1" -p 512 -n 128 -r 5 -ngl 99 2>/dev/null \
        | awk -F'|' '/tg128/ {split($8,a,"±"); gsub(/ /,"",a[1]); print a[1]}'
}

# imp's --bench prompt is self-repetitive, so n-gram speculation accepts ~100 % of
# its drafts. llama-bench cannot exploit that, so the defaults column and the
# spec-off column mean different things and both are reported.
imp_tg() {  # $1 = container path, $2 = extra args
    docker run --rm --gpus all -v "$MODELS_DIR":/models "$IMP_IMAGE" \
        imp-cli --model "$1" --bench --bench-pp 512 --bench-reps 10 \
        --max-tokens 128 --temperature 0 $2 2>&1 \
        | grep -oP '^tg\s+128 tokens.*?\(\s*\K[0-9.]+(?= tok/s)'
}

# 20 s between arms, not 5. At 5 s the Qwen3-14B row read 5.3 % low against two
# isolated re-measurements (154.77 against 162.08 / 162.04) - the first imp run
# after a 16 GiB competitor model unloads is not on a settled card. The imp
# default / spec-off pair doubles as this sweep's repeatability control: on every
# model where speculation is inert the two columns must agree, and they agree to
# 0.2 % on four of five. That is what caught this.
check_gpu
: > "$OUT"
printf 'model\timp_default\timp_spec_off\tllama_tg128\thero\n' >> "$OUT"
MISSING=""
while IFS=$'\t' read -r name rel tier; do
    [ -n "$name" ] || continue
    # A model that is set but not on disk is named, not skipped past. Under
    # RELEASE_BAR it is a failure; outside it, it is still reported.
    if [ ! -r "$MODELS_DIR/$rel" ]; then
        MISSING="$MISSING$name ($rel)\n"
        printf '%s\tNOMODEL\tNOMODEL\tNOMODEL\t%s\n' "$name" "$tier" >> "$OUT"
        continue
    fi
    echo ">>> $name" >&2
    l=$(llama_tg "/models/$rel"); sleep 20
    i=$(imp_tg   "/models/$rel" ""); sleep 20
    o=$(imp_tg   "/models/$rel" "--set speculative.ngram=false"); sleep 20
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "${i:-FAILED}" "${o:-FAILED}" "${l:-FAILED}" "$tier" >> "$OUT"
done <<< "$MATRIX"

awk -F'\t' 'NR>1 {
    lead = ($4 ~ /^[0-9.]+$/ && $2 ~ /^[0-9.]+$/) ? sprintf("%+.1f%%", 100*($2/$4-1)) : "n/a"
    printf "%-34s %10s %10s %10s %8s %8s\n", $1, $2, $3, $4, $5, lead
}' "$OUT"

echo
echo "Heroes this sweep cannot contest (stated, not omitted):"
while IFS=$'\t' read -r hname reason; do
    [ -n "$hname" ] || continue
    printf '  %-24s %s\n' "$hname" "$reason"
done <<< "$UNCONTESTED"

if grep -q FAILED "$OUT"; then
    echo "FATAL: at least one arm produced no number" >&2
    exit 1
fi

if [ -n "$MISSING" ]; then
    echo
    printf 'Model files absent from %s:\n' "$MODELS_DIR" >&2
    printf "$MISSING" | sed 's/^/  /' >&2
fi

if [ "$RELEASE_BAR" != "1" ]; then
    [ -z "$MISSING" ] || exit 1
    exit 0
fi

# GOAL.md release bar 2: decode must lead llama.cpp by >= HERO_LEAD_PCT on every
# hero. Evaluated on the DEFAULT column, which is what a user gets; the spec-off
# column is the repeatability control, not the claim.
echo
BREACH=$(awk -F'\t' -v bar="$HERO_LEAD_PCT" 'NR>1 && $5=="hero" {
    if ($2 !~ /^[0-9.]+$/ || $4 !~ /^[0-9.]+$/) { printf "%s: not measured\n", $1; next }
    lead = 100*($2/$4-1)
    if (lead < bar) printf "%s: +%.1f%% (bar %+d%%)\n", $1, lead, bar
}' "$OUT")
if [ -n "$BREACH" ] || [ -n "$MISSING" ]; then
    echo "FAIL: release bar 2, decode lead over llama.cpp below ${HERO_LEAD_PCT}% on:" >&2
    [ -z "$BREACH" ] || printf '%s\n' "$BREACH" | sed 's/^/  /' >&2
    [ -z "$MISSING" ] || echo "  (and the absent models above)" >&2
    exit 1
fi
# The pass line carries its own scope. A bare "PASS" here would be read as
# "the seven-hero blocker is satisfied" by everyone who does not open this
# script, which is the same over-reading that let five heroes go unmeasured for
# six weeks. The two uncontestable heroes are named at exactly the moment
# somebody is deciding to ship.
# awk on the tab-delimited field, not grep on a line suffix: 'hero$' also
# matches 'nonhero' and counted 6 heroes where there are 5.
N_HERO_MATRIX=$(awk -F'\t' '$3=="hero" {n++} END {print n+0}' <<< "$MATRIX")
N_UNCONTESTABLE=$(awk 'NF {n++} END {print n+0}' <<< "$UNCONTESTED")
N_HERO_TOTAL=$(( N_HERO_MATRIX + N_UNCONTESTABLE ))
N_CONTESTED=$(awk -F'\t' 'NR>1 && $5=="hero" && $2 ~ /^[0-9.]+$/ && $4 ~ /^[0-9.]+$/ {n++} END {print n+0}' "$OUT")
echo "RELEASE BAR: ${N_CONTESTED}/${N_HERO_TOTAL} heroes contested, all above +${HERO_LEAD_PCT}%;"
echo "             ${N_UNCONTESTABLE} uncontestable (NVFP4-only, no llama.cpp sm_120 path)."
echo "             This is NOT a statement about the ${N_UNCONTESTABLE} it could not measure."
exit 0
