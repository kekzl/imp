#!/usr/bin/env bash
# Competitive decode sweep: imp vs vLLM on the same SafeTensors hero checkpoints, one stream.
# Both engines serve the checkpoint through their OpenAI server, fresh container per arm; client
# tools/analysis/conc_client.py at concurrency 1 (300-token greedy answers, ignore_eos, median of
# WAVES waves). Competitor image pinned BY DIGEST (VLLM_IMAGE), not tag; update deliberately and
# record it in the PROV block. Model dirs required: unreadable path is a failure, not a skip.
set -euo pipefail

# Run from a frozen copy of ourselves: bash reads the script incrementally, so an edit landing
# mid-run mixes two versions with nothing in the output to say so.
# IMP_BENCH_FROZEN=1 skips the copy; the hash goes in the PROV block ("which script" != "which tree").
if [ "${IMP_BENCH_FROZEN:-0}" != "1" ]; then
    _self=$(mktemp /tmp/bench_competitive.XXXXXX.sh)
    cp -- "$0" "$_self"
    export IMP_BENCH_FROZEN=1 IMP_BENCH_REPO
    IMP_BENCH_REPO="$(cd "$(dirname "$0")/.." && pwd)"
    export IMP_BENCH_HARNESS_MD5
    IMP_BENCH_HARNESS_MD5=$(md5sum < "$_self" | cut -d' ' -f1)
    echo "harness: $_self  md5=$IMP_BENCH_HARNESS_MD5" >&2
    exec bash "$_self" "$@"
fi

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3}"  # v0.31.0
IMP_IMAGE="${IMP_IMAGE:-imp:test}"
CLIENT_IMAGE="${CLIENT_IMAGE:-imp:toolchain}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OUT="${OUT:-/tmp/bench_competitive.tsv}"
PORT="${PORT:-8096}"
WAVES="${WAVES:-3}"
TOOLS="${IMP_BENCH_REPO:-$(cd "$(dirname "$0")/.." && pwd)}/tools/analysis"

# name <TAB> SafeTensors dir relative to MODELS_DIR <TAB> hero|nonhero
# "hero" marks rows GOAL.md's release blocker is defined over; RELEASE_BAR=1 fails on any hero
# under HERO_LEAD_PCT.
read -r -d '' MATRIX <<'TSV' || true
Qwen3-8B NVFP4	Qwen3-8B-NVFP4-cortecs	hero
Qwen3-8B FP8	Qwen3-8B-FP8	hero
Qwen3-14B NVFP4	Qwen3-14B-NVFP4	hero
Qwen3-Coder-30B-A3B NVFP4	Qwen3-Coder-30B-A3B-Instruct-FP4	hero
Qwen3.6-35B-A3B NVFP4	Qwen3.6-35B-A3B-NVFP4	hero
Gemma-4-26B-A4B NVFP4	Gemma-4-26B-A4B-it-NVFP4	hero
Nemotron-3-Nano-30B-A3B NVFP4	Nemotron-3-Nano-30B-A3B-NVFP4	hero
Qwen3.8-27B NVFP4	Qwen3.8-27B-NVFP4-vllm	hero
TSV

# Heroes this sweep cannot contest, with the reason. They are PRINTED, never silently omitted:
# five heroes went unmeasured for six weeks because nothing said they were unmeasured
# (DEBT_LEDGER section (h)).
read -r -d '' UNCONTESTED <<'TSV' || true
gpt-oss-20b	official SafeTensors not staged on this host
TSV

RELEASE_BAR=${RELEASE_BAR:-0}      # 1 = evaluate GOAL.md's hero bar and exit non-zero on a breach
HERO_LEAD_PCT=${HERO_LEAD_PCT:-5}  # GOAL.md release bar 2: decode must lead by at least this much

check_gpu() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -gt 3000 ]; then
        echo "FATAL: $used MiB already held on the GPU. On WSL2 the process list is" >&2
        echo "       blank even when memory is held, so this is the only usable guard." >&2
        exit 1
    fi
    # Host load, not the container list: an idle container depresses nothing, a CPU-bound one
    # shows here. Limit: IMP_BENCH_MAX_LOAD (default nproc/4) on the 1-min loadavg.
    local load max_load
    load=$(cut -d' ' -f1 /proc/loadavg)
    max_load=${IMP_BENCH_MAX_LOAD:-$(( $(nproc) / 4 ))}
    if awk -v l="$load" -v m="$max_load" 'BEGIN { exit !(l > m) }'; then
        echo "FATAL: host 1-min load $load > $max_load, it depresses every number" >&2
        echo "       running containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" >&2
        exit 1
    fi
}

wait_healthy() {  # $1 = seconds
    local _
    for _ in $(seq 1 $(( $1 / 2 ))); do
        sleep 2
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        [ -n "$(docker ps -q -f name=bench_competitive)" ] || break
    done
    echo "server not healthy:" >&2
    docker logs bench_competitive 2>&1 | tail -20 >&2
    return 1
}

# $1 = arm (imp|imp_spec_off|vllm), $2 = model dir name. Prints the median tok/s or nothing.
serve_tg() {
    docker rm -f bench_competitive >/dev/null 2>&1 || true
    local common=(-d --name bench_competitive --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT:$PORT")
    case "$1" in
        imp) docker run "${common[@]}" "$IMP_IMAGE" imp-server --model "/models/$2" \
                 --host 0.0.0.0 --port "$PORT" >/dev/null ;;
        imp_spec_off) docker run "${common[@]}" "$IMP_IMAGE" imp-server --model "/models/$2" \
                 --host 0.0.0.0 --port "$PORT" --set speculative.mtp_k=0 \
                 --set speculative.ngram=false >/dev/null ;;
        vllm) docker run "${common[@]}" --ipc=host -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 "$VLLM_IMAGE" \
                 --model "/models/$2" --served-model-name "$2" --host 0.0.0.0 --port "$PORT" \
                 --max-model-len 16384 --max-num-seqs 1 >/dev/null ;;
    esac
    if wait_healthy 1200; then
        docker run --rm --network host -e IGNORE_EOS=1 -e MODEL_NAME="$2" -v "$TOOLS":/t:ro \
            "$CLIENT_IMAGE" python3 /t/conc_client.py "$PORT" 1 "$WAVES" "$1" 0 2>/dev/null \
            | awk '/^MEDIAN/ {print $2}'
    fi
    docker rm -f bench_competitive >/dev/null 2>&1 || true
}

# 20 s cooldown between arms: right after a large model unloads the card isn't settled yet.
# The default/spec-off pair shows what speculation adds on imp; the bar reads the default column.
check_gpu
: > "$OUT"
printf 'model\timp_default\timp_spec_off\tvllm\thero\n' >> "$OUT"
MISSING=""
while IFS=$'\t' read -r name rel tier; do
    [ -n "$name" ] || continue
    # A model that is set but not on disk is named, not skipped past. Under
    # RELEASE_BAR it is a failure; outside it, it is still reported.
    if [ ! -r "$MODELS_DIR/$rel/config.json" ]; then
        MISSING="$MISSING$name ($rel)\n"
        printf '%s\tNOMODEL\tNOMODEL\tNOMODEL\t%s\n' "$name" "$tier" >> "$OUT"
        continue
    fi
    echo ">>> $name" >&2
    v=$(serve_tg vllm "$rel"); sleep 20
    i=$(serve_tg imp "$rel"); sleep 20
    o=$(serve_tg imp_spec_off "$rel"); sleep 20
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "${i:-FAILED}" "${o:-FAILED}" "${v:-FAILED}" "$tier" >> "$OUT"
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

# An imp arm without a number is a failure; a vLLM arm without one means vLLM could not serve
# the checkpoint on this card, reported as uncontested below and not a bar breach.
if awk -F'\t' 'NR>1 && ($2=="FAILED" || $3=="FAILED") {f=1} END {exit !f}' "$OUT"; then
    echo "FATAL: at least one imp arm produced no number" >&2
    exit 1
fi

if [ -n "$MISSING" ]; then
    echo
    printf 'Model dirs absent from %s:\n' "$MODELS_DIR" >&2
    printf "$MISSING" | sed 's/^/  /' >&2
fi

if [ "$RELEASE_BAR" != "1" ]; then
    [ -z "$MISSING" ] || exit 1
    exit 0
fi

# GOAL.md release bar 2: decode must lead vLLM by >= HERO_LEAD_PCT on every hero vLLM serves.
# Evaluated on the DEFAULT column, which is what a user gets.
echo
BREACH=$(awk -F'\t' -v bar="$HERO_LEAD_PCT" 'NR>1 && $5=="hero" && $4 ~ /^[0-9.]+$/ {
    lead = 100*($2/$4-1)
    if (lead < bar) printf "%s: %+.1f%% (bar +%d%%)\n", $1, lead, bar
}' "$OUT")
if [ -n "$BREACH" ] || [ -n "$MISSING" ]; then
    echo "FAIL: release bar 2, decode lead over vLLM below ${HERO_LEAD_PCT}% on:" >&2
    [ -z "$BREACH" ] || printf '%s\n' "$BREACH" | sed 's/^/  /' >&2
    [ -z "$MISSING" ] || echo "  (and the absent models above)" >&2
    exit 1
fi
# awk on the tab-delimited hero field, not grep on suffix ("hero$" also matches "nonhero").
N_HERO_MATRIX=$(awk -F'\t' '$3=="hero" {n++} END {print n+0}' <<< "$MATRIX")
N_UNCONTESTABLE=$(awk 'NF {n++} END {print n+0}' <<< "$UNCONTESTED")
N_VLLM_FAILED=$(awk -F'\t' 'NR>1 && $5=="hero" && $4=="FAILED" {n++} END {print n+0}' "$OUT")
N_CONTESTED=$(awk -F'\t' 'NR>1 && $5=="hero" && $4 ~ /^[0-9.]+$/ {n++} END {print n+0}' "$OUT")
echo "RELEASE BAR: ${N_CONTESTED}/$(( N_HERO_MATRIX + N_UNCONTESTABLE )) heroes contested, all above +${HERO_LEAD_PCT}%;"
echo "             ${N_VLLM_FAILED} not served by vLLM, ${N_UNCONTESTABLE} not staged (listed above)."
echo "             This is NOT a statement about the heroes it could not measure."
exit 0
