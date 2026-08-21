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

LLAMA_IMAGE="ghcr.io/ggml-org/llama.cpp@sha256:c49f4d485fb08d3002fcbd6b43be8b18758b4a2f021243b42968f64a37b57e1d"
IMP_IMAGE="${IMP_IMAGE:-imp:test}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OUT="${OUT:-/tmp/bench_competitive.tsv}"

# name <TAB> gguf path relative to MODELS_DIR
read -r -d '' MATRIX <<'TSV' || true
Qwen3-8B Q8_0	Qwen3-8B-Q8_0.gguf
Qwen3-14B Q6_K	Qwen3-14B-Q6_K.gguf
Qwen3.6-35B-A3B UD-Q4_K_M	qwen3.6-35B-A3B-gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
Gemma-4-26B-A4B UD-Q4_K_M	gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
Qwen3-30B-A3B Q4_K_M	Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
gpt-oss-20b MXFP4	gpt-oss-20b-mxfp4.gguf
TSV

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
printf 'model\timp_default\timp_spec_off\tllama_tg128\n' >> "$OUT"
while IFS=$'\t' read -r name rel; do
    [ -n "$name" ] || continue
    require_readable "$MODELS_DIR/$rel"
    echo ">>> $name" >&2
    l=$(llama_tg "/models/$rel"); sleep 20
    i=$(imp_tg   "/models/$rel" ""); sleep 20
    o=$(imp_tg   "/models/$rel" "--set speculative.ngram=false"); sleep 20
    printf '%s\t%s\t%s\t%s\n' "$name" "${i:-FAILED}" "${o:-FAILED}" "${l:-FAILED}" >> "$OUT"
done <<< "$MATRIX"

awk -F'\t' '{printf "%-34s %12s %12s %12s\n", $1, $2, $3, $4}' "$OUT"
if grep -q FAILED "$OUT"; then
    echo "FATAL: at least one arm produced no number" >&2
    exit 1
fi
