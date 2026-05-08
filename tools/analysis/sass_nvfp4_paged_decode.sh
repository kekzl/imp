#!/usr/bin/env bash
# Re-runnable SASS audit: count Tensor-Core (HMMA) vs CUDA-core (FFMA/FADD/FMUL)
# instructions in imp's NVFP4 paged decode kernel.
#
# Why: per memory file `kv_research_grade_eval_2026_05_09.md` and
# `bitdecoding_sass_audit_2026_05_09.md`, imp's NVFP4 paged decode uses
# scalar FFMA on the Q.K dot and PV accumulation (CUDA-cores). BitDecoding
# (HPCA 2026, arxiv:2503.18773) shows that routing dequantized-NVFP4 KV
# math through Tensor Cores (HMMA) yields up to 8.6× over FA-v2 FP16.
#
# Run after each major change to the NVFP4 paged decode kernel to track
# whether HMMA count moves up (= adopting BitDecoding-style TC dispatch)
# or stays at zero (= still CUDA-cores-only).
#
# Usage: bash tools/analysis/sass_nvfp4_paged_decode.sh [HEAD_DIM]
# Default HEAD_DIM = 128 (Qwen3-style attention dim)
set -euo pipefail

HEAD_DIM="${1:-128}"
BIN="${IMP_CLI_BIN:-/usr/local/bin/imp-cli}"

if [ ! -x "$BIN" ]; then
    # Extract from imp:test container if not on path
    CID=$(docker create imp:test)
    BIN=/tmp/imp-cli-sass-audit
    docker cp "${CID}:/usr/local/bin/imp-cli" "$BIN"
    docker rm "$CID" > /dev/null
fi

CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}
NVDISASM=${NVDISASM:-/usr/local/cuda/bin/nvdisasm}

# Extract sm_120a cubin (cuobjdump --list-elf shows the embedded ELF name first)
CUBIN=/tmp/imp-cli-sass-audit.sm_120a.cubin
rm -f "$CUBIN"
TMPDIR=$(mktemp -d)
pushd "$TMPDIR" > /dev/null
ELF_NAME=$("$CUOBJDUMP" --list-elf "$BIN" 2>/dev/null | grep -oE "[A-Za-z0-9_.-]+\.sm_120a\.cubin" | head -1)
if [ -n "$ELF_NAME" ]; then
    "$CUOBJDUMP" --extract-elf "$ELF_NAME" "$BIN" > /dev/null 2>&1 || true
    if [ -f "$ELF_NAME" ]; then
        mv "$ELF_NAME" "$CUBIN"
    fi
fi
popd > /dev/null
rm -rf "$TMPDIR"

if [ ! -f "$CUBIN" ]; then
    echo "ERROR: could not extract sm_120a cubin from $BIN" >&2
    exit 1
fi

# Disassemble and isolate the paged_attention_decode_nvfp4_kernel<HD> section
SASS=/tmp/imp-cli-sass-audit.sass
"$NVDISASM" -c -ndf "$CUBIN" 2>/dev/null > "$SASS"

KERNEL_PATTERN="paged_attention_decode_nvfp4_kernelILi${HEAD_DIM}EE"

if ! grep -q "$KERNEL_PATTERN" "$SASS"; then
    echo "ERROR: kernel paged_attention_decode_nvfp4_kernel<${HEAD_DIM}> not found in cubin" >&2
    echo "Available NVFP4 paged decode kernels:" >&2
    grep -oE "paged_attention_decode_nvfp4_kernelILi[0-9]+EE" "$SASS" | sort -u | sed 's/^/  /' >&2
    exit 1
fi

echo "=== imp NVFP4 paged decode SASS audit (HEAD_DIM=$HEAD_DIM) ==="
echo
awk -v p="$KERNEL_PATTERN" '
    $0 ~ p {found=1; print "kernel section start:"; print "  " $0}
    found && /^\/\/-+ \.text\.[^ ]+ -+$/ && $0 !~ p {if (started) exit}
    found {
        if (/^[[:space:]]+\/\*[0-9a-f]+\*\//) started=1
        if (started) print
    }
' "$SASS" | grep -oE "(HMMA|FFMA|FADD|FMUL|MUFU|LDG|STG)\.[A-Z0-9.]*" | sort | uniq -c | sort -rn | head -20

echo
echo "=== Tensor Core (HMMA) vs scalar FP (FFMA+FADD+FMUL) summary ==="
KERNEL_SASS=$(awk -v p="$KERNEL_PATTERN" '
    $0 ~ p {found=1}
    found && /^\/\/-+ \.text\.[^ ]+ -+$/ && $0 !~ p {if (started) exit}
    found {if (/^[[:space:]]+\/\*[0-9a-f]+\*\//) started=1; if (started) print}
' "$SASS")
HMMA=$(echo "$KERNEL_SASS" | grep -cE "HMMA" || true)
SCALAR=$(echo "$KERNEL_SASS" | grep -cE "FFMA|FADD\.FTZ|FMUL\.FTZ" || true)
echo "  HMMA  (Tensor Core MMA): $HMMA"
echo "  FFMA+FADD+FMUL (scalar): $SCALAR"
if [ "$HMMA" -eq 0 ]; then
    echo "  ⚠ Zero Tensor Cores — BitDecoding-style TC dispatch not yet adopted."
else
    echo "  ✓ Tensor Cores in use (count: $HMMA)."
fi
