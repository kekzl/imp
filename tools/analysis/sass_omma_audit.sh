#!/usr/bin/env bash
# Re-runnable SASS audit: enumerate every kernel in imp's sm_120a build that
# emits OMMA (hardware FP4 block-scaled MMA), bucket by source file, and
# flag suspicious absences (kernels you'd expect to use OMMA but don't).
#
# Why: the FP4 hot-path on consumer-Blackwell SM120 is `OMMA.SF.16864` —
# *not* HMMA, *not* TCGEN05/UMMA/BMMA. Two earlier audit memos
# (`sass_audit_120a_no_tcgen05_2026_05_04`,
#  `nvfp4_moe_prefill_landscape_2026_05_10`) drew opposite conclusions
# about which opcode actually drives FP4. This script gives a definitive
# per-kernel breakdown so future kernel changes can be checked against it.
#
# Run after each significant NVFP4/MXFP4 change to verify:
#   1) New CUTLASS dispatch sites still emit OMMA (not silent FP16 fallback).
#   2) Hand-rolled FP4 kernels (smallM, FMHA-MXFP4) keep their OMMA count.
#   3) No regression to `MUFU`-heavy software-dequant pattern.
#
# Usage: bash tools/analysis/sass_omma_audit.sh
set -euo pipefail

BIN="${IMP_CLI_BIN:-/usr/local/bin/imp-cli}"
CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}
NVDISASM=${NVDISASM:-/usr/local/cuda/bin/nvdisasm}

if [ ! -x "$BIN" ]; then
    CID=$(docker create imp:test)
    BIN=/tmp/imp-cli-omma-audit
    docker cp "${CID}:/usr/local/bin/imp-cli" "$BIN"
    docker rm "$CID" > /dev/null
fi

CUBIN=/tmp/imp-omma-audit.sm_120a.cubin
SASS=/tmp/imp-omma-audit.sass
rm -f "$CUBIN" "$SASS"

TMPDIR=$(mktemp -d)
pushd "$TMPDIR" > /dev/null
ELF_NAME=$("$CUOBJDUMP" --list-elf "$BIN" 2>/dev/null \
            | grep -oE "[A-Za-z0-9_.-]+\.sm_120a\.cubin" | head -1)
if [ -z "$ELF_NAME" ]; then
    echo "ERROR: no sm_120a cubin embedded in $BIN" >&2
    exit 1
fi
"$CUOBJDUMP" --extract-elf "$ELF_NAME" "$BIN" > /dev/null
mv "$ELF_NAME" "$CUBIN"
popd > /dev/null
rm -rf "$TMPDIR"

"$NVDISASM" -c -ndf "$CUBIN" 2>/dev/null > "$SASS"

echo "=== imp sm_120a SASS audit — OMMA inventory ==="
echo "binary:  $BIN"
echo "cubin:   $CUBIN ($(stat -c %s "$CUBIN") bytes)"
echo "sass:    $SASS ($(stat -c %s "$SASS") bytes)"
echo

echo "--- aggregate tensor-core opcode counts ---"
grep -oE "(OMMA|HMMA|IMMA|BMMA|TCGEN05|UMMA)\.[A-Z0-9.]*" "$SASS" \
    | sort | uniq -c | sort -rn | head -20
echo

# Per-kernel OMMA / HMMA breakdown
awk '
/^\/\/-+ \.text\.[^ ]+ -+$/ {
    if (kernel != "" && (omma > 0 || hmma > 0)) {
        print kernel "\t" omma "\t" hmma
    }
    kernel = $2; sub(/^\.text\./, "", kernel)
    omma = 0; hmma = 0
    next
}
{
    n = gsub(/OMMA\.SF/, "&"); omma += n
    n = gsub(/HMMA\./,  "&"); hmma += n
}
END {
    if (omma > 0 || hmma > 0) print kernel "\t" omma "\t" hmma
}
' "$SASS" > /tmp/imp-omma-audit.kernels.tsv

echo "--- OMMA emitters bucketed by source file ---"
python3 - <<'PY'
import collections, re

buckets = collections.defaultdict(lambda: {"omma": 0, "hmma": 0, "n": 0})
suspicious = []  # kernels with hot-path-sounding names but 0 OMMA

with open('/tmp/imp-omma-audit.kernels.tsv') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line:
            continue
        kern, omma, hmma = line.split('\t')
        omma, hmma = int(omma), int(hmma)

        if 'gemm_grouped_nvfp4_smallM_cu' in kern:
            label = 'src/compute/gemm_grouped_nvfp4_smallM.cu (opt-in moe.nvfp4_smallM)'
        elif 'gemm_cutlass_mxfp4_sm120_cu' in kern:
            label = 'src/compute/gemm_cutlass_mxfp4_sm120.cu'
        elif 'gemm_cutlass_grouped_3x_cu' in kern:
            label = 'src/compute/gemm_cutlass_grouped_3x.cu (MoE NVFP4 grouped)'
        elif 'gemm_cutlass_sm120_cu' in kern:
            label = 'src/compute/gemm_cutlass_sm120.cu (single-batch NVFP4)'
        elif 'fmha_sm120_mxfp4_kernel' in kern:
            label = 'src/compute/attention_fmha_mxfp4_sm120.cu (FMHA MXFP4-KV)'
        elif re.search(r'(probe|bench|smoke).*(mxf4|nvfp4|mxfp4|blockscale)', kern):
            label = 'probe/bench/smoke (microbench, not hot-path)'
        elif 'cutlass' in kern and 'device_kernel' in kern:
            label = 'cutlass<other> ' + ('grouped' if 'PtrArray' in kern else 'single')
        else:
            label = 'OTHER: ' + kern[:70]

        if omma > 0:
            b = buckets[label]
            b["omma"] += omma
            b["hmma"] += hmma
            b["n"] += 1
        elif hmma > 16 and any(s in kern for s in (
                'nvfp4', 'mxfp4', 'blackwell', 'fp4'
        )):
            suspicious.append((kern, hmma))

print(f"  {'kernels':>7}  {'OMMA':>6}  {'HMMA':>6}  source")
print('  ' + '-' * 86)
total = collections.Counter()
for label, b in sorted(buckets.items(), key=lambda kv: -kv[1]['omma']):
    print(f"  {b['n']:>7}  {b['omma']:>6}  {b['hmma']:>6}  {label}")
    total["n"] += b["n"]; total["omma"] += b["omma"]; total["hmma"] += b["hmma"]
print('  ' + '-' * 86)
print(f"  {total['n']:>7}  {total['omma']:>6}  {total['hmma']:>6}  TOTAL")

if suspicious:
    print()
    print('--- suspicious absences (FP4-named kernels with 0 OMMA, HMMA-only) ---')
    for kern, hmma in suspicious[:10]:
        print(f"  {hmma:>4} HMMA, 0 OMMA  —  {kern[:90]}")
    print('  (these may legitimately dequant FP4→FP16 before HMMA — verify intentional)')
PY

echo
echo "Re-run after kernel changes; compare against memory file"
echo "  memory/sass_audit_120a_no_tcgen05_2026_05_04.md"
