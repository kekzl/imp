#!/usr/bin/env bash
# PTX cvt instruction survey for sm_120 (RTX 5090, GB202 Blackwell).
#
# Tests every relevant FP4/FP6/FP8 cvt variant against ptxas to determine
# which are supported on the current CUDA toolkit. Output: markdown table.
#
# Usage:
#   tools/analysis/ptx_cvt_survey.sh                # full run
#   tools/analysis/ptx_cvt_survey.sh --quick        # FP4 only
#   tools/analysis/ptx_cvt_survey.sh --image IMAGE  # custom CUDA docker image
#
# Re-run after CUDA toolkit upgrades to refresh dead-ends.
#
# Register sizing per type (this matters — wrong register class = false negative):
#   e2m1x2 (FP4 pair) =  8 bits → .b8  register class (route via uint32 + cvt.u32.u8)
#   e2m3x2 (FP6 pair) = 12 bits → .b16 register class (16-bit reg, 4 unused bits)
#   e3m2x2 (FP6 pair) = 12 bits → .b16
#   e4m3x2 (FP8 pair) = 16 bits → .b16
#   e5m2x2 (FP8 pair) = 16 bits → .b16

set -u

CUDA_IMG=${CUDA_IMG:-nvidia/cuda:13.2.1-devel-ubuntu24.04}
ARCH=${ARCH:-compute_120f,code=sm_120}
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --image) CUDA_IMG="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

# Compile a kernel with `body` and report SUPPORTED/REJECTED.
compile_test() {
    local label="$1"
    local body="$2"
    local sig="$3"

    cat > "$WORK/test.cu" <<EOF
#include <cstdint>
#include <cuda_fp16.h>
extern "C" __global__ void k(${sig}) {
    ${body}
}
EOF
    local err
    err=$(docker run --rm -v "$WORK:/w" -w /w "$CUDA_IMG" \
            bash -c "nvcc -gencode=arch=${ARCH} -c test.cu -o /tmp/o 2>&1" 2>&1 | \
            grep -E "error|fatal|Feature|Modifier|Rounding|Instruction" | head -1)
    if [[ -z "$err" ]]; then
        echo "✅ | \`$label\` | OK"
    else
        local reason=$(echo "$err" | sed -E '
            s/.*error\s*:\s*//
            s/^Arguments mismatch.*$/Arguments mismatch/
            s/^Unexpected instruction types.*$/Unexpected instruction types/
            s/^Feature.*not supported on.*sm_120.*$/Feature not supported on sm_120/
            s/^Instruction.*not supported.*$/Instruction not supported/
            s/.*Modifier.*$/Modifier requirement/
            s/.*Rounding.*$/Rounding mod required/
        ' | head -c 90)
        [[ -z "$reason" ]] && reason="(rejected)"
        echo "❌ | \`$label\` | ${reason}"
    fi
}

# ---- FP4 (8-bit packed) -------------------------------------------------------
fp4_encode_f32_pair() {
    compile_test "$1" "
    uint32_t r;
    asm(\"{ .reg .b8 b; ${2} b, %2, %1; cvt.u32.u8 %0, b; }\"
        : \"=r\"(r) : \"f\"(a), \"f\"(b_in));
    *out = r;" "float a, float b_in, uint32_t* out"
}
fp4_encode_f16x2() {
    compile_test "$1" "
    uint32_t r;
    asm(\"{ .reg .b8 b; ${2} b, %1; cvt.u32.u8 %0, b; }\"
        : \"=r\"(r) : \"r\"(in));
    *out = r;" "uint32_t in, uint32_t* out"
}
fp4_decode_to_f16x2() {
    compile_test "$1" "
    uint32_t r;
    asm(\"{ .reg .b8 b; cvt.u8.u32 b, %1; ${2} %0, b; }\"
        : \"=r\"(r) : \"r\"(in));
    *out = r;" "uint32_t in, uint32_t* out"
}
fp4_decode_to_bf16x2() {
    compile_test "$1" "
    uint32_t r;
    asm(\"{ .reg .b8 b; cvt.u8.u32 b, %1; ${2} %0, b; }\"
        : \"=r\"(r) : \"r\"(in));
    *out = r;" "uint32_t in, uint32_t* out"
}

# ---- FP8 / FP6 (16-bit packed) -----------------------------------------------
fp8_encode_f32_pair() {
    compile_test "$1" "
    uint16_t r;
    asm(\"${2} %0, %2, %1;\" : \"=h\"(r) : \"f\"(a), \"f\"(b_in));
    *out = r;" "float a, float b_in, uint16_t* out"
}
fp8_encode_f16x2() {
    compile_test "$1" "
    uint16_t r;
    asm(\"${2} %0, %1;\" : \"=h\"(r) : \"r\"(in));
    *out = r;" "uint32_t in, uint16_t* out"
}
fp8_decode_to_f16x2() {
    compile_test "$1" "
    uint32_t r;
    asm(\"${2} %0, %1;\" : \"=r\"(r) : \"h\"(in));
    *out = r;" "uint16_t in, uint32_t* out"
}
fp8_decode_to_bf16x2() {
    compile_test "$1" "
    uint32_t r;
    asm(\"${2} %0, %1;\" : \"=r\"(r) : \"h\"(in));
    *out = r;" "uint16_t in, uint32_t* out"
}
fp8_decode_to_f32x2() {
    compile_test "$1" "
    float r0, r1;
    asm(\"${2} { %0, %1 }, %2;\" : \"=f\"(r0), \"=f\"(r1) : \"h\"(in));
    out[0] = r0; out[1] = r1;" "uint16_t in, float* out"
}

run_fp4() {
    echo ""
    echo "### \`e2m1\` (FP4, 8-bit packed pair)"
    echo ""
    echo "Status | Instruction | Reason"
    echo "---|---|---"
    fp4_encode_f32_pair "cvt.rn.satfinite.e2m1x2.f32" \
        "cvt.rn.satfinite.e2m1x2.f32"
    fp4_encode_f32_pair "cvt.rn.e2m1x2.f32 (no .satfinite)" \
        "cvt.rn.e2m1x2.f32"
    fp4_encode_f16x2 "cvt.rn.satfinite.e2m1x2.f16x2" \
        "cvt.rn.satfinite.e2m1x2.f16x2"
    fp4_encode_f16x2 "cvt.rn.satfinite.e2m1x2.bf16x2" \
        "cvt.rn.satfinite.e2m1x2.bf16x2"
    fp4_decode_to_f16x2 "cvt.rn.f16x2.e2m1x2" \
        "cvt.rn.f16x2.e2m1x2"
    fp4_decode_to_f16x2 "cvt.f16x2.e2m1x2 (no .rn)" \
        "cvt.f16x2.e2m1x2"
    fp4_decode_to_f16x2 "cvt.rn.relu.f16x2.e2m1x2" \
        "cvt.rn.relu.f16x2.e2m1x2"
    fp4_decode_to_bf16x2 "cvt.rn.bf16x2.e2m1x2" \
        "cvt.rn.bf16x2.e2m1x2"
}

run_fp6_fp8() {
    local T="$1"
    local label="$2"

    echo ""
    echo "### \`${T}\` ($label)"
    echo ""
    echo "Status | Instruction | Reason"
    echo "---|---|---"
    fp8_encode_f32_pair "cvt.rn.satfinite.${T}x2.f32" \
        "cvt.rn.satfinite.${T}x2.f32"
    fp8_encode_f32_pair "cvt.rn.${T}x2.f32 (no .satfinite)" \
        "cvt.rn.${T}x2.f32"
    fp8_encode_f16x2 "cvt.rn.satfinite.${T}x2.f16x2" \
        "cvt.rn.satfinite.${T}x2.f16x2"
    fp8_encode_f16x2 "cvt.rn.satfinite.${T}x2.bf16x2" \
        "cvt.rn.satfinite.${T}x2.bf16x2"
    fp8_decode_to_f16x2 "cvt.rn.f16x2.${T}x2" \
        "cvt.rn.f16x2.${T}x2"
    fp8_decode_to_f16x2 "cvt.f16x2.${T}x2 (no .rn)" \
        "cvt.f16x2.${T}x2"
    fp8_decode_to_f16x2 "cvt.rn.relu.f16x2.${T}x2" \
        "cvt.rn.relu.f16x2.${T}x2"
    fp8_decode_to_bf16x2 "cvt.rn.bf16x2.${T}x2" \
        "cvt.rn.bf16x2.${T}x2"
    fp8_decode_to_f32x2 "cvt.f32x2.${T}x2 (→ pair of f32)" \
        "cvt.f32x2.${T}x2"
    fp8_decode_to_f32x2 "cvt.rn.f32x2.${T}x2" \
        "cvt.rn.f32x2.${T}x2"
}

echo ""
echo "## PTX cvt survey — sm_120f / $CUDA_IMG"

run_fp4

if [[ "$QUICK" == "1" ]]; then
    exit 0
fi

run_fp6_fp8 e2m3 "FP6 E2M3, 12-bit packed pair (16-bit reg, 4 bits unused)"
run_fp6_fp8 e3m2 "FP6 E3M2, 12-bit packed pair (16-bit reg, 4 bits unused)"
run_fp6_fp8 e4m3 "FP8 E4M3, 16-bit packed pair"
run_fp6_fp8 e5m2 "FP8 E5M2, 16-bit packed pair"

echo ""
echo "### Block scale types (UE4M3 / UE8M0 — typically MMA operands only)"
echo ""
echo "Status | Instruction | Reason"
echo "---|---|---"
# UE8M0 = 8-bit pure-exponent scale (one byte per scale)
compile_test "cvt.rn.satfinite.ue8m0.f32" "
    uint16_t r;
    asm(\"{ .reg .b8 b; cvt.rn.satfinite.ue8m0.f32 b, %1; cvt.u16.u8 %0, b; }\"
        : \"=h\"(r) : \"f\"(a));
    *out = r;" "float a, uint16_t* out"
compile_test "cvt.rn.satfinite.ue8m0.f32x2 (encode 2 scales)" "
    uint16_t r;
    asm(\"cvt.rn.satfinite.ue8m0x2.f32 %0, %2, %1;\"
        : \"=h\"(r) : \"f\"(a), \"f\"(b));
    *out = r;" "float a, float b, uint16_t* out"
compile_test "cvt.f32.ue8m0 (decode scale → f32)" "
    float r;
    asm(\"{ .reg .b8 b; cvt.u8.u16 b, %1; cvt.f32.ue8m0 %0, b; }\"
        : \"=f\"(r) : \"h\"(in));
    *out = r;" "uint16_t in, float* out"

echo ""
echo "Done. Re-run after CUDA toolkit upgrades to refresh dead-end status."
