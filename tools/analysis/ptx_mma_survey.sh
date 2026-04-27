#!/usr/bin/env bash
# PTX MMA acceptance survey for sm_120f.
# Compiles ONE variant at a time so a single ptxas error never poisons others.

set -e
CUDA_IMG=${CUDA_IMG:-nvidia/cuda:13.2.1-devel-ubuntu24.04}
ARCH=${ARCH:-compute_120f,code=sm_120}

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

PRELUDE='#include <cuda_runtime.h>
#include <cstdint>

#define MMA_PRELUDE \
    uint32_t a0=threadIdx.x*37u+1u, a1=threadIdx.x*41u+2u, a2=threadIdx.x*43u+3u, a3=threadIdx.x*47u+4u; \
    uint32_t b0=threadIdx.x*53u+5u, b1=threadIdx.x*59u+6u, b2=threadIdx.x*61u+7u, b3=threadIdx.x*67u+8u; \
    uint32_t sfa=0x38383838u, sfb=0x38383838u, metadata=0x44444444u; \
    float d0=0.f, d1=0.f, d2=0.f, d3=0.f; \
    constexpr uint16_t bidA=0, tidA=0, bidB=0, tidB=0; \
    (void)a0;(void)a1;(void)a2;(void)a3; \
    (void)b0;(void)b1;(void)b2;(void)b3; \
    (void)sfa;(void)sfb;(void)metadata; \
    (void)bidA;(void)tidA;(void)bidB;(void)tidB
#define SINK if (threadIdx.x == 0) sink[0] = d0+d1+d2+d3
'

# ---- operand-shape templates ------------------------------------------------
# Each template knows its instruction prefix + operand list.

# Dense, no scale: 14 operands {d0..3},{a0..3},{b0..1},{c0..3}
template_dense_noscale() {
    local kind="$1"; local rest="$2"
    cat <<EOF
__global__ void k(float* sink) {
    MMA_PRELUDE;
    asm volatile(
        "mma.sync.aligned.kind::${kind}.${rest} "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    SINK;
}
EOF
}

# Dense, block-scale: 20 operands
template_dense_blockscale() {
    local rest="$1"
    cat <<EOF
__global__ void k(float* sink) {
    MMA_PRELUDE;
    asm volatile(
        "mma.sync.aligned.${rest} "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB));
    SINK;
}
EOF
}

# Sparse, no scale: 17 operands (4d + 4a + 4b + 4c + metadata, sparsity selector immediate)
template_sparse_noscale() {
    local rest="$1"
    cat <<EOF
__global__ void k(float* sink) {
    MMA_PRELUDE;
    asm volatile(
        "mma.sync.aligned.${rest} "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0;\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(metadata));
    SINK;
}
EOF
}

# Sparse, block-scale: 23 operands
template_sparse_blockscale() {
    local rest="$1"
    cat <<EOF
__global__ void k(float* sink) {
    MMA_PRELUDE;
    asm volatile(
        "mma.sync.aligned.${rest} "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0,"
        "{%17},{%18,%19},{%20},{%21,%22};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(metadata),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB));
    SINK;
}
EOF
}

# ---- test runner ------------------------------------------------------------
test_variant() {
    local label="$1"
    local template_fn="$2"
    shift 2
    {
        echo "$PRELUDE"
        $template_fn "$@"
    } > "$WORK/single.cu"

    local err
    err=$(docker run --rm -v "$WORK:/w" -w /w "$CUDA_IMG" \
            bash -c "nvcc -gencode=arch=${ARCH} -c single.cu -o /tmp/o.o 2>&1" 2>&1 | \
            grep -E "error|fatal|Feature" | head -1)
    if [[ -z "$err" ]]; then
        echo "✅ | \`$label\` | OK"
    else
        local reason
        reason=$(echo "$err" | sed -E '
            s/.*error\s*:\s*//
            s/^Feature[^.]*not supported on .*$/Feature not supported on sm_120/
            s/^Instruction[^.]* not supported.*$/Instruction not supported/
            s/^Arguments mismatch.*/Arguments mismatch/
            s/^Unexpected.*/Unexpected types/
            s/^.*Unknown modifier.*$/Unknown modifier/
            s/^Incompatible.*/Incompatible vector elements/
        ' | head -c 90)
        [[ -z "$reason" ]] && reason="(rejected)"
        echo "❌ | \`$label\` | $reason"
    fi
}

# ---- Run -------------------------------------------------------------------
echo ""
echo "## PTX MMA acceptance survey — sm_120f / $CUDA_IMG"
echo ""

echo "### DENSE no-scale (kind::f8f6f4)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
test_variant "f8f6f4 m16n8k32 e2m1×e2m1 (legacy FP4)"  template_dense_noscale "f8f6f4" "m16n8k32.row.col.f32.e2m1.e2m1.f32"
test_variant "f8f6f4 m16n8k32 e4m3×e4m3 (FP8)"          template_dense_noscale "f8f6f4" "m16n8k32.row.col.f32.e4m3.e4m3.f32"
test_variant "f8f6f4 m16n8k32 e5m2×e5m2 (FP8 alt)"      template_dense_noscale "f8f6f4" "m16n8k32.row.col.f32.e5m2.e5m2.f32"
test_variant "f8f6f4 m16n8k32 e4m3×e2m1 (mixed FP8×FP4)" template_dense_noscale "f8f6f4" "m16n8k32.row.col.f32.e4m3.e2m1.f32"
test_variant "f8f6f4 m16n8k32 e2m1×e4m3 (mixed FP4×FP8)" template_dense_noscale "f8f6f4" "m16n8k32.row.col.f32.e2m1.e4m3.f32"
test_variant "f8f6f4 m16n8k32 e2m3×e2m3 (FP6 E2M3)"     template_dense_noscale "f8f6f4" "m16n8k32.row.col.f32.e2m3.e2m3.f32"
test_variant "f8f6f4 m16n8k32 e3m2×e3m2 (FP6 E3M2)"     template_dense_noscale "f8f6f4" "m16n8k32.row.col.f32.e3m2.e3m2.f32"
test_variant "f8f6f4 m16n8k64 e2m1×e2m1 (illegal? K=64 needs sparse)" template_dense_noscale "f8f6f4" "m16n8k64.row.col.f32.e2m1.e2m1.f32"

echo ""
echo "### DENSE block-scale (kind::mxf4nvf4 — K=64)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
test_variant "mxf4nvf4 scale_vec::4X K=64 ue4m3 (Project B)" template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"
test_variant "mxf4nvf4 scale_vec::4X K=64 ue8m0"          template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "mxf4nvf4 scale_vec::2X K=64 ue4m3"          template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"
test_variant "mxf4nvf4 scale_vec::2X K=64 ue8m0 (per-32 scales)" template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "mxf4nvf4 scale_vec::1X K=64 ue4m3"          template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"
test_variant "mxf4nvf4 scale_vec::1X K=64 ue8m0"          template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "mxf4nvf4 scale_vec::8X K=64 ue4m3 (8X exists?)" template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::8X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"
test_variant "mxf4nvf4 scale_vec::4X K=128 (dense at sparse-K?)" template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3"
test_variant "mxf4nvf4 scale_vec::4X K=32"                template_dense_blockscale "kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue4m3"

echo ""
echo "### DENSE block-scale (kind::mxf8f6f4 — K=32)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
test_variant "mxf8f6f4 1X K=32 e2m1×e2m1 ue8m0 (FP4 with HW scale)" template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "mxf8f6f4 1X K=32 e4m3×e4m3 ue8m0 (FP8 with HW scale)" template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"
test_variant "mxf8f6f4 1X K=32 e5m2×e5m2 ue8m0"                    template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e5m2.f32.ue8m0"
test_variant "mxf8f6f4 1X K=32 e4m3×e2m1 (mixed FP8×FP4)"          template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0"
test_variant "mxf8f6f4 1X K=32 e2m1×e4m3 (mixed FP4×FP8)"          template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0"
test_variant "mxf8f6f4 1X K=32 e2m3×e2m3 (FP6 E2M3)"               template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e2m3.f32.ue8m0"
test_variant "mxf8f6f4 1X K=32 e3m2×e3m2 (FP6 E3M2)"               template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e3m2.f32.ue8m0"
test_variant "mxf8f6f4 2X K=32 e2m1×e2m1"                          template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::2X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "mxf8f6f4 1X K=32 e2m1×e2m1 ue4m3 (NVFP4 scale type)" template_dense_blockscale "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue4m3"

echo ""
echo "### SPARSE no-scale (kind::f8f6f4.sp::ordered_metadata, K=64)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
test_variant "sparse f8f6f4 K=64 e2m1×e2m1 (FP4 2:4 sparse)"  template_sparse_noscale "kind::f8f6f4.sp::ordered_metadata.m16n8k64.row.col.f32.e2m1.e2m1.f32"
test_variant "sparse f8f6f4 K=64 e4m3×e4m3 (FP8 2:4 sparse)"  template_sparse_noscale "kind::f8f6f4.sp::ordered_metadata.m16n8k64.row.col.f32.e4m3.e4m3.f32"
test_variant "sparse f8f6f4 K=64 e5m2×e5m2"                   template_sparse_noscale "kind::f8f6f4.sp::ordered_metadata.m16n8k64.row.col.f32.e5m2.e5m2.f32"
test_variant "sparse f8f6f4 K=64 e2m3×e2m3 (FP6)"             template_sparse_noscale "kind::f8f6f4.sp::ordered_metadata.m16n8k64.row.col.f32.e2m3.e2m3.f32"
test_variant "sparse f8f6f4 K=64 e3m2×e3m2 (FP6)"             template_sparse_noscale "kind::f8f6f4.sp::ordered_metadata.m16n8k64.row.col.f32.e3m2.e3m2.f32"
test_variant "sparse f8f6f4 K=64 e4m3×e2m1 (mixed)"           template_sparse_noscale "kind::f8f6f4.sp::ordered_metadata.m16n8k64.row.col.f32.e4m3.e2m1.f32"

echo ""
echo "### SPARSE block-scale (kind::mxf4nvf4.sp — USER REQUESTED check)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
test_variant "sparse mxf4nvf4 4X K=128 ue4m3 (the headline-rejected one)" template_sparse_blockscale "kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3"
test_variant "sparse mxf4nvf4 4X K=128 ue8m0" template_sparse_blockscale "kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "sparse mxf4nvf4 2X K=128 ue4m3" template_sparse_blockscale "kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::2X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3"
test_variant "sparse mxf4nvf4 2X K=128 ue8m0" template_sparse_blockscale "kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::2X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "sparse mxf4nvf4 1X K=128 ue8m0" template_sparse_blockscale "kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::1X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "sparse mxf4nvf4 4X K=64 (smaller K)" template_sparse_blockscale "kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"

echo ""
echo "### SPARSE block-scale (kind::mxf8f6f4.sp — K=64)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
test_variant "sparse mxf8f6f4 1X K=64 e2m1×e2m1 ue8m0" template_sparse_blockscale "kind::mxf8f6f4.sp::ordered_metadata.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
test_variant "sparse mxf8f6f4 1X K=64 e4m3×e4m3 ue8m0" template_sparse_blockscale "kind::mxf8f6f4.sp::ordered_metadata.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e4m3.e4m3.f32.ue8m0"
test_variant "sparse mxf8f6f4 1X K=64 e2m3×e2m3 ue8m0 (FP6 sparse blockscale)" template_sparse_blockscale "kind::mxf8f6f4.sp::ordered_metadata.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e2m3.e2m3.f32.ue8m0"

echo ""
echo "### Sanity baseline (always-supported FP16/BF16 dense MMA)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
# Standard FP16/BF16 m16n8k16 — different operand count, use plain noscale template
template_fp16_baseline() {
    local etype="$1"
    cat <<EOF
__global__ void k(float* sink) {
    MMA_PRELUDE;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.${etype}.${etype}.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    SINK;
}
EOF
}
test_variant "FP16 m16n8k16 (sanity)" template_fp16_baseline "f16"
test_variant "BF16 m16n8k16 (sanity)" template_fp16_baseline "bf16"

echo ""
echo "Done. Re-run after CUDA toolkit upgrades."
