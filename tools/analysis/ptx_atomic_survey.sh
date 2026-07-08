#!/usr/bin/env bash
# PTX atomics + reductions survey for sm_120f.
# Covers: atom.global.* on FP/INT/vector types, red.global.*, redux.sync.*

set -e
# shellcheck source=tools/analysis/latest_cuda_img.sh
source "$(dirname "${BASH_SOURCE[0]}")/latest_cuda_img.sh"
CUDA_IMG=${CUDA_IMG:-$(latest_cuda_devel_img)}
ARCH=${ARCH:-compute_120f,code=sm_120}
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

PRELUDE='#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
'

run_test() {
    local label="$1"; local body="$2"
    cat > "$WORK/t.cu" <<EOF
$PRELUDE
__global__ void k(uint64_t* gmem, float* fbuf) {
    (void)gmem; (void)fbuf;
    $body
}
EOF
    local err
    err=$(docker run --rm -v "$WORK:/w" -w /w "$CUDA_IMG" \
            bash -c "nvcc -gencode=arch=${ARCH} -c t.cu -o /tmp/o.o 2>&1" 2>&1 | \
            grep -E "error|fatal|Feature|Instruction|Modifier|Unsupported|Unknown|Illegal" | head -1)
    if [[ -z "$err" ]]; then
        echo "✅ | \`$label\` | OK"
    else
        local r
        r=$(echo "$err" | sed -E '
            s/.*error\s*:\s*//
            s/^Feature[^.]*not supported on .*$/Feature not supported on sm_120/
            s/^Instruction[^.]*not supported.*/Instruction not supported/
            s/^Modifier[^.]* not supported.*/Modifier not supported/
            s/^Unsupported.*/Unsupported/
            s/^Unknown.*/Unknown modifier/
            s/^Illegal.*/Illegal modifier/
        ' | head -c 90)
        [[ -z "$r" ]] && r="(rejected)"
        echo "❌ | \`$label\` | $r"
    fi
}

echo ""
echo "## PTX atomics + reductions survey — sm_120f / $CUDA_IMG"
echo ""

echo "### atom.global add (numeric types)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "atom.global.add.u32" \
    'uint32_t r; asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gmem), "r"(1u));'
run_test "atom.global.add.u64" \
    'uint64_t r; asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r) : "l"(gmem), "l"(1ul));'
run_test "atom.global.add.f32" \
    'float r; asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(r) : "l"(fbuf), "f"(1.f));'
run_test "atom.global.add.f64" \
    'double r; asm volatile("atom.global.add.f64 %0, [%1], %2;" : "=d"(r) : "l"(fbuf), "d"(1.0));'
run_test "atom.global.add.noftz.f16 (scalar half)" \
    'uint16_t r; asm volatile("atom.global.add.noftz.f16 %0, [%1], %2;" : "=h"(r) : "l"(fbuf), "h"((uint16_t)0x3c00));'
run_test "atom.global.add.noftz.bf16 (scalar bfloat)" \
    'uint16_t r; asm volatile("atom.global.add.noftz.bf16 %0, [%1], %2;" : "=h"(r) : "l"(fbuf), "h"((uint16_t)0x3f80));'
run_test "atom.global.add.noftz.f16x2 (vector half2)" \
    'uint32_t r; asm volatile("atom.global.add.noftz.f16x2 %0, [%1], %2;" : "=r"(r) : "l"(fbuf), "r"(0x3c003c00u));'
run_test "atom.global.add.noftz.bf16x2 (vector bf16x2)" \
    'uint32_t r; asm volatile("atom.global.add.noftz.bf16x2 %0, [%1], %2;" : "=r"(r) : "l"(fbuf), "r"(0x3f803f80u));'

echo ""
echo "### atom.global min/max"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "atom.global.min.u32" \
    'uint32_t r; asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(r) : "l"(gmem), "r"(1u));'
run_test "atom.global.max.u32" \
    'uint32_t r; asm volatile("atom.global.max.u32 %0, [%1], %2;" : "=r"(r) : "l"(gmem), "r"(1u));'
run_test "atom.global.min.s32" \
    'int32_t r; asm volatile("atom.global.min.s32 %0, [%1], %2;" : "=r"(r) : "l"(gmem), "r"(1));'

echo ""
echo "### atom.global cas / exch / and / or / xor"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "atom.global.cas.b32" \
    'uint32_t r; asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r) : "l"(gmem), "r"(0u), "r"(1u));'
run_test "atom.global.cas.b64" \
    'uint64_t r; asm volatile("atom.global.cas.b64 %0, [%1], %2, %3;" : "=l"(r) : "l"(gmem), "l"(0ul), "l"(1ul));'
run_test "atom.global.exch.b32" \
    'uint32_t r; asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(r) : "l"(gmem), "r"(1u));'
run_test "atom.global.and.b32" \
    'uint32_t r; asm volatile("atom.global.and.b32 %0, [%1], %2;" : "=r"(r) : "l"(gmem), "r"(0xFFu));'

echo ""
echo "### red.global (reduce, no return)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "red.global.add.u32" \
    'asm volatile("red.global.add.u32 [%0], %1;" :: "l"(gmem), "r"(1u));'
run_test "red.global.add.f32" \
    'asm volatile("red.global.add.f32 [%0], %1;" :: "l"(fbuf), "f"(1.f));'
run_test "red.global.add.noftz.f16x2 (vector half reduce)" \
    'asm volatile("red.global.add.noftz.f16x2 [%0], %1;" :: "l"(fbuf), "r"(0x3c003c00u));'
run_test "red.global.add.noftz.bf16x2" \
    'asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" :: "l"(fbuf), "r"(0x3f803f80u));'

echo ""
echo "### multimem (DSMEM cluster reduction — sm_100+ feature)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "multimem.ld_reduce.add.f32" \
    'float r; asm volatile("multimem.ld_reduce.add.f32 %0, [%1];" : "=f"(r) : "l"(fbuf));'
run_test "multimem.st.b32" \
    'asm volatile("multimem.st.b32 [%0], %1;" :: "l"(fbuf), "r"(0u));'
run_test "multimem.red.add.f32" \
    'asm volatile("multimem.red.add.f32 [%0], %1;" :: "l"(fbuf), "f"(1.f));'

echo ""
echo "### redux.sync (warp-level reduction — Volta+)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "redux.sync.add.s32" \
    'int r; asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(1));'
run_test "redux.sync.add.u32" \
    'uint32_t r; asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(1u));'
run_test "redux.sync.min.u32" \
    'uint32_t r; asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(1u));'
run_test "redux.sync.max.s32" \
    'int r; asm volatile("redux.sync.max.s32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(1));'
run_test "redux.sync.and.b32" \
    'uint32_t r; asm volatile("redux.sync.and.b32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(0xFFu));'
run_test "redux.sync.or.b32" \
    'uint32_t r; asm volatile("redux.sync.or.b32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(0xFFu));'
run_test "redux.sync.add.f32 (FP variant?)" \
    'float r; asm volatile("redux.sync.add.f32 %0, %1, 0xFFFFFFFF;" : "=f"(r) : "f"(1.f));'
run_test "redux.sync.min.f32" \
    'float r; asm volatile("redux.sync.min.f32 %0, %1, 0xFFFFFFFF;" : "=f"(r) : "f"(1.f));'

echo ""
echo "### Cluster sync"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "bar.cluster.sync" \
    'asm volatile("barrier.cluster.sync;\n");'
run_test "bar.cluster.arrive" \
    'asm volatile("barrier.cluster.arrive;\n");'
run_test "bar.cluster.wait" \
    'asm volatile("barrier.cluster.wait;\n");'

echo ""
echo "Done."
