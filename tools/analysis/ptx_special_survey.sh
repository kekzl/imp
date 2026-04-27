#!/usr/bin/env bash
# PTX special-function-unit (SFU) + math survey for sm_120f.
# Covers: rcp/rsqrt/ex2/lg2/sin/cos/tanh + approx variants, divide,
# packed FP16 arithmetic (add/mul/fma f16x2/bf16x2), and special activations.

set -e
CUDA_IMG=${CUDA_IMG:-nvidia/cuda:13.2.1-devel-ubuntu24.04}
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
__global__ void k(float* fbuf, uint32_t* ubuf) {
    (void)fbuf; (void)ubuf;
    $body
}
EOF
    local err
    err=$(docker run --rm -v "$WORK:/w" -w /w "$CUDA_IMG" \
            bash -c "nvcc -gencode=arch=${ARCH} -c t.cu -o /tmp/o.o 2>&1" 2>&1 | \
            grep -E "error|fatal|Feature|Instruction|Modifier|Unsupported|Unknown|Illegal|Argument" | head -1)
    if [[ -z "$err" ]]; then
        echo "✅ | \`$label\` | OK"
    else
        local r
        r=$(echo "$err" | sed -E '
            s/.*error\s*:\s*//
            s/^Feature[^.]*not supported on .*$/Feature not supported on sm_120/
            s/^Instruction[^.]*not supported.*/Instruction not supported/
            s/^Unsupported.*/Unsupported/
            s/^Unknown.*/Unknown modifier/
            s/^Illegal.*/Illegal modifier/
            s/^Arguments.*/Arguments mismatch/
        ' | head -c 90)
        [[ -z "$r" ]] && r="(rejected)"
        echo "❌ | \`$label\` | $r"
    fi
}

echo ""
echo "## PTX SFU + math survey — sm_120f / $CUDA_IMG"
echo ""

echo "### Special-function-unit (SFU) approximations"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "rcp.approx.f32" \
    'float r; asm volatile("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "rcp.approx.ftz.f32" \
    'float r; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "rsqrt.approx.f32" \
    'float r; asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "rsqrt.approx.ftz.f32" \
    'float r; asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "ex2.approx.f32" \
    'float r; asm volatile("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "ex2.approx.ftz.f32" \
    'float r; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "lg2.approx.f32" \
    'float r; asm volatile("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "sin.approx.f32" \
    'float r; asm volatile("sin.approx.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "cos.approx.f32" \
    'float r; asm volatile("cos.approx.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "tanh.approx.f32" \
    'float r; asm volatile("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "tanh.approx.f16 (scalar half)" \
    'uint16_t r; asm volatile("tanh.approx.f16 %0, %1;" : "=h"(r) : "h"((uint16_t)0x4000));'
run_test "tanh.approx.f16x2 (packed half)" \
    'uint32_t r; asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(r) : "r"(0x40004000u));'
run_test "tanh.approx.bf16x2 (packed bfloat)" \
    'uint32_t r; asm volatile("tanh.approx.bf16x2 %0, %1;" : "=r"(r) : "r"(0x40004000u));'
run_test "ex2.approx.f16 (does it exist?)" \
    'uint16_t r; asm volatile("ex2.approx.f16 %0, %1;" : "=h"(r) : "h"((uint16_t)0x4000));'
run_test "ex2.approx.f16x2" \
    'uint32_t r; asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(0x40004000u));'
run_test "rcp.approx.f16x2" \
    'uint32_t r; asm volatile("rcp.approx.f16x2 %0, %1;" : "=r"(r) : "r"(0x40004000u));'
run_test "rsqrt.approx.f16x2" \
    'uint32_t r; asm volatile("rsqrt.approx.f16x2 %0, %1;" : "=r"(r) : "r"(0x40004000u));'
run_test "rcp.approx.bf16x2" \
    'uint32_t r; asm volatile("rcp.approx.bf16x2 %0, %1;" : "=r"(r) : "r"(0x40004000u));'

echo ""
echo "### Division & full-precision approximations"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "div.approx.f32" \
    'float r; asm volatile("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(2.f), "f"(3.f));'
run_test "rcp.rn.f32 (full precision)" \
    'float r; asm volatile("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "rsqrt.approx.f64 (double)" \
    'double r; asm volatile("rsqrt.approx.f64 %0, %1;" : "=d"(r) : "d"(2.0));'
run_test "sqrt.approx.f32" \
    'float r; asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(2.f));'
run_test "sqrt.rn.f32" \
    'float r; asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(r) : "f"(2.f));'

echo ""
echo "### Packed FP arithmetic (half2 / bf16x2 native ops)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "add.f16x2" \
    'uint32_t r; asm volatile("add.f16x2 %0, %1, %2;" : "=r"(r) : "r"(0x40004000u), "r"(0x3c003c00u));'
run_test "mul.f16x2" \
    'uint32_t r; asm volatile("mul.f16x2 %0, %1, %2;" : "=r"(r) : "r"(0x40004000u), "r"(0x3c003c00u));'
run_test "fma.rn.f16x2" \
    'uint32_t r; asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(0x40004000u), "r"(0x3c003c00u), "r"(0x40004000u));'
run_test "fma.rn.bf16x2" \
    'uint32_t r; asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(0x3f803f80u), "r"(0x3f803f80u), "r"(0x3f803f80u));'
run_test "fma.rn.relu.f16x2 (fused ReLU)" \
    'uint32_t r; asm volatile("fma.rn.relu.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(0x40004000u), "r"(0x3c003c00u), "r"(0x40004000u));'
run_test "min.f16x2" \
    'uint32_t r; asm volatile("min.f16x2 %0, %1, %2;" : "=r"(r) : "r"(0x40004000u), "r"(0x3c003c00u));'
run_test "max.f16x2" \
    'uint32_t r; asm volatile("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(0x40004000u), "r"(0x3c003c00u));'

echo ""
echo "### Bit manipulation / permute"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "prmt.b32" \
    'uint32_t r; asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(0u), "r"(1u), "r"(0x4321u));'
run_test "prmt.f4e.b32 (forward 4 extract)" \
    'uint32_t r; asm volatile("prmt.f4e.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(0u), "r"(1u), "r"(0u));'
run_test "bfe.u32 (bit field extract)" \
    'uint32_t r; asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(0xFFu), "r"(0u), "r"(8u));'
run_test "bfi.b32 (bit field insert)" \
    'uint32_t r; asm volatile("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(r) : "r"(0xFFu), "r"(0u), "r"(0u), "r"(8u));'
run_test "fns.b32 (find n-th set bit)" \
    'int r; asm volatile("fns.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(0xFFu), "r"(0u), "r"(1));'
run_test "lop3.b32 (lookup-table 3-input boolean)" \
    'uint32_t r; asm volatile("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r) : "r"(0xFFu), "r"(0xF0u), "r"(0x0Fu));'
run_test "popc.b32 (population count)" \
    'uint32_t r; asm volatile("popc.b32 %0, %1;" : "=r"(r) : "r"(0xFFu));'
run_test "clz.b32 (count leading zeros)" \
    'uint32_t r; asm volatile("clz.b32 %0, %1;" : "=r"(r) : "r"(0xFFu));'
run_test "shf.l.wrap.b32 (funnel shift left)" \
    'uint32_t r; asm volatile("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(0xFFu), "r"(0xFFu), "r"(4u));'

echo ""
echo "### dp4a / dp2a (INT8/INT16 dot-product)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "dp4a.s32.s32 (signed INT8 dp4a)" \
    'int r; asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(1u), "r"(1u), "r"(0));'
run_test "dp4a.u32.u32 (unsigned INT8 dp4a)" \
    'uint32_t r; asm volatile("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(1u), "r"(1u), "r"(0u));'
run_test "dp2a.lo.s32.s32 (INT16 dp2a low)" \
    'int r; asm volatile("dp2a.lo.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(1u), "r"(1u), "r"(0));'
run_test "dp4a.s32.u32 (mixed sign)" \
    'int r; asm volatile("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(1u), "r"(1u), "r"(0));'

echo ""
echo "### Misc system / utility"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "nanosleep.u32" \
    'asm volatile("nanosleep.u32 100;\n");'
run_test "clock.lo.s64" \
    'int64_t r; asm volatile("mov.u64 %0, %clock64;" : "=l"(r));'
run_test "%smid (SM ID)" \
    'uint32_t r; asm volatile("mov.u32 %0, %smid;" : "=r"(r));'
run_test "%nsmid (number of SMs)" \
    'uint32_t r; asm volatile("mov.u32 %0, %nsmid;" : "=r"(r));'
run_test "activemask" \
    'uint32_t r; asm volatile("activemask.b32 %0;" : "=r"(r));'
run_test "match.any.sync.b32" \
    'uint32_t r; asm volatile("match.any.sync.b32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(0u));'
run_test "match.all.sync.b32" \
    'uint32_t r; uint32_t pred; asm volatile("match.all.sync.b32 %0|%1, %2, 0xFFFFFFFF;" : "=r"(r), "=r"(pred) : "r"(0u));'

echo ""
echo "Done."
