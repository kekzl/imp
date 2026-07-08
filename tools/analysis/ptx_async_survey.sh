#!/usr/bin/env bash
# PTX async memory & barrier instruction survey for sm_120f.
# Covers: cp.async (legacy), cp.async.bulk (TMA), mbarrier, st.async, fence.proxy.

set -e
# shellcheck source=tools/analysis/latest_cuda_img.sh
source "$(dirname "${BASH_SOURCE[0]}")/latest_cuda_img.sh"
CUDA_IMG=${CUDA_IMG:-$(latest_cuda_devel_img)}
ARCH=${ARCH:-compute_120f,code=sm_120}
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

PRELUDE='#include <cuda_runtime.h>
#include <cstdint>
extern __shared__ uint8_t smem[];
'

run_test() {
    local label="$1"; local body="$2"
    cat > "$WORK/t.cu" <<EOF
$PRELUDE
__global__ void k(uint8_t* gmem, uint64_t* desc, uint32_t* st_buf) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    (void)gmem; (void)desc; (void)st_buf; (void)smem_addr;
    $body
    if (threadIdx.x == 0 && blockIdx.x == 0) gmem[0] = smem[0];
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
            s/^Unsupported.*/Unsupported/
            s/^Unknown.*/Unknown modifier/
            s/^Illegal.*/Illegal modifier/
            s/^Modifier.*/Modifier rejected/
        ' | head -c 90)
        [[ -z "$r" ]] && r="(rejected)"
        echo "❌ | \`$label\` | $r"
    fi
}

echo ""
echo "## PTX async/barrier survey — sm_120f / $CUDA_IMG"
echo ""

echo "### cp.async (Ampere-style legacy async copy)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "cp.async.ca.shared.global 4-byte" \
    'asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr), "l"(gmem));'
run_test "cp.async.ca.shared.global 8-byte" \
    'asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(smem_addr), "l"(gmem));'
run_test "cp.async.ca.shared.global 16-byte" \
    'asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem));'
run_test "cp.async.cg.shared.global 16-byte (bypass L1)" \
    'asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem));'
run_test "cp.async.commit_group" \
    'asm volatile("cp.async.commit_group;\n");'
run_test "cp.async.wait_group 0" \
    'asm volatile("cp.async.wait_group 0;\n");'
run_test "cp.async.wait_all" \
    'asm volatile("cp.async.wait_all;\n");'

echo ""
echo "### cp.async.bulk (Hopper TMA)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "cp.async.bulk shared::cluster (CTA-pair)" \
    'asm volatile("cp.async.bulk.shared::cluster.global.bulk_group [%0], [%1], 128, [%2];\n" :: "r"(smem_addr), "l"(gmem), "r"(smem_addr));'
run_test "cp.async.bulk shared (single-CTA)" \
    'asm volatile("cp.async.bulk.shared.global.bulk_group [%0], [%1], 128;\n" :: "r"(smem_addr), "l"(gmem));'
run_test "cp.async.bulk.commit_group" \
    'asm volatile("cp.async.bulk.commit_group;\n");'
run_test "cp.async.bulk.wait_group 0" \
    'asm volatile("cp.async.bulk.wait_group 0;\n");'
run_test "cp.async.bulk.tensor.1d.tile.mbarrier" \
    'asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];\n" :: "r"(smem_addr), "l"(desc), "r"(0u), "r"(smem_addr));'
run_test "cp.async.bulk.tensor.2d.tile.mbarrier" \
    'asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];\n" :: "r"(smem_addr), "l"(desc), "r"(0u), "r"(0u), "r"(smem_addr));'
run_test "cp.async.bulk.tensor.3d.tile.mbarrier" \
    'asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n" :: "r"(smem_addr), "l"(desc), "r"(0u), "r"(0u), "r"(0u), "r"(smem_addr));'
run_test "cp.async.bulk.tensor.5d.tile.mbarrier" \
    'asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4,%5,%6}], [%7];\n" :: "r"(smem_addr), "l"(desc), "r"(0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u), "r"(smem_addr));'
run_test "cp.async.bulk.tensor.2d.im2col.mbarrier (im2col mode)" \
    'asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], {%5,%6}, [%7];\n" :: "r"(smem_addr), "l"(desc), "r"(0u), "r"(0u), "r"(0u), "h"((unsigned short)0), "h"((unsigned short)0), "r"(smem_addr));'
run_test "cp.async.bulk.tensor.2d.tile + bulk_group (no mbarrier)" \
    'asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.bulk_group [%0], [%1, {%2,%3}];\n" :: "r"(smem_addr), "l"(desc), "r"(0u), "r"(0u));'
run_test "cp.async.bulk.shared::cluster.global (raw bulk no tensor)" \
    'asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];\n" :: "r"(smem_addr), "l"(desc), "r"(smem_addr));'

echo ""
echo "### mbarrier (memory barrier for async ops)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "mbarrier.init.shared::cta.b64" \
    'asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(smem_addr));'
run_test "mbarrier.arrive.shared::cta.b64" \
    'uint64_t state; asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n" : "=l"(state) : "r"(smem_addr));'
run_test "mbarrier.arrive.expect_tx.shared::cta.b64" \
    'uint64_t state; asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], 128;\n" : "=l"(state) : "r"(smem_addr));'
run_test "mbarrier.try_wait.parity.shared::cta.b64" \
    'unsigned ok; asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(smem_addr));'
run_test "mbarrier.try_wait.shared::cta.b64" \
    'unsigned ok; uint64_t st = 0; asm volatile("{ .reg .pred p; mbarrier.try_wait.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(smem_addr), "l"(st));'

echo ""
echo "### st.async (async store)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "st.async.weak.shared::cluster.b32" \
    'asm volatile("st.async.weak.shared::cluster.b32 [%0], %1, [%2];\n" :: "r"(smem_addr), "r"(0u), "r"(smem_addr));'
run_test "st.async.weak.shared::cluster.b128 (4× b32)" \
    'asm volatile("st.async.weak.shared::cluster.v4.b32 [%0], {%1,%2,%3,%4}, [%5];\n" :: "r"(smem_addr), "r"(0u), "r"(0u), "r"(0u), "r"(0u), "r"(smem_addr));'
run_test "st.async.global.b32 (does it exist for global?)" \
    'asm volatile("st.async.weak.global.b32 [%0], %1, [%2];\n" :: "l"(st_buf), "r"(0u), "r"(smem_addr));'

echo ""
echo "### fence.proxy / async fences"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "fence.proxy.async" \
    'asm volatile("fence.proxy.async;\n");'
run_test "fence.proxy.async.shared::cta" \
    'asm volatile("fence.proxy.async.shared::cta;\n");'
run_test "fence.proxy.async.global" \
    'asm volatile("fence.proxy.async.global;\n");'
run_test "fence.proxy.tensormap::generic" \
    'asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" :: "l"(desc));'

echo ""
echo "### Programmatic Dependent Launch (PDL)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "griddepcontrol.wait" \
    'asm volatile("griddepcontrol.wait;\n");'
run_test "griddepcontrol.launch_dependents" \
    'asm volatile("griddepcontrol.launch_dependents;\n");'

echo ""
echo "Done. Re-run after CUDA upgrades."
