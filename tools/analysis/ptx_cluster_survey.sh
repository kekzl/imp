#!/usr/bin/env bash
# PTX cluster / multimem / tcgen05 / wgmma survey for sm_120f.
# Goal: confirm which sm_100+ "data center Blackwell" features ARE or are NOT
# available on consumer Blackwell sm_120f.

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
__global__ void k(uint64_t* gmem, float* fbuf, uint32_t* tm) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    (void)gmem; (void)fbuf; (void)tm; (void)smem_addr;
    $body
}
EOF
    local err
    err=$(docker run --rm -v "$WORK:/w" -w /w "$CUDA_IMG" \
            bash -c "nvcc -gencode=arch=${ARCH} -c t.cu -o /tmp/o.o 2>&1" 2>&1 | \
            grep -E "error|fatal|Feature|Instruction|Modifier|Unsupported|Unknown|Illegal|Argument|name" | head -1)
    if [[ -z "$err" ]]; then
        echo "✅ | \`$label\` | OK"
    else
        local r
        r=$(echo "$err" | sed -E '
            s/.*error\s*:\s*//
            s/^Feature[^.]*not supported on .*$/Feature not supported on sm_120/
            s/^Instruction[^.]*not supported.*/Instruction not supported on sm_120/
            s/^Unsupported.*/Unsupported/
            s/^Unknown.*/Unknown modifier/
            s/^Illegal.*/Illegal modifier/
            s/^Arguments.*/Arguments mismatch/
            s/^Not a name.*/Not a known instruction/
        ' | head -c 90)
        [[ -z "$r" ]] && r="(rejected)"
        echo "❌ | \`$label\` | $r"
    fi
}

echo ""
echo "## PTX cluster / multimem / TCGEN05 / wgmma survey — sm_120f / $CUDA_IMG"
echo ""
echo "Tests sm_100+ \"data center Blackwell\" features against consumer Blackwell."
echo ""

echo "### Cluster sync / mapa (cluster shared memory address translation)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "barrier.cluster.arrive" \
    'asm volatile("barrier.cluster.arrive;\n");'
run_test "barrier.cluster.wait" \
    'asm volatile("barrier.cluster.wait;\n");'
run_test "barrier.cluster.arrive.relaxed" \
    'asm volatile("barrier.cluster.arrive.relaxed;\n");'
run_test "%cluster_ctaid.x (cluster CTA ID)" \
    'uint32_t r; asm volatile("mov.u32 %0, %cluster_ctaid.x;" : "=r"(r));'
run_test "%cluster_nctaid.x (cluster size)" \
    'uint32_t r; asm volatile("mov.u32 %0, %cluster_nctaid.x;" : "=r"(r));'
run_test "%cluster_ctarank" \
    'uint32_t r; asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(r));'
run_test "mapa.shared::cluster.u32 (DSMEM addr translate)" \
    'uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(0u));'
run_test "mapa.shared::cluster.u64" \
    'uint64_t r; asm volatile("mapa.shared::cluster.u64 %0, %1, %2;" : "=l"(r) : "r"(smem_addr), "r"(0u));'
run_test "getctarank.shared::cluster.u32" \
    'uint32_t r; asm volatile("getctarank.shared::cluster.u32 %0, %1;" : "=r"(r) : "r"(smem_addr));'

echo ""
echo "### Multimem (DSMEM cluster reduction / store)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "multimem.ld_reduce.add.f32" \
    'float r; asm volatile("multimem.ld_reduce.add.f32 %0, [%1];" : "=f"(r) : "l"(fbuf));'
run_test "multimem.ld_reduce.add.f16" \
    'uint16_t r; asm volatile("multimem.ld_reduce.add.noftz.f16 %0, [%1];" : "=h"(r) : "l"(fbuf));'
run_test "multimem.ld_reduce.add.f16x2" \
    'uint32_t r; asm volatile("multimem.ld_reduce.add.noftz.f16x2 %0, [%1];" : "=r"(r) : "l"(fbuf));'
run_test "multimem.ld_reduce.add.bf16x2" \
    'uint32_t r; asm volatile("multimem.ld_reduce.add.noftz.bf16x2 %0, [%1];" : "=r"(r) : "l"(fbuf));'
run_test "multimem.ld_reduce.add.v4.f32" \
    'float r0,r1,r2,r3; asm volatile("multimem.ld_reduce.add.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3) : "l"(fbuf));'
run_test "multimem.ld_reduce.min.f32" \
    'float r; asm volatile("multimem.ld_reduce.min.f32 %0, [%1];" : "=f"(r) : "l"(fbuf));'
run_test "multimem.st.f32" \
    'asm volatile("multimem.st.f32 [%0], %1;" :: "l"(fbuf), "f"(1.f));'
run_test "multimem.red.add.f32" \
    'asm volatile("multimem.red.add.f32 [%0], %1;" :: "l"(fbuf), "f"(1.f));'
run_test "multimem.red.add.f16x2 (vector half)" \
    'asm volatile("multimem.red.add.noftz.f16x2 [%0], %1;" :: "l"(fbuf), "r"(0u));'

echo ""
echo "### TCGEN05 (Tensor Core Gen 5 — sm_100/sm_103 only?)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "tcgen05.alloc.shared::cta.b32" \
    'asm volatile("tcgen05.alloc.cta_group::1.shared::cta.b32 [%0], %1;" :: "r"(smem_addr), "r"(64u));'
run_test "tcgen05.dealloc.cta_group::1.b32" \
    'asm volatile("tcgen05.dealloc.cta_group::1.b32 %0, %1;" :: "r"(0u), "r"(64u));'
run_test "tcgen05.relinquish_alloc_permit.cta_group::1" \
    'asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1;\n");'
run_test "tcgen05.commit.cta_group::1.mbarrier" \
    'asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_addr));'
run_test "tcgen05.cp.cta_group::1.4x256b (TMEM copy)" \
    'asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;" :: "r"(0u), "l"(0xDEADBEEFul));'
run_test "tcgen05.fence::after_thread_sync" \
    'asm volatile("tcgen05.fence::after_thread_sync;\n");'
run_test "tcgen05.shift.cta_group::1.down" \
    'asm volatile("tcgen05.shift.cta_group::1.down [%0];" :: "r"(0u));'
run_test "tcgen05.wait::ld.sync.aligned" \
    'asm volatile("tcgen05.wait::ld.sync.aligned;\n");'
run_test "tcgen05.mma.cta_group::1.kind::f16 (TMEM-input MMA)" \
    'asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, false;" :: "r"(0u), "l"(0ul), "l"(0ul), "r"(0u));'

echo ""
echo "### wgmma (Warp-Group MMA, Hopper-style — sm_90 only?)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "wgmma.fence.sync.aligned" \
    'asm volatile("wgmma.fence.sync.aligned;\n");'
run_test "wgmma.commit_group.sync.aligned" \
    'asm volatile("wgmma.commit_group.sync.aligned;\n");'
run_test "wgmma.wait_group.sync.aligned 0" \
    'asm volatile("wgmma.wait_group.sync.aligned 0;\n");'
run_test "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16" \
    'float d[4]={0,0,0,0}; asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, %4, %5, 1, 1, 1, 0, 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "l"(0ul), "l"(0ul));'

echo ""
echo "### Async stmatrix / ldmatrix (smem ↔ register fragment helpers)"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "ldmatrix.sync.aligned.m8n8.x1.shared.b16" \
    'uint32_t r; asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r) : "r"(smem_addr));'
run_test "ldmatrix.sync.aligned.m8n8.x4.shared.b16" \
    'uint32_t r0,r1,r2,r3; asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(smem_addr));'
run_test "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 (transposed)" \
    'uint32_t r0,r1,r2,r3; asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(smem_addr));'
run_test "ldmatrix.sync.aligned.m16n16.x1.shared.b8 (8-bit fragments)" \
    'uint32_t r; asm volatile("ldmatrix.sync.aligned.m16n16.x1.shared.b8 {%0}, [%1];" : "=r"(r) : "r"(smem_addr));'
run_test "stmatrix.sync.aligned.m8n8.x1.shared.b16" \
    'asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};" :: "r"(smem_addr), "r"(0u));'
run_test "stmatrix.sync.aligned.m8n8.x4.shared.b16" \
    'asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(smem_addr), "r"(0u), "r"(0u), "r"(0u), "r"(0u));'
run_test "stmatrix.sync.aligned.m16n8.x1.shared.b8" \
    'asm volatile("stmatrix.sync.aligned.m16n8.x1.shared.b8 [%0], {%1};" :: "r"(smem_addr), "r"(0u));'

echo ""
echo "### Tensormap / TMA descriptor manipulation"
echo ""
echo "Status | Variant | Reason"
echo "---|---|---"
run_test "tensormap.replace.tile.global_address.shared::cta.b1024.b64" \
    'asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "r"(smem_addr), "l"(0xDEADBEEFul));'
run_test "tensormap.cp_fenceproxy.global.shared::cta" \
    'asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;" :: "l"(gmem), "r"(smem_addr));'

echo ""
echo "Done."
