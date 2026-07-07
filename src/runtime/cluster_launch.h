#pragma once

// Cluster launch helper — wraps the cudaLaunchKernelEx + cluster-dimension
// + GPC-spread scheduling boilerplate that paged_attention_cluster_kernel
// established for DSMEM K-broadcast across Q-heads (sm_90+; works on
// sm_120a / Consumer Blackwell with the same calling convention).
//
// M5 Slice 1 from review/phase5_synthesis.md §2.2:
//   - The existing decode-path GQA cluster launcher at
//     src/compute/attention_paged.cu:1456-1474 is the working reference.
//   - Slice 1 extracts that boilerplate into a reusable
//     cluster_launch(kernel, ...) template so a future FMHA-prefill
//     migration (Slice 2) can opt into cluster launch without copying
//     20 lines of attribute setup.
//
// GB202 specifics:
//   - 12 GPCs. Default cluster scheduling packs clusters per-GPC which
//     oversubscribes when only a handful are live. `Spread` gives each
//     cluster its own GPC for small grids, keeping DSMEM traffic local
//     and freeing remaining GPCs for concurrent work on other streams.
//   - Cluster dimension must be a power of 2 (CUDA requirement).
//   - DSMEM is part of distributed shared memory: blocks within a cluster
//     can read each other's smem via `cluster.map_shared_rank(...)`.

#include <cuda_runtime.h>

namespace imp {

namespace cluster {

// Default cluster scheduling on GB202: Spread across GPCs for small
// grids. Override with cudaClusterSchedulingPolicyDefault if grids are
// large enough to saturate every GPC organically (in practice, every
// imp cluster launch today is in the spread-favorable regime).
inline cudaLaunchAttribute spread_attr() {
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attr.val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
    return attr;
}

inline cudaLaunchAttribute cluster_dim_attr(unsigned int x, unsigned int y = 1, unsigned int z = 1) {
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {x, y, z};
    return attr;
}

// Build a cudaLaunchConfig_t for a cluster launch. `attrs` must point to
// storage with lifetime spanning the launch call (caller-owned). The
// helper writes (cluster_dim, spread_policy) into attrs[0..1] and sets
// config.attrs/numAttrs accordingly. Callers can append more attributes
// (PDL, access-policy window) after the helper returns by extending
// attrs[] and bumping numAttrs.
//
// Returns by value; callers pass `&config` to cudaLaunchKernelEx.
inline cudaLaunchConfig_t build_cluster_config(dim3 grid, dim3 block, size_t dyn_smem,
                                               cudaStream_t stream, cudaLaunchAttribute* attrs,
                                               unsigned int cluster_x, unsigned int cluster_y = 1,
                                               unsigned int cluster_z = 1) {
    attrs[0] = cluster_dim_attr(cluster_x, cluster_y, cluster_z);
    attrs[1] = spread_attr();
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = dyn_smem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;
    return config;
}


// Cluster dimension validity check (CUDA requires power-of-2 in each axis
// and total cluster size ≤ 16 on GB202). Use at launch sites that
// dispatch on a runtime value (e.g. n_q_per_kv in GQA attention).
inline bool valid_cluster_dim(unsigned int x, unsigned int y = 1, unsigned int z = 1) {
    auto is_pow2 = [](unsigned int v) { return v > 0 && (v & (v - 1)) == 0; };
    if (!is_pow2(x) || !is_pow2(y) || !is_pow2(z)) return false;
    return (x * y * z) <= 16u;
}

}  // namespace cluster

}  // namespace imp
