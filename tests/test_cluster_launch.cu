#include "runtime/cluster_launch.h"

#include <gtest/gtest.h>

using namespace imp;

// ---------------------------------------------------------------------------
// M5 Slice 1 — cluster_launch helper unit tests.
//
// These pin the pure-host helpers (config builders, validity checks)
// without requiring a GPU launch. The full cluster kernel A/B that
// validates DSMEM K-broadcast performance lives in the FMHA prefill
// migration (Slice 2 of M5).
// ---------------------------------------------------------------------------

TEST(ClusterLaunch, ValidClusterDimsArePow2AndLeq16) {
    EXPECT_TRUE(cluster::valid_cluster_dim(1));
    EXPECT_TRUE(cluster::valid_cluster_dim(2));
    EXPECT_TRUE(cluster::valid_cluster_dim(4));
    EXPECT_TRUE(cluster::valid_cluster_dim(8));
    EXPECT_TRUE(cluster::valid_cluster_dim(16));
    EXPECT_TRUE(cluster::valid_cluster_dim(2, 2));   // 4 total
    EXPECT_TRUE(cluster::valid_cluster_dim(4, 4));   // 16 total
    EXPECT_TRUE(cluster::valid_cluster_dim(8, 2));   // 16 total
}

TEST(ClusterLaunch, InvalidClusterDimsRejected) {
    EXPECT_FALSE(cluster::valid_cluster_dim(0));      // zero is not pow-2
    EXPECT_FALSE(cluster::valid_cluster_dim(3));      // not pow-2
    EXPECT_FALSE(cluster::valid_cluster_dim(5));
    EXPECT_FALSE(cluster::valid_cluster_dim(6));
    EXPECT_FALSE(cluster::valid_cluster_dim(7));
    EXPECT_FALSE(cluster::valid_cluster_dim(32));     // >16 total
    EXPECT_FALSE(cluster::valid_cluster_dim(8, 4));   // 32 total
    EXPECT_FALSE(cluster::valid_cluster_dim(4, 8));   // 32 total
}

TEST(ClusterLaunch, BuildConfigPopulatesAttrsAndDims) {
    cudaLaunchAttribute attrs[2];
    const dim3 grid(64, 1);
    const dim3 block(128);
    const size_t smem = 32 * 1024;
    cudaLaunchConfig_t cfg =
        cluster::build_cluster_config(grid, block, smem, /*stream=*/nullptr, attrs, /*cluster_x=*/4);

    EXPECT_EQ(cfg.gridDim.x, grid.x);
    EXPECT_EQ(cfg.gridDim.y, grid.y);
    EXPECT_EQ(cfg.blockDim.x, block.x);
    EXPECT_EQ(cfg.dynamicSmemBytes, smem);
    EXPECT_EQ(cfg.stream, nullptr);
    EXPECT_EQ(cfg.numAttrs, 2u);
    EXPECT_EQ(cfg.attrs, attrs);

    EXPECT_EQ(attrs[0].id, cudaLaunchAttributeClusterDimension);
    EXPECT_EQ(attrs[0].val.clusterDim.x, 4u);
    EXPECT_EQ(attrs[0].val.clusterDim.y, 1u);
    EXPECT_EQ(attrs[0].val.clusterDim.z, 1u);

    EXPECT_EQ(attrs[1].id, cudaLaunchAttributeClusterSchedulingPolicyPreference);
    EXPECT_EQ(attrs[1].val.clusterSchedulingPolicyPreference, cudaClusterSchedulingPolicySpread);
}

TEST(ClusterLaunch, BuildConfig3D) {
    cudaLaunchAttribute attrs[2];
    cudaLaunchConfig_t cfg = cluster::build_cluster_config(dim3(1, 1), dim3(32), 0, nullptr, attrs,
                                                            /*cluster_x=*/2, /*cluster_y=*/2,
                                                            /*cluster_z=*/2);  // 8 total
    EXPECT_EQ(attrs[0].val.clusterDim.x, 2u);
    EXPECT_EQ(attrs[0].val.clusterDim.y, 2u);
    EXPECT_EQ(attrs[0].val.clusterDim.z, 2u);
}
