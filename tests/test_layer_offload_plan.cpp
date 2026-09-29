// --gpu-layers planning (#2291): Qwen3-8B-Q8_0 geometry, 36 layers, gpu_layers=20.
#include "memory/layer_offload.h"
#include "memory/layer_offload_plan.h"

#include <gtest/gtest.h>

namespace imp {
namespace {

// Qwen3-8B: d_model 4096, 8 KV heads x 128, FFN 12288. Q8_0 = 34 B per 32 elements.
constexpr int kLayers = 36;
constexpr size_t kQ8LayerBytes = 2 * 17'825'792 + 2 * 4'456'448 + 3 * 53'477'376;  // 204'996'608
constexpr size_t kNormBytes = 4096 * 4 * 2 + 128 * 4 * 2;                            // 33'792
constexpr size_t kLayerBytes = kQ8LayerBytes + kNormBytes;                           // 205'030'400

char g_dummy;

Tensor make(QType q, int64_t rows, int64_t cols, bool on_device) {
    const int64_t shape2[4] = {rows, cols, 0, 0};
    const int64_t shape1[4] = {cols, 0, 0, 0};
    return rows > 0 ? Tensor(&g_dummy, q, 2, shape2, on_device) : Tensor(&g_dummy, q, 1, shape1, on_device);
}

void build_qwen3_8b_q8_0(Model& m, bool on_device) {
    m.layers_.resize(kLayers);
    for (auto& L : m.layers_) {
        L.wq = make(QType::Q8_0, 4096, 4096, on_device);
        L.wk = make(QType::Q8_0, 1024, 4096, on_device);
        L.wv = make(QType::Q8_0, 1024, 4096, on_device);
        L.wo = make(QType::Q8_0, 4096, 4096, on_device);
        L.w_gate = make(QType::Q8_0, 12288, 4096, on_device);
        L.w_up = make(QType::Q8_0, 12288, 4096, on_device);
        L.w_down = make(QType::Q8_0, 4096, 12288, on_device);
        L.attn_norm = make(QType::F32, 0, 4096, on_device);
        L.attn_q_norm = make(QType::F32, 0, 128, on_device);
        L.attn_k_norm = make(QType::F32, 0, 128, on_device);
        L.ffn_norm = make(QType::F32, 0, 4096, on_device);
    }
}

TEST(LayerOffloadPlan, Q8_0StorageBytesAreBlockSized) {
    EXPECT_EQ(tensor_storage_bytes(make(QType::Q8_0, 12288, 4096, false)), 53'477'376u);
    EXPECT_EQ(tensor_storage_bytes(make(QType::F32, 0, 4096, false)), 16'384u);
}

TEST(LayerOffloadPlan, Qwen3_8B_Q8_0_GpuLayers20Plans16HostLayers) {
    Model m;
    build_qwen3_8b_q8_0(m, /*on_device=*/false);
    const LayerOffloadPlan p = plan_layer_offload(m, 20);
    ASSERT_TRUE(p.active);
    EXPECT_EQ(p.n_offloaded, 16);
    for (int i = 0; i < kLayers; i++) {
        EXPECT_EQ(p.offloaded[i], i >= 20) << "layer " << i;
        EXPECT_EQ(p.layer_bytes[i], kLayerBytes) << "layer " << i;
        EXPECT_EQ(p.host_bytes[i], kLayerBytes) << "layer " << i;
    }
    EXPECT_EQ(p.planned_bytes, 16 * kLayerBytes);
    EXPECT_EQ(p.max_host_bytes, kLayerBytes);
}

// Post-upload state: every weight is device-resident, so the plan cannot be staged from host.
TEST(LayerOffloadPlan, DeviceResidentWeightsPlanBytesButNoHostPayload) {
    Model m;
    build_qwen3_8b_q8_0(m, /*on_device=*/true);
    const LayerOffloadPlan p = plan_layer_offload(m, 20);
    ASSERT_TRUE(p.active);
    EXPECT_EQ(p.n_offloaded, 16);
    EXPECT_EQ(p.planned_bytes, 16 * kLayerBytes);
    EXPECT_EQ(p.max_host_bytes, 0u);
}

// #2291: init must fail with the reason, not log "nothing to offload" and return true.
TEST(LayerOffloadPlan, ManagerInitFailsWhenPlannedLayersAreDeviceResident) {
    Model m;
    build_qwen3_8b_q8_0(m, /*on_device=*/true);
    LayerOffloadManager mgr;
    EXPECT_FALSE(mgr.init(&m, 20));
    EXPECT_FALSE(mgr.is_enabled());
}

TEST(LayerOffloadPlan, AllLayersOnGpuIsInactive) {
    Model m;
    build_qwen3_8b_q8_0(m, /*on_device=*/true);
    EXPECT_FALSE(plan_layer_offload(m, -1).active);
    EXPECT_FALSE(plan_layer_offload(m, kLayers).active);
    LayerOffloadManager mgr;
    EXPECT_TRUE(mgr.init(&m, -1));
    EXPECT_TRUE(mgr.init(&m, kLayers));
    EXPECT_FALSE(mgr.is_enabled());
}

}  // namespace
}  // namespace imp
