// --gpu-layers Pass 1 (#2298): plan -> which tensors stay on host, on a stub device and a stub pin.
#include "memory/layer_offload.h"
#include "model/layer_host_keep.h"
#include "model/weight_upload_traits.h"
#include "quant/dequant_gpu.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <set>
#include <vector>

namespace imp {
namespace {

char g_dummy;

Tensor make(void* data, QType q, int64_t rows, int64_t cols, bool on_device = false) {
    const int64_t shape2[4] = {rows, cols, 0, 0};
    const int64_t shape1[4] = {cols, 0, 0, 0};
    return rows > 0 ? Tensor(data, q, 2, shape2, on_device) : Tensor(data, q, 1, shape1, on_device);
}

// Qwen3-8B-Q8_0: 36 layers, d 4096, 8 KV heads x 128, FFN 12288. Matmuls 204'996'608 B per layer.
constexpr int kLayers = 36;
constexpr size_t kQ8LayerBytes = 2 * 17'825'792 + 2 * 4'456'448 + 3 * 53'477'376;

void build_qwen3_8b(Model& m) {
    m.layers_.resize(kLayers);
    for (auto& L : m.layers_) {
        L.wq = make(&g_dummy, QType::Q8_0, 4096, 4096);
        L.wk = make(&g_dummy, QType::Q8_0, 1024, 4096);
        L.wv = make(&g_dummy, QType::Q8_0, 1024, 4096);
        L.wo = make(&g_dummy, QType::Q8_0, 4096, 4096);
        L.w_gate = make(&g_dummy, QType::Q8_0, 12288, 4096);
        L.w_up = make(&g_dummy, QType::Q8_0, 12288, 4096);
        L.w_down = make(&g_dummy, QType::Q8_0, 4096, 12288);
        L.attn_norm = make(&g_dummy, QType::F32, 0, 4096);
        L.attn_q_norm = make(&g_dummy, QType::F32, 0, 128);
        L.attn_k_norm = make(&g_dummy, QType::F32, 0, 128);
        L.ffn_norm = make(&g_dummy, QType::F32, 0, 4096);
    }
}

// Pass 1 stand-in: every tensor that still has host data "uploads" (goes on device).
void upload_rest(TransformerLayer& L) {
    for (Tensor* t : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down, &L.attn_norm,
                      &L.attn_q_norm, &L.attn_k_norm, &L.ffn_norm})
        if (t->data && !t->on_device)
            t->on_device = true;
}

TEST(LayerHostKeep, Qwen3_8B_GpuLayers20Keeps16LayersOfMatmulsOnHost) {
    Model m;
    build_qwen3_8b(m);
    ASSERT_EQ(dense_layer_streaming_refusal(m), nullptr);
    m.upload_host_layers_ = gpu_layers_host_plan(m, 20);
    ASSERT_EQ(m.upload_host_layers_.size(), static_cast<size_t>(kLayers));
    EXPECT_EQ(host_keep_bytes(m, &dequant_gpu_supported), 16 * kQ8LayerBytes);
    const auto& plan_host = m.upload_host_layers_;

    int kept_tensors = 0;
    size_t kept_bytes = 0;
    for (int i = 0; i < kLayers; i++) {
        HostKeptLayer kept;
        if (plan_host[i])
            detach_host_matmuls(m.layers_[i], &dequant_gpu_supported, [](size_t) -> void* { return nullptr; }, kept);
        upload_rest(m.layers_[i]);
        kept_tensors += static_cast<int>(kept.kept.size());
        kept_bytes += kept.bytes;
        EXPECT_EQ(kept.pinned, 0);
        kept.restore();
    }
    EXPECT_EQ(kept_tensors, 16 * 7);
    EXPECT_EQ(kept_bytes, 16 * kQ8LayerBytes);

    for (int i = 0; i < kLayers; i++) {
        const auto& L = m.layers_[i];
        const bool host = i >= 20;
        for_each_streamable_matmul(L, [&](const Tensor& w) {
            EXPECT_EQ(w.on_device, !host) << "layer " << i;
            EXPECT_EQ(w.data, &g_dummy) << "layer " << i;
            EXPECT_EQ(w.qtype, QType::Q8_0);
        });
        for (const Tensor* n : {&L.attn_norm, &L.attn_q_norm, &L.attn_k_norm, &L.ffn_norm})
            EXPECT_TRUE(n->on_device) << "norms upload on every layer, layer " << i;
    }

    // Post-upload plan: exactly the matmuls are the manager's slot payload.
    const LayerOffloadPlan post = plan_layer_offload(m, 20);
    EXPECT_EQ(post.n_offloaded, 16);
    for (int i = 0; i < kLayers; i++)
        EXPECT_EQ(post.host_bytes[i], i >= 20 ? kQ8LayerBytes : 0u) << "layer " << i;
    EXPECT_EQ(post.max_host_bytes, kQ8LayerBytes);  // staging slot 195.50 MiB
    EXPECT_DOUBLE_EQ(post.max_host_bytes / (1024.0 * 1024.0), 195.5);
}

// Minimal device for wupload::upload_dispatch: host buffers, records which sources were uploaded.
struct StubDevice {
    std::deque<std::vector<uint8_t>> bufs;
    std::set<const void*> uploaded_src;
    void* alloc(size_t n) {
        bufs.emplace_back(n + 1, 0);
        return bufs.back().data();
    }
    int copy(void* dst, const void* src, size_t n) {
        std::memcpy(dst, src, n);
        uploaded_src.insert(src);
        return 0;
    }
    void release(void*, size_t) {}
    void track(void*) {}
    void dequant(const void*, void*, QType, int, int) {}
    const char* err_str(int) { return "stub"; }
    bool dequant_supported(QType q) { return dequant_gpu_supported(q); }
};

TEST(LayerHostKeep, StubDevicePass1UploadsResidentLayersAndNormsOnly) {
    // 4 layers, d 64, FFN 128, Q8_0 matmuls with real bytes; gpu_layers 2 -> layers 2, 3 on host.
    constexpr int kL = 4, kD = 64, kF = 128;
    std::vector<std::vector<uint8_t>> store;
    auto q8 = [&](int64_t rows, int64_t cols) {
        store.emplace_back(static_cast<size_t>(rows) * qtype_row_bytes(QType::Q8_0, cols));
        for (size_t b = 0; b < store.back().size(); b++)
            store.back()[b] = static_cast<uint8_t>(b * 7 + store.size());
        return make(store.back().data(), QType::Q8_0, rows, cols);
    };
    auto f32 = [&](int64_t cols) {
        store.emplace_back(static_cast<size_t>(cols) * 4, 0);
        return make(store.back().data(), QType::F32, 0, cols);
    };
    store.reserve(kL * 8);
    Model m;
    m.layers_.resize(kL);
    for (auto& L : m.layers_) {
        L.wq = q8(kD, kD);
        L.wk = q8(kD / 2, kD);
        L.wv = q8(kD / 2, kD);
        L.wo = q8(kD, kD);
        L.w_gate = q8(kF, kD);
        L.w_up = q8(kF, kD);
        L.w_down = q8(kD, kF);
        L.attn_norm = f32(kD);
    }
    const LayerOffloadPlan plan = plan_layer_offload(m, 2);
    ASSERT_TRUE(plan.active);

    StubDevice dev;
    std::deque<std::vector<uint8_t>> pinned;
    auto pin = [&](size_t n) -> void* {
        pinned.emplace_back(n, 0);
        return pinned.back().data();
    };
    std::vector<const void*> src_of(kL * 7);
    for (int i = 0; i < kL; i++) {
        int k = 0;
        for_each_streamable_matmul(m.layers_[i], [&](const Tensor& w) { src_of[i * 7 + k++] = w.data; });
    }
    m.upload_host_layers_ = gpu_layers_host_plan(m, 2);
    ASSERT_EQ(m.upload_host_layers_, plan.offloaded);
    // The real Pass 1 driver; the per-layer upload is weight_upload.cpp's calls on the stub device.
    const bool ok = upload_pass1_keeping_host(m, &dequant_gpu_supported, pin, [&](TransformerLayer& L, int) {
        for (Tensor* t : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down})
            if (t->data && !t->on_device && !wupload::upload_dispatch(*t, t->qtype, /*raw_quant=*/true, 0.0f, dev))
                return false;
        return wupload::upload_dispatch(L.attn_norm, QType::NONE, true, 0.0f, dev);
    });
    ASSERT_TRUE(ok);
    EXPECT_EQ(pinned.size(), 2u * 7u);

    for (int i = 0; i < kL; i++) {
        const auto& L = m.layers_[i];
        int k = 0;
        for_each_streamable_matmul(L, [&](const Tensor& w) {
            const void* src = src_of[i * 7 + k++];
            const size_t bytes = static_cast<size_t>(w.shape[0]) * qtype_row_bytes(QType::Q8_0, w.shape[1]);
            EXPECT_EQ(dev.uploaded_src.count(src), plan.offloaded[i] ? 0u : 1u) << "layer " << i;
            if (plan.offloaded[i]) {
                EXPECT_FALSE(w.on_device);
                EXPECT_NE(w.data, src);  // pinned copy
                EXPECT_EQ(std::memcmp(w.data, src, bytes), 0) << "pinned bytes differ, layer " << i;
                EXPECT_EQ(w.qtype, QType::Q8_0);
            } else {
                EXPECT_TRUE(w.on_device);
            }
        });
        EXPECT_TRUE(L.attn_norm.on_device) << "layer " << i;
    }
    const LayerOffloadPlan post = plan_layer_offload(m, 2);
    EXPECT_GT(post.max_host_bytes, 0u);
    EXPECT_EQ(post.host_bytes[0], 0u);
    EXPECT_EQ(post.host_bytes[1], 0u);
}

TEST(LayerHostKeep, OnlyVerbatimUploadsStayOnHost) {
    EXPECT_TRUE(upload_keeps_bytes_verbatim(make(&g_dummy, QType::Q8_0, 64, 64), true));
    EXPECT_TRUE(upload_keeps_bytes_verbatim(make(&g_dummy, QType::Q4_K, 64, 256), true));
    EXPECT_FALSE(upload_keeps_bytes_verbatim(make(&g_dummy, QType::MXFP4, 64, 64), true));  // repacked
    EXPECT_FALSE(upload_keeps_bytes_verbatim(make(&g_dummy, QType::F16, 64, 64), false));  // no dequant path
    EXPECT_FALSE(upload_keeps_bytes_verbatim(make(&g_dummy, QType::F32, 0, 64), false));   // 1-D norm
    EXPECT_FALSE(upload_keeps_bytes_verbatim(make(&g_dummy, QType::Q8_0, 64, 64, true), true));
    EXPECT_FALSE(upload_keeps_bytes_verbatim(make(nullptr, QType::Q8_0, 64, 64), true));
}

TEST(LayerHostKeep, NonDenseModelsAreRefusedWithAReason) {
    Model moe;
    build_qwen3_8b(moe);
    moe.config_.n_experts = 8;
    EXPECT_NE(dense_layer_streaming_refusal(moe), nullptr);

    Model hybrid;
    build_qwen3_8b(hybrid);
    hybrid.layers_[3].ssm_in = make(&g_dummy, QType::Q8_0, 64, 64);
    EXPECT_NE(dense_layer_streaming_refusal(hybrid), nullptr);

    Model prequant;
    build_qwen3_8b(prequant);
    prequant.config_.is_nvfp4_prequant = true;
    EXPECT_NE(dense_layer_streaming_refusal(prequant), nullptr);
}

TEST(LayerHostKeep, DetachedLayerIsRestoredOnScopeExit) {
    Model m;
    build_qwen3_8b(m);
    {
        HostKeptLayer kept;
        detach_host_matmuls(m.layers_[0], &dequant_gpu_supported, [](size_t) -> void* { return nullptr; }, kept);
        EXPECT_EQ(m.layers_[0].wq.data, nullptr);  // Pass 1 upload calls skip it
    }
    EXPECT_EQ(m.layers_[0].wq.data, &g_dummy);
    EXPECT_EQ(m.layers_[0].w_down.shape[1], 12288);
}

TEST(LayerHostKeep, HostPlanIsEmptyWhenInactiveOrRefused) {
    Model m;
    build_qwen3_8b(m);
    EXPECT_TRUE(gpu_layers_host_plan(m, -1).empty());
    EXPECT_TRUE(gpu_layers_host_plan(m, kLayers).empty());
    EXPECT_EQ(gpu_layers_host_plan(m, 0).size(), static_cast<size_t>(kLayers));
    m.config_.n_experts = 8;
    EXPECT_TRUE(gpu_layers_host_plan(m, 20).empty());
    EXPECT_STRNE(layer_streaming_failure(m), "");
}

// pre_dequant phase 1: host-streamed weights never get an FP16 cache entry.
TEST(LayerHostKeep, Phase1CachesDeviceResidentSourcesOnly) {
    EXPECT_TRUE(fp16_cache_source(make(&g_dummy, QType::Q8_0, 64, 64, true), QType::Q8_0, true));
    EXPECT_FALSE(fp16_cache_source(make(&g_dummy, QType::Q8_0, 64, 64, false), QType::Q8_0, true));
    EXPECT_TRUE(fp16_cache_source(make(&g_dummy, QType::FP8_E4M3, 64, 64, true), QType::FP8_E4M3, false));
    EXPECT_FALSE(fp16_cache_source(make(&g_dummy, QType::F16, 64, 64, true), QType::F16, false));
    EXPECT_FALSE(fp16_cache_source(make(nullptr, QType::Q8_0, 64, 64, true), QType::Q8_0, true));
}

// pre_dequant phases 2/3 walk dense weights through this: host-streamed ones are never cached.
TEST(LayerHostKeep, PreDequantDenseWalkSkipsHostStreamedWeights) {
    Model m;
    build_qwen3_8b(m);
    for (int i = 0; i < 20; i++)
        upload_rest(m.layers_[i]);
    int device = 0, host = 0, null = 0;
    for_each_device_dense_weight(m, kLayers, [&](const Tensor& w, QType q) {
        EXPECT_EQ(q, w.qtype);
        if (!w.data)
            null++;
        else if (w.on_device)
            device++;
        else
            host++;
    });
    EXPECT_EQ(device, 20 * 7);
    EXPECT_EQ(host, 0);
    EXPECT_EQ(null, kLayers * 5);  // ssm_in/out + 3 shared-expert slots, visited as before
}

}  // namespace
}  // namespace imp
