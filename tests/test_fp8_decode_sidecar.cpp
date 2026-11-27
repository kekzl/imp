// Phase-2b FP8 decode sidecar selection and byte charge (exec/fp8_decode_sidecar.h, B-141).
// CPU only: tensors carry fake device pointers, nothing is dereferenced.

#include <gtest/gtest.h>

#include "exec/fp8_decode_sidecar.h"

#include <cstdint>
#include <vector>

using namespace imp;

namespace {

// Gemma-4-26B-A4B layer shapes (BF16 -> F16 at upload): q 4096x2816, o 2816x4096, gate/up 2112x2816,
// down 2816x2112. The exact numbers only need distinct byte counts per role.
Tensor f16(uintptr_t addr, int64_t rows, int64_t cols) {
    Tensor t;
    t.data = reinterpret_cast<void*>(addr);
    t.qtype = QType::F16;
    t.ndim = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.on_device = true;
    return t;
}

// shared: the Gemma-4 MoE layout, whose dense MLP loads into the shared-expert slots (weight_map.cpp).
TransformerLayer gemma4_like_layer(uintptr_t base, bool shared = false) {
    TransformerLayer L;
    L.wq = f16(base + 0x100, 4096, 2816);
    L.wk = f16(base + 0x200, 2048, 2816);
    L.wv = f16(base + 0x300, 2048, 2816);
    L.wo = f16(base + 0x400, 2816, 4096);
    (shared ? L.w_gate_shared : L.w_gate) = f16(base + 0x500, 2112, 2816);
    (shared ? L.w_up_shared : L.w_up) = f16(base + 0x600, 2112, 2816);
    (shared ? L.w_down_shared : L.w_down) = f16(base + 0x700, 2816, 2112);
    return L;
}

std::vector<const Tensor*> ffn_trio(const TransformerLayer& L) {
    if (L.w_gate.data != nullptr)
        return {&L.w_gate, &L.w_up, &L.w_down};
    return {&L.w_gate_shared, &L.w_up_shared, &L.w_down_shared};
}

Fp8SidecarPlan plan_for(const std::vector<TransformerLayer>& layers, const Fp8SidecarScope& s) {
    return plan_fp8_sidecar(
        static_cast<int>(layers.size()), [&](int i) -> const TransformerLayer& { return layers[i]; }, s,
        [](const void*) { return false; });
}

constexpr size_t kAttnBytes = 4096ull * 2816 + 2 * 2048ull * 2816 + 2816ull * 4096;
constexpr size_t kFfnBytes = 2 * 2112ull * 2816 + 2816ull * 2112;
constexpr size_t kFfnRows = 2112 + 2112 + 2816;

}  // namespace

TEST(Fp8DecodeSidecar, ScopeFollowsBothFlags) {
    const Fp8SidecarScope off = fp8_sidecar_scope("auto", false, "off", false);
    EXPECT_FALSE(off.attn_qo || off.attn_kv || off.ffn);
    const Fp8SidecarScope ffn = fp8_sidecar_scope("off", true, "on", false);
    EXPECT_FALSE(ffn.attn_qo || ffn.attn_kv);
    EXPECT_TRUE(ffn.ffn);
    const Fp8SidecarScope qo = fp8_sidecar_scope("qo", false, "off", false);
    EXPECT_TRUE(qo.attn_qo);
    EXPECT_FALSE(qo.attn_kv || qo.ffn);
    const Fp8SidecarScope autofull = fp8_sidecar_scope("auto", true, "off", false);
    EXPECT_TRUE(autofull.attn_qo && autofull.attn_kv);
    EXPECT_FALSE(autofull.ffn);
    // ffn "auto" follows the profile (Gemma-4 on), "on" / "off" override it either way.
    EXPECT_TRUE(fp8_sidecar_scope("auto", false, "auto", true).ffn);
    EXPECT_FALSE(fp8_sidecar_scope("auto", false, "auto", false).ffn);
    EXPECT_FALSE(fp8_sidecar_scope("auto", false, "off", true).ffn);
    EXPECT_TRUE(fp8_sidecar_scope("auto", false, "on", false).ffn);
    const Fp8SidecarScope gemma4 = fp8_sidecar_scope("auto", false, "auto", true);
    EXPECT_FALSE(gemma4.attn_qo || gemma4.attn_kv) << "fp8_attn_proj stays off on Gemma-4";
}

// gemm.fp8_ffn_proj off (the non-Gemma-4 default): the dense FFN weights get no sidecar, so their decode
// stays on the 16-bit source; on: exactly the gate / up / down trio (dense or shared-expert slots) is added
// and charged 1 byte per element.
TEST(Fp8DecodeSidecar, FfnFlagSelectsDenseFfnAndChargesItsBytes) {
    const std::vector<TransformerLayer> layers = {gemma4_like_layer(0x10000),
                                                  gemma4_like_layer(0x20000, /*shared=*/true)};

    const Fp8SidecarPlan none = plan_for(layers, fp8_sidecar_scope("off", false, "off", false));
    EXPECT_TRUE(none.entries.empty());
    EXPECT_EQ(none.bytes, 0u);

    const Fp8SidecarPlan ffn = plan_for(layers, fp8_sidecar_scope("off", false, "on", false));
    ASSERT_EQ(ffn.entries.size(), 6u);
    for (const auto& L : layers) {
        for (const Tensor* w : ffn_trio(L)) {
            bool found = false;
            for (const auto& e : ffn.entries)
                found |= e.ptr == w->data && e.rows == w->shape[0] && e.cols == w->shape[1];
            EXPECT_TRUE(found) << "missing FFN weight " << w->data;
        }
        for (const Tensor* w : {&L.wq, &L.wk, &L.wv, &L.wo})
            for (const auto& e : ffn.entries)
                EXPECT_NE(e.ptr, w->data) << "attention weight planned without gemm.fp8_attn_proj";
    }
    EXPECT_EQ(ffn.bytes, 2 * kFfnBytes);
    EXPECT_EQ(ffn.rows, 2 * kFfnRows);

    const Fp8SidecarPlan both = plan_for(layers, fp8_sidecar_scope("on", false, "on", false));
    EXPECT_EQ(both.entries.size(), 14u);
    EXPECT_EQ(both.bytes, 2 * (kAttnBytes + kFfnBytes));

    const Fp8SidecarPlan attn = plan_for(layers, fp8_sidecar_scope("on", false, "off", false));
    EXPECT_EQ(both.bytes - attn.bytes, 2 * kFfnBytes) << "the FFN flag's VRAM charge";
}

// Quantized, already-cached, misaligned or host weights never get a sidecar; a shared pointer is charged
// once.
TEST(Fp8DecodeSidecar, IneligibleAndDuplicateWeightsAreSkipped) {
    std::vector<TransformerLayer> layers = {gemma4_like_layer(0x10000)};
    layers[0].w_gate.qtype = QType::NVFP4;
    layers[0].w_up.shape[1] = 2817;
    layers[0].w_down.on_device = false;
    EXPECT_TRUE(plan_for(layers, fp8_sidecar_scope("off", false, "on", false)).entries.empty());

    layers = {gemma4_like_layer(0x10000)};
    const void* gate = layers[0].w_gate.data;
    const Fp8SidecarPlan cached = plan_fp8_sidecar(
        1, [&](int) -> const TransformerLayer& { return layers[0]; },
        fp8_sidecar_scope("off", false, "on", false), [&](const void* p) { return p == gate; });
    EXPECT_EQ(cached.entries.size(), 2u);

    layers.push_back(layers[0]);  // tied layer: same pointers
    EXPECT_EQ(plan_for(layers, fp8_sidecar_scope("off", false, "on", false)).bytes, kFfnBytes);
}
