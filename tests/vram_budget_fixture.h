#pragma once
// Fake models shared by tests/test_vram_budget_reserve.cpp (GPU) and test_vram_budget_reserve_host.cpp.

#include "core/tensor.h"
#include "model/model.h"

#include <cstdint>
#include <utility>

namespace imp::vram_fixture {

inline Tensor make_weight(TensorKind kind, int64_t rows, int64_t cols, QType qt, uintptr_t sentinel) {
    Tensor t;
    t.data = reinterpret_cast<void*>(sentinel);
    t.qtype = qt;
    t.ndim = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.on_device = false;
    t.kind = kind;
    return t;
}

// 32 layers; attention weights of `attn_qt`, FFN weights of `ffn_qt`
// (Model is non-copyable — populate the caller's instance).
inline void fill_model(Model& m, QType attn_qt, QType ffn_qt) {
    m.config_.n_layers = 32;
    m.config_.n_kv_heads = 8;
    for (int i = 0; i < 32; ++i) {
        TransformerLayer L;
        uintptr_t base = static_cast<uintptr_t>(i) * 100 + 1;
        L.wq = make_weight(TensorKind::WQ, 4096, 4096, attn_qt, base + 0);
        L.w_gate = make_weight(TensorKind::W_GATE, 11008, 4096, ffn_qt, base + 1);
        L.w_up = make_weight(TensorKind::W_UP, 11008, 4096, ffn_qt, base + 2);
        L.w_down = make_weight(TensorKind::W_DOWN, 4096, 11008, ffn_qt, base + 3);
        m.layers_.push_back(std::move(L));
    }
}

// Fake prequant hybrid: 2 layers, dense attention proj + GDN projections +
// 4-expert MoE (per-expert 2D views, the SafeTensors prequant layout).
// Shapes store the PACKED byte dim (K/2), mirroring the NVFP4 wire format.
inline void fill_prequant_model(Model& m) {
    m.config_.n_layers = 2;
    m.config_.n_kv_heads = 8;
    m.config_.n_experts = 4;
    m.config_.is_nvfp4_prequant = true;
    uintptr_t next = 1;
    for (int i = 0; i < 2; ++i) {
        TransformerLayer L;
        L.wq = make_weight(TensorKind::WQ, 1024, 512, QType::INT8, next++);
        L.ssm_in = make_weight(TensorKind::WQ, 2048, 1024, QType::INT8, next++);
        L.gdn_gate = make_weight(TensorKind::WQ, 512, 1024, QType::INT8, next++);
        for (int e = 0; e < 4; ++e) {
            L.expert_w_up.push_back(make_weight(TensorKind::EXPERT_UP, 768, 512, QType::INT8, next++));
            L.expert_w_down.push_back(make_weight(TensorKind::EXPERT_DOWN, 256, 768, QType::INT8, next++));
        }
        m.layers_.push_back(std::move(L));
    }
}

}  // namespace imp::vram_fixture
