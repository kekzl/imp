// Gemma-4 E-variants (E2B/E4B): per-layer inputs (PLE) and KV sharing. HF reference:
// transformers models/gemma4/modeling_gemma4.py (Gemma4TextModel.project_per_layer_inputs,
// Gemma4TextDecoderLayer per_layer_input branch, Gemma4TextAttention is_kv_shared_layer).
//
// PLE: pli = (rmsnorm(W_mp @ emb * d^-0.5) * w + table[id] * sqrt(pli_dim)) * 2^-0.5 once per
// forward; per layer after the FFN residual: h += rmsnorm(W_proj @ (gelu_tanh(W_gate @ h) * pli_l)),
// then layer_out_scale. Every step is a device kernel or GEMM on graph-stable buffers.
// KV sharing: a shared layer projects Q only and attends to its source layer's K/V. Prefill rows
// of the current chunk come from GemmaPliState::kv_rows (saved by the source layer after norm + RoPE),
// past rows from the source's paged cache (kv_layer_map_ maps the shared layer to the source slot).

#include "compute/gemm.h"
#include "compute/gemma_pli.h"
#include "compute/layernorm.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/executor_kernels.h"
#include "exec/gemma_pli_state.h"

#include <algorithm>
#include <cmath>

namespace imp {

namespace {

Tensor rows_tensor(void* p, int64_t rows, int64_t cols) {
    const int64_t shape[2] = {rows, cols};
    return Tensor(p, QType::F16, 2, shape, true);
}

}  // namespace

bool GraphExecutor::gemma_pli_alloc_(int max_tokens) {
    const auto& cfg = model_->config();
    const bool pli = cfg.pli_dim > 0 && model_->pli_table().data != nullptr;
    if (!pli && cfg.kv_source_layer.empty())
        return true;
    if (compute_dtype_ != QType::F16) {
        IMP_LOG_ERROR("Gemma 4 per-layer inputs / KV sharing need FP16 compute");
        return false;
    }
    auto alloc = [&](int64_t cols, const char* tag) -> void* {
        return vram_alloc(vram_alloc_, align256(static_cast<size_t>(max_tokens) * cols * sizeof(uint16_t)),
                          tag);
    };
    size_t bytes = 0;
    if (pli) {
        const int64_t wide = static_cast<int64_t>(cfg.n_layers) * cfg.pli_dim;
        void* in = alloc(wide, "pli_inputs");
        void* gate = alloc(cfg.pli_dim, "pli_gate");
        void* out = alloc(cfg.d_model, "pli_out");
        if (in == nullptr || gate == nullptr || out == nullptr)
            return false;
        gpli_.inputs = rows_tensor(in, max_tokens, wide);
        gpli_.gate = rows_tensor(gate, max_tokens, cfg.pli_dim);
        gpli_.out = rows_tensor(out, max_tokens, cfg.d_model);
        bytes += static_cast<size_t>(max_tokens) * (wide + cfg.pli_dim + cfg.d_model) * 2;
    }
    gpli_.kv_rows.assign(static_cast<size_t>(cfg.n_layers), GemmaPliState::KvShareRows{});
    for (int l = 0; l < cfg.n_layers; ++l) {
        const int src = cfg.kv_source(l);
        if (src < 0 || gpli_.kv_rows[static_cast<size_t>(src)].k != nullptr)
            continue;
        const int nkv = src < static_cast<int>(cfg.n_kv_heads_per_layer.size())
                            ? cfg.n_kv_heads_per_layer[static_cast<size_t>(src)]
                            : cfg.n_kv_heads;
        const int hd = src < static_cast<int>(cfg.head_dim_per_layer.size())
                           ? cfg.head_dim_per_layer[static_cast<size_t>(src)]
                           : cfg.head_dim;
        GemmaPliState::KvShareRows& r = gpli_.kv_rows[static_cast<size_t>(src)];
        r.cols = static_cast<int64_t>(nkv) * hd;
        r.k = alloc(r.cols, "kv_share_k");
        r.v = alloc(r.cols, "kv_share_v");
        if (r.k == nullptr || r.v == nullptr)
            return false;
        bytes += static_cast<size_t>(max_tokens) * r.cols * 4;
    }
    IMP_LOG_INFO("Gemma 4 E-variant workspace: per-layer inputs %s, %zu KV-shared layers, %.1f MiB",
                 pli ? "on" : "off",
                 static_cast<size_t>(std::count_if(cfg.kv_source_layer.begin(), cfg.kv_source_layer.end(),
                                                   [](int s) { return s >= 0; })),
                 static_cast<double>(bytes) / (1024.0 * 1024.0));
    return true;
}

void GraphExecutor::gemma_pli_free_() {
    for (Tensor* t : {&gpli_.inputs, &gpli_.gate, &gpli_.out}) {
        if (t->data != nullptr)
            vram_free(vram_alloc_, t->data);
        t->data = nullptr;
    }
    for (GemmaPliState::KvShareRows& r : gpli_.kv_rows) {
        if (r.k != nullptr)
            vram_free(vram_alloc_, r.k);
        if (r.v != nullptr)
            vram_free(vram_alloc_, r.v);
    }
    gpli_.kv_rows.clear();
}

void GraphExecutor::pli_inputs_run_(const InferenceState& state, int n, cudaStream_t stream) {
    if (gpli_.inputs.data == nullptr || n <= 0)
        return;
    const auto& cfg = model_->config();
    const int wide = cfg.n_layers * cfg.pli_dim;
    Tensor emb = view_tokens(hidden_, n);
    Tensor proj = rows_tensor(gpli_.inputs.data, n, wide);
    // per_layer_model_projection(emb) * hidden_size^-0.5 (the scale rides the GEMM alpha)
    gemm(emb, model_->pli_model_proj(), proj, 1.0f / std::sqrt(static_cast<float>(cfg.d_model)), 0.0f,
         stream);
    const int32_t mm_id = state.n_vision_tokens > 0 ? state.vision_token_id : -1;
    pli_project_combine(proj.data, model_->pli_table().data, state.token_ids, model_->pli_proj_norm().data, n,
                        cfg.n_layers, cfg.pli_dim, cfg.pli_vocab, mm_id, /*pad_id=*/0, cfg.rms_norm_eps,
                        std::sqrt(static_cast<float>(cfg.pli_dim)), 1.0f / std::sqrt(2.0f), stream);
}

void GraphExecutor::pli_layer_run_(int layer, int n, cudaStream_t stream) {
    const auto& ly = model_->layer(layer);
    if (gpli_.inputs.data == nullptr || ly.pli_gate.data == nullptr || n <= 0)
        return;
    const auto& cfg = model_->config();
    Tensor h = view_tokens(hidden_, n);
    Tensor g = rows_tensor(gpli_.gate.data, n, cfg.pli_dim);
    Tensor p = rows_tensor(gpli_.out.data, n, cfg.d_model);
    gemm(h, ly.pli_gate, g, 1.0f, 0.0f, stream);  // [n, d] x [pli_dim, d]^T
    pli_gate_mul(g.data, gpli_.inputs.data, n, cfg.n_layers, cfg.pli_dim, layer, stream);
    gemm(g, ly.pli_proj, p, 1.0f, 0.0f, stream);  // [n, pli_dim] x [d, pli_dim]^T
    if (fp32_accum_buf_ != nullptr && model_->profile().fp32_residual_norms) {
        // fp32 residual += rmsnorm(p) * w, h = fp16(fp32 residual): same arm as post_ffn_norm
        rmsnorm_fp32_accum_to_fp16(static_cast<const half*>(p.data),
                                   static_cast<const half*>(ly.pli_norm.data),
                                   static_cast<float*>(view_tokens(fp32_hidden_, n).data),
                                   static_cast<half*>(h.data), n, cfg.d_model, cfg.rms_norm_eps, norm_w_off_,
                                   stream);
    } else {
        rmsnorm(p, ly.pli_norm, p, cfg.rms_norm_eps, stream, norm_w_off_);
        elementwise_add(h, p, stream);
    }
}

void kv_share_save(const GemmaPliState& s, int layer, const Tensor& k, const Tensor& v, int n,
                   cudaStream_t stream) {
    if (layer < 0 || layer >= static_cast<int>(s.kv_rows.size()) || n <= 0)
        return;
    const GemmaPliState::KvShareRows& r = s.kv_rows[static_cast<size_t>(layer)];
    if (r.k == nullptr)
        return;
    const size_t bytes = static_cast<size_t>(n) * r.cols * sizeof(uint16_t);
    device_copy_async(r.k, k.data, bytes, stream);
    device_copy_async(r.v, v.data, bytes, stream);
}

bool kv_share_bind(const GemmaPliState& s, const ModelConfig& cfg, int layer, Tensor& k, Tensor& v, int n) {
    const int src = cfg.kv_source(layer);
    if (src < 0 || src >= static_cast<int>(s.kv_rows.size()))
        return false;
    const GemmaPliState::KvShareRows& r = s.kv_rows[static_cast<size_t>(src)];
    if (r.k == nullptr || r.cols != k.shape[1])
        return false;
    k = rows_tensor(r.k, n, r.cols);
    v = rows_tensor(r.v, n, r.cols);
    return true;
}

}  // namespace imp
