#include "exec/workspace_sizes.h"

#include "model/model.h"

#include <algorithm>
#include <cstdio>

namespace imp {

namespace {

// Logical (unpacked) K of a weight tensor. NVFP4 stores two e2m1 values per
// byte, so shape[...] is the PACKED byte count and the dequant target is twice
// as wide — the same distinction allocate_nvfp4_dequant_workspace makes.
struct Dims {
    int64_t n = 0;
    int64_t k = 0;
    bool valid() const { return n > 0 && k > 0; }
};

Dims weight_dims(const Tensor& t) {
    if (!t.data || t.ndim < 2)
        return {};
    const bool packed = (t.qtype == QType::NVFP4);
    if (t.ndim >= 3)  // [n_experts, N, K] — one expert is the dequant unit
        return {t.shape[1], t.shape[2] * (packed ? 2 : 1)};
    return {t.shape[0], t.shape[1] * (packed ? 2 : 1)};
}

// The tensor set the MMVQ / NVFP4-dequant paths can see. Deliberately the same
// list compute_native_cache_demand() scans (runtime/vram_budget.cpp), so the
// two cannot drift apart on a model where one of them matters.
template <class F>
void for_each_weight(const Model& model, F&& f) {
    f(model.output_proj());
    const int n = model.n_layers();
    for (int i = 0; i < n; ++i) {
        const auto& L = model.layer(i);
        f(L.wq); f(L.wk); f(L.wv); f(L.wo);
        f(L.w_gate); f(L.w_up); f(L.w_down);
        f(L.w_gate_shared); f(L.w_up_shared); f(L.w_down_shared);
        f(L.ssm_in); f(L.ssm_out); f(L.gdn_gate);
        if (L.expert_gate_packed.data)
            f(L.expert_gate_packed);
        else if (!L.expert_w_gate.empty())
            f(L.expert_w_gate[0]);
        if (L.expert_up_packed.data)
            f(L.expert_up_packed);
        else if (!L.expert_w_up.empty())
            f(L.expert_w_up[0]);
        if (L.expert_down_packed.data)
            f(L.expert_down_packed);
        else if (!L.expert_w_down.empty())
            f(L.expert_w_down[0]);
    }
}

}  // namespace

std::string ExecT2Demand::describe() const {
    constexpr double kMiB = 1024.0 * 1024.0;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "mmvq %.1f + nvfp4 %.1f + sample %.1f + moe %.2f + fp8red %.2f + quant %.2f "
                  "+ splitk %.2f + mla %.1f MiB",
                  mmvq_scratch / kMiB, nvfp4_dequant / kMiB, sample_scratch / kMiB,
                  moe_arrays / kMiB, fp8_reduction / kMiB, quant_scratch / kMiB,
                  splitk_scratch / kMiB, mla_scratch / kMiB);
    return buf;
}

// ── adapter ──────────────────────────────────────────────────────────

ExecShape exec_shape_of(const Model& model) {
    const auto& cfg = model.config();
    const auto& prof = model.profile();
    ExecShape s;
    s.max_seq_len_cfg = cfg.max_seq_len;
    s.is_ssm = prof.is_ssm;
    s.is_moe = prof.is_moe;
    s.n_experts = cfg.n_experts;
    s.expert_d_ff = cfg.expert_d_ff;
    s.d_ff = cfg.d_ff;
    s.d_model = cfg.d_model;
    s.n_heads = cfg.n_heads;
    s.head_dim = cfg.head_dim;
    s.ssm_inner_size = cfg.ssm_inner_size;
    s.ssm_conv_channels = cfg.ssm_inner_size > 0 ? cfg.ssm_conv_channels() : 0;
    s.ssm_dt_rank = cfg.ssm_dt_rank;
    s.n_experts_active = cfg.n_experts_active;
    s.n_layers = cfg.n_layers;
    s.kv_lora_rank = cfg.kv_lora_rank;
    s.qk_rope_head_dim = cfg.qk_rope_head_dim;
    s.qk_nope_head_dim = cfg.qk_nope_head_dim;
    s.v_head_dim = cfg.v_head_dim;
    // max_batch_size, use_fp8_prefill and mla_absorb are filled by the caller: the
    // model knows neither the batch nor the runtime config. mla_absorb matters
    // because the absorbed latent cache is two orders of magnitude larger than the
    // MLA quartet, so the plan can treat it as neither always-on nor always-off.
    for_each_weight(model, [&](const Tensor& t) {
        const Dims d = weight_dims(t);
        if (d.valid())
            s.weights.emplace_back(d.n, d.k);
    });
    // The dp4a scratch's own scan (executor_workspace_buffers.cu): a narrower
    // tensor list, shape[1]/shape[2] read RAW. Kept separate from `weights`
    // above on purpose — see the ExecShape comment.
    for (int i = 0; i < model.n_layers(); ++i) {
        const auto& L = model.layer(i);
        for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down,
                              &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared, &L.ssm_in,
                              &L.ssm_out}) {
            if (w->data && w->ndim >= 2)
                s.mmvq_max_k = std::max(s.mmvq_max_k, static_cast<int>(w->shape[1]));
        }
        for (const auto* w : {&L.expert_up_packed, &L.expert_down_packed, &L.expert_gate_packed}) {
            if (w->data && w->ndim >= 3)
                s.mmvq_max_k = std::max(s.mmvq_max_k, static_cast<int>(w->shape[2]));
        }
        if (L.expert_down_packed.data && L.expert_down_packed.ndim >= 3) {
            s.mmvq_max_expert_down_k =
                std::max(s.mmvq_max_expert_down_k, static_cast<int>(L.expert_down_packed.shape[2]));
        }
        for (const auto* w : {&L.w_gate, &L.w_up, &L.w_down, &L.w_gate_shared, &L.w_up_shared,
                              &L.w_down_shared, &L.wq, &L.wk, &L.wv, &L.wo}) {
            if (w->data && (w->qtype == QType::Q4_K || w->qtype == QType::Q5_K))
                s.has_sub5bit_dense = true;
        }
    }
    return s;
}

// ── pure core ────────────────────────────────────────────────────────

int exec_max_tokens(const ExecShape& shape, int max_seq_len) {
    int effective = (max_seq_len > 0) ? max_seq_len : shape.max_seq_len_cfg;
    int t = std::min(effective, 4096);
    if (t <= 0)
        t = 4096;
    // AS-BUILT condition, not the intended one — executor_workspace.cu reads
    // has_gdn_ before it is assigned, so this fires for SSM+MoE only. Matching
    // the intent here would under-reserve by 2x on those models (AUDIT B18).
    if (shape.is_ssm && shape.is_moe)
        t = std::min(t, 2048);
    return t;
}

int exec_max_weight_k(const ExecShape& shape) {
    int64_t max_k = 0;
    for (const auto& [n, k] : shape.weights) {
        (void)n;
        max_k = std::max(max_k, k);
    }
    return static_cast<int>(max_k);
}

ExecT2Demand exec_t2_demand(const ExecShape& shape, int max_seq_len) {
    ExecT2Demand out;
    const int t = exec_max_tokens(shape, max_seq_len);
    const int max_k = exec_max_weight_k(shape);
    if (t <= 0 || max_k <= 0)
        return out;

    // gemm_scratch.cu: per_call = max_tokens * ceil(K/32) * 36, need = 2x.
    out.mmvq_scratch = static_cast<size_t>(t) * ((static_cast<size_t>(max_k) + 31) / 32) * 36 * 2;

    // executor_workspace_buffers.cu: the largest dequant target at or below the
    // 512 MiB cap. Targets above the cap are served by the uncapturable path,
    // so they must NOT raise the reservation.
    constexpr size_t kCap = 512ULL * 1024 * 1024;
    size_t covered = 0;
    auto consider = [&](int64_t n, int64_t k) {
        if (n <= 0 || k <= 0)
            return;
        const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(k) * 2;
        if (bytes <= kCap)
            covered = std::max(covered, bytes);
    };
    for (const auto& [n, k] : shape.weights)
        consider(n, k);

    // Config-derived shapes, because this runs BEFORE the weight upload and
    // some checkpoints do not carry their final layout yet. gpt-oss is the
    // case that forced this: its experts arrive as
    // expert_gate_up_packed_blocks, a 4D U8 [ne, 2*d_ff, K/32, 16] slot that
    // the upload consumes, so the tensor scan sees nothing resembling the
    // dequant target. The real target is one expert's FUSED gate_up —
    // 2*expert_d_ff x d_model — which is exactly the 31.64 MiB the workspace
    // asked for and was refused (AUDIT B23). Deriving it from the config
    // instead of the tensors makes the reservation independent of when it runs.
    if (shape.n_experts > 0) {
        const int64_t eff = shape.expert_d_ff > 0 ? shape.expert_d_ff : shape.d_ff;
        consider(2 * eff, shape.d_model);  // fused gate_up
        consider(eff, shape.d_model);      // gate / up / down individually
        consider(shape.d_model, eff);
    }
    if (shape.d_ff > 0) {
        consider(2 * static_cast<int64_t>(shape.d_ff), shape.d_model);
        consider(shape.d_ff, shape.d_model);
        consider(shape.d_model, shape.d_ff);
    }

    out.nvfp4_dequant = covered;

    // Sampling result scratch (executor_workspace_buffers.cu): two parities of
    // max_logit_tokens slots, each SAMPLE_SCRATCH_BYTES. max_logit_tokens is
    // max(max_batch_size, 8) — the BATCH, not the context, which is why this is
    // ~1 MiB and not the ~115 MiB I first assumed (AUDIT B53).
    constexpr size_t kSampleScratchBytes =
        sizeof(int32_t) + 64 * (2 * sizeof(float) + 128 * (sizeof(float) + sizeof(int32_t)));
    const int logit_tokens = std::max(shape.max_batch_size, 8);
    out.sample_scratch = 2ull * static_cast<size_t>(logit_tokens) * kSampleScratchBytes;

    // Batched-MoE pointer/scale arrays (executor_workspace_buffers.cu): work
    // pointers 3*ne void*, fp8 scales ne float, M_per ne int32, alpha_compact ne
    // float, the active-expert counter, and the SFA offset prefix sum (ne+1)
    // int64. A few KiB in total; charged so the arena is sized for them.
    if (shape.n_experts > 0) {
        const size_t ne = static_cast<size_t>(shape.n_experts);
        out.moe_arrays = 3 * ne * sizeof(void*)        // d_work_ptrs
                         + ne * sizeof(float)         // d_fp8_scales
                         + ne * sizeof(int32_t)       // d_M_per
                         + ne * sizeof(float)         // d_alpha_compact
                         + sizeof(int32_t)            // d_na
                         + (ne + 1) * sizeof(int64_t) // d_sfa_offsets
                         + 2 * ne * sizeof(void*)     // d_B_ptrs_cache, d_SFB_ptrs_cache
                         + ne * sizeof(float)         // d_alpha_full
                         + ne * sizeof(void*)         // d_weight_ptrs
                         + ne * sizeof(void*)         // cutlass3x_sfa_ptrs
                         + 11 * 256;                  // per-take 256 B alignment
    }

    // dp4a input staging (executor_workspace_buffers.cu), all three tenants of
    // one sizing family. The site's max_blocks is max(max_k/32, top_k *
    // down_k/32) — the MoE down projection quantizes top_k expert activations
    // contiguously, which is what makes the second term able to exceed the
    // first.
    {
        const int moe_down_blocks =
            shape.n_experts_active * (shape.mmvq_max_expert_down_k / 32);
        const int max_blocks = std::max(shape.mmvq_max_k / 32, moe_down_blocks);
        if (max_blocks > 0) {
            // Rows: min(max(max_logit_tokens, 8), 16). The batch, capped at 16 so
            // a large-batch server does not inflate this K-sized scratch.
            const size_t rows =
                static_cast<size_t>(std::min(std::max(shape.max_batch_size, 8), 16));
            out.quant_scratch = static_cast<size_t>(max_blocks) * rows *
                                (kExecBlockQ81Bytes + sizeof(float));
            // FFN sparsity mask: one bit per Q8 block, packed uint32.
            out.quant_scratch +=
                ((static_cast<size_t>(max_blocks) + 31) / 32) * sizeof(uint32_t);
            // dp4a dense prefill pair. Gated on a Q4_K/Q5_K dense weight existing
            // at all, and sized from kDp4aDenseMaxM=64 — the M above which the
            // weight-stationary tile stops winning and the path is not taken.
            if (shape.has_sub5bit_dense && t > 1) {
                const size_t prefill_blocks = static_cast<size_t>(std::min(t, 64)) *
                                              (static_cast<size_t>(shape.mmvq_max_k) / 32);
                out.quant_scratch += prefill_blocks * (kExecBlockQ81Bytes + sizeof(float));
            }
            out.quant_scratch += 5 * 256;  // per-take 256 B alignment
        }
    }

    // Split-K paged attention partials (executor_workspace_buffers.cu). Splits
    // scale with the context in KV blocks and cap at 128; the batch dimension is
    // max_logit_tokens, not max_batch_size, so it inherits the floor of 8.
    if (shape.n_heads > 0) {
        const int hd = shape.head_dim > 0 ? shape.head_dim : (shape.d_model / shape.n_heads);
        const int ctx_blocks = (t + kExecKVBlockSize - 1) / kExecKVBlockSize;
        const int splits = std::min(128, std::max(1, ctx_blocks));
        const int stride = 2 + hd;
        const int batch = std::max(shape.max_batch_size, 8);
        if (hd > 0) {
            out.splitk_scratch = static_cast<size_t>(batch) * shape.n_heads * splits * stride *
                                     sizeof(float) +
                                 256;
        }
    }

    // MLA QKV scratch (executor_workspace_buffers.cu). kv_lora_rank > 0 IS
    // is_mla(). The quartet is sized for max_tokens and, unlike every other tenant
    // here, has NO degradation contract: executor_attention_qkv.cu dereferences
    // all four unconditionally, so a short arena fails the load instead of
    // handing out a null (see the site).
    if (shape.kv_lora_rank > 0) {
        const size_t T = static_cast<size_t>(t);
        const size_t kva_out = static_cast<size_t>(shape.kv_lora_rank + shape.qk_rope_head_dim);
        const size_t kvb_out =
            static_cast<size_t>(shape.n_heads) * (shape.qk_nope_head_dim + shape.v_head_dim);
        out.mla_scratch = T * 2 *
                          (kva_out + static_cast<size_t>(shape.kv_lora_rank) +
                           static_cast<size_t>(shape.qk_rope_head_dim) + kvb_out);
        out.mla_scratch += 4 * 256;  // four takes

        // Absorbed-decode latent cache. Sized from the FULL sequence length, NOT
        // from max_tokens — mla_absorb_max_seq_ is deliberately uncapped where
        // max_tokens_ clamps at 4096, so this is the term that reaches ~974 MiB at
        // a 32k context. Charged only when the opt-in flag is on, because charging
        // it always would reserve that on every DeepSeek load.
        if (shape.mla_absorb) {
            const int effective = (max_seq_len > 0) ? max_seq_len : shape.max_seq_len_cfg;
            const size_t absorb_seq = static_cast<size_t>(effective > 0 ? effective : 4096);
            out.mla_scratch += static_cast<size_t>(shape.n_layers) * absorb_seq * kva_out * 2;
            out.mla_scratch += static_cast<size_t>(shape.n_heads) * absorb_seq * sizeof(float);
            out.mla_scratch += 2 * 256;
        }
    }

    // FP8 activation reduction scratch. Mirrors the max_dim ladder and the grid
    // arithmetic in executor_workspace_buffers.cu exactly: kElemsPerThread=4,
    // kBlockSize=256, plus the act-scale and absmax scalars.
    if (shape.use_fp8_prefill && shape.d_model > 0) {
        int max_dim = shape.d_model;
        if (shape.d_ff > 0)
            max_dim = std::max(max_dim, shape.d_ff);
        const int hd = shape.head_dim > 0
                           ? shape.head_dim
                           : (shape.n_heads > 0 ? shape.d_model / shape.n_heads : 0);
        max_dim = std::max(max_dim, shape.n_heads * hd);
        if (shape.ssm_inner_size > 0) {
            max_dim = std::max(max_dim, shape.ssm_inner_size + shape.ssm_conv_channels +
                                            shape.ssm_dt_rank);
            max_dim = std::max(max_dim, shape.ssm_conv_channels + shape.ssm_inner_size +
                                            2 * shape.ssm_dt_rank);
            max_dim = std::max(max_dim, shape.ssm_inner_size);
        }
        const size_t act = static_cast<size_t>(t) * static_cast<size_t>(max_dim);
        const size_t grid = ((act + 3) / 4 + 255) / 256;
        out.fp8_reduction = grid * sizeof(float) + 2 * sizeof(float) + 3 * 256;
    }

    return out;
}

// ── Model overloads ──────────────────────────────────────────────────

int exec_max_tokens(const Model& model, int max_seq_len) {
    return exec_max_tokens(exec_shape_of(model), max_seq_len);
}

int exec_max_weight_k(const Model& model) { return exec_max_weight_k(exec_shape_of(model)); }

ExecT2Demand exec_t2_demand(const Model& model, int max_seq_len, int max_batch_size,
                            bool use_fp8_prefill, bool mla_absorb) {
    ExecShape shape = exec_shape_of(model);
    shape.max_batch_size = max_batch_size;
    shape.use_fp8_prefill = use_fp8_prefill;
    shape.mla_absorb = mla_absorb;
    return exec_t2_demand(shape, max_seq_len);
}

ExecT2Demand exec_t2_demand(const Model& model, int max_seq_len, int max_batch_size) {
    return exec_t2_demand(model, max_seq_len, max_batch_size, false);
}

ExecT2Demand exec_t2_demand(const Model& model, int max_seq_len) {
    return exec_t2_demand(model, max_seq_len, 1, false);
}

}  // namespace imp
