#include "runtime/vram_budget.h"
#include "runtime/engine.h"  // EngineConfig full definition
#include "core/logging.h"
#include <algorithm>
#include <cuda_runtime.h>

namespace imp {

VRAMBudget compute_vram_budget(const Model& model, const EngineConfig& config, int n_kv_layers, int head_dim,
                               size_t free_vram) {
    VRAMBudget budget;
    const auto& mcfg = model.config();

    // --- 1. Classify model quantization ---
    auto qtype = model.layer(0).wq.qtype;
    bool sub_8bit = (qtype == QType::Q4_0 || qtype == QType::Q4_K || qtype == QType::Q5_0 ||
                     qtype == QType::Q5_K || qtype == QType::Q3_K || qtype == QType::Q2_K ||
                     qtype == QType::Q4_1 || qtype == QType::Q5_1);

    // --- 2. Choose strategy ---
    if (config.use_nvfp4_decode == 0) {
        budget.strategy = VRAMBudget::FP16_ONLY;
    } else if (sub_8bit) {
        budget.strategy = VRAMBudget::NVFP4_DECODE_ONLY;
    } else {
        budget.strategy = VRAMBudget::FP8_PREFILL_NVFP4_DECODE;
    }

    // --- 3. Compute available VRAM ---
    // Feature-aware reserve instead of flat 1 GiB.
    budget.reserve_bytes = 256ULL * 1024 * 1024;  // base: cuBLAS + driver
    if (config.use_cuda_graphs)
        budget.reserve_bytes += 256ULL * 1024 * 1024;
    if (config.use_green_contexts)
        budget.reserve_bytes += 128ULL * 1024 * 1024;
    if (config.use_fp8_prefill)
        budget.reserve_bytes += 128ULL * 1024 * 1024;
    budget.reserve_bytes = std::max(budget.reserve_bytes, static_cast<size_t>(512ULL * 1024 * 1024));
    // At least 10% of total VRAM as headroom — but skipped for second-pass
    // NVFP4 (mode 2). In that mode the NVFP4 MoE caching path already enforced
    // its own ~1 GiB reserve before this runs; piling another 10% on top here
    // pushed `available` to 0 on Nemotron-H NVFP4 (32 GiB GPU → 3.2 GiB extra
    // reserve > what the MoE pass left free), which collapsed the KV cache to
    // the 16-block floor and left long-prompt requests stuck in pending_.
    size_t total_vram = 0;
    {
        size_t f;
        cudaMemGetInfo(&f, &total_vram);
    }
    if (config.use_nvfp4_decode != 2) {
        budget.reserve_bytes = std::max(budget.reserve_bytes, total_vram / 10);
    }

    // Estimate SSM footprint
    size_t ssm_footprint = 0;
    if (mcfg.ssm_inner_size > 0) {
        int n_ssm = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model.layer(i).ssm_in.data != nullptr)
                n_ssm++;
        if (n_ssm > 0) {
            int conv_ch = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
            int n_heads = mcfg.ssm_dt_rank;
            int hd_ssm = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            ssm_footprint = static_cast<size_t>(n_ssm) * config.max_batch_size *
                            (conv_ch * std::max(mcfg.ssm_conv_kernel - 1, 0) * sizeof(float) +
                             n_heads * hd_ssm * mcfg.ssm_state_size * dtype_size(config.ssm_state_dtype));
        }
    }

    size_t available = free_vram;
    size_t overhead = budget.reserve_bytes + ssm_footprint;
    available = (available > overhead) ? (available - overhead) : 0;

    // --- 4. Compute KV cache per-block cost ---
    int bs = config.kv_block_size > 0 ? config.kv_block_size : kKVBlockSize;
    size_t single_block_bytes;
    // Packed 4-bit KV dtypes: 2 elements per byte (FP4 nibbles or INT4 packed).
    // NVFP4 was historically missing from this OR-chain — fell through to the
    // dtype_size() fallback which returns 0 for QType::NVFP4, silently zeroing
    // out NVFP4's KV-cache budget contribution. Pre-existing pre-MXFP4-KV; the
    // Slice 2 spec reviewer flagged it during the MXFP4-KV scope review.
    if (config.kv_cache_dtype == QType::INT4 || config.kv_cache_dtype == QType::NVFP4 ||
        config.kv_cache_dtype == QType::MXFP4_KV) {
        single_block_bytes = static_cast<size_t>(bs) * mcfg.n_kv_heads * head_dim / 2;
    } else {
        single_block_bytes = static_cast<size_t>(bs) * mcfg.n_kv_heads * head_dim *
                             dtype_size(config.kv_cache_dtype);
    }
    // K+V (2x) for all supported dtypes.
    size_t per_block_total = single_block_bytes * 2 * n_kv_layers;
    if (config.kv_cache_dtype == QType::INT8 || config.kv_cache_dtype == QType::INT4) {
        size_t scale_per_block = static_cast<size_t>(bs) * mcfg.n_kv_heads * sizeof(half);
        per_block_total += scale_per_block * 2 * n_kv_layers;  // K scales + V scales (always 2x)
    }
    // NVFP4 / MXFP4_KV: 1 scale byte per 16 elems per head per token, K+V (2x).
    if (config.kv_cache_dtype == QType::NVFP4 || config.kv_cache_dtype == QType::MXFP4_KV) {
        size_t scale_per_block = static_cast<size_t>(bs) * mcfg.n_kv_heads * (head_dim / 16);
        per_block_total += scale_per_block * 2 * n_kv_layers;
    }

    int blocks_per_seq = (config.max_seq_len + bs - 1) / bs;
    int needed_blocks = blocks_per_seq * config.max_batch_size;

    // --- 5. Estimate NVFP4-eligible weight cache size ---
    auto nvfp4_beneficial = [](QType qt) -> bool {
        using enum QType;
        switch (qt) {
            case Q8_0:
            case Q8_K:
            case Q6_K:
            case Q5_K:
                return true;
            default:
                return false;
        }
    };

    size_t nvfp4_elems = 0;
    auto count_nvfp4 = [&](const Tensor& w, QType qt) {
        if (!w.data || !nvfp4_beneficial(qt))
            return;
        if (w.shape[1] % 16 != 0)
            return;
        nvfp4_elems += static_cast<size_t>(w.shape[0]) * w.shape[1];
    };

    count_nvfp4(model.output_proj(), model.out_proj_.qtype);
    for (int i = 0; i < mcfg.n_layers; i++) {
        const auto& L = model.layer(i);
        count_nvfp4(L.wq, L.wq.qtype);
        count_nvfp4(L.wk, L.wk.qtype);
        count_nvfp4(L.wv, L.wv.qtype);
        count_nvfp4(L.wo, L.wo.qtype);
        count_nvfp4(L.w_gate, L.w_gate.qtype);
        count_nvfp4(L.w_up, L.w_up.qtype);
        count_nvfp4(L.w_down, L.w_down.qtype);
        count_nvfp4(L.ssm_in, L.ssm_in.qtype);
        count_nvfp4(L.ssm_out, L.ssm_out.qtype);
        count_nvfp4(L.w_gate_shared, L.w_gate_shared.qtype);
        count_nvfp4(L.w_up_shared, L.w_up_shared.qtype);
        count_nvfp4(L.w_down_shared, L.w_down_shared.qtype);
    }

    size_t nvfp4_estimate = nvfp4_elems / 2 + nvfp4_elems / 16;
    size_t cutlass_sf_estimate = nvfp4_elems / 16;

    // --- 6. Allocate based on strategy ---
    switch (budget.strategy) {
        case VRAMBudget::NVFP4_DECODE_ONLY: {
            budget.fp8_cache_bytes = 0;
            budget.nvfp4_cache_bytes = nvfp4_estimate;
            size_t weight_total = nvfp4_estimate + cutlass_sf_estimate;
            size_t kv_available = (available > weight_total) ? (available - weight_total) : 0;
            budget.kv_cache_bytes = static_cast<size_t>(kv_available * 0.8);
            budget.kv_max_blocks = (per_block_total > 0)
                                       ? static_cast<int>(budget.kv_cache_bytes / per_block_total)
                                       : needed_blocks;
            budget.nvfp4_second_pass = false;
            break;
        }
        case VRAMBudget::FP8_PREFILL_NVFP4_DECODE: {
            // NVFP4 decode cache is critical for performance — ensure it fits first.
            // FP8 prefill cache is nice-to-have but not essential (fallback: dequant on-the-fly).
            budget.nvfp4_cache_bytes = nvfp4_estimate;
            double kv_fraction = (config.use_nvfp4_decode == 2) ? 0.1 : 0.8;
            budget.kv_cache_bytes = static_cast<size_t>(available * kv_fraction);
            budget.kv_max_blocks = (per_block_total > 0)
                                       ? static_cast<int>(budget.kv_cache_bytes / per_block_total)
                                       : needed_blocks;
            if (config.use_nvfp4_decode != 2)
                budget.kv_max_blocks = std::max(budget.kv_max_blocks, needed_blocks);
            // FP8 budget is computed below — after the kv_max_blocks clamp /
            // min_kv_blocks enforcement — so it reflects the FINAL KV size.
            budget.nvfp4_second_pass = (config.use_nvfp4_decode == 2);
            break;
        }
        case VRAMBudget::FP16_ONLY: {
            budget.fp8_cache_bytes = 0;
            budget.nvfp4_cache_bytes = 0;
            budget.kv_cache_bytes = static_cast<size_t>(available * 0.8);
            budget.kv_max_blocks = (per_block_total > 0)
                                       ? static_cast<int>(budget.kv_cache_bytes / per_block_total)
                                       : needed_blocks;
            budget.nvfp4_second_pass = false;
            break;
        }
    }
    budget.kv_max_blocks = std::max(budget.kv_max_blocks, 16);

    int target_blocks = (config.use_nvfp4_decode == 2 || budget.strategy == VRAMBudget::NVFP4_DECODE_ONLY)
                            ? needed_blocks
                            : needed_blocks * 2;
    budget.kv_max_blocks = std::min(budget.kv_max_blocks, target_blocks);
    budget.kv_max_blocks = std::max(budget.kv_max_blocks, 16);

    // Enforce minimum KV token budget. Auto default: 16K tokens or 4x max_seq_len,
    // whichever is smaller (capped to not exceed what VRAM can physically hold).
    int min_kv_tok = config.min_kv_tokens;
    bool user_requested_min = (min_kv_tok > 0);
    if (!user_requested_min) {
        min_kv_tok = std::min(16384, config.max_seq_len * 4);
    }
    int min_kv_blocks = (min_kv_tok + bs - 1) / bs;
    int max_affordable = (per_block_total > 0) ? static_cast<int>(available / per_block_total)
                                               : budget.kv_max_blocks;
    // Defensive cap for auto mode (leaves room for weight caches). When the
    // user explicitly sets min_kv_tokens, respect their request up to the
    // physical max_affordable — they're opting into a tighter weight-cache
    // budget in exchange for more context.
    int cap = user_requested_min ? max_affordable : static_cast<int>(max_affordable * 0.8);
    min_kv_blocks = std::min(min_kv_blocks, cap);
    if (budget.kv_max_blocks < min_kv_blocks) {
        IMP_LOG_INFO("VRAM budget: raising KV from %d to %d blocks (min_kv_tokens=%d)", budget.kv_max_blocks,
                     min_kv_blocks, min_kv_tok);
        budget.kv_max_blocks = min_kv_blocks;
        budget.kv_cache_bytes = static_cast<size_t>(min_kv_blocks) * per_block_total;
    }

    // FP8 prefill: use remaining VRAM after NVFP4 decode + the *final* KV size.
    // Computing this earlier (against the unclamped kv_max_blocks) silently
    // zeroed FP8 in mode 1 because the 0.8 kv_fraction filled the budget on
    // paper, even though `target_blocks` clamped the actual KV allocation
    // back down. The post-clamp computation lets mode 1 (additive) populate
    // both caches when VRAM allows — Qwen3-14B Q6_K mode 1 default flags
    // previously cached fp8=0 tensors and paid ~28 % prefill for it.
    if (budget.strategy == VRAMBudget::FP8_PREFILL_NVFP4_DECODE) {
        size_t kv_actual = static_cast<size_t>(budget.kv_max_blocks) * per_block_total;
        size_t nvfp4_actual = nvfp4_estimate + cutlass_sf_estimate;
        size_t remaining_for_fp8 = (available > kv_actual + nvfp4_actual)
                                       ? (available - kv_actual - nvfp4_actual)
                                       : 0;
        budget.fp8_cache_bytes = std::min(nvfp4_elems, remaining_for_fp8);
    }

    const char* strat_name = (budget.strategy == VRAMBudget::FP8_PREFILL_NVFP4_DECODE)
                                 ? "FP8_PREFILL_NVFP4_DECODE"
                             : (budget.strategy == VRAMBudget::NVFP4_DECODE_ONLY) ? "NVFP4_DECODE_ONLY"
                                                                                  : "FP16_ONLY";
    IMP_LOG_INFO(
        "VRAM budget: strategy=%s, available=%.1f MiB, "
        "kv=%d blocks (%.1f MiB), nvfp4=%.1f MiB, fp8=%.1f MiB, "
        "reserve=%.0f MiB, second_pass=%s",
        strat_name, available / (1024.0 * 1024.0), budget.kv_max_blocks,
        (per_block_total > 0 ? budget.kv_max_blocks * per_block_total : 0) / (1024.0 * 1024.0),
        nvfp4_estimate / (1024.0 * 1024.0), budget.fp8_cache_bytes / (1024.0 * 1024.0),
        budget.reserve_bytes / (1024.0 * 1024.0), budget.nvfp4_second_pass ? "yes" : "no");

    return budget;
}

}  // namespace imp
