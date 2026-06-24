// Pre-dequant Phase 3 (MoE): MoE expert decode-cache build — gpt-oss
// MXFP4→NVFP4 conversion, the GGUF / NVFP4-prequant expert caching driver,
// and the per-projection contiguous native-NVFP4 cache builder.
// Split out of pre_dequant_phase3_nvfp4_decode.cu to keep each .cu under the
// kernel file-size threshold. See pre_dequant_internal.h / quant_pipeline.h
// for shared declarations.

#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "compute/gemm_cutlass_sm120.h"
#include "quant/dequant_gpu.h"
#include "quant/gpt_oss_mxfp4_convert.h"
#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace imp {

using imp::pre_dequant_internal::nvfp4_beneficial;

// Cache MoE expert weights — done after FP16 free so mode 2 has full budget.
// Handles two sub-paths:
//  - cache_moe_native_nvfp4: NVFP4-prequant SafeTensors (per-expert tensors)
//    consolidated into one contiguous packed_data + scales buffer per layer
//    per projection.
//  - cache_moe_expert_nvfp4: GGUF / re-quant path, expert_*_packed is the
//    3-D contiguous tensor.
// ---------------------------------------------------------------------------
// gpt-oss (#547): convert the HF-MXFP4 pre-packed experts (host-mmap'd
// blocks/scales) into the native NVFP4 MoE cache. e2m1 nibbles are
// bit-identical (linear pair order); ue8m0 scales expand 1→2 e4m3
// micro-scales under a per-expert tensor scale. gate_up rows arrive
// interleaved (g0,u0,g1,…) and are de-interleaved into separate gate/up
// results, so the whole proven NVFP4-MoE machinery (CUTLASS grouped prefill,
// gemv_nvfp4_moe decode, CUDA graphs) applies unchanged. Placeholder packed
// tensors keyed on the converted device pointers let Phase 4 wire
// nvfp4_moe_{gate,up,down}_ptr exactly like the Modelopt path.
// ---------------------------------------------------------------------------
void QuantPipeline::gpt_oss_convert_moe_experts_(const ModelConfig& cfg, Nvfp4DecodeContext& dctx) {
    int converted = 0;

    // GGUF path helper: convert one host-resident ggml-MXFP4 expert tensor
    // ([ne, N, K], separate per projection) into a device NvFP4MoEQuantResult.
    // ggml type-39 packs each 32-element block as [scale(1) | qs(16)] with
    // SPLIT nibble order (element j = low nibble of qs[j], j+16 = high nibble);
    // type-31 packs [qs(16) | scale(1)] in LINEAR pair order. The converter
    // expects linear pairs + a separate ue8m0 scale plane, so normalize here
    // (mirrors weight_upload.cu's upload_qtype_mxfp4_). stride=1 because GGUF
    // stores gate and up as distinct tensors (HF interleaves them, stride=2).
    auto gguf_convert = [&](const Tensor& t, float extra_scale, NvFP4MoEQuantResult& r,
                            std::vector<float>& ts) -> bool {
        if (!t.data || t.ndim < 3)
            return false;
        const int ne = static_cast<int>(t.shape[0]);
        const int64_t N = t.shape[1];
        const int64_t K = t.shape[2];
        const int64_t kb = K / 32;  // MXFP4 blocks per row
        const int64_t nblk = static_cast<int64_t>(ne) * N * kb;
        std::vector<uint8_t> blocks(static_cast<size_t>(nblk) * 16);
        std::vector<uint8_t> scales(static_cast<size_t>(nblk));
        const uint8_t* src = static_cast<const uint8_t*>(t.data);
        const bool v2 = t.mxfp4_layout_v2;
        for (int64_t blk = 0; blk < nblk; ++blk) {
            const uint8_t* sb = src + static_cast<size_t>(blk) * 17;  // 17B ggml block
            uint8_t* db = blocks.data() + static_cast<size_t>(blk) * 16;
            if (v2) {
                scales[blk] = sb[0];
                const uint8_t* qs = sb + 1;
                for (int b = 0; b < 16; ++b) {
                    const int e0 = 2 * b, e1 = 2 * b + 1;
                    const uint8_t n0 = (e0 < 16) ? (qs[e0] & 0xF) : (qs[e0 - 16] >> 4);
                    const uint8_t n1 = (e1 < 16) ? (qs[e1] & 0xF) : (qs[e1 - 16] >> 4);
                    db[b] = static_cast<uint8_t>(n0 | (n1 << 4));
                }
            } else {
                std::memcpy(db, sb, 16);
                scales[blk] = sb[16];
            }
        }
        return gpt_oss_convert_experts_to_nvfp4(blocks.data(), scales.data(), ne, N, K,
                                                /*offset=*/0, /*stride=*/1, r, extra_scale, &ts);
    };

    for (int i = 0; i < cfg.n_layers; ++i) {
        TransformerLayer& L = const_cast<Model*>(model_)->layer(i);

        int ne = 0;
        NvFP4MoEQuantResult g{}, u{}, d{};
        std::vector<float> g_ts, u_ts, d_ts;

        const Tensor& gu_b = L.expert_gate_up_packed_blocks;
        const Tensor& gu_s = L.expert_gate_up_packed_scales;
        const Tensor& dn_b = L.expert_down_packed_blocks;
        const Tensor& dn_s = L.expert_down_packed_scales;

        if (gu_b.data && gu_s.data && dn_b.data && dn_s.data) {
            // SafeTensors: HF packed blocks (gate_up interleaved, down separate).
            if (gu_b.on_device || dn_b.on_device) {
                IMP_LOG_ERROR("gpt-oss L%d: packed expert tensors unexpectedly on device — skipping", i);
                continue;
            }
            ne = static_cast<int>(gu_b.shape[0]);
            const int64_t gu_rows = gu_b.shape[1];   // 2*d_ff, interleaved
            const int64_t gu_K = gu_b.shape[2] * 32;  // K from block count
            const int64_t dn_rows = dn_b.shape[1];   // d_model
            const int64_t dn_K = dn_b.shape[2] * 32;  // d_ff
            bool ok = gpt_oss_convert_experts_to_nvfp4(static_cast<const uint8_t*>(gu_b.data),
                                                       static_cast<const uint8_t*>(gu_s.data), ne, gu_rows,
                                                       gu_K, /*offset=*/0, /*stride=*/2, g, 1.0f, &g_ts) &&
                      gpt_oss_convert_experts_to_nvfp4(static_cast<const uint8_t*>(gu_b.data),
                                                       static_cast<const uint8_t*>(gu_s.data), ne, gu_rows,
                                                       gu_K, /*offset=*/1, /*stride=*/2, u, 1.0f, &u_ts) &&
                      // down: extra 2^-4 — residual-stream rescale (see arch
                      // registry comment in model.cpp; bias scaled in the loader).
                      gpt_oss_convert_experts_to_nvfp4(static_cast<const uint8_t*>(dn_b.data),
                                                       static_cast<const uint8_t*>(dn_s.data), ne, dn_rows,
                                                       dn_K, /*offset=*/0, /*stride=*/1, d,
                                                       /*extra_scale=*/0.0625f, &d_ts);
            if (!ok) {
                IMP_LOG_ERROR("gpt-oss L%d: MXFP4→NVFP4 expert conversion failed (VRAM?)", i);
                return;
            }
        } else if (L.expert_gate_packed.data && !L.expert_gate_packed.on_device &&
                   L.expert_gate_packed.qtype == QType::MXFP4) {
            // GGUF: separate ggml-MXFP4 gate/up/down expert tensors (host-mmap),
            // kept host-resident by weight_upload's gpt-oss carve-out.
            ne = static_cast<int>(L.expert_gate_packed.shape[0]);
            if (!gguf_convert(L.expert_gate_packed, 1.0f, g, g_ts) ||
                !gguf_convert(L.expert_up_packed, 1.0f, u, u_ts) ||
                !gguf_convert(L.expert_down_packed, 0.0625f, d, d_ts)) {
                IMP_LOG_ERROR("gpt-oss L%d: GGUF MXFP4→NVFP4 expert conversion failed", i);
                return;
            }
        } else {
            continue;  // no convertible expert source on this layer
        }

        auto install = [&](NvFP4MoEQuantResult& r, Tensor& packed_slot) {
            int64_t shp[3] = {r.n_experts, r.N, r.K};
            packed_slot = Tensor(r.packed_data, QType::NVFP4, 3, shp, /*on_device=*/true);
            wcache_->nvfp4_moe[r.packed_data] = r;
            auto* m = const_cast<Model*>(model_);
            m->gpu_allocations_.push_back(r.packed_data);
            m->gpu_allocations_.push_back(r.micro_scales);
            m->gpu_allocations_.push_back(r.tensor_scales);
        };
        install(g, L.expert_gate_packed);
        install(u, L.expert_up_packed);
        install(d, L.expert_down_packed);

        // Per-expert CUTLASS_NVFP4 registration (#547 prefill): without it,
        // covers_ids() rejects the CUTLASS 3.x grouped path and prefill runs
        // the dequant->FP16->cuBLAS batch fallback (~38 GB of FP16 dequant
        // writes per forward — pp512 ~1.9k vs ~15k+ tok/s). Mirrors
        // cache_moe_native_nvfp4's re-stamp block: shared SfAtom buffer per
        // projection + per-expert Tensor slices into the contiguous result so
        // Phase 4's register_tensor() sees CUTLASS-tier wcache entries.
        auto register_cutlass = [&](NvFP4MoEQuantResult& r, std::vector<Tensor>& experts,
                                    const std::vector<float>& h_ts) -> bool {
            if (!cutlass_sm120_nvfp4_available())
                return false;
            const size_t sf_per_expert =
                cutlass_nvfp4_sf_size(static_cast<int>(r.N), static_cast<int>(r.K));
            const size_t sfatom_total = static_cast<size_t>(ne) * sf_per_expert;
            void* d_sfatom = vram_alloc_force(vram_alloc_, sfatom_total, "gptoss_moe_sfatom");
            if (!d_sfatom) {
                IMP_LOG_WARN("gpt-oss L%d: SfAtom alloc failed (%.1f MiB) — prefill stays on the "
                             "dequant fallback",
                             i, sfatom_total / (1024.0 * 1024.0));
                return false;
            }
            convert_nvfp4_moe_scales_to_sfatom(r.micro_scales, d_sfatom, ne, static_cast<int>(r.N),
                                               static_cast<int>(r.K), /*stream=*/nullptr);
            IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());
            const_cast<Model*>(model_)->gpu_allocations_.push_back(d_sfatom);
            experts.assign(ne, Tensor{});
            for (int e = 0; e < ne; ++e) {
                void* data_slice =
                    static_cast<char*>(r.packed_data) + static_cast<size_t>(e) * r.expert_stride_packed;
                void* ms_slice =
                    static_cast<char*>(r.micro_scales) + static_cast<size_t>(e) * r.expert_stride_ms;
                int64_t eshape[2] = {r.N, r.K / 2};  // packed-byte convention (K/2)
                Tensor w(data_slice, QType::NVFP4, 2, eshape, /*on_device=*/true);
                w.scales = ms_slice;
                w.tensor_scale = h_ts[e];
                experts[e] = w;

                NvFP4QuantResult nv;
                nv.packed_data = data_slice;
                nv.micro_scales = ms_slice;
                nv.owned = false;  // slices into the contiguous conversion result
                nv.tensor_scale = h_ts[e];
                nv.N = r.N;
                nv.K = r.K;
                wcache_->nvfp4[data_slice] = nv;

                CutlassNvFP4Weight cw;
                cw.data = data_slice;
                cw.scale_factors = static_cast<char*>(d_sfatom) + static_cast<size_t>(e) * sf_per_expert;
                cw.tensor_scale = h_ts[e];
                cw.N = r.N;
                cw.K = r.K;
                cw.sf_bytes = sf_per_expert;
                cw.sf_borrowed = true;
                wcache_->cutlass_nvfp4[data_slice] = cw;
            }
            wcache_->cutlass_nvfp4_bytes += sfatom_total;
            return true;
        };
        bool ct_ok = register_cutlass(g, L.expert_w_gate, g_ts) &&
                     register_cutlass(u, L.expert_w_up, u_ts) &&
                     register_cutlass(d, L.expert_w_down, d_ts);
        if (i == 0)
            IMP_LOG_INFO("gpt-oss: CUTLASS grouped prefill %s", ct_ok ? "registered" : "UNAVAILABLE");

        dctx.nvfp4_moe_count += 3;
        converted++;
    }
    if (converted)
        IMP_LOG_INFO("gpt-oss: converted MXFP4→NVFP4 experts for %d layers "
                     "(ne=%d, gate/up de-interleaved)",
                     converted, cfg.n_experts);
}

void QuantPipeline::nvfp4_decode_cache_moe_experts_(const ModelConfig& cfg,
                                                    size_t& remaining_budget,
                                                    cudaStream_t stream,
                                                    Nvfp4DecodeContext& dctx) {
    (void)remaining_budget;
    size_t moe_budget;
    // Cache MoE expert weights — done after FP16 free so mode 2 has full budget
    if (wcache_->nvfp4_decode_mode == 2) {
        size_t free_mem = 0, total_mem = 0;
        IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem, &total_mem));
        // Reserve VRAM so the KV cache (sized after this in init_kv_cache)
        // can fit `min_kv_tokens` (default 16K) + workspaces. Computed from
        // the model's actual attention layout — the previous 1 GiB constant
        // was over-cautious for hybrid models (Nemotron-H: 6/52 attn layers,
        // <100 MiB KV at 16K) where it starved the NVFP4 MoE cache and
        // forced decode through the legacy D2H-sync fallback.
        //
        // Capped at 1 GiB (the previous static value) so this can only
        // RELEASE budget, never tighten it vs the previous behavior. Floor
        // at 256 MiB to keep workspace + scratch room.
        //
        // IMP_MOE_RESERVE_MIB still overrides for manual tuning (range
        // 128-4096 MiB).
        int n_attn_layers = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            if (model_->layer(i).wq.data != nullptr &&
                model_->layer(i).gdn_gate.data == nullptr)
                n_attn_layers++;
        }
        if (n_attn_layers == 0)
            n_attn_layers = cfg.n_layers;
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        int kv_heads = cfg.n_kv_heads > 0 ? cfg.n_kv_heads : cfg.n_heads;
        // 16K tokens × n_attn × 2 (K+V) × kv_heads × hd × FP16
        constexpr int kKvFloorTokens = 16384;
        size_t per_token_kv = static_cast<size_t>(n_attn_layers) * 2 *
                              static_cast<size_t>(kv_heads) * static_cast<size_t>(hd) * 2;
        size_t kv_reserve = static_cast<size_t>(kKvFloorTokens) * per_token_kv;
        constexpr size_t kWorkspaceSafety = 256ULL * 1024 * 1024;
        constexpr size_t kReserveCap = 1024ULL * 1024 * 1024;
        constexpr size_t kReserveFloor = 256ULL * 1024 * 1024;
        size_t kMoeReserve = std::clamp(kv_reserve + kWorkspaceSafety, kReserveFloor, kReserveCap);
        {
            const int v = runtime_config().moe.reserve_mib;
            if (v >= 128 && v <= 4096)
                kMoeReserve = static_cast<size_t>(v) * 1024ULL * 1024ULL;
        }
        IMP_LOG_DEBUG("MoE reserve: %.0f MiB (n_attn=%d, kv_heads=%d, hd=%d → %.0f MiB KV at 16K + 256 MiB workspace)",
                      kMoeReserve / (1024.0 * 1024.0), n_attn_layers, kv_heads, hd,
                      kv_reserve / (1024.0 * 1024.0));
        constexpr size_t kRuntimeHeadroom = 512ULL * 1024 * 1024;
        size_t total_reserve = kMoeReserve + kRuntimeHeadroom;
        moe_budget = (free_mem > total_reserve) ? (free_mem - total_reserve) : 0;
    } else {
        moe_budget = (remaining_budget > wcache_->nvfp4_bytes) ? (remaining_budget - wcache_->nvfp4_bytes)
                                                              : 0;
    }
    bool moe_budget_exhausted = false;
    // Self-tracked logical budget for cache_moe_native_nvfp4 (NVFP4 prequant
    // SafeTensors). cudaMemGetInfo doesn't reflect the per-expert cudaFree's
    // promptly on this driver, so we track allocations and frees logically.
    // Initial value is moe_budget plus the per-expert weights that the
    // function will swap out — those sum to the cached size, so net per
    // call is zero and all 40 layers fit if the initial budget covers one
    // layer's worth of overhead.
    size_t moe_logical_avail = moe_budget;

    const bool decode_all_moe = runtime_config().gemm.nvfp4_decode_all;
    auto cache_moe_expert_nvfp4 = [&](const Tensor& packed, QType qtype) {
        if (!packed.data)
            return;
        if (!nvfp4_beneficial(qtype, decode_all_moe))
            return;
        if (wcache_->nvfp4_moe.count(packed.data))
            return;
        if (moe_budget_exhausted)
            return;
        if (!packed.on_device)
            return;
        if (packed.ndim < 3)
            return;

        int ne = static_cast<int>(packed.shape[0]);
        int rows = static_cast<int>(packed.shape[1]);
        int cols = static_cast<int>(packed.shape[2]);
        if (cols % 16 != 0)
            return;
        if (!dequant_gpu_supported(qtype) || !qscratch_->dequant)
            return;

        size_t nvfp4_bytes = static_cast<size_t>(ne) * rows * cols / 2 +
                             static_cast<size_t>(ne) * rows * cols / 16 +
                             static_cast<size_t>(ne) * sizeof(float);

        if (dctx.nvfp4_moe_total + nvfp4_bytes > moe_budget) {
            moe_budget_exhausted = true;
            IMP_LOG_INFO(
                "NVFP4 MoE cache: VRAM budget reached after %d MoE tensors "
                "(%.1f / %.1f MiB)",
                dctx.nvfp4_moe_count, dctx.nvfp4_moe_total / (1024.0 * 1024.0), moe_budget / (1024.0 * 1024.0));
            return;
        }

        NvFP4MoEQuantResult result;
        quantize_packed_experts_to_nvfp4(packed.data, qtype, ne, rows, cols, qscratch_->dequant, result,
                                         stream);

        wcache_->nvfp4_moe[packed.data] = result;
        dctx.nvfp4_moe_total += nvfp4_bytes;
        dctx.nvfp4_moe_count++;
    };

    // NVFP4-prequant SafeTensors path: experts arrive as per-expert tensors
    // (expert_w_gate[e] / expert_w_up[e] / expert_w_down[e]) with NVFP4
    // qtype + .scales / .tensor_scale sidecars promoted in Phase 0. The 3D
    // expert_*_packed tensors are NULL (the loader only stamps them for
    // GGUF and Gemma-4). Without this branch, cache_moe_expert_nvfp4 would
    // early-return at `!packed.data` and the legacy FP16 dequant + cuBLAS
    // sm_80 WMMA fallback fires per layer per token, killing CUDA Graphs.
    //
    // We allocate one contiguous packed_data + micro_scales + tensor_scales
    // buffer per layer per projection, copy the per-expert pointers in,
    // and stamp `packed.data` so wcache lookups (line below the layer loop)
    // and the consumer dispatch in executor_forward_moe.cu (lookup via
    // expert_*_packed.data) wire up automatically. After a successful copy
    // for a layer the per-expert allocations are freed inline — at 35B-A3B
    // the duplicate (per-expert + contiguous) would peak at ~30 GiB which
    // doesn't fit in 32 GiB, and the legacy fallback can't fire for layers
    // where nvfp4_moe_*_ptr is non-null anyway.

    for (int i = 0; i < cfg.n_layers; i++) {
        // Need mutable access to expert_*_packed for cache_moe_native_nvfp4
        // to stamp the contiguous buffer pointer. const_cast follows the
        // existing pattern at e.g. lines 1517 / 1598 of weight_upload.cu.
        auto& L = const_cast<Model*>(model_)->layer(i);

        bool g = false, u = false, d = false;
        if (cfg.is_nvfp4_prequant) {
            g = cache_moe_native_nvfp4_(L.expert_gate_packed, L.expert_w_gate, stream, dctx, moe_budget_exhausted, moe_logical_avail);
            u = cache_moe_native_nvfp4_(L.expert_up_packed, L.expert_w_up, stream, dctx, moe_budget_exhausted, moe_logical_avail);
            d = cache_moe_native_nvfp4_(L.expert_down_packed, L.expert_w_down, stream, dctx, moe_budget_exhausted, moe_logical_avail);
            // Non-gated MoE (e.g. Nemotron-H NemotronHForCausalLM): no gate
            // projection exists, so g=0 is expected when up and down cached.
            // Suppress the misleading warning in that case; expert_gemm's
            // wcache_->nvfp4_moe lookup handles the missing-gate path.
            bool non_gated = (L.expert_gate_packed.data == nullptr &&
                              (L.expert_w_gate.empty() ||
                               L.expert_w_gate[0].data == nullptr));
            if ((g || u || d) && !(g && u && d) && !(non_gated && u && d)) {
                IMP_LOG_WARN(
                    "Layer %d: partial NVFP4 MoE native cache "
                    "(g=%d u=%d d=%d) — fast path may not engage",
                    i, (int)g, (int)u, (int)d);
            }
        }

        // GGUF / re-quant path: only run when native didn't populate.
        // For GGUF NVFP4-target models the source qtype is Q*_K/Q8_0 and
        // packed.data is non-null; for prequant SafeTensors all three
        // native calls succeeded above and these are no-ops because
        // packed.data now points into wcache_->nvfp4_moe.
        if (!g)
            cache_moe_expert_nvfp4(L.expert_gate_packed, L.expert_gate_packed.qtype);
        if (!u)
            cache_moe_expert_nvfp4(L.expert_up_packed, L.expert_up_packed.qtype);
        if (!d)
            cache_moe_expert_nvfp4(L.expert_down_packed, L.expert_down_packed.qtype);
    }

    if (dctx.nvfp4_moe_count > 0) {
        wcache_->nvfp4_moe_bytes = dctx.nvfp4_moe_total;
        IMP_LOG_INFO("NVFP4 MoE cache: %d tensors, %.2f MiB", dctx.nvfp4_moe_count,
                     dctx.nvfp4_moe_total / (1024.0 * 1024.0));
        if (dctx.nvfp4_moe_ms_freed > 0)
            IMP_LOG_INFO("NVFP4 MoE cache: freed %.2f MiB duplicated per-expert micro-scales "
                         "(contiguous ms_ref copies are now the single source)",
                         dctx.nvfp4_moe_ms_freed / (1024.0 * 1024.0));
    } else if (wcache_->nvfp4.empty()) {
        IMP_LOG_INFO("NVFP4 decode: no eligible weights found (all ≤ 4.5 bits/elem)");
    }
}


// Extracted from nvfp4_decode_cache_moe_experts_ (was a 325-line [&] lambda).
// Builds the contiguous NVFP4 decode cache for one MoE projection's experts;
// see the declaration in executor.h for the full contract. The budget flags
// (moe_budget_exhausted / moe_logical_avail) are threaded in so the per-layer
// accounting is shared across the gate/up/down calls.
bool QuantPipeline::cache_moe_native_nvfp4_(Tensor& packed, std::vector<Tensor>& experts,
                                            cudaStream_t stream, Nvfp4DecodeContext& dctx,
                                            bool& moe_budget_exhausted, size_t& moe_logical_avail) {
        if (experts.empty() || !experts[0].data)
            return false;
        if (experts[0].qtype != QType::NVFP4 || experts[0].scales == nullptr)
            return false;
        if (packed.data && wcache_->nvfp4_moe.count(packed.data))
            return false;
        // ZERO-COPY decode cache (LEAD-2): NVFP4-prequant SafeTensors upload the
        // per-expert weights AND scales into contiguous VRAM (one buffer per
        // projection, sliced per expert). When that holds we point an
        // NvFP4MoEQuantResult directly at the existing buffers — no 15 GiB
        // contiguous duplicate, no per-expert copy. Only the tiny per-expert
        // tensor_scales array is allocated. This engages the fast
        // gemv_nvfp4_moe_* decode kernels (base + expert_stride) instead of the
        // CUTLASS grouped GEMM, which under-utilizes the GPU at M=1 decode.
        // Guarded by a strict contiguity + shape check; on any mismatch we fall
        // through to leaving the experts on the CUTLASS path (prior behavior).
        if (wcache_->cutlass_nvfp4.count(experts[0].data)) {
          if (runtime_config().gemm.nvfp4_moe_decode) {
            const int ne_z = static_cast<int>(experts.size());
            const int64_t N_z = experts[0].shape[0];
            const int64_t Kp_z = experts[0].shape[1];
            const int64_t K_z = Kp_z * 2;
            if (K_z % 16 == 0 && N_z > 0 && experts[0].scales) {
                const size_t e_packed = static_cast<size_t>(N_z) * Kp_z;
                const size_t e_ms = static_cast<size_t>(N_z) * (K_z / 16);
                // Data must be contiguous to borrow it zero-copy (the big ~15 GiB
                // win). Scales are small (~1/16 of weights) — copy them into a
                // contiguous buffer if they aren't already, so non-contiguous
                // scale uploads still take the fast path.
                bool data_contig = true, scales_contig = true, shapes_ok = true;
                std::vector<float> h_ts(ne_z);
                for (int e = 0; e < ne_z; ++e) {
                    const auto& w = experts[e];
                    if (!w.data || !w.scales || w.shape[0] != N_z || w.shape[1] != Kp_z) {
                        shapes_ok = false;
                        break;
                    }
                    if (static_cast<const char*>(w.data) !=
                        static_cast<const char*>(experts[0].data) + static_cast<size_t>(e) * e_packed)
                        data_contig = false;
                    if (static_cast<const char*>(w.scales) !=
                        static_cast<const char*>(experts[0].scales) + static_cast<size_t>(e) * e_ms)
                        scales_contig = false;
                    h_ts[e] = w.tensor_scale;
                }
                if (shapes_ok && data_contig) {
                    const size_t total_ms = static_cast<size_t>(ne_z) * e_ms;
                    void* ms_base = experts[0].scales;
                    void* d_ms_copy = nullptr;
                    if (!scales_contig) {
                        d_ms_copy = vram_alloc_force(vram_alloc_, total_ms, "nvfp4_moe_ms_ref");
                        if (d_ms_copy) {
                            for (int e = 0; e < ne_z; ++e)
                                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                                    static_cast<char*>(d_ms_copy) + static_cast<size_t>(e) * e_ms,
                                    experts[e].scales, e_ms, cudaMemcpyDeviceToDevice, stream));
                            ms_base = d_ms_copy;
                        }
                    }
                    float* d_ts = static_cast<float*>(
                        vram_alloc_force(vram_alloc_, static_cast<size_t>(ne_z) * sizeof(float),
                                         "nvfp4_moe_ts_ref"));
                    if ((scales_contig || d_ms_copy) && d_ts) {
                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_ts, h_ts.data(),
                                                           static_cast<size_t>(ne_z) * sizeof(float),
                                                           cudaMemcpyHostToDevice, stream));
                        NvFP4MoEQuantResult r;
                        r.packed_data = experts[0].data;  // borrowed (resident, contiguous)
                        r.micro_scales = ms_base;         // borrowed or small contiguous copy
                        r.tensor_scales = d_ts;
                        r.n_experts = ne_z;
                        r.N = N_z;
                        r.K = K_z;
                        r.expert_stride_packed = e_packed;
                        r.expert_stride_ms = e_ms;
                        r.borrowed = true;  // data borrowed from model; scales/ts via VRAMAllocator
                        int64_t shp[3] = {static_cast<int64_t>(ne_z), N_z, K_z};
                        packed = Tensor(experts[0].data, QType::NVFP4, 3, shp, /*on_device=*/true);
                        wcache_->nvfp4_moe[experts[0].data] = r;
                        dctx.nvfp4_moe_count++;
                        // The contiguous micro-scale copy is now the single
                        // source for every consumer (decode cache via
                        // r.micro_scales; CUTLASS prefill reads its own
                        // Phase-0 SfAtom buffers, never the raw scales; the
                        // Phase-4 registry snapshot runs after this and
                        // picks up re-stamped pointers). Free the scattered
                        // per-expert source scales — they were resident
                        // TWICE, ~1.7 GiB across 144 groups on
                        // Qwen3-30B-A3B-NVFP4 — and re-stamp the layer
                        // tensors + per-expert nvfp4 wcache entries onto
                        // the copy slices. Mirrors the non-borrow branch's
                        // free-after-copy below.
                        if (d_ms_copy) {
                            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                            auto* mut_model = const_cast<Model*>(model_);
                            size_t freed = 0;
                            for (int e = 0; e < ne_z; ++e) {
                                auto& w = experts[e];
                                void* old_sc = w.scales;
                                void* new_sc = static_cast<char*>(d_ms_copy) +
                                               static_cast<size_t>(e) * e_ms;
                                if (old_sc && mut_model->is_base_gpu_allocation(old_sc)) {
                                    mut_model->release_gpu_allocation(old_sc);
                                    IMP_CUDA_CHECK_LOG(cudaFree(old_sc));
                                    freed += e_ms;
                                }
                                w.scales = new_sc;
                                auto nit = wcache_->nvfp4.find(w.data);
                                if (nit != wcache_->nvfp4.end() &&
                                    nit->second.micro_scales == old_sc)
                                    nit->second.micro_scales = new_sc;
                            }
                            if (freed)
                                dctx.nvfp4_moe_ms_freed += freed;
                        }
                        IMP_LOG_INFO("NVFP4 MoE native: data-borrow decode cache (ne=%d N=%lld K=%lld, "
                                     "scales_contig=%d; gemv_nvfp4_moe fast decode)",
                                     ne_z, (long long)N_z, (long long)K_z, (int)scales_contig);
                        return true;
                    }
                    if (d_ms_copy)
                        vram_free(vram_alloc_, d_ms_copy);
                }
                IMP_LOG_INFO("NVFP4 MoE native: zero-copy decode declined (shapes_ok=%d data_contig=%d) — "
                             "leaving on CUTLASS path",
                             (int)shapes_ok, (int)data_contig);
            }
          }
          // Flag off, non-contiguous data, or alloc failed: leave experts on the
          // CUTLASS path (Phase-0 already registered per-expert NVFP4+CUTLASS).
          return true;
        }
        if (moe_budget_exhausted)
            return false;

        int ne = static_cast<int>(experts.size());
        // SafeTensors NVFP4 prequant: per-expert weight tensor on-disk
        // dtype is U8 (loader → INT8 → Phase-0 promote → NVFP4) and shape
        // is [N, K_packed] where K_packed = K_logical/2 (two FP4 nibbles
        // per byte). The same packed-shape convention is what the
        // existing executor_attention.cu / executor_ffn.cu NVFP4 dispatch
        // expects when computing `tmp.K = hw->shape[1] * 2`. Match that.
        int64_t N = experts[0].shape[0];
        int64_t K_packed = experts[0].shape[1];
        int64_t K = K_packed * 2;  // logical inner dim
        if (K % 16 != 0)
            return false;

        size_t expert_packed_bytes = static_cast<size_t>(N) * K_packed;
        size_t expert_ms_bytes = static_cast<size_t>(N) * (K / 16);
        size_t total_packed = static_cast<size_t>(ne) * expert_packed_bytes;
        size_t total_ms = static_cast<size_t>(ne) * expert_ms_bytes;
        size_t total_ts = static_cast<size_t>(ne) * sizeof(float);
        size_t add_bytes = total_packed + total_ms + total_ts;

        // Self-tracked logical budget. cudaMemGetInfo does NOT reflect
        // cudaFree's of upload-time per-expert weights in time on this
        // driver — after ~5 layers it reports free=0 even though the
        // heap has ~5 GiB freed but not yet reclaimed. The previous
        // per-call cudaMemGetInfo gate aborted at ~7 layers (21/120
        // entries) and left layers 7-39 on the legacy fallback path
        // with D2H expert_offsets sync, killing CUDA graph capture and
        // pinning decode at ~30 tok/s. Track the budget logically:
        // initialised once from cudaMemGetInfo, decremented on alloc,
        // incremented after per-expert frees below — net per-call
        // change is zero so all 40 layers fit.
        if (add_bytes > moe_logical_avail) {
            moe_budget_exhausted = true;
            IMP_LOG_INFO(
                "NVFP4 MoE native cache: logical budget reached after %d "
                "tensors (%.1f MiB cached, %.1f MiB logical avail, need %.1f MiB)",
                dctx.nvfp4_moe_count, dctx.nvfp4_moe_total / (1024.0 * 1024.0),
                moe_logical_avail / (1024.0 * 1024.0), add_bytes / (1024.0 * 1024.0));
            return false;
        }

        void* d_packed = vram_alloc_force(vram_alloc_, total_packed, "nvfp4_moe_packed_native");
        void* d_ms = vram_alloc_force(vram_alloc_, total_ms, "nvfp4_moe_ms_native");
        void* d_ts_raw = vram_alloc_force(vram_alloc_, total_ts, "nvfp4_moe_ts_native");
        if (!d_packed || !d_ms || !d_ts_raw) {
            if (d_packed)
                vram_free(vram_alloc_, d_packed);
            if (d_ms)
                vram_free(vram_alloc_, d_ms);
            if (d_ts_raw)
                vram_free(vram_alloc_, d_ts_raw);
            moe_budget_exhausted = true;
            IMP_LOG_WARN(
                "NVFP4 MoE native cache: cudaMalloc failed at %d "
                "tensors (%.1f MiB cached) — driver heap exhausted",
                dctx.nvfp4_moe_count, dctx.nvfp4_moe_total / (1024.0 * 1024.0));
            return false;
        }
        moe_logical_avail = (moe_logical_avail > add_bytes) ? (moe_logical_avail - add_bytes) : 0;
        float* d_ts = static_cast<float*>(d_ts_raw);

        std::vector<float> h_ts(ne);
        for (int e = 0; e < ne; ++e) {
            const auto& w = experts[e];
            if (w.shape[0] != N || w.shape[1] != K_packed || !w.data || !w.scales) {
                IMP_LOG_WARN(
                    "NVFP4 MoE native: expert %d shape/data mismatch, "
                    "rolling back layer",
                    e);
                vram_free(vram_alloc_, d_packed);
                vram_free(vram_alloc_, d_ms);
                vram_free(vram_alloc_, d_ts_raw);
                return false;
            }
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(static_cast<char*>(d_packed) +
                                                   static_cast<size_t>(e) * expert_packed_bytes,
                                               w.data, expert_packed_bytes, cudaMemcpyDeviceToDevice,
                                               stream));
            IMP_CUDA_CHECK_LOG(
                cudaMemcpyAsync(static_cast<char*>(d_ms) + static_cast<size_t>(e) * expert_ms_bytes,
                                w.scales, expert_ms_bytes, cudaMemcpyDeviceToDevice, stream));
            h_ts[e] = w.tensor_scale;
        }
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_ts, h_ts.data(), total_ts, cudaMemcpyHostToDevice, stream));

        NvFP4MoEQuantResult r;
        r.packed_data = d_packed;
        r.micro_scales = d_ms;
        r.tensor_scales = d_ts;
        r.n_experts = ne;
        r.N = N;
        r.K = K;
        r.expert_stride_packed = expert_packed_bytes;
        r.expert_stride_ms = expert_ms_bytes;

        // Stamp the packed Tensor so wcache_->nvfp4_moe key + consumer
        // wiring (expert_*_packed.data lookup) work uniformly with the
        // GGUF path. Logical K (NOT K/2) per cache_moe_expert_nvfp4
        // convention at shape[2].
        int64_t shape[3] = {static_cast<int64_t>(ne), N, K};
        packed = Tensor(d_packed, QType::NVFP4, 3, shape, /*on_device=*/true);

        wcache_->nvfp4_moe[d_packed] = r;
        dctx.nvfp4_moe_total += add_bytes;
        dctx.nvfp4_moe_count++;

        // Free per-expert GPU allocations now — the legacy fallback path
        // (executor_forward_moe.cu:expert_gemm + chunked_dequant_gemm) can
        // no longer fire for this layer because nvfp4_moe_*_ptr is non-null
        // after the cache populates and stamps `packed`. Without freeing,
        // we hold the same NVFP4 weights twice (per-expert + contiguous);
        // the duplicate exhausts VRAM around layer 33 of Qwen3.6-35B-A3B
        // and breaks layers 33-39's fast path. Per-layer free keeps total
        // overhead bounded — only the just-copied 384 expert pointers are
        // released, and only after the contiguous copy succeeded.
        //
        // Sync the stream so the in-flight D2D copies (which read from
        // experts[e].data / .scales) finish before we cudaFree the source.
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        auto* mut_model = const_cast<Model*>(model_);
        size_t freed_bytes = 0;
        for (int e = 0; e < ne; ++e) {
            auto& w = experts[e];
            if (w.data) {
                mut_model->release_gpu_allocation(w.data);
                IMP_CUDA_CHECK_LOG(cudaFree(w.data));
                freed_bytes += expert_packed_bytes;
                w.data = nullptr;
                w.on_device = false;
            }
            if (w.scales) {
                mut_model->release_gpu_allocation(w.scales);
                IMP_CUDA_CHECK_LOG(cudaFree(w.scales));
                freed_bytes += expert_ms_bytes;
                w.scales = nullptr;
            }
        }
        moe_logical_avail += freed_bytes;

        // Re-stamp per-expert Tensors to slice into the contiguous packed +
        // micro-scale buffers and register CUTLASS_NVFP4 entries so the MoE
        // prefill fast path (executor_forward_moe.cu CUTLASS 3.x grouped
        // branch) can fire instead of dequant→FP16→cuBLAS. The cleanup loop
        // above nulled experts[e].data because the original per-expert
        // source allocs were freed; the executor needs valid slice pointers
        // for register_tensor() and the per-expert wcache_->cutlass_nvfp4
        // lookup. Without this block, expert_*_ids[e] = kInvalidTensorID
        // (because t.data == nullptr) and covers_ids() rejects the fast
        // path → 88% of prefill time is spent in dequantize_nvfp4_moe_kernel.
        if (cutlass_sm120_nvfp4_available()) {
            // Phase 0 may have already created per-expert CUTLASS entries
            // (with SfAtom scales). If so, just re-stamp the expert Tensors
            // to point into the contiguous buffer and reuse Phase 0's entries.
            bool phase0_has_cutlass = wcache_->cutlass_nvfp4.count(experts[0].data) != 0;
            size_t sf_per_expert = cutlass_nvfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
            size_t sfatom_total = static_cast<size_t>(ne) * sf_per_expert;
            void* d_sfatom = nullptr;
            if (!phase0_has_cutlass) {
                d_sfatom = (sfatom_total <= moe_logical_avail)
                               ? vram_alloc_force(vram_alloc_, sfatom_total, "nvfp4_moe_sfatom")
                               : nullptr;
                if (d_sfatom) {
                    convert_nvfp4_moe_scales_to_sfatom(d_ms, d_sfatom, ne, static_cast<int>(N),
                                                       static_cast<int>(K), stream);
                }
            }
            if (d_sfatom || phase0_has_cutlass) {
                for (int e = 0; e < ne; ++e) {
                    auto& w = experts[e];
                    void* old_data = w.data;
                    void* data_slice = static_cast<char*>(d_packed) +
                                       static_cast<size_t>(e) * expert_packed_bytes;
                    w.data = data_slice;
                    w.scales = static_cast<char*>(d_ms) + static_cast<size_t>(e) * expert_ms_bytes;
                    w.on_device = true;
                    w.tensor_scale = h_ts[e];
                    if (phase0_has_cutlass) {
                        // Move Phase 0's CUTLASS entry from old key to new key
                        auto it = wcache_->cutlass_nvfp4.find(old_data);
                        if (it != wcache_->cutlass_nvfp4.end()) {
                            CutlassNvFP4Weight cw = it->second;
                            cw.data = data_slice;
                            wcache_->cutlass_nvfp4.erase(it);
                            wcache_->cutlass_nvfp4[data_slice] = cw;
                        }
                        // Move NVFP4 entry too
                        auto nit = wcache_->nvfp4.find(old_data);
                        if (nit != wcache_->nvfp4.end()) {
                            NvFP4QuantResult nv = nit->second;
                            nv.packed_data = data_slice;
                            nv.micro_scales = w.scales;
                            nv.owned = false;  // borrows resident model storage — don't cudaFree on teardown
                            wcache_->nvfp4.erase(nit);
                            wcache_->nvfp4[data_slice] = nv;
                        }
                    } else {
                        void* sf_slice = static_cast<char*>(d_sfatom) +
                                         static_cast<size_t>(e) * sf_per_expert;
                        CutlassNvFP4Weight cw;
                        cw.data = data_slice;
                        cw.scale_factors = sf_slice;
                        cw.tensor_scale = h_ts[e];
                        cw.N = N;
                        cw.K = K;
                        cw.sf_bytes = sf_per_expert;
                        cw.sf_borrowed = true;
                        wcache_->cutlass_nvfp4[data_slice] = cw;
                    }
                }
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                wcache_->cutlass_nvfp4_bytes += sfatom_total;
                moe_logical_avail = (moe_logical_avail > sfatom_total)
                                        ? (moe_logical_avail - sfatom_total)
                                        : 0;
            } else {
                IMP_LOG_WARN(
                    "MoE NVFP4 SfAtom alloc failed (%.1f MiB for %d experts) "
                    "— prefill stays on dequant→cuBLAS fallback",
                    sfatom_total / (1024.0 * 1024.0), ne);
            }
        }

        return true;
}

}  // namespace imp
