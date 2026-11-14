// Grouped NVFP4 MoE prefill path (gemm_grouped_nvfp4_tile.cu), extracted from
// executor_forward_moe.cu: GraphExecutor::try_run_moe_cutlass3x_nvfp4_prefill_().

#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_forward_moe_internal.h"
#include "exec/moe_prefill_decision.h"  // select_moe_prefill_path — the mirror this file verifies against
#include "core/dispatch_policy.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_grouped_nvfp4_decode.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "compute/dispatch_record.h"  // resolved-path recording (#1205)
#include "compute/activation.h"
#include "compute/moe_routing.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"  // compute_M_per_from_offsets_device, bzero_sfa_active
#include "quant/nvfp4_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "core/logging.h"
#include "core/pdl.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

namespace imp {

// Path-selection order + gates mirror select_moe_prefill_path
// (moe_prefill_decision.h). That mirror used to be test-only (F-3): a
// reorder here left the test green without enforcing sync. This function
// now CONSULTS the mirror: each tier records what it actually did, and
// once a tier wins, the model is replayed against those observations and
// must name the same winner. KNOWN LIMIT: a tier reordered ahead of the
// winner that would have accepted stays invisible (chain short-circuits).
// One-shot logging so a divergence cannot flood the serving log.
static void verify_against_moe_routing_model(bool expert_bias_glu, const MoePrefillWorkspace& obs,
                                             MoePrefillPath chosen) {
    const MoePrefillPath modeled = select_moe_prefill_path(expert_bias_glu, obs);
    if (modeled == chosen)
        return;
    static bool warned = false;
    if (warned)
        return;
    warned = true;
    IMP_LOG_ERROR(
        "MoE routing model disagrees with the dispatch: dispatch ran %s, "
        "select_moe_prefill_path() says %s. executor_forward_moe_cutlass.cu and "
        "moe_prefill_decision.h have drifted apart (F-3) — the routing unit test "
        "is now describing a dispatch that does not exist.",
        moe_prefill_path_name(chosen), moe_prefill_path_name(modeled));
}

// Pre-cache miss: per-call H2D of the expert pointer / alpha tables into the workspace caches. NOT
// graph-capturable: the memcpy sources are stack vectors, so a recorded node would read a dead stack address
// on every replay (#860). False on a failed upload (the caller's !ok fallback runs, no GEMM on stale tables).
static bool resolve_expert_tables(const WeightRegistry& registry, const std::vector<TensorID>& weight_ids,
                                  int ne, const MoEWorkspace& moe, cudaStream_t stream, const void**& d_B,
                                  const void**& d_SFB, float*& d_a, float act_scale) {
    if (d_B && d_SFB && d_a)
        return true;
    moe_host_args_capture_guard(stream);
    std::vector<const void*> h_B_ptrs(ne), h_SFB_ptrs(ne);
    std::vector<float> h_alpha(ne);
    for (int e = 0; e < ne; ++e) {
        const auto& h = registry.handle(weight_ids[e]);
        h_B_ptrs[e] = h.payload.cutlass_nvfp4.weight;
        h_SFB_ptrs[e] = h.payload.cutlass_nvfp4.sf;
        h_alpha[e] = h.cutlass_nvfp4_alpha(act_scale);
    }
    const size_t ptr_bytes = ne * sizeof(const void*);
    IMP_CUDA_CHECK_BOOL(cudaMemcpyAsync(static_cast<void*>(moe.d_B_ptrs_cache),
                                        static_cast<const void*>(h_B_ptrs.data()), ptr_bytes,
                                        cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_BOOL(cudaMemcpyAsync(static_cast<void*>(moe.d_SFB_ptrs_cache),
                                        static_cast<const void*>(h_SFB_ptrs.data()), ptr_bytes,
                                        cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_BOOL(cudaMemcpyAsync(moe.d_alpha_full, h_alpha.data(), ne * sizeof(float),
                                        cudaMemcpyHostToDevice, stream));
    d_B = moe.d_B_ptrs_cache;
    d_SFB = moe.d_SFB_ptrs_cache;
    d_a = moe.d_alpha_full;
    return true;
}

// Decode rows: gate + up + activation + NVFP4 quantize in one grouped launch, down in a second
// (gemm_grouped_nvfp4_decode.h), per-row bits equal to the 4-launch tile path. The down input lives in the
// FP16 up buffer, which this path never writes. False = not taken; the caller runs the tile path.
static bool launch_moe_decode_grouped(const MoEWorkspace::PerLayerNvfp4DeviceArgsCache& da,
                                      const MoEWorkspace& moe, const int32_t* d_offsets, int ne, int d,
                                      int eff, int expanded, MoeDecodeEpi epi, void* down_out,
                                      float down_act_scale, cudaStream_t stream) {
    const bool gated = epi != MoeDecodeEpi::kRelu2;
    if (!da.d_up_B_ptrs || !da.d_up_SFB_ptrs || !da.d_up_alpha || !da.d_down_B_ptrs || !da.d_down_SFB_ptrs ||
        !da.d_down_alpha || (gated && (!da.d_gate_B_ptrs || !da.d_gate_SFB_ptrs || !da.d_gate_alpha)))
        return false;
    if (!gemm_grouped_nvfp4_decode_supports(ne, eff, d, epi) ||
        !gemm_grouped_nvfp4_decode_supports(ne, d, eff, MoeDecodeEpi::kF16) ||
        gemm_grouped_nvfp4_decode_act_bytes(ne, expanded, eff) > moe.expert_up.nbytes())
        return false;
    char* mid = static_cast<char*>(moe.expert_up.data);
    char* mid_sf = mid + gemm_grouped_nvfp4_decode_act_sf_offset(expanded, eff);
    GroupedNvfp4DecodeArgs a{};
    a.d_M_per = moe.d_M_per;
    a.d_expert_offsets = d_offsets;
    a.d_sfa_offsets = moe.d_sfa_offsets;  // prep_sfa(d) of the gate/up input
    a.base_A_packed = moe.cutlass3x_packed;
    a.base_A_sf = moe.cutlass3x_sf;
    a.d_B_ptrs = da.d_up_B_ptrs;
    a.d_SFB_ptrs = da.d_up_SFB_ptrs;
    a.d_alpha = da.d_up_alpha;
    if (gated) {
        a.d_G_ptrs = da.d_gate_B_ptrs;
        a.d_SFG_ptrs = da.d_gate_SFB_ptrs;
        a.d_galpha = da.d_gate_alpha;
    }
    a.base_D = mid;
    a.base_D_sf = mid_sf;
    a.act_scale = down_act_scale;  // the down alpha table carries the same factor
    if (!gemm_grouped_nvfp4_decode(ne, eff, d, expanded, epi, a, stream))
        return false;
    GroupedNvfp4DecodeArgs b{};
    b.d_M_per = moe.d_M_per;
    b.d_expert_offsets = d_offsets;
    b.base_A_packed = mid;
    b.base_A_sf = mid_sf;  // SF slab offsets computed in the kernel (K = eff)
    b.d_B_ptrs = da.d_down_B_ptrs;
    b.d_SFB_ptrs = da.d_down_SFB_ptrs;
    b.d_alpha = da.d_down_alpha;
    b.base_D = down_out;
    return gemm_grouped_nvfp4_decode(ne, d, eff, expanded, MoeDecodeEpi::kF16, b, stream);
}

struct MoeDecodeGroupedIn {
    const MoEWorkspace::PerLayerNvfp4DeviceArgsCache& da;
    const MoEWorkspace& moe;
    const int32_t* d_offsets;
    int n, ne, d, eff, expanded;
    int max_rows;   // imp::cfg::kMoeNvfp4DecodeRows
    bool eligible;  // no calibration, no expert bias GLU (gpt-oss)
    bool non_gated;
    FFNActivation act;
    void* down_out;
    float down_act_scale;  // MoEWorkspace::ActScales::dn
};

static bool run_moe_decode_grouped(const MoeDecodeGroupedIn& in, cudaStream_t stream) {
    if (in.n > in.max_rows || !in.eligible ||
        (!in.non_gated && in.act != FFNActivation::GEGLU && in.act != FFNActivation::SWIGLU))
        return false;
    const MoeDecodeEpi epi = in.non_gated                     ? MoeDecodeEpi::kRelu2
                             : in.act == FFNActivation::GEGLU ? MoeDecodeEpi::kGeglu
                                                              : MoeDecodeEpi::kSwiglu;
    if (!launch_moe_decode_grouped(in.da, in.moe, in.d_offsets, in.ne, in.d, in.eff, in.expanded, epi,
                                   in.down_out, in.down_act_scale, stream)) {
        static std::atomic<bool> s_refused{false};
        if (!s_refused.exchange(true))
            IMP_LOG_WARN(
                "MoE decode rows: grouped NVFP4 decode GEMM refused (n=%d ne=%d d=%d eff=%d, down input "
                "%zu B vs up buffer %zu B), tile path",
                in.n, in.ne, in.d, in.eff, gemm_grouped_nvfp4_decode_act_bytes(in.ne, in.expanded, in.eff),
                in.moe.expert_up.nbytes());
        return false;
    }
    static std::atomic<bool> s_logged{false};  // activity proof, once
    if (!s_logged.exchange(true))
        IMP_LOG_INFO("MoE decode rows: grouped NVFP4 decode GEMM ACTIVE (gate+up+act fused, n=%d, epi=%d)",
                     in.n, static_cast<int>(epi));
    return true;
}

bool GraphExecutor::try_run_moe_cutlass3x_nvfp4_prefill_(int layer, cudaStream_t stream,
                                                          MoeFfnContext& ctx) {
    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    const MoEWorkspace::ActScales las = moe_.act_scales(layer);  // static act scales, gate|up and down input
    int&  n        = ctx.n;
    int&  d        = ctx.d;
    int&  ne       = ctx.ne;
    int&  eff      = ctx.eff;
    int&  expanded = ctx.expanded;
    bool& non_gated_experts = ctx.non_gated_experts;
    MoeRoutingResult& routing = ctx.routing;

    auto covers_ids = [&](const std::vector<TensorID>& ids) {
        if (static_cast<int>(ids.size()) < ctx.ne)
            return false;
        for (int e = 0; e < ctx.ne; ++e) {
            if (ids[e] == kInvalidTensorID)
                return false;
            if (registry_.handle(ids[e]).primary_tier != StorageTier::CUTLASS_NVFP4)
                return false;
        }
        return true;
    };
    // Grouped NVFP4 path predicate.
    // Observations for the routing-model replay are filled in
    // as the chain is walked; a tier the chain never reaches keeps its false
    // (see verify_against_moe_routing_model's KNOWN LIMIT). Host-resident
    // NVFP4 experts: staging the layer here is why `covers_ids` cannot see these weights.
    const bool staged_covers = stage_layer_for_prefill_(layer, stream, ctx);

    MoePrefillWorkspace obs{};
    obs.staged_blocks = ctx.staged_blocks;
    const bool ids_cover = covers_ids(ly.expert_up_ids) && covers_ids(ly.expert_down_ids) &&
                           (ctx.non_gated_experts || covers_ids(ly.expert_gate_ids));
    obs.grouped_available = gemm_grouped_nvfp4_moe_supports(ne, d, eff) && moe_.cutlass3x_packed &&
                            moe_.cutlass3x_sf && (staged_covers || ids_cover);
    if (!obs.grouped_available) {
        verify_against_moe_routing_model(model_->profile().moe_expert_bias_glu, obs, MoePrefillPath::LEGACY);
        return false;
    }

    // Grouped NVFP4xNVFP4->FP16 tile GEMM (gemm_grouped_nvfp4_dispatch), no dequant pass, per-expert alpha
    // from dargs.d_alpha in the epilogue. Device-args: gate/up/act/down on
    // device before any D2H+sync (MoE prefill graph capture). Legacy on dispatch failure or empty workspace.
    bool device_args_done = false;
    {
        // gpt-oss (#547): the fused act+quantize kernel knows SwiGLU/GeGLU/ReLU2 only and has no
        // per-expert bias hooks; the unstaged path below runs bias + apply_expert_activation
        // (GPT_OSS_GLU) + a plain quantize instead. The staged path keeps gpt-oss on legacy.
        const bool use_device_args = (cfg.arch != ModelArch::GPT_OSS || !ctx.staged_blocks) && moe_.d_M_per &&
                                     moe_.d_M_per_count >= ne && moe_.d_sfa_offsets && moe_.d_B_ptrs_cache &&
                                     moe_.d_SFB_ptrs_cache && moe_.d_alpha_full && moe_.cutlass3x_packed &&
                                     moe_.cutlass3x_sf && moe_.cutlass3x_sfa_ptrs &&
                                     moe_.cutlass3x_sfa_ptrs_count >= ne;
        if (use_device_args) {
            // Log once per process; layer==0 fires on every
            // forward, but only the first ever needs to flag
            // the path choice.
            static std::atomic<bool> s_da_logged{false};
            if (layer == 0 && !s_da_logged.exchange(true))
                IMP_LOG_INFO(
                    "MoE prefill: CUTLASS 3.x device-args full path "
                    "(default)");
            // Populate device-resident d_M_per (no D2H).
            imp::compute_M_per_from_offsets_device(static_cast<const int32_t*>(routing.expert_offsets.data),
                                                   moe_.d_M_per, ne, stream);

            // gathered_base intentionally not bound: gate/up input quantize reads
            // ctx.no via sorted_token_ids directly (skip-the-read half of the
            // gather+quant fusion). Down projection input comes from
            // fused_act_quantize_device, which reads gate/up outputs directly.
            char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
            char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
            char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

            // SFA buffer prep — shared by both gate/up quant
            // (K_in = d) and the fused down-input quant (K_in = eff).
            auto prep_sfa = [&](int K_in) {
                imp::compute_sfa_offsets_device(moe_.d_M_per, moe_.d_sfa_offsets, ne, K_in, stream,
                                                reinterpret_cast<uint8_t**>(moe_.cutlass3x_sfa_ptrs),
                                                moe_.cutlass3x_sf);
                // Zeroes only the ACTIVE prefix of the SFA staging buffer: padded rows of
                // the SfAtom layout must be 0 for clean grouped-GEMM reads. QW5 (#604) replaces
                // the full cudaMemsetAsync with a bounded device kernel that reads
                // d_sfa_offsets[ne] as the byte count, capped at cutlass3x_sf_size.
                imp::bzero_sfa_active(moe_.cutlass3x_sf, moe_.d_sfa_offsets, ne, moe_.cutlass3x_sf_size,
                                      stream);
            };
            // (Former quantize_device lambda removed — the gate/up call site
            // below uses quantize_fp16_to_nvfp4_cutlass_moe_gather inline.
            // The down-projection input is handled by fused_act_quantize_device.)

            // Fused activation+quantize for the down-projection input: reads gate/up
            // directly, computes SwiGLU/GeGLU/ReLU^2 in registers, writes only packed
            // FP4+SFA (M1, #604). Non-gated (RELU_SQR): gate is nullptr, reads `up`
            // only; unlike the legacy relu_sqr_inplace, this does NOT clobber `up`.
            auto fused_act_quantize_device = [&](const char* gate_base, const char* up_base, int K_in,
                                                 FFNActivation act_type, const void* gate_bias = nullptr,
                                                 const void* up_bias = nullptr) {
                prep_sfa(K_in);
                imp::fused_act_quantize_fp16_to_nvfp4_cutlass_moe(
                    gate_base, up_base, moe_.cutlass3x_packed,
                    reinterpret_cast<uint8_t* const*>(moe_.cutlass3x_sfa_ptrs),
                    static_cast<const int*>(routing.expert_offsets.data), expanded, K_in, ne, act_type,
                    stream, gate_bias, up_bias, las.dn);
            };

            // Per-layer pre-cached arrays (Phase 3c-full Step 3): when ready, all
            // three projections feed dispatch_device with device-resident ptr arrays,
            // no per-call host iteration or H2D. Falls back to per-call upload via the
            // Step 1 workspace caches when the pre-cache isn't built.
            const bool da_cache_ready = layer < static_cast<int>(moe_.per_layer_da_cache.size()) &&
                                        moe_.per_layer_da_cache[layer].ready;

            // Host-resident experts: the staged copy stands in for the resident
            // one, so from here this layer is dispatched like any other. See
            // build_staged_device_args_().
            MoEWorkspace::PerLayerNvfp4DeviceArgsCache da_cache{};
            if (!build_staged_device_args_(ctx, non_gated_experts, da_cache) && da_cache_ready)
                da_cache = moe_.per_layer_da_cache[layer];

            auto dispatch_device = [&](const std::vector<TensorID>& weight_ids, const void** pre_d_B,
                                       const void** pre_d_SFB, float* pre_d_alpha, char* c_base, int K_in,
                                       int N_out, int e0 = 0, int n_e = -1) -> bool {
                // Expert block [e0, e0 + n_e): the offset arrays are absolute prefix sums, so a
                // block is a slice of every per-expert array with the A/D bases unchanged.
                const int n_grp = n_e < 0 ? ne : n_e;
                const void** d_B   = pre_d_B;
                const void** d_SFB = pre_d_SFB;
                float*       d_a   = pre_d_alpha;
                const float as = &weight_ids == &ly.expert_down_ids ? las.dn : las.gu;
                if (!resolve_expert_tables(registry_, weight_ids, ne, moe_, stream, d_B, d_SFB, d_a, as))
                    return false;

                imp::GroupedNvfp4DeviceArgs dargs{};
                dargs.d_M_per          = moe_.d_M_per + e0;
                dargs.d_expert_offsets = static_cast<const int32_t*>(
                    routing.expert_offsets.data) + e0;
                dargs.d_sfa_offsets    = moe_.d_sfa_offsets + e0;
                dargs.d_alpha          = d_a + e0;
                dargs.base_A_packed    = moe_.cutlass3x_packed;
                dargs.base_A_sf        = moe_.cutlass3x_sf;
                dargs.d_B_ptrs         = d_B + e0;
                dargs.d_SFB_ptrs       = d_SFB + e0;
                dargs.base_D           = c_base;
                return imp::gemm_grouped_nvfp4_dispatch(n_grp, N_out, K_in, expanded, dargs, stream);
            };

            // Gate/Up share input quantization (K_in=d): fused gather+quantize reads
            // ctx.no in token order via sorted_token_ids and writes packed FP4+SFA in
            // expert-sorted layout in one kernel pass, saving the gathered_base HBM read.
            prep_sfa(d);
            imp::quantize_fp16_to_nvfp4_cutlass_moe_gather(
                ctx.no.data, static_cast<const int32_t*>(routing.sorted_token_ids.data),
                moe_.cutlass3x_packed, reinterpret_cast<uint8_t* const*>(moe_.cutlass3x_sfa_ptrs),
                static_cast<const int*>(routing.expert_offsets.data), expanded, d, ne, stream, las.gu);
            bool ok = true;
            if (ctx.staged_blocks) {
                const auto& da = da_cache;
                ok = run_staged_blocks_(
                    layer, stream, ctx,
                    [&](ExpertProj p, int e0, int n_ex) {
                        if (p == ExpertProj::Gate)
                            return dispatch_device(ly.expert_gate_ids, da.d_gate_B_ptrs, da.d_gate_SFB_ptrs,
                                                   da.d_gate_alpha, expert_gate_base, d, eff, e0, n_ex);
                        if (p == ExpertProj::Up)
                            return dispatch_device(ly.expert_up_ids, da.d_up_B_ptrs, da.d_up_SFB_ptrs,
                                                   da.d_up_alpha, expert_up_base, d, eff, e0, n_ex);
                        return dispatch_device(ly.expert_down_ids, da.d_down_B_ptrs, da.d_down_SFB_ptrs,
                                               da.d_down_alpha, expert_down_base, eff, d, e0, n_ex);
                    },
                    [&] {
                        fused_act_quantize_device(non_gated_experts ? nullptr : expert_gate_base,
                                                  expert_up_base, eff,
                                                  non_gated_experts ? FFNActivation::RELU_SQR
                                                                    : cfg.ffn_activation);
                    });
            } else if (run_moe_decode_grouped({da_cache, moe_,
                                               static_cast<const int32_t*>(routing.expert_offsets.data), n,
                                               ne, d, eff, expanded, imp::cfg::kMoeNvfp4DecodeRows,
                                               calib_ == nullptr && !model_->profile().moe_expert_bias_glu,
                                               non_gated_experts, cfg.ffn_activation, expert_down_base,
                                               las.dn},
                                              stream)) {
                // Decode rows done: gate + up + act + quantize, down (gemm_grouped_nvfp4_decode.h).
            } else {
                if (!non_gated_experts)
                    ok = ok &&
                         dispatch_device(ly.expert_gate_ids, da_cache.d_gate_B_ptrs, da_cache.d_gate_SFB_ptrs,
                                         da_cache.d_gate_alpha, expert_gate_base, d, eff);
                ok = ok && dispatch_device(ly.expert_up_ids, da_cache.d_up_B_ptrs, da_cache.d_up_SFB_ptrs,
                                           da_cache.d_up_alpha, expert_up_base, d, eff);
                if (ok) {
                    // Fused: activation (SwiGLU/GeGLU/ReLU²) + NVFP4 quant
                    // for the down-projection input. Saves one HBM
                    // round-trip of the swiglu intermediate per layer.
                    const FFNActivation act_type = non_gated_experts ? FFNActivation::RELU_SQR
                                                                     : cfg.ffn_activation;
                    const char* gate_for_fused = non_gated_experts ? nullptr : expert_gate_base;
                    const auto* d_offs = static_cast<const int32_t*>(routing.expert_offsets.data);
                    const bool gpt_oss = model_->profile().moe_expert_bias_glu;
                    // gpt-oss (#547, #2466): gate/up biases + clamped GLU fused into the quantize (cfg act is
                    // GPT_OSS_GLU); the down bias rides the fused scatter when phase 7 takes it.
                    fused_act_quantize_device(gate_for_fused, expert_up_base, eff, act_type,
                                              ly.expert_gate_bias.data, ly.expert_up_bias.data);
                    ok = dispatch_device(ly.expert_down_ids, da_cache.d_down_B_ptrs, da_cache.d_down_SFB_ptrs,
                                         da_cache.d_down_alpha, expert_down_base, eff, d);
                    if (ok && gpt_oss) {
                        if (moe_scatter_can_add_down_bias(ctx, compute_dtype_))
                            ctx.scatter_down_bias = ly.expert_down_bias.data;
                        else
                            moe_add_expert_bias_sorted(expert_down_base, ly.expert_down_bias.data, d_offs, ne,
                                                       expanded, d, stream);
                    }
                }
            }
            if (ok) {
                device_args_done = true;
                // What the tier DID, not what its preconditions promised: the block
                // above can still fail on dispatch and fall through to the legacy
                // host-args path below.
                obs.device_args_ready = true;
                dispatch_record::set_moe_prefill_tier(MoePrefillPath::DEVICE_ARGS);
                verify_against_moe_routing_model(model_->profile().moe_expert_bias_glu, obs,
                                                 MoePrefillPath::DEVICE_ARGS);
            } else {
                IMP_LOG_ERROR("device-args full path failed; falling back to legacy");
            }
        }
    }

    if (!device_args_done) {
        // Lazy moe_gather: caller skipped the upstream gather when this path was
        // predicted to fire; the fast path didn't, so the legacy fallback needs
        // the gathered FP16 intermediate. Idempotent (moe_gather_done guard).
        if (!ctx.moe_gather_done) {
            int64_t gath_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(d)};
            Tensor gathered(moe_.gathered.data, compute_dtype_, 2, gath_shape, true);
            moe_gather(ctx.no, routing, gathered, stream);
            ctx.moe_gather_done = true;
            IMP_LOG_WARN(
                "MoE prefill: device-args predicted but inner gate refused — "
                "lazy moe_gather + legacy fallback (one wasted decision; "
                "investigate moe_cutlass3x_will_use_device_args_ mismatch)");
        }

        // Legacy D2H+sync grouped dispatch path.
        moe_host_args_capture_guard(stream);
        std::vector<int32_t> h_offsets(ne + 1);
        moe_host_args_ok_or_throw(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                                                  static_cast<size_t>(ne + 1) * sizeof(int32_t),
                                                  cudaMemcpyDeviceToHost, stream),
                                  "cutlass3x legacy prefill");
        // Populate device-resident d_M_per in parallel with the D2H copy.
        // Phase 1 of MoE-prefill-graphs lever: foundation for graph-safe
        // dispatch (Phase 2+ migrates host M_per[] uses to this buffer).
        if (moe_.d_M_per && moe_.d_M_per_count >= ne) {
            imp::compute_M_per_from_offsets_device(static_cast<const int32_t*>(routing.expert_offsets.data),
                                                   moe_.d_M_per, ne, stream);
        }
        moe_host_args_ok_or_throw(cudaStreamSynchronize(stream), "cutlass3x legacy prefill");

        std::vector<int> M_per(ne);
        for (int e = 0; e < ne; ++e)
            M_per[e] = h_offsets[e + 1] - h_offsets[e];

        char* gathered_base = static_cast<char*>(moe_.gathered.data);
        char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
        char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
        char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
        char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

        dispatch_record::set_moe_prefill_tier(MoePrefillPath::GROUPED);
        obs.grouped_ready = true;
        verify_against_moe_routing_model(model_->profile().moe_expert_bias_glu, obs, MoePrefillPath::GROUPED);
        if (layer == 0)
            IMP_LOG_INFO("MoE prefill: grouped NVFP4 host-args (n=%d, expanded=%d)", n, expanded);

        // Active-expert SFA offset table, computed per K_in (differs for d vs
        // eff). Gate and up share K_in=d, so one quantize of gathered_base +
        // pointer array is reused for both grouped GEMMs; down reuses the staging
        // buffer with a fresh quantize for K_in=eff.
        auto quantize_once = [&](const char* a_base, int K_in, std::vector<size_t>& sfa_offsets,
                                 std::vector<uint8_t*>& h_sfa_bases, float act_scale) -> bool {
            sfa_offsets.assign(ne + 1, 0);
            h_sfa_bases.assign(ne, nullptr);
            size_t total_packed = 0;
            for (int e = 0; e < ne; ++e) {
                total_packed += static_cast<size_t>(M_per[e]) * K_in / 2;
                sfa_offsets[e + 1] = sfa_offsets[e] + cutlass_nvfp4_sf_size(M_per[e], K_in);
            }
            const size_t total_sfa = sfa_offsets[ne];
            if (total_packed > moe_.cutlass3x_packed_size || total_sfa > moe_.cutlass3x_sf_size) {
                IMP_LOG_ERROR("CUTLASS 3.x MoE staging too small: need packed=%zu sf=%zu, have %zu/%zu",
                              total_packed, total_sfa, moe_.cutlass3x_packed_size, moe_.cutlass3x_sf_size);
                return false;
            }
            uint8_t* all_sfa = static_cast<uint8_t*>(moe_.cutlass3x_sf);
            for (int e = 0; e < ne; ++e) {
                if (M_per[e] > 0)
                    h_sfa_bases[e] = all_sfa + sfa_offsets[e];
            }
            // Upload SFA pointer array to device (~1 KB).
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(static_cast<void*>(moe_.cutlass3x_sfa_ptrs),
                                               static_cast<const void*>(h_sfa_bases.data()),
                                               static_cast<size_t>(ne) * sizeof(uint8_t*),
                                               cudaMemcpyHostToDevice, stream));
            // Zero active SFA region (SfAtom pads rows to 128).
            IMP_CUDA_CHECK_LOG(cudaMemsetAsync(all_sfa, 0, total_sfa, stream));
            // Fused per-expert quantize in one kernel launch.
            quantize_fp16_to_nvfp4_cutlass_moe(a_base, moe_.cutlass3x_packed, moe_.cutlass3x_sfa_ptrs,
                                               routing.expert_offsets.data
                                                   ? static_cast<const int*>(routing.expert_offsets.data)
                                                   : nullptr,
                                               expanded, K_in, ne, stream, act_scale);
            return true;
        };

        // Legacy host-args dispatch, entered only when the default-on device-args
        // full path failed its precondition check (workspace buffers not
        // populated). Kept as the safety-net fallback path.
        auto grouped_gemm = [&](const std::vector<TensorID>& weight_ids, char* c_base, int K_in, int N_out,
                                const std::vector<size_t>& sfa_offsets, float act_scale) {
            std::vector<const void*> hB(ne), hSFB(ne);
            std::vector<float> hAlpha(ne);
            for (int e = 0; e < ne; ++e) {
                const auto& h = registry_.handle(weight_ids[e]);
                hB[e] = h.payload.cutlass_nvfp4.weight;
                hSFB[e] = h.payload.cutlass_nvfp4.sf;
                hAlpha[e] = h.cutlass_nvfp4_alpha(act_scale);
            }
            GroupedNvfp4HostArgs args{};
            args.offsets = h_offsets.data();
            args.sfa_offsets = sfa_offsets.data();
            args.B = hB.data();
            args.SFB = hSFB.data();
            args.alpha = hAlpha.data();
            args.base_A_packed = moe_.cutlass3x_packed;
            args.base_A_sf = moe_.cutlass3x_sf;
            args.base_D = c_base;
            // Last tier: an unwritten expert output fails the request instead of reaching the logits (#2446).
            if (!gemm_grouped_nvfp4_host_args(ne, N_out, K_in, expanded, args, stream))
                throw std::runtime_error(
                    "grouped NVFP4 host-args dispatch failed (K=" + std::to_string(K_in) +
                    " N=" + std::to_string(N_out) + " ne=" + std::to_string(ne) + ")");
        };

        // Gate+Up share the same input (gathered_base, K_in=d) → quantize once,
        // run two grouped GEMMs reusing staging buffers.
        std::vector<size_t> sfa_offs;
        std::vector<uint8_t*> sfa_bases;
        if (quantize_once(gathered_base, d, sfa_offs, sfa_bases, las.gu)) {
            if (!non_gated_experts)
                grouped_gemm(ly.expert_gate_ids, expert_gate_base, d, eff, sfa_offs, las.gu);
            grouped_gemm(ly.expert_up_ids, expert_up_base, d, eff, sfa_offs, las.gu);
        }

        // gpt-oss (#547): per-expert biases on gate/up BEFORE activation (rows are
        // expert-sorted — same seam as the dequant batch path).
        const int32_t* gpt_oss_offsets = model_->profile().moe_expert_bias_glu
                                             ? static_cast<const int32_t*>(routing.expert_offsets.data)
                                             : nullptr;
        if (gpt_oss_offsets) {
            moe_add_expert_bias_sorted(expert_gate_base, ly.expert_gate_bias.data, gpt_oss_offsets, ne,
                                       expanded, eff, stream);
            moe_add_expert_bias_sorted(expert_up_base, ly.expert_up_bias.data, gpt_oss_offsets, ne, expanded,
                                       eff, stream);
        }

        apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data, moe_.expert_swiglu.data,
                                non_gated_experts, expanded, eff, compute_dtype_, cfg.ffn_activation, stream);

        // Down has a different activation (post-SwiGLU, K_in=eff): re-quantize.
        // A fused silu(gate)*up+quantize kernel regressed short-prompt decode
        // (~11%, low SM occupancy at small expanded); the separate swiglu kernel
        // has better per-element parallelism, kept separate.
        char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        if (quantize_once(down_act, eff, sfa_offs, sfa_bases, las.dn)) {
            grouped_gemm(ly.expert_down_ids, expert_down_base, eff, d, sfa_offs, las.dn);
            if (gpt_oss_offsets)
                moe_add_expert_bias_sorted(expert_down_base, ly.expert_down_bias.data, gpt_oss_offsets, ne,
                                           expanded, d, stream);
        }
    }  // !device_args_done
    return true;
}

}  // namespace imp
