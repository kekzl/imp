// CUTLASS 3.x NVFP4 MoE prefill path, extracted from
// executor_forward_moe.cu: GraphExecutor::try_run_moe_cutlass3x_nvfp4_prefill_().

#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_forward_moe_internal.h"
#include "exec/moe_prefill_decision.h"  // select_moe_prefill_path — the mirror this file verifies against
#include "core/dispatch_policy.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "compute/dispatch_record.h"  // resolved-path recording (#1205)
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "compute/activation.h"
#include "compute/moe_routing.h"
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
static void verify_against_moe_routing_model(ModelArch arch, const DispatchPolicy& rcfg,
                                             const MoePrefillWorkspace& obs, MoePrefillPath chosen) {
    const MoePrefillPath modeled = select_moe_prefill_path(arch, rcfg, obs);
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

// ---------------------------------------------------------------------------
bool GraphExecutor::try_run_moe_cutlass3x_nvfp4_prefill_(int layer, cudaStream_t stream,
                                                          MoeFfnContext& ctx) {
    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
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
    // CUTLASS 3.x NVFP4 grouped path predicate. moe.no_cutlass3x forces legacy
    // for debugging. Observations for the routing-model replay are filled in
    // as the chain is walked; a tier the chain never reaches keeps its false
    // (see verify_against_moe_routing_model's KNOWN LIMIT). Host-resident
    // NVFP4 experts: staging the layer here is why `covers_ids` cannot see these weights.
    const bool staged_covers = stage_layer_for_prefill_(layer, stream, ctx);

    MoePrefillWorkspace obs{};
    static const bool force_off = dispatch_policy().moe.no_cutlass3x;
    const bool ids_cover = covers_ids(ly.expert_up_ids) && covers_ids(ly.expert_down_ids) &&
                           (ctx.non_gated_experts || covers_ids(ly.expert_gate_ids));
    obs.grouped_available = !force_off && cutlass_grouped_3x_nvfp4_available() &&
                            moe_.cutlass3x_packed && moe_.cutlass3x_sf &&
                            (staged_covers || ids_cover);
    if (!obs.grouped_available) {
        verify_against_moe_routing_model(cfg.arch, dispatch_policy(), obs, MoePrefillPath::LEGACY);
        return false;
    }

// CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM (NVFP4xNVFP4->FP16), on by
// default (moe.no_cutlass3x forces legacy). Zero dequant overhead vs the
// NVFP4->FP16 batch path; per-group alpha via fusion_args.alpha_ptr_array.
// Fully device-args dispatch runs gate/up/activation/down end-to-end via
// device-resident kernels before any D2H+sync, a prerequisite for graph
// capture of MoE prefill. Falls back to legacy on dispatch failure or an unpopulated workspace.
bool device_args_done = false;
{
    // moe.nvfp4_device_args: default on; set false to force the legacy
    // host-args + smallM dispatch for A/B or workarounds.
    const bool da_enabled = dispatch_policy().moe.nvfp4_device_args;
    // gpt-oss (#547): the fused act+quantize kernel knows SwiGLU/GeGLU/ReLU2
    // only and has no per-expert bias hooks — the legacy host-args path below
    // runs apply_expert_activation (GPT_OSS_GLU-aware) with bias seams.
    const bool use_device_args =
        da_enabled && cfg.arch != ModelArch::GPT_OSS &&
        moe_.d_M_per && moe_.d_M_per_count >= ne &&
        moe_.d_sfa_offsets && moe_.d_B_ptrs_cache &&
        moe_.d_SFB_ptrs_cache && moe_.d_alpha_full &&
        moe_.cutlass3x_packed && moe_.cutlass3x_sf &&
        moe_.cutlass3x_sfa_ptrs &&
        moe_.cutlass3x_sfa_ptrs_count >= ne;
    if (use_device_args) {
        // Log once per process; layer==0 fires on every
        // forward, but only the first ever needs to flag
        // the path choice.
        static std::atomic<bool> s_da_logged{false};
        if (layer == 0 && !s_da_logged.exchange(true))
            IMP_LOG_INFO(
                "MoE prefill: CUTLASS 3.x device-args full path "
                "(default; set moe.nvfp4_device_args=false to disable)");
        // Populate device-resident d_M_per (no D2H).
        imp::compute_M_per_from_offsets_device(
            static_cast<const int32_t*>(routing.expert_offsets.data),
            moe_.d_M_per, ne, stream);

        // gathered_base intentionally not bound: gate/up input quantize reads
        // ctx.no via sorted_token_ids directly (skip-the-read half of the
        // gather+quant fusion). Down projection input comes from
        // fused_act_quantize_device, which reads gate/up outputs directly.
        char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
        char* expert_up_base   = static_cast<char*>(moe_.expert_up.data);
        char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

        // SFA buffer prep — shared by both gate/up quant
        // (K_in = d) and the fused down-input quant (K_in = eff).
        auto prep_sfa = [&](int K_in) {
            imp::compute_sfa_offsets_device(
                moe_.d_M_per, moe_.d_sfa_offsets, ne, K_in, stream);
            imp::build_sfa_bases_device(
                reinterpret_cast<uint8_t**>(moe_.cutlass3x_sfa_ptrs),
                moe_.cutlass3x_sf, moe_.d_sfa_offsets, ne, stream);
            // Zeroes only the ACTIVE prefix of the SFA staging buffer: padded rows of
            // the SfAtom layout must be 0 for clean CUTLASS reads. QW5 (#604) replaces
            // the full cudaMemsetAsync with a bounded device kernel that reads
            // d_sfa_offsets[ne] as the byte count, capped at cutlass3x_sf_size.
            imp::bzero_sfa_active(
                moe_.cutlass3x_sf,
                moe_.d_sfa_offsets,
                ne,
                moe_.cutlass3x_sf_size,
                stream);
        };
        // (Former quantize_device lambda removed — the gate/up call site
        // below uses quantize_fp16_to_nvfp4_cutlass_moe_gather inline.
        // The down-projection input is handled by fused_act_quantize_device.)

        // Fused activation+quantize for the down-projection input: reads gate/up
        // directly, computes SwiGLU/GeGLU/ReLU^2 in registers, writes only packed
        // FP4+SFA (M1, #604). Non-gated (RELU_SQR): gate is nullptr, reads `up`
        // only; unlike the legacy relu_sqr_inplace, this does NOT clobber `up`.
        auto fused_act_quantize_device = [&](const char* gate_base,
                                              const char* up_base, int K_in,
                                              FFNActivation act_type) {
            prep_sfa(K_in);
            imp::fused_act_quantize_fp16_to_nvfp4_cutlass_moe(
                gate_base, up_base, moe_.cutlass3x_packed,
                reinterpret_cast<uint8_t* const*>(moe_.cutlass3x_sfa_ptrs),
                static_cast<const int*>(routing.expert_offsets.data),
                expanded, K_in, ne, act_type, stream);
        };

        // Per-layer pre-cached arrays (Phase 3c-full Step 3): when ready, all
        // three projections feed dispatch_device with device-resident ptr arrays,
        // no per-call host iteration or H2D. Falls back to per-call upload via the
        // Step 1 workspace caches when the pre-cache isn't built.
        const bool da_cache_ready =
            layer < static_cast<int>(moe_.per_layer_da_cache.size()) &&
            moe_.per_layer_da_cache[layer].ready;

        // Host-resident experts: the staged copy stands in for the resident
        // one, so from here this layer is dispatched like any other. See
        // build_staged_device_args_().
        MoEWorkspace::PerLayerNvfp4DeviceArgsCache da_cache{};
        if (!build_staged_device_args_(ctx, non_gated_experts, da_cache) && da_cache_ready)
            da_cache = moe_.per_layer_da_cache[layer];

        auto dispatch_device =
            [&](const std::vector<TensorID>& weight_ids,
                const void** pre_d_B, const void** pre_d_SFB,
                float* pre_d_alpha, char* c_base, int K_in,
                int N_out) -> bool {
                const void** d_B   = pre_d_B;
                const void** d_SFB = pre_d_SFB;
                float*       d_a   = pre_d_alpha;
                if (!d_B || !d_SFB || !d_a) {
                    // Pre-cache miss falls back to per-call H2D into the Step 1 workspace
                    // caches. NOT graph-capturable: the memcpy sources are stack vectors, so a
                    // recorded node would read a dead stack address on every replay (#860,
                    // nondeterministic garbage pointers / misaligned-address graph launches).
                    moe_host_args_capture_guard(stream);
                    std::vector<const void*> h_B_ptrs(ne), h_SFB_ptrs(ne);
                    std::vector<float> h_alpha(ne);
                    for (int e = 0; e < ne; ++e) {
                        const auto& h = registry_.handle(weight_ids[e]);
                        h_B_ptrs[e]   = h.payload.cutlass_nvfp4.weight;
                        h_SFB_ptrs[e] = h.payload.cutlass_nvfp4.sf;
                        h_alpha[e] =
                            h.payload.cutlass_nvfp4.global_scale
                                ? *h.payload.cutlass_nvfp4.global_scale
                                : 1.0f;
                    }
                    cudaMemcpyAsync(moe_.d_B_ptrs_cache, h_B_ptrs.data(),
                                    ne * sizeof(const void*),
                                    cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(moe_.d_SFB_ptrs_cache,
                                    h_SFB_ptrs.data(),
                                    ne * sizeof(const void*),
                                    cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(moe_.d_alpha_full, h_alpha.data(),
                                    ne * sizeof(float),
                                    cudaMemcpyHostToDevice, stream);
                    d_B   = moe_.d_B_ptrs_cache;
                    d_SFB = moe_.d_SFB_ptrs_cache;
                    d_a   = moe_.d_alpha_full;
                }

                imp::GroupedNvfp4DeviceArgs dargs{};
                dargs.d_M_per          = moe_.d_M_per;
                dargs.d_expert_offsets = static_cast<const int32_t*>(
                    routing.expert_offsets.data);
                dargs.d_sfa_offsets    = moe_.d_sfa_offsets;
                dargs.d_alpha          = d_a;
                dargs.base_A_packed    = moe_.cutlass3x_packed;
                dargs.base_A_sf        = moe_.cutlass3x_sf;
                dargs.d_B_ptrs         = d_B;
                dargs.d_SFB_ptrs       = d_SFB;
                dargs.base_D           = c_base;
                return imp::gemm_grouped_cutlass_3x_nvfp4_device_args(
                    ne, N_out, K_in, dargs, stream);
            };

        // Gate/Up share input quantization (K_in=d): fused gather+quantize reads
        // ctx.no in token order via sorted_token_ids and writes packed FP4+SFA in
        // expert-sorted layout in one kernel pass, saving the gathered_base HBM read.
        prep_sfa(d);
        imp::quantize_fp16_to_nvfp4_cutlass_moe_gather(
            ctx.no.data,
            static_cast<const int32_t*>(routing.sorted_token_ids.data),
            moe_.cutlass3x_packed,
            reinterpret_cast<uint8_t* const*>(moe_.cutlass3x_sfa_ptrs),
            static_cast<const int*>(routing.expert_offsets.data),
            expanded, d, ne, stream);
        bool ok = true;
        if (!non_gated_experts)
            ok = ok && dispatch_device(ly.expert_gate_ids,
                                       da_cache.d_gate_B_ptrs,
                                       da_cache.d_gate_SFB_ptrs,
                                       da_cache.d_gate_alpha,
                                       expert_gate_base, d, eff);
        ok = ok && dispatch_device(ly.expert_up_ids,
                                   da_cache.d_up_B_ptrs,
                                   da_cache.d_up_SFB_ptrs,
                                   da_cache.d_up_alpha,
                                   expert_up_base, d, eff);
        if (ok) {
            // Fused: activation (SwiGLU/GeGLU/ReLU²) + NVFP4 quant
            // for the down-projection input. Saves one HBM
            // round-trip of the swiglu intermediate per layer.
            const FFNActivation act_type =
                non_gated_experts ? FFNActivation::RELU_SQR : cfg.ffn_activation;
            const char* gate_for_fused =
                non_gated_experts ? nullptr : expert_gate_base;
            fused_act_quantize_device(gate_for_fused, expert_up_base, eff,
                                      act_type);
            ok = dispatch_device(ly.expert_down_ids,
                                 da_cache.d_down_B_ptrs,
                                 da_cache.d_down_SFB_ptrs,
                                 da_cache.d_down_alpha,
                                 expert_down_base, eff, d);
        }
        if (ok) {
            device_args_done = true;
            // What the tier DID, not what its preconditions promised: the block
            // above can still fail on dispatch and fall through to the legacy
            // host-args path below.
            obs.device_args_ready = true;
            dispatch_record::set_moe_prefill_tier(MoePrefillPath::DEVICE_ARGS);
            verify_against_moe_routing_model(cfg.arch, dispatch_policy(), obs, MoePrefillPath::DEVICE_ARGS);
        } else {
            IMP_LOG_ERROR(
                "device-args full path failed; falling back to legacy");
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

// Legacy D2H+sync + smallM + non-smallM dispatch path.
moe_host_args_capture_guard(stream);
std::vector<int32_t> h_offsets(ne + 1);
IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                                   static_cast<size_t>(ne + 1) * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream));
// Populate device-resident d_M_per in parallel with the D2H copy.
// Phase 1 of MoE-prefill-graphs lever: foundation for graph-safe
// dispatch (Phase 2+ migrates host M_per[] uses to this buffer).
if (moe_.d_M_per && moe_.d_M_per_count >= ne) {
    imp::compute_M_per_from_offsets_device(
        static_cast<const int32_t*>(routing.expert_offsets.data),
        moe_.d_M_per, ne, stream);
}
cudaStreamSynchronize(stream);

std::vector<int> M_per(ne);
for (int e = 0; e < ne; ++e)
    M_per[e] = h_offsets[e + 1] - h_offsets[e];

char* gathered_base = static_cast<char*>(moe_.gathered.data);
char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

// Optional smallM kernel branch (opt-in moe.nvfp4_smallM): activates when
// max(M_per) <= moe.nvfp4_smallM_threshold (default 64) AND all three
// native NVFP4 MoE pointers are populated ([n_experts,N,K/16] layout).
// Falls through to CUTLASS 3.x on any failure/unavailability.
bool smallM_done = false;
{
    const auto& moe_cfg = dispatch_policy().moe;
    const bool smallM_optin = moe_cfg.nvfp4_smallM && cfg.arch != ModelArch::GPT_OSS;
    if (smallM_optin && imp::gemm_grouped_nvfp4_smallM_available()) {
        const int smallM_threshold = moe_cfg.nvfp4_smallM_threshold;
        int max_M = 0;
        for (int e = 0; e < ne; ++e)
            max_M = std::max(max_M, M_per[e]);
        const bool native_up_ok = (ly.nvfp4_moe_up_ptr != nullptr);
        const bool native_down_ok = (ly.nvfp4_moe_down_ptr != nullptr);
        const bool native_gate_ok =
            non_gated_experts || (ly.nvfp4_moe_gate_ptr != nullptr);
        const bool gate_ne_ok =
            non_gated_experts ||
            (ly.nvfp4_moe_gate_ptr && ly.nvfp4_moe_gate_ptr->n_experts == ne);
        const bool up_ne_ok =
            native_up_ok && ly.nvfp4_moe_up_ptr->n_experts == ne;
        const bool down_ne_ok =
            native_down_ok && ly.nvfp4_moe_down_ptr->n_experts == ne;
        const bool use_smallM = max_M > 0 && max_M <= smallM_threshold &&
                                native_up_ok && native_down_ok &&
                                native_gate_ok && up_ne_ok && down_ne_ok &&
                                gate_ne_ok;
        if (use_smallM) {
            if (layer == 0) {
                IMP_LOG_INFO(
                    "MoE prefill: smallM kernel branch (n=%d, expanded=%d, "
                    "max_M=%d, thr=%d)",
                    n, expanded, max_M, smallM_threshold);
            }

            // Per-expert offsets into the activation scratch (native
            // row-major: K/2 bytes per FP4 row, K/16 per UE4M3-SF row).
            auto compute_offsets =
                [&](int K_in, std::vector<size_t>& packed_offs,
                    std::vector<size_t>& sf_offs) {
                    packed_offs.assign(ne + 1, 0);
                    sf_offs.assign(ne + 1, 0);
                    for (int e = 0; e < ne; ++e) {
                        packed_offs[e + 1] =
                            packed_offs[e] +
                            static_cast<size_t>(M_per[e]) * K_in / 2;
                        sf_offs[e + 1] =
                            sf_offs[e] +
                            static_cast<size_t>(M_per[e]) * K_in / 16;
                    }
                };

            char* act_packed_base =
                static_cast<char*>(moe_.cutlass3x_packed);
            char* act_sf_base = static_cast<char*>(moe_.cutlass3x_sf);

            std::vector<size_t> packed_offs_du, sf_offs_du;
            compute_offsets(d, packed_offs_du, sf_offs_du);

            bool ok = (packed_offs_du[ne] <= moe_.cutlass3x_packed_size) &&
                      (sf_offs_du[ne] <= moe_.cutlass3x_sf_size);
            if (!ok) {
                IMP_LOG_ERROR(
                    "smallM gate/up scratch too small (need %zu/%zu, "
                    "have %zu/%zu); falling back to CUTLASS 3.x",
                    packed_offs_du[ne], sf_offs_du[ne],
                    moe_.cutlass3x_packed_size, moe_.cutlass3x_sf_size);
            }

            std::vector<void*> act_packed_ptrs(ne), act_sf_ptrs(ne);
            // d_act_tscales stays on device — no D2H sync.
            float* d_act_tscales = nullptr;
            if (ok) {
                for (int e = 0; e < ne; ++e) {
                    act_packed_ptrs[e] = act_packed_base + packed_offs_du[e];
                    act_sf_ptrs[e] = act_sf_base + sf_offs_du[e];
                }

                // Allocate a transient device buffer for per-expert
                // activation tensor_scales. Tiny — ne*4 bytes.
                IMP_CUDA_CHECK_LOG(cudaMallocAsync(
                    &d_act_tscales,
                    static_cast<size_t>(ne) * sizeof(float), stream));
                // Quantize gathered FP16 activations native row-major,
                // and have the kernel emit per-expert tensor_scales.
                imp::quantize_fp16_to_nvfp4_moe_native_with_scales(
                    reinterpret_cast<const __half*>(gathered_base),
                    act_packed_ptrs.data(), act_sf_ptrs.data(),
                    d_act_tscales,
                    static_cast<const int*>(routing.expert_offsets.data),
                    expanded, d, ne, stream);
                // No D2H sync — d_act_tscales stays on device.
            }

            // Builds per-expert weight pointer arrays from the native
            // NvFP4MoEQuantResult cache; alpha = act_ts*weight_ts, computed entirely
            // on device (compute_moe_alpha_device). run_proj takes d_act_ts (device
            // float*), not a host vector; weight scales H2D once per projection (~ne*4 bytes), multiplied on
            // device.
            auto run_proj = [&](const NvFP4MoEQuantResult* W,
                                const std::vector<void*>& act_packed,
                                const std::vector<void*>& act_sf,
                                const float* d_act_ts,  // device ptr
                                char* c_base,
                                int K_in, int N_out) -> bool {
                // Upload weight tensor_scales H2D once per projection.
                // W->tensor_scales is already on device if non-null.
                float* d_alpha = nullptr;
                IMP_CUDA_CHECK_LOG(cudaMallocAsync(
                    &d_alpha, static_cast<size_t>(ne) * sizeof(float), stream));
                if (W && W->tensor_scales) {
                    // Compute alpha = act_ts * weight_ts on device.
                    imp::compute_moe_alpha_device(
                        d_act_ts, W->tensor_scales, d_alpha, ne, stream);
                } else {
                    // No weight tensor_scale: alpha = act_ts * 1.0
                    // Copy act_ts into d_alpha directly.
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                        d_alpha, d_act_ts,
                        static_cast<size_t>(ne) * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));
                }
                std::vector<int> active_M_local;
                std::vector<const void*> hA, hSFA, hB, hSFB;
                std::vector<void*> hD;
                // alpha stays on device (no hAlpha): builds a device-indexed view of
                // d_alpha for active experts, since gemm_grouped_nvfp4_smallM accepts a
                // contiguous [n_experts] device array indexed by blockIdx.x, and needs it
                // indexed by active-expert position instead.
                std::vector<float> h_alpha_compact;
                active_M_local.reserve(ne);
                hA.reserve(ne);
                hSFA.reserve(ne);
                hB.reserve(ne);
                hSFB.reserve(ne);
                hD.reserve(ne);
                h_alpha_compact.reserve(ne);
                // Compacting alpha for active experts is only needed when some M_per[e]==0
                // experts are skipped; reads d_alpha back only then, otherwise passes d_alpha directly.
                bool all_active = true;
                for (int e = 0; e < ne; ++e)
                    if (M_per[e] == 0) { all_active = false; break; }

                for (int e = 0; e < ne; ++e) {
                    if (M_per[e] == 0)
                        continue;
                    active_M_local.push_back(M_per[e]);
                    hA.push_back(act_packed[e]);
                    hSFA.push_back(act_sf[e]);
                    hB.push_back(static_cast<char*>(W->packed_data) +
                                 static_cast<size_t>(e) *
                                     W->expert_stride_packed);
                    hSFB.push_back(static_cast<char*>(W->micro_scales) +
                                   static_cast<size_t>(e) *
                                       W->expert_stride_ms);
                    hD.push_back(c_base +
                                 static_cast<size_t>(h_offsets[e]) * N_out *
                                     sizeof(half));
                }
                const int na = static_cast<int>(active_M_local.size());
                if (na == 0) {
                    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_alpha, stream));
                    return true;
                }

                // d_alpha_active: compact device buffer for the na active
                // experts. When all experts are active, d_alpha == d_alpha_active.
                float* d_alpha_active = d_alpha;
                float* d_alpha_compact_dev = nullptr;
                if (!all_active) {
                    // Compacting alpha: D2H the small d_alpha buffer (ne floats), compact,
                    // H2D the compact array. Cheapest option; still eliminates the larger D2H of activation
                    // scales.
                    std::vector<float> h_alpha_full(ne);
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                        h_alpha_full.data(), d_alpha,
                        static_cast<size_t>(ne) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
                    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                    for (int e = 0; e < ne; ++e)
                        if (M_per[e] > 0)
                            h_alpha_compact.push_back(h_alpha_full[e]);
                    IMP_CUDA_CHECK_LOG(cudaMallocAsync(
                        &d_alpha_compact_dev,
                        static_cast<size_t>(na) * sizeof(float), stream));
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                        d_alpha_compact_dev, h_alpha_compact.data(),
                        static_cast<size_t>(na) * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
                    d_alpha_active = d_alpha_compact_dev;
                }

                bool ret = imp::gemm_grouped_nvfp4_smallM(
                    na, active_M_local.data(), N_out, K_in, hA.data(),
                    hSFA.data(), hB.data(), hSFB.data(), hD.data(),
                    d_alpha_active, stream);
                if (d_alpha_compact_dev)
                    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_alpha_compact_dev, stream));
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_alpha, stream));
                return ret;
            };

            bool ok_gate = ok;
            if (ok && !non_gated_experts) {
                ok_gate = run_proj(ly.nvfp4_moe_gate_ptr, act_packed_ptrs,
                                   act_sf_ptrs, d_act_tscales,
                                   expert_gate_base, d, eff);
            }
            bool ok_up = ok;
            if (ok) {
                ok_up = run_proj(ly.nvfp4_moe_up_ptr, act_packed_ptrs,
                                 act_sf_ptrs, d_act_tscales,
                                 expert_up_base, d, eff);
            }

            // Free gate/up activation scales (down will use a fresh d_act_tscales_dn).
            if (d_act_tscales) {
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_act_tscales, stream));
                d_act_tscales = nullptr;
            }

            if (ok && ok_gate && ok_up) {
                apply_expert_activation(
                    moe_.expert_gate.data, moe_.expert_up.data,
                    moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                    compute_dtype_, cfg.ffn_activation, stream);

                // Down projection: re-quantize post-activation buffer
                // (K_in = eff). Reuse staging via stream-ordered
                // overwrite.
                std::vector<size_t> packed_offs_dn, sf_offs_dn;
                compute_offsets(eff, packed_offs_dn, sf_offs_dn);
                bool down_ok =
                    (packed_offs_dn[ne] <= moe_.cutlass3x_packed_size) &&
                    (sf_offs_dn[ne] <= moe_.cutlass3x_sf_size);
                if (!down_ok) {
                    IMP_LOG_ERROR(
                        "smallM down scratch too small (need %zu/%zu, "
                        "have %zu/%zu); falling back to CUTLASS 3.x",
                        packed_offs_dn[ne], sf_offs_dn[ne],
                        moe_.cutlass3x_packed_size, moe_.cutlass3x_sf_size);
                }
                if (down_ok) {
                    for (int e = 0; e < ne; ++e) {
                        act_packed_ptrs[e] =
                            act_packed_base + packed_offs_dn[e];
                        act_sf_ptrs[e] = act_sf_base + sf_offs_dn[e];
                    }
                    char* down_act = non_gated_experts ? expert_up_base
                                                        : expert_swiglu_base;
                    // Re-quantize post-SwiGLU activations; keep scales on device.
                    float* d_act_tscales_dn = nullptr;
                    IMP_CUDA_CHECK_LOG(cudaMallocAsync(
                        &d_act_tscales_dn,
                        static_cast<size_t>(ne) * sizeof(float), stream));
                    imp::quantize_fp16_to_nvfp4_moe_native_with_scales(
                        reinterpret_cast<const __half*>(down_act),
                        act_packed_ptrs.data(), act_sf_ptrs.data(),
                        d_act_tscales_dn,
                        static_cast<const int*>(routing.expert_offsets.data),
                        expanded, eff, ne, stream);
                    // No D2H sync — pass d_act_tscales_dn directly.
                    bool ok_down =
                        run_proj(ly.nvfp4_moe_down_ptr, act_packed_ptrs,
                                 act_sf_ptrs, d_act_tscales_dn,
                                 expert_down_base, eff, d);
                    IMP_CUDA_CHECK_LOG(
                        cudaFreeAsync(d_act_tscales_dn, stream));
                    if (ok_down) {
                        smallM_done = true;
                        // Again the outcome, not the gate: a smallM gate/up or
                        // down dispatch failure falls through to GROUPED below.
                        obs.smallM_available = true;
                        obs.smallM_under_threshold = true;
                        dispatch_record::set_moe_prefill_tier(MoePrefillPath::SMALL_M);
                        verify_against_moe_routing_model(cfg.arch, dispatch_policy(), obs,
                                                         MoePrefillPath::SMALL_M);
                    } else {
                        IMP_LOG_ERROR(
                            "smallM down dispatch failed; falling back to "
                            "CUTLASS 3.x");
                    }
                }
            } else if (ok) {
                IMP_LOG_ERROR(
                    "smallM gate/up dispatch failed; falling back to "
                    "CUTLASS 3.x");
            }
            // Free activation scales if not yet freed (early-exit paths).
            if (d_act_tscales) {
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_act_tscales, stream));
                d_act_tscales = nullptr;
            }
        }
    }
}

if (!smallM_done) {
    dispatch_record::set_moe_prefill_tier(MoePrefillPath::GROUPED);
    obs.grouped_ready = true;
    verify_against_moe_routing_model(cfg.arch, dispatch_policy(), obs, MoePrefillPath::GROUPED);
    if (layer == 0)
        IMP_LOG_INFO("MoE prefill: CUTLASS 3.x NVFP4 grouped (n=%d, expanded=%d)", n, expanded);
}

if (!smallM_done) {

// Active-expert SFA offset table, computed per K_in (differs for d vs
// eff). Gate and up share K_in=d, so one quantize of gathered_base +
// pointer array is reused for both grouped GEMMs; down reuses the staging
// buffer with a fresh quantize for K_in=eff.
auto quantize_once = [&](const char* a_base, int K_in, std::vector<size_t>& sfa_offsets,
                         std::vector<uint8_t*>& h_sfa_bases) -> bool {
    sfa_offsets.assign(ne + 1, 0);
    h_sfa_bases.assign(ne, nullptr);
    size_t total_packed = 0;
    for (int e = 0; e < ne; ++e) {
        total_packed += static_cast<size_t>(M_per[e]) * K_in / 2;
        sfa_offsets[e + 1] = sfa_offsets[e] + cutlass_nvfp4_sf_size(M_per[e], K_in);
    }
    const size_t total_sfa = sfa_offsets[ne];
    if (total_packed > moe_.cutlass3x_packed_size || total_sfa > moe_.cutlass3x_sf_size) {
        IMP_LOG_ERROR(
            "CUTLASS 3.x MoE staging too small: need packed=%zu sf=%zu, have %zu/%zu",
            total_packed, total_sfa, moe_.cutlass3x_packed_size, moe_.cutlass3x_sf_size);
        return false;
    }
    uint8_t* all_sfa = static_cast<uint8_t*>(moe_.cutlass3x_sf);
    for (int e = 0; e < ne; ++e) {
        if (M_per[e] > 0)
            h_sfa_bases[e] = all_sfa + sfa_offsets[e];
    }
    // Upload SFA pointer array to device (~1 KB).
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(moe_.cutlass3x_sfa_ptrs, h_sfa_bases.data(),
                                       static_cast<size_t>(ne) * sizeof(uint8_t*),
                                       cudaMemcpyHostToDevice, stream));
    // Zero active SFA region (SfAtom pads rows to 128).
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(all_sfa, 0, total_sfa, stream));
    // Fused per-expert quantize in one kernel launch.
    quantize_fp16_to_nvfp4_cutlass_moe(
        a_base, moe_.cutlass3x_packed, moe_.cutlass3x_sfa_ptrs,
        routing.expert_offsets.data ? static_cast<const int*>(routing.expert_offsets.data)
                                    : nullptr,
        expanded, K_in, ne, stream);
    return true;
};

// Legacy host-args dispatch, entered only when the default-on device-args
// full path failed its precondition check (workspace buffers not
// populated). Kept as the safety-net fallback path.
auto grouped_gemm = [&](const std::vector<TensorID>& weight_ids, char* c_base, int K_in,
                        int N_out, const std::vector<size_t>& sfa_offsets) {
    char* all_packed = static_cast<char*>(moe_.cutlass3x_packed);
    uint8_t* all_sfa = static_cast<uint8_t*>(moe_.cutlass3x_sf);
    // Active experts only (CUTLASS 3.x wants non-empty groups).
    std::vector<int> active_M;
    std::vector<const void*> hA, hSFA, hB, hSFB;
    std::vector<void*> hD;
    std::vector<float> hAlpha;
    active_M.reserve(ne);
    hA.reserve(ne);
    hSFA.reserve(ne);
    hB.reserve(ne);
    hSFB.reserve(ne);
    hD.reserve(ne);
    hAlpha.reserve(ne);
    for (int e = 0; e < ne; ++e) {
        if (M_per[e] == 0)
            continue;
        const auto& h = registry_.handle(weight_ids[e]);
        active_M.push_back(M_per[e]);
        hA.push_back(all_packed + static_cast<size_t>(h_offsets[e]) * K_in / 2);
        hSFA.push_back(all_sfa + sfa_offsets[e]);
        hB.push_back(h.payload.cutlass_nvfp4.weight);
        hSFB.push_back(h.payload.cutlass_nvfp4.sf);
        hD.push_back(c_base + static_cast<size_t>(h_offsets[e]) * N_out * sizeof(half));
        hAlpha.push_back(h.payload.cutlass_nvfp4.global_scale
                             ? *h.payload.cutlass_nvfp4.global_scale
                             : 1.0f);
    }
    const int na = static_cast<int>(active_M.size());
    if (na == 0)
        return;
    bool ok = gemm_grouped_cutlass_3x_nvfp4(na, active_M.data(), N_out, K_in, hA.data(),
                                            hSFA.data(), hB.data(), hSFB.data(),
                                            hD.data(), hAlpha.data(), stream);
    if (!ok)
        IMP_LOG_ERROR("CUTLASS 3.x grouped dispatch failed (K=%d N=%d ne=%d)", K_in,
                      N_out, na);
};

// Gate+Up share the same input (gathered_base, K_in=d) → quantize once,
// run two grouped GEMMs reusing staging buffers.
std::vector<size_t> sfa_offs;
std::vector<uint8_t*> sfa_bases;
if (quantize_once(gathered_base, d, sfa_offs, sfa_bases)) {
    if (!non_gated_experts)
        grouped_gemm(ly.expert_gate_ids, expert_gate_base, d, eff, sfa_offs);
    grouped_gemm(ly.expert_up_ids, expert_up_base, d, eff, sfa_offs);
}

// gpt-oss (#547): per-expert biases on gate/up BEFORE activation (rows are
// expert-sorted — same seam as the dequant batch path).
const int32_t* gpt_oss_offsets =
    (model_->profile().is_gpt_oss) ? static_cast<const int32_t*>(routing.expert_offsets.data)
                                     : nullptr;
if (gpt_oss_offsets) {
    moe_add_expert_bias_sorted(expert_gate_base, ly.expert_gate_bias.data, gpt_oss_offsets, ne,
                               expanded, eff, stream);
    moe_add_expert_bias_sorted(expert_up_base, ly.expert_up_bias.data, gpt_oss_offsets, ne,
                               expanded, eff, stream);
}

apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                        moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                        compute_dtype_, cfg.ffn_activation, stream);

// Down has a different activation (post-SwiGLU, K_in=eff): re-quantize.
// A fused silu(gate)*up+quantize kernel regressed short-prompt decode
// (~11%, low SM occupancy at small expanded); the separate swiglu kernel
// has better per-element parallelism, kept separate.
char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
if (quantize_once(down_act, eff, sfa_offs, sfa_bases)) {
    grouped_gemm(ly.expert_down_ids, expert_down_base, eff, d, sfa_offs);
    if (gpt_oss_offsets)
        moe_add_expert_bias_sorted(expert_down_base, ly.expert_down_bias.data, gpt_oss_offsets, ne,
                                   expanded, d, stream);
}
}  // !smallM_done
}  // !device_args_done
    return true;
}

}  // namespace imp
