// Decode path for MoE layers whose NVFP4 experts live on host.
//
// The GGUF host-offload path (#1370) works by handing the fused decode kernels
// the LRU cache's per-layer slot pool instead of the model's expert array: the
// kernels read `base + idx * stride`, and the pool is exactly that shape, so
// `idx` becomes a slot index and no kernel changes.
//
// NVFP4 experts did not have that path at all. A host-resident placement was
// loaded and served WRONG - Phase 0 skipped promoting scales onto host weights,
// the generic GEMM then recognised the scale-less packed weight and returned
// without multiplying, and the model answered from whichever experts happened
// to be resident, at exit code 0 (#1403 refused the placement rather than
// serve it). This file is the path that refusal was standing in for.
//
// What is different from the GGUF case, and why it is only different by this
// much: an NVFP4 expert is TWO byte ranges (packed FP4 weights + FP8 E4M3
// micro-scales) rather than one. Both are addressed by the same slot index,
// because the kernels take separate bases and separate strides for them - so a
// slot holding `packed || micro_scales` resolves both at once. The layout
// arithmetic is in nvfp4_expert_offload.h.
//
// The one piece that genuinely does not fall out is the per-tensor scale: the
// kernels read `tensor_scales[idx]` with the SAME index as the weight, so the
// checkpoint's per-EXPERT array cannot be handed to a slot-indexed kernel. The
// cache keeps a per-SLOT mirror, written whenever a slot's occupant changes.

#include "exec/executor.h"
#include "exec/executor_forward_moe_internal.h"
#include "exec/executor_helpers.h"
#include "exec/executor_kernels.h"
#include "exec/nvfp4_expert_offload.h"
#include "compute/activation.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/moe_routing.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace imp {

// Refuse a placement nothing can serve - the gate #1403 installed, moved to
// where the answer is actually known.
//
// At weight-upload time it was not: whether a host-resident NVFP4 layer can be
// served depends on the expert cache, which is sized later (init_weights runs
// before init_kv_cache). The old gate resolved that by refusing every
// host-resident NVFP4 placement outright, which was correct while no path
// existed. Now that one does, re-deriving the cache's sizing at placement time
// would mean a second copy of that arithmetic - and a copy that drifts is
// exactly the failure #1384 and #1403 both were.
//
// So the check runs here instead, after pre-dequant, against the real tensors
// and the real cache. It uses the SAME predicates the dispatch uses, so a
// placement that passes here is one run_moe_decode_fast will route to the slot
// path rather than quietly fall through to a GEMM that cannot multiply it.
void GraphExecutor::verify_host_expert_placement() const {
    const auto& cfg = model_->config();
    if (!cfg.is_nvfp4_prequant)
        return;  // GGUF-class experts have a staging fallback that is slow, not wrong.

    const int top_k = std::max(1, cfg.n_experts_active);
    int host_layers = 0, unservable = 0, first_bad = -1;
    for (int i = 0; i < cfg.n_layers; ++i) {
        const auto& L = model_->layer(i);
        if (L.expert_w_up.empty() || !L.expert_w_up[0].data || L.expert_w_up[0].on_device)
            continue;
        ++host_layers;
        const bool has_gate = (!L.expert_w_gate.empty() && L.expert_w_gate[0].data != nullptr);
        const bool ok = nvfp4_host_experts_servable(L.expert_w_up) &&
                        nvfp4_host_experts_servable(L.expert_w_down) &&
                        (!has_gate || nvfp4_host_experts_servable(L.expert_w_gate)) &&
                        nvfp4_host_pool_ready(expert_cache_, moe_, top_k);
        if (!ok) {
            ++unservable;
            if (first_bad < 0)
                first_bad = i;
        }
    }

    if (unservable > 0) {
        const int need = kExpertProjCount * top_k;
        std::string msg = "NVFP4 experts are host-resident on ";
        msg += std::to_string(unservable);
        msg += " MoE layer(s) (first: layer ";
        msg += std::to_string(first_bad);
        msg += ") that the expert cache cannot serve. Serving them anyway would skip those "
               "experts' GEMMs and answer from the rest, at exit code 0. The slot path needs at "
               "least ";
        msg += std::to_string(need);
        msg += " slots per layer (3 projections x top_k) and has ";
        msg += std::to_string(expert_cache_.slots_per_layer_);
        msg += ". Raise moe.expert_cache_budget_pct, reduce runtime.max_seq_len, pick a smaller "
               "KV dtype, or use a GGUF quantisation of this model.";
        throw std::runtime_error(msg);
    }

    if (host_layers > 0) {
        IMP_LOG_INFO("NVFP4 experts: %d MoE layer(s) host-resident, served from the expert cache "
                     "(%d slots/layer, %.2f MiB per slot)",
                     host_layers, expert_cache_.slots_per_layer_,
                     expert_cache_.slot_size_ / (1024.0 * 1024.0));
    }
}

// Stage one host-resident NVFP4 layer's experts into the device buffer.
//
// The per-expert route issues two H2D per expert per projection, ~768 KiB and
// ~96 KiB - sizes that do not reach PCIe bandwidth. nsys measured 90 759 such
// transfers moving 38 GB for one profiling run, with the host inside those
// calls far longer than the GPU spent transferring. A whole projection at once
// is the same bytes as ONE transfer of ~110 MiB.
//
// That collapse is only possible because the sources are already contiguous:
// `moe.pin_host_experts` lays a projection's experts into one pinned slab back
// to back, and a plain mmap usually does the same. Contiguity is CHECKED here
// rather than assumed - a checkpoint that interleaves its tensors would
// otherwise have its experts silently read from the wrong addresses.
bool GraphExecutor::stage_nvfp4_layer_(int layer, cudaStream_t stream,
                                       StagedProj out[kExpertProjCount]) {
    for (int i = 0; i < kExpertProjCount; ++i)
        out[i] = StagedProj{};
    if (!moe_.layer_stage_buf || moe_.layer_stage_proj_bytes == 0)
        return false;

    const auto& ly = model_->layer(layer);
    const std::vector<Tensor>* projs[kExpertProjCount] = {&ly.expert_w_gate, &ly.expert_w_up,
                                                          &ly.expert_w_down};
    bool any = false;
    for (int p = 0; p < kExpertProjCount; ++p) {
        const std::vector<Tensor>& experts = *projs[p];
        if (!nvfp4_host_experts_servable(experts))
            continue;  // non-gated model, or a projection this path cannot take
        const int ne = static_cast<int>(experts.size());
        if (ne > moe_.layer_stage_experts)
            continue;  // buffer was sized for fewer experts

        const auto layout = nvfp4_slot_layout(experts[0].shape[0], experts[0].shape[1] * 2);
        const size_t pb = layout.packed_bytes, mb = layout.ms_bytes;
        if (static_cast<size_t>(ne) * (pb + mb) > moe_.layer_stage_proj_bytes)
            continue;

        char* base = static_cast<char*>(moe_.layer_stage_buf) +
                     static_cast<size_t>(p) * moe_.layer_stage_proj_bytes;
        char* packed_dst = base;
        char* ms_dst = base + static_cast<size_t>(ne) * pb;

        // Are the host experts laid out back to back? If so this is two
        // memcpys for the whole projection instead of 2*ne.
        bool packed_contig = true, ms_contig = true;
        for (int e = 1; e < ne; ++e) {
            const char* w0 = static_cast<const char*>(experts[0].data);
            const char* s0 = static_cast<const char*>(experts[0].scales);
            if (static_cast<const char*>(experts[e].data) != w0 + static_cast<size_t>(e) * pb)
                packed_contig = false;
            if (static_cast<const char*>(experts[e].scales) != s0 + static_cast<size_t>(e) * mb)
                ms_contig = false;
            if (!packed_contig && !ms_contig)
                break;
        }

        if (packed_contig) {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(packed_dst, experts[0].data,
                                               static_cast<size_t>(ne) * pb,
                                               cudaMemcpyHostToDevice, stream));
        } else {
            for (int e = 0; e < ne; ++e)
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(packed_dst + static_cast<size_t>(e) * pb,
                                                   experts[e].data, pb, cudaMemcpyHostToDevice,
                                                   stream));
        }
        if (ms_contig) {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ms_dst, experts[0].scales,
                                               static_cast<size_t>(ne) * mb,
                                               cudaMemcpyHostToDevice, stream));
        } else {
            for (int e = 0; e < ne; ++e)
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ms_dst + static_cast<size_t>(e) * mb,
                                                   experts[e].scales, mb, cudaMemcpyHostToDevice,
                                                   stream));
        }

        out[p].packed = packed_dst;
        out[p].ms = ms_dst;
        out[p].packed_stride = pb;
        out[p].ms_stride = mb;
        out[p].n_experts = ne;
        any = true;

        // CUTLASS device-args view: the grouped GEMM reads SfAtom-ordered
        // scale factors and per-expert pointer arrays, neither of which the
        // staged bytes carry. Building them here is what lets a host-resident
        // layer take the same prefill path as a resident one instead of the
        // per-expert dequant fallback.
        if (!moe_.layer_stage_sf || moe_.layer_stage_sf_proj_bytes == 0)
            continue;
        const int64_t N = experts[0].shape[0];
        const int64_t K = experts[0].shape[1] * 2;
        const size_t sf_expert = cutlass_nvfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
        if (static_cast<size_t>(ne) * sf_expert > moe_.layer_stage_sf_proj_bytes)
            continue;
        char* sf_base = static_cast<char*>(moe_.layer_stage_sf) +
                        static_cast<size_t>(p) * moe_.layer_stage_sf_proj_bytes;
        convert_nvfp4_moe_scales_to_sfatom(ms_dst, sf_base, ne, static_cast<int>(N),
                                           static_cast<int>(K), stream);

        std::vector<const void*> hb(ne), hsfb(ne);
        std::vector<float> halpha(ne);
        for (int e = 0; e < ne; ++e) {
            hb[e] = packed_dst + static_cast<size_t>(e) * pb;
            hsfb[e] = sf_base + static_cast<size_t>(e) * sf_expert;
            halpha[e] = experts[e].tensor_scale;
        }
        const size_t off = static_cast<size_t>(p) * moe_.layer_stage_experts;
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(moe_.layer_stage_b_ptrs + off, hb.data(),
                                           static_cast<size_t>(ne) * sizeof(const void*),
                                           cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(moe_.layer_stage_sfb_ptrs + off, hsfb.data(),
                                           static_cast<size_t>(ne) * sizeof(const void*),
                                           cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(moe_.layer_stage_alpha + off, halpha.data(),
                                           static_cast<size_t>(ne) * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
        // The pointer arrays are filled from stack vectors, so this call is
        // not graph-capturable - which is fine, graphs are already off under
        // host-resident experts.
        out[p].cutlass_ready = true;
    }
    return any;
}

// Stage a host-resident layer for prefill and report whether the staged copy
// can carry the CUTLASS path.
//
// Staged once per MoE call and carried on ctx, so a fallback to the legacy
// path does not transfer the same bytes again. The weights then live in the
// staging buffer rather than the registry, which is why the CUTLASS entry
// predicate (`covers_ids`) cannot see them: those handles exist only for
// device-resident experts.
bool GraphExecutor::stage_layer_for_prefill_(int layer, cudaStream_t stream, MoeFfnContext& ctx) {
    if (!ctx.staged_done && ctx.n > 1 && moe_.layer_stage_buf)
        ctx.staged_done = stage_nvfp4_layer_(layer, stream, ctx.staged);
    if (!runtime_config().moe.staged_cutlass_prefill || !ctx.staged_done)
        return false;
    const auto ready = [&](ExpertProj p) { return ctx.staged[std::to_underlying(p)].cutlass_ready; };
    return ready(ExpertProj::Up) && ready(ExpertProj::Down) &&
           (ctx.non_gated_experts || ready(ExpertProj::Gate));
}

// Present a staged layer to the CUTLASS grouped prefill as if it were
// device-resident. The staging pass already wrote the per-expert pointer and
// alpha arrays; this only slices them per projection.
//
// Opt-in (moe.staged_cutlass_prefill): the prefill win is large and the decode
// effect that comes with it is real but unexplained - see dispatch_policy.h.
bool GraphExecutor::build_staged_device_args_(
    const MoeFfnContext& ctx, bool non_gated,
    MoEWorkspace::PerLayerNvfp4DeviceArgsCache& out) const {
    if (!runtime_config().moe.staged_cutlass_prefill || !ctx.staged_done ||
        !moe_.layer_stage_b_ptrs || moe_.layer_stage_experts <= 0)
        return false;
    const auto ready = [&](ExpertProj p) { return ctx.staged[std::to_underlying(p)].cutlass_ready; };
    if (!ready(ExpertProj::Up) || !ready(ExpertProj::Down) ||
        (!non_gated && !ready(ExpertProj::Gate)))
        return false;

    const size_t stride = static_cast<size_t>(moe_.layer_stage_experts);
    const auto at = [&](ExpertProj p) { return static_cast<size_t>(std::to_underlying(p)) * stride; };
    out.d_gate_B_ptrs = moe_.layer_stage_b_ptrs + at(ExpertProj::Gate);
    out.d_gate_SFB_ptrs = moe_.layer_stage_sfb_ptrs + at(ExpertProj::Gate);
    out.d_gate_alpha = moe_.layer_stage_alpha + at(ExpertProj::Gate);
    out.d_up_B_ptrs = moe_.layer_stage_b_ptrs + at(ExpertProj::Up);
    out.d_up_SFB_ptrs = moe_.layer_stage_sfb_ptrs + at(ExpertProj::Up);
    out.d_up_alpha = moe_.layer_stage_alpha + at(ExpertProj::Up);
    out.d_down_B_ptrs = moe_.layer_stage_b_ptrs + at(ExpertProj::Down);
    out.d_down_SFB_ptrs = moe_.layer_stage_sfb_ptrs + at(ExpertProj::Down);
    out.d_down_alpha = moe_.layer_stage_alpha + at(ExpertProj::Down);
    out.ready = true;
    return true;
}

namespace {

// Point an NvFP4MoEQuantResult at one layer's slot pool. `ms_off` is where a
// slot's micro-scale block starts; the cache filled the slot with the same
// value, and disagreeing about it is fatal there rather than silent here.
NvFP4MoEQuantResult pool_view(char* layer_pool, size_t slot_size, size_t ms_off, float* slot_scales,
                              int slots_per_layer, int64_t N, int64_t K) {
    NvFP4MoEQuantResult r;
    r.packed_data = layer_pool;
    r.micro_scales = layer_pool + ms_off;
    r.tensor_scales = slot_scales;
    r.n_experts = slots_per_layer;
    r.N = N;
    r.K = K;
    r.expert_stride_packed = slot_size;
    r.expert_stride_ms = slot_size;
    r.borrowed = true;  // every pointer belongs to the cache
    return r;
}

}  // namespace

void GraphExecutor::run_moe_decode_nvfp4_host(int layer, cudaStream_t stream, int d, int eff,
                                              int top_k, const MoeRoutingResult& routing,
                                              const Tensor& no, Tensor& h, const Tensor& r,
                                              bool moe_use_fp32_residual,
                                              bool will_skip_residual_copy, bool& residual_fused,
                                              bool non_gated_experts) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    const int32_t* expert_indices = static_cast<const int32_t*>(routing.expert_indices.data);
    const float* expert_weights = static_cast<const float*>(routing.expert_weights.data);

    half* norm_ptr = static_cast<half*>(no.data);
    half* gate_buf = static_cast<half*>(moe_.expert_gate.data);   // [top_k, eff]
    half* up_buf = static_cast<half*>(moe_.expert_up.data);       // [top_k, eff]
    half* act_buf = static_cast<half*>(moe_.expert_swiglu.data);  // [top_k, eff]
    half* down_buf = static_cast<half*>(moe_.expert_down.data);   // [top_k, d]

    // Establishing residency needs the routing on the host, so this path pays
    // one D2H + sync per layer - the same one the GGUF slot path pays, and the
    // reason CUDA graphs stay disabled under host-resident experts.
    moe_host_args_capture_guard(stream);
    std::vector<int32_t> h_experts(top_k);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_experts.data(), expert_indices,
                                       static_cast<size_t>(top_k) * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    std::vector<int32_t> h_slots(static_cast<size_t>(kExpertProjCount) * top_k, -1);
    size_t ms_off[kExpertProjCount] = {0, 0, 0};

    auto stage = [&](const std::vector<Tensor>& experts, ExpertProj proj) -> bool {
        const int proj_idx = std::to_underlying(proj);
        if (experts.empty() || !experts[0].data)
            return true;  // non-gated model has no gate projection
        const auto layout = nvfp4_slot_layout(experts[0].shape[0], experts[0].shape[1] * 2);
        ms_off[proj_idx] = layout.packed_off();
        const int off = proj_idx * top_k;
        for (int k = 0; k < top_k; ++k) {
            const int e = h_experts[k];
            if (e < 0 || e >= static_cast<int>(experts.size()))
                return false;
            const Tensor& w = experts[e];
            // packed_ptr identifies the projection; expert 0's address is
            // stable and unique per (layer, projection).
            ExpertCacheKey ck{experts[0].data, e};
            void* p = expert_cache_.get_or_load_nvfp4(layer, proj, ck, w.data, layout.packed_bytes,
                                                      w.scales, layout.ms_bytes, layout.packed_off(),
                                                      w.tensor_scale, stream);
            if (!p)
                return false;
            const size_t flat =
                static_cast<size_t>(static_cast<char*>(p) - static_cast<char*>(expert_cache_.pool_)) /
                expert_cache_.slot_size_;
            h_slots[off + k] = static_cast<int32_t>(flat) - layer * expert_cache_.slots_per_layer_;
        }
        return true;
    };

    if (!stage(ly.expert_w_gate, ExpertProj::Gate) || !stage(ly.expert_w_up, ExpertProj::Up) ||
        !stage(ly.expert_w_down, ExpertProj::Down)) {
        // Unreachable by construction: the dispatch predicate checked every
        // precondition, and staging can only fail on an out-of-range expert
        // id. Falling through would hand a HOST pointer to a kernel, so say so
        // instead.
        // IMP_CHECK, not IMP_LOG_FATAL. The comment above says continuing hands a
        // HOST pointer to a device kernel, and IMP_LOG_FATAL only LOGS
        // (logging.h:58) - so it said so and then did it. Abort rather than
        // throw: this is state corruption, not a request that can be failed.
        IMP_CHECK(false,
                  "MoE NVFP4 host decode: experts could not be staged into the LRU pool (layer %d, "
                  "top_k %d, slots/layer %d). The dispatch predicate and this path have diverged.",
                  layer, top_k, expert_cache_.slots_per_layer_);
    }

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(moe_.d_slot_idx, h_slots.data(),
                                       h_slots.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                                       stream));

    char* layer_pool = static_cast<char*>(expert_cache_.pool_) +
                       static_cast<size_t>(layer) * expert_cache_.slots_per_layer_ *
                           expert_cache_.slot_size_;
    float* slot_scales = expert_cache_.layer_slot_scales(layer);
    const size_t slot_size = expert_cache_.slot_size_;
    const int slots_per_layer = expert_cache_.slots_per_layer_;

    const int32_t* gate_idx = moe_.d_slot_idx;
    const int32_t* up_idx = moe_.d_slot_idx + top_k;
    const int32_t* down_idx = moe_.d_slot_idx + 2 * top_k;

    const NvFP4MoEQuantResult up_view =
        pool_view(layer_pool, slot_size, ms_off[std::to_underlying(ExpertProj::Up)], slot_scales,
                  slots_per_layer, eff, d);
    const NvFP4MoEQuantResult down_view =
        pool_view(layer_pool, slot_size, ms_off[std::to_underlying(ExpertProj::Down)], slot_scales,
                  slots_per_layer, d, eff);

    if (!non_gated_experts) {
        const NvFP4MoEQuantResult gate_view =
            pool_view(layer_pool, slot_size, ms_off[std::to_underlying(ExpertProj::Gate)], slot_scales,
                      slots_per_layer, eff, d);
        // gate and up sit in DIFFERENT slots, so the one-index fused gate+up
        // kernel cannot express both. Two decodes still collapse 2*top_k
        // weight launches into 2 - the same trade the GGUF slot path makes.
        gemv_nvfp4_moe_decode(gate_view, gate_idx, norm_ptr, gate_buf, eff, d, /*x_stride=*/0, top_k,
                              stream);
        gemv_nvfp4_moe_decode(up_view, up_idx, norm_ptr, up_buf, eff, d, /*x_stride=*/0, top_k, stream);
        apply_expert_activation(gate_buf, up_buf, act_buf, /*non_gated=*/false, top_k, eff,
                                compute_dtype_, cfg.ffn_activation, stream);
        gemv_nvfp4_moe_decode(down_view, down_idx, act_buf, down_buf, d, eff, /*x_stride=*/eff, top_k,
                              stream);
    } else {
        gemv_nvfp4_moe_decode(up_view, up_idx, norm_ptr, up_buf, eff, d, /*x_stride=*/0, top_k, stream);
        int64_t act_shape[2] = {static_cast<int64_t>(top_k), static_cast<int64_t>(eff)};
        Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
        relu_sqr_inplace(up_t, stream);
        gemv_nvfp4_moe_decode(down_view, down_idx, up_buf, down_buf, d, eff, /*x_stride=*/eff, top_k,
                              stream);
    }

    const bool has_shared_expert = (ly.w_up_shared.data != nullptr);
    const void* res_ptr = (has_shared_expert || moe_use_fp32_residual)
                              ? nullptr
                              : (will_skip_residual_copy ? h.data : r.data);
    moe_weighted_sum_residual(down_buf, expert_weights, res_ptr, h.data, d, top_k, stream);
    if (!has_shared_expert && !moe_use_fp32_residual)
        residual_fused = true;
}

}  // namespace imp
