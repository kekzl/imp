#include "model/model.h"
#include "memory/vram_query.h"
#include "memory/weight_snapshot.h"
#include "memory/weight_cache_file.h"
#include "model/host_expert_pin.h"
#include "model/expert_placement.h"
#include "model/weight_upload_traits.h"
#include "model/layer_host_keep.h"
#include "model/host_embedding.h"
#include "exec/nvfp4_expert_offload.h"
#include "model/gguf_loader.h"
#include "memory/mem_account.h"
#include "quant/dequant_gpu.h"
#include "quant/dequant_awq.h"
#include "core/config/lm_head_mode.h"
#include "core/cuda_raii.h"
#include "core/logging.h"
#include "core/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef __linux__
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#endif

namespace imp {

// Checked GPU allocation prevents oversubscription: an unchecked cudaMalloc on Linux
// silently backs with system RAM (unified memory), causing cuBLASLt INTERNAL_ERROR on the
// resulting pointers. VRAM state is cached, refreshed once per upload pass instead of
// per-tensor, to avoid hundreds of cudaMemGetInfo roundtrips.
static size_t g_cached_free_mem = 0;
static size_t g_total_allocated = 0;
static size_t g_vram_reserve = 0;  // set from Engine's computed reserve

// Suspend-to-RAM (memory/weight_snapshot.h): active upload log + armed warm
// snapshot for the current upload_weights_gpu pass. Installed/cleared by
// StagingGuard alongside g_stager; nullptr everywhere else.
static WeightUploadLog* g_upload_log = nullptr;
static WeightSnapshot* g_warm = nullptr;

// Pinned pages are not reclaimable, so pinning past what /proc/meminfo reports does not
// slow the host down, it kills the process. 6 GiB headroom: the PLE n-gram table is
// mmap'd and only needs page cache it can lose, but the allocator, the CUDA driver and
// the request path still need room. host_mem_available_bytes() is weight_snapshot.h's.
static bool host_pin_ram_available(size_t bytes) {
    constexpr size_t kHeadroom = 6ull << 30;
    const size_t avail = host_mem_available_bytes();
    return avail == 0 || avail > bytes + kHeadroom;
}

// Whether the WHOLE set of host-resident NVFP4 experts fits in pinned memory. Decided once
// per upload (upload_expert_weights) and read by both the weight and the micro-scale slab
// builders: pinning some of them costs that RAM and buys nothing, because the device expert
// cache needs a device view on every host-resident layer or it serves none of them.
static bool g_pin_host_experts_fits = true;

// Allocations run through malloc_async_in_scope: weights Phase 4b may free (GDN projections)
// are uploaded inside a ReleasePoolScope so every freed byte reaches the driver.
static cudaError_t checked_cuda_malloc(void** ptr, size_t size, cudaStream_t stream = nullptr) {
    size_t reserve = g_vram_reserve;
    // Use cached free memory (updated at start of each upload pass)
    if (g_cached_free_mem > 0) {
        if (g_total_allocated + size + reserve > g_cached_free_mem) {
            *ptr = nullptr;
            return cudaErrorMemoryAllocation;
        }
        cudaError_t err = malloc_async_in_scope(ptr, size, stream);
        if (err == cudaSuccess) {
            g_total_allocated += size;
            MemAccount::instance().note_alloc("WEIGHTS", *ptr, size);  // uncharged by Model (#2354)
            if (g_upload_log)
                g_upload_log->note_alloc(*ptr, size);
        }
        return err;
    }
    // Fallback: per-tensor check (used outside upload passes)
    size_t free_mem = 0, total_mem = 0;
    // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
    (void)vram_budget_mem_get_info(&free_mem, &total_mem);
    if (size + reserve > free_mem) {
        *ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    cudaError_t err = malloc_async_in_scope(ptr, size, stream);
    if (err == cudaSuccess && g_upload_log)
        g_upload_log->note_alloc(*ptr, size);
    return err;
}

// Batch-shaped reserve (#2393): Pass 1 runs against the batch-1 reserve; a configured batch
// that fits passes every check it passed before, one that does not is clamped, not aborted.
static size_t pass1_upload_reserve(const UploadBatchFit& fit, size_t full_reserve) {
    return (fit.configured <= 1 || g_cached_free_mem == 0) ? full_reserve
                                                           : upload_reserve_bytes(fit.reserve, 1);
}

// After Pass 1: fit.fitted = the largest batch fitting the measured bytes; returns the reserve.
static size_t fit_batch_after_pass1(UploadBatchFit& fit, size_t full_reserve) {
    const size_t free_b = g_cached_free_mem, used = g_total_allocated;
    const int b = free_b > 0 ? upload_fitting_batch(fit.reserve, fit.configured, free_b, used) : 0;
    fit.fitted = (b >= 1 && b < fit.configured) ? b : fit.configured;
    g_vram_reserve = fit.fitted < fit.configured ? upload_reserve_bytes(fit.reserve, b) : full_reserve;
    if (fit.fitted < fit.configured)
        IMP_LOG_WARN("%s", upload_batch_clamp_message(fit.reserve, fit.configured, b, free_b, used).c_str());
    return g_vram_reserve;
}

// Double-buffered pinned staging for H2D transfers: on WSL2, mmap'd memory cannot be
// pinned (cudaHostRegister fails/corrupts), so cudaMemcpyAsync from mmap'd memory falls
// back to synchronous driver-side staging. Pre-allocates two pinned buffers and pipelines
// CPU memcpy(pinned[i], mmap) against GPU DMA(gpu, pinned[i^1]) for full-bandwidth async DMA.
struct PinnedStager {
    // Ring of N pinned buffers: depth and chunk size come from vram.upload_ring_{depth,
    // chunk_mib} (#1653), keys rather than constants because the optimum is a property of the
    // host's WDDM pinning cost, not of imp. Pinning cost dominates the H2D overlap it exists to
    // buy; smaller rings/chunks win on this host, down to a point where per-chunk event/memcpy
    // overhead starts costing more than the pinning saves.
    static int ring() {
        static const int n = std::clamp(process_diag_upload_ring_depth(), 1, kRingMax);
        return n;
    }
    static size_t chunk_size() {
        static const size_t n = static_cast<size_t>(std::clamp(process_diag_upload_ring_chunk_mib(), 1, 1024))
                                << 20;
        return n;
    }
    static constexpr int kRingMax = 16;
    // T5a: transient load-time staging. PinnedBuffer owns it, so the partial
    // failure path below no longer has to unwind by hand (memory/host_pinned.h).
    PinnedBuffer buf[kRingMax];
    CudaEvent done[kRingMax];
    int idx = 0;

    bool init() {
        for (int i = 0; i < ring(); i++) {
            buf[i] = PinnedBuffer::acquire(cuda_host_pinned_allocator(), chunk_size());
            if (buf[i].empty() || !done[i].create(cudaEventDisableTiming)) {
                destroy();
                return false;
            }
        }
        return true;
    }

    // First failing call ends the copy and is returned.
    cudaError_t copy(void* dst, const void* src, size_t n, cudaStream_t s) {
        for (size_t off = 0; off < n;) {
            size_t chunk = std::min(n - off, chunk_size());
            int b = idx % ring();
            if (cudaError_t e = cudaEventSynchronize(done[b]); e != cudaSuccess)
                return e;
            memcpy(buf[b].data(), static_cast<const char*>(src) + off, chunk);
            if (cudaError_t e = cudaMemcpyAsync(static_cast<char*>(dst) + off, buf[b].data(), chunk,
                                                cudaMemcpyHostToDevice, s);
                e != cudaSuccess)
                return e;
            if (cudaError_t e = cudaEventRecord(done[b], s); e != cudaSuccess)
                return e;
            off += chunk;
            idx++;
        }
        return cudaSuccess;
    }

    void destroy() {
        for (int i = 0; i < ring(); i++) {
            if (done[i]) {
                IMP_CUDA_CHECK_LOG(cudaEventSynchronize(done[i]));
                done[i].reset();
            }
            buf[i].reset();
        }
    }
};

// Active stager for current upload pass (nullptr = use plain cudaMemcpyAsync)
static PinnedStager* g_stager = nullptr;

// H2D copy that routes through pinned staging when available
[[nodiscard]] static cudaError_t h2d_copy(void* dst, const void* src, size_t n, cudaStream_t s) {
    if (g_stager)
        return g_stager->copy(dst, src, n, s);
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s);
}

// Syncs `stream` (the device when null); false + log on failure.
static bool sync_upload(cudaStream_t stream, const char* what) {
    const cudaError_t e = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
    if (e != cudaSuccess)
        IMP_LOG_ERROR("upload_weights_gpu: %s sync failed: %s", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

// Device copy of a host scale tensor. Any allocation goes into `allocs`; nullptr on failure.
static void* scale_to_device(const void* src, size_t bytes, cudaStream_t stream, std::vector<void*>& allocs) {
    void* d_ptr = nullptr;
    if (cudaMallocAsync(&d_ptr, bytes, stream) != cudaSuccess)
        return nullptr;
    allocs.push_back(d_ptr);
    const cudaError_t e = cudaMemcpyAsync(d_ptr, src, bytes, cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess)
        IMP_LOG_ERROR("scale upload: cudaMemcpyAsync %zu bytes: %s", bytes, cudaGetErrorString(e));
    return e == cudaSuccess ? d_ptr : nullptr;
}

// Copies each expert's host weight_scale into `slab`; false = copy e failed, experts >= e stay on host.
static bool upload_scale_slab(void* slab, const std::vector<NvFP4PreQuantWeight*>& grp, size_t e_ms,
                              cudaStream_t stream, int& scale_count) {
    for (size_t e = 0; e < grp.size(); ++e) {
        Tensor& ws = grp[e]->weight_scale;
        void* dst = static_cast<char*>(slab) + e * e_ms;
        if (cudaError_t ce = cudaMemcpyAsync(dst, ws.data, e_ms, cudaMemcpyHostToDevice, stream);
            ce != cudaSuccess) {
            IMP_LOG_ERROR("scale slab: cudaMemcpyAsync expert %zu: %s", e, cudaGetErrorString(ce));
            return false;
        }
        ws.data = dst;        // interior pointer, NOT tracked as a base alloc
        ws.on_device = true;  // the per-tensor upload_scale loop skips it
        scale_count++;
    }
    return true;
}

// WSL2 detection: cudaHostRegister on mmap'd memory can succeed but produce corrupted DMA
// transfers on WSL2 (stale data from GPU reads). Detected at runtime to skip pinning and
// fall back to pageable H2D copies.
static bool is_wsl2() {
#ifdef __linux__
    static int cached = -1;
    if (cached >= 0)
        return cached;
    std::ifstream f("/proc/version");
    if (f) {
        std::string line;
        std::getline(f, line);
        cached = (line.find("microsoft") != std::string::npos ||
                  line.find("Microsoft") != std::string::npos || line.find("WSL") != std::string::npos)
                     ? 1
                     : 0;
    } else {
        cached = 0;
    }
    return cached;
#else
    return false;
#endif
}

using wupload::float_to_fp16;
using wupload::fp16_to_float;

// upload_weight: uploads one weight tensor from host (mmap) to GPU. Q4_0 splits into
// packed_nibbles [N,K/2] + scales [N,K/32] on GPU (dtype->INT4). Q8_0/Q6_K raw_quant=true
// uploads raw bytes (executor dequants on-the-fly before GEMM); raw_quant=false dequants to
// FP16 on host first. F16/BF16 upload direct; F32 converts to FP16 on host. The per-format
// host staging is model/weight_upload_traits.h; CudaUploadDev is its device side.
struct CudaUploadDev {
    cudaStream_t stream;
    std::vector<void*>& gpu_allocs;

    void* alloc(size_t n) {
        void* d = nullptr;
        checked_cuda_malloc(&d, n, stream);
        return d;
    }
    int copy(void* dst, const void* src, size_t n) { return h2d_copy(dst, src, n, stream); }
    void release(void* d) { IMP_CUDA_CHECK_LOG(cudaFreeAsync(d, stream)); }
    void track(void* d) { gpu_allocs.push_back(d); }
    void dequant(const void* raw, void* out, QType qtype, int rows, int cols) {
        dequant_gpu(raw, out, qtype, rows, cols, stream);
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    }
    static bool dequant_supported(QType qtype) { return dequant_gpu_supported(qtype); }
    static const char* err_str(int e) { return cudaGetErrorString(static_cast<cudaError_t>(e)); }
};

// Warm-restore hooks: route the snapshot's device allocs through the same
// budget-checked allocator and staged H2D copy as a cold upload.
static const WarmRestoreOps kWarmOps{&checked_cuda_malloc, &h2d_copy};

// Per-qtype dispatch, split out of upload_weight so the wrapper can bracket it
// with the suspend-to-RAM restore/record hooks.
static bool upload_weight_dispatch_(Tensor& weight, QType qtype, cudaStream_t stream,
                                    std::vector<void*>& gpu_allocs, bool raw_quant, float weight_offset) {
    CudaUploadDev dev{stream, gpu_allocs};
    return wupload::upload_dispatch(weight, qtype, raw_quant, weight_offset, dev);
}

// wname/wlayer: canonical suspend-to-RAM key parts (make_weight_key). Sites
// that pass nullptr stay cold-only — they re-upload normally at resume.
static bool upload_weight(Tensor& weight, QType qtype, [[maybe_unused]] QType compute_dtype, cudaStream_t stream,
                          std::vector<void*>& gpu_allocs, bool raw_quant = true, float weight_offset = 0.0f,
                          const char* wname = nullptr, int wlayer = -1) {
    // weight_offset: added to each FP32 element BEFORE FP16 conversion, applied only on
    // BF16-source paths (qtype==BF16, or F32/NONE with weight.qtype==BF16). F32-source paths
    // leave it unused: GGUF norms already carry the offset from the converter; SafeTensors
    // stores the delta W (actual gamma = 1+W) for Qwen3.5/3.6 block norms.
    if (weight.data == nullptr || weight.on_device)
        return true;
    if (weight.ndim < 1)
        return true;

    int64_t n_elements = weight.numel();
    if (n_elements == 0)
        return true;

    char keybuf[96];
    const char* key = (wname && (g_warm || g_upload_log)) ? make_weight_key(keybuf, wname, wlayer) : nullptr;
    const QType src_qtype = weight.qtype;
    // True source byte count for the raw-from-source heuristic: the row-based
    // formula the raw upload branches use. Tensor::nbytes() rounds per-element
    // for block quants (Q8_0 = 34 bytes / 32 elements) and would never match.
    size_t src_nbytes = weight.nbytes();
    if (weight.ndim >= 2 && weight.shape[weight.ndim - 1] > 0) {
        const int64_t k = weight.shape[weight.ndim - 1];
        src_nbytes = static_cast<size_t>(weight.numel() / k) * qtype_row_bytes(src_qtype, k);
    }

    if (key && g_warm &&
        g_warm->try_restore(key, weight, stream, gpu_allocs, kWarmOps, g_upload_log))
        return true;

    const size_t allocs_before = gpu_allocs.size();
    if (!upload_weight_dispatch_(weight, qtype, stream, gpu_allocs, raw_quant,
                                 weight_offset))
        return false;

    if (key && g_upload_log && gpu_allocs.size() > allocs_before) {
        g_upload_log->record(key, const_cast<const void* const*>(gpu_allocs.data()) + allocs_before,
                             gpu_allocs.size() - allocs_before, weight, src_qtype, n_elements,
                             src_nbytes);
    }
    return true;
}

// Uploads a weight tensor with no associated quant type (norm weights, embeddings);
// detects the dtype from the tensor itself.

static bool upload_unquantized_weight(Tensor& weight, QType qtype, QType compute_dtype, cudaStream_t stream,
                                      std::vector<void*>& gpu_allocs, bool raw_quant = true,
                                      const char* wname = nullptr, int wlayer = -1) {
    return upload_weight(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, 0.0f, wname, wlayer);
}

// Bundles the repeated parameters needed by all upload helpers, passed by reference to
// avoid more than 8 params on every helper call.
struct UploadCtx {
    QType compute_dtype;
    cudaStream_t stream;
    std::vector<void*>& gpu_allocs;
    std::vector<HostRegistration>& host_pinned;
    std::vector<PinnedBuffer>& host_pinned_allocs;
    // Architecture-specific norm-weight offset: Qwen3.5/3.6 SafeTensors stores block-norm
    // gammas as deltas (gamma=1+W) while GGUF bakes the +1 in at conversion. Applied only on
    // BF16-source paths in upload_weight(). 1.0f for QWEN35[_MOE]/QWEN36_MOE/QWEN4_EXP, 0.0f otherwise.
    float arch_norm_offset = 0.0f;
    // gpt-oss MXFP4 MoE experts must stay host-resident through weight upload: the
    // MXFP4->NVFP4 conversion runs at pre_dequant (needs the executor's wcache). True for
    // ModelArch::GPT_OSS so upload_packed_experts skips them instead of uploading raw MXFP4 the
    // executor cannot consume.
    bool mxfp4_experts_deferred = false;
    // NVFP4-prequant SafeTensors: the MoE host-offload path does not support native-NVFP4
    // experts (they'd stay QType::INT8 packed and leak to the generic cuBLAS GEMM, status-15
    // garbage). Forces all experts on-device for these models so Phase-0 promotes them; see
    // decide_expert_layer_placement_.
    bool is_nvfp4_prequant = false;
    // Back-pointer for upload-time in-place device transforms (Gemma-4 fused
    // expert split) to mark the model suspend-unsupported.
    Model* model = nullptr;
};

// UPLOAD_OR_FAIL / UPLOAD_UNQUANT_OR_FAIL: reduces the per-weight boilerplate of
// upload_weight() + error log + early return.
#define UPLOAD_OR_FAIL(tensor, qtype, msg, layer_idx, ctx)                                          \
    do {                                                                                            \
        if (!upload_weight((tensor), (qtype), (ctx).compute_dtype, (ctx).stream, (ctx).gpu_allocs, \
                           true, 0.0f, (msg), (layer_idx))) {                                       \
            IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx));                    \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

#define UPLOAD_OR_FAIL_RAW(tensor, qtype, raw, msg, layer_idx, ctx)                                 \
    do {                                                                                            \
        if (!upload_weight((tensor), (qtype), (ctx).compute_dtype, (ctx).stream, (ctx).gpu_allocs, \
                           (raw), 0.0f, (msg), (layer_idx))) {                                      \
            IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx));                    \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

#define UPLOAD_UNQUANT_OR_FAIL(tensor, msg, layer_idx, ctx)                                          \
    do {                                                                                             \
        if ((tensor).data && !(tensor).on_device) {                                                  \
            if (!upload_unquantized_weight((tensor), QType::NONE, (ctx).compute_dtype, (ctx).stream, \
                                           (ctx).gpu_allocs, true, (msg), (layer_idx))) {            \
                IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx));                 \
                return false;                                                                        \
            }                                                                                        \
        }                                                                                            \
    } while (0)

// ---------------------------------------------------------------------------
// upload_embeddings_and_output: token embedding, output norm, output projection
// ---------------------------------------------------------------------------
static bool upload_embeddings_and_output(Tensor& tok_emb, Tensor& out_norm, Tensor& out_proj,
                                         const UploadCtx& ctx) {
    // Embedding lookup only supports Q8_0/Q6_K natively; other quant types need dequanting to
    // FP16 (raw_quant=false) for the standard FP16 embedding gather. tok_emb.qtype is updated
    // in-place by upload_weight if a host-side dequant occurs.
    const void* tok_emb_host_ptr = tok_emb.data;  // save for weight-tying check below
    const QType tok_emb_orig_qtype = tok_emb.qtype;
    // Untied: the table only feeds embedding_lookup, which reads a mapped host view (#2484).
    const bool tied_source = out_proj.data == tok_emb_host_ptr;
    if (tok_emb.data && !tok_emb.on_device && !tied_source && process_diag_host_token_embedding())
        (void)place_embedding_on_host(tok_emb, ctx.host_pinned_allocs);  // false: VRAM upload below
    if (tok_emb.data && !tok_emb.on_device) {
        const bool emb_raw = (tok_emb.qtype == QType::Q8_0 || tok_emb.qtype == QType::Q6_K);
        if (!upload_unquantized_weight(tok_emb, tok_emb.qtype, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs,
                                       emb_raw, "tok_emb")) {
            IMP_LOG_ERROR("Failed to upload token embedding");
            return false;
        }
    }

    // Output norm (final RMSNorm feeding the LM head) must take ctx.arch_norm_offset like every
    // other norm. It used to skip the offset while attn_norm/ffn_norm/QK-norms/MTP norms all
    // took it, so a Qwen3.5/3.6 SafeTensors checkpoint ran the last scaling before logits with
    // gamma=W instead of gamma=1+W: every layer correct, only the final scale wrong, so the
    // model stayed coherent and merely got much worse.
    if (out_norm.data && !out_norm.on_device) {
        if (!upload_weight(out_norm, out_norm.qtype, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs,
                           /*raw_quant=*/true, ctx.arch_norm_offset, "out_norm")) {
            IMP_LOG_ERROR("Failed to upload output norm");
            return false;
        }
    }

    // Upload output projection — raw Q6_K/Q8_0 for dp4a GEMV (saves ~60% VRAM).
    // Falls back to FP16 dequant for unsupported quant types.
    // For weight-tied models (out_proj == tok_emb), share the GPU data directly.
    if (out_proj.data && !out_proj.on_device) {
        // Weight tying: share GPU data only if both point to the same host tensor
        // (i.e. GGUF had no output.weight and the loader aliased out_proj = tok_emb).
        // Compare against the ORIGINAL pre-upload qtype since out_proj still holds it.
        const bool actually_tied = (out_proj.data == tok_emb_host_ptr &&
                                    out_proj.qtype == tok_emb_orig_qtype);
        if (actually_tied && tok_emb.on_device) {
            out_proj = tok_emb;
            IMP_LOG_INFO("Output projection shares GPU data with token embedding (weight tying)");
        } else {
            // Own release-on-free pool: gemm.nvfp4_lm_head=fp8 frees this source after load.
            ReleasePoolScope head_scope(ReleasePool::LmHead);
            const bool raw_ok = (out_proj.qtype == QType::Q6_K || out_proj.qtype == QType::Q8_0 ||
                                 out_proj.qtype == QType::Q4_0);
            if (!upload_unquantized_weight(out_proj, out_proj.qtype, ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs,
                                           /*raw_quant=*/raw_ok, "out_proj")) {
                IMP_LOG_ERROR("Failed to upload output projection");
                return false;
            }
        }
    }

    return true;
}

// Per-row FP8 checkpoint head (#2479): codes went up raw above, the F32 row scales follow as-is.
static bool upload_lm_head_row_scales(const Tensor& head, const Tensor& norm, Tensor& sc,
                                      const UploadCtx& ctx) {
    if (!sc.data || sc.on_device)
        return true;
    void* d = wupload::lm_head_row_scales_ok(head, norm, sc)
                  ? scale_to_device(sc.data, sc.nbytes(), ctx.stream, ctx.gpu_allocs)
                  : nullptr;
    if (!d)
        IMP_LOG_ERROR("%s: refused or upload failed (head %s ndim %d, scales %s, norm %s)",
                      kLmHeadRowScaleTensor, qtype_name(head.qtype), head.ndim, qtype_name(sc.qtype),
                      qtype_name(norm.qtype));
    sc.data = d ? d : sc.data;
    sc.on_device = d != nullptr;
    return d != nullptr;
}

// Qwen4Exp MTP experts: F8_E4M3 bytes go up unchanged, one allocation per expert (gate|up|down,
// 4.7 MiB) so no multi-GiB request hits the async pool; BF16 128x128 block scales become FP32.
// The forward addresses experts through the device pointer tables (gemv_fp8_block_moe.h).
static bool upload_mtp_fp8_experts(MtpHead& head, const UploadCtx& ctx) {
    const int ne = static_cast<int>(head.experts_fp8.size());
    if (ne == 0)
        return true;
    const MtpFp8Expert& x0 = head.experts_fp8[0];
    const int64_t eff = x0.gate_proj.shape[0];
    const int64_t hid = x0.gate_proj.shape[1];
    const size_t gu_sc = static_cast<size_t>(eff / 128) * (hid / 128);  // scales per [eff, hid] matrix
    auto shape_ok = [&](const Tensor& w, int64_t r, int64_t c, const Tensor& s) {
        return w.data && w.qtype == QType::FP8_E4M3 && w.ndim == 2 && w.shape[0] == r && w.shape[1] == c &&
               s.data && s.qtype == QType::BF16 && s.ndim == 2 && s.shape[0] == r / 128 && s.shape[1] == c / 128;
    };
    if (eff % 128 != 0 || hid % 128 != 0) {
        IMP_LOG_ERROR("MTP FP8 experts: [%lld, %lld] is not a multiple of the 128x128 scale block",
                      (long long)eff, (long long)hid);
        return false;
    }
    const size_t mat = static_cast<size_t>(eff) * hid;  // bytes per E4M3 matrix
    std::vector<const uint8_t*> gu_tab(static_cast<size_t>(ne) * 2), dn_tab(static_cast<size_t>(ne));
    std::vector<float> gu_s(static_cast<size_t>(ne) * 2 * gu_sc), dn_s(static_cast<size_t>(ne) * gu_sc);
    auto widen = [](const Tensor& s, float* dst, size_t n) {
        const uint16_t* p = static_cast<const uint16_t*>(s.data);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t bits = static_cast<uint32_t>(p[i]) << 16;  // BF16 = top half of an FP32
            std::memcpy(&dst[i], &bits, sizeof(float));
        }
    };
    for (int e = 0; e < ne; ++e) {
        const MtpFp8Expert& x = head.experts_fp8[static_cast<size_t>(e)];
        if (!shape_ok(x.gate_proj, eff, hid, x.gate_scale_inv) || !shape_ok(x.up_proj, eff, hid, x.up_scale_inv) ||
            !shape_ok(x.down_proj, hid, eff, x.down_scale_inv)) {
            IMP_LOG_ERROR("MTP FP8 expert %d: dtype or shape differs from expert 0", e);
            return false;
        }
        void* d = nullptr;
        checked_cuda_malloc(&d, 3 * mat, ctx.stream);
        if (d == nullptr) {
            IMP_LOG_ERROR("MTP FP8 expert %d: device allocation failed (%zu MiB)", e, 3 * mat >> 20);
            return false;
        }
        ctx.gpu_allocs.push_back(d);
        uint8_t* b = static_cast<uint8_t*>(d);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(b, x.gate_proj.data, mat, cudaMemcpyHostToDevice, ctx.stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(b + mat, x.up_proj.data, mat, cudaMemcpyHostToDevice, ctx.stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(b + 2 * mat, x.down_proj.data, mat, cudaMemcpyHostToDevice, ctx.stream));
        gu_tab[size_t{2} * e] = b;
        gu_tab[size_t{2} * e + 1] = b + mat;
        dn_tab[e] = b + 2 * mat;
        widen(x.gate_scale_inv, &gu_s[(2 * static_cast<size_t>(e)) * gu_sc], gu_sc);
        widen(x.up_scale_inv, &gu_s[(2 * static_cast<size_t>(e) + 1) * gu_sc], gu_sc);
        widen(x.down_scale_inv, &dn_s[static_cast<size_t>(e) * gu_sc], gu_sc);
    }
    // Tables and scales: four small uploads, synchronous so the host vectors may die here.
    auto put = [&](const void* src, size_t bytes, const void** out) -> bool {
        void* d = nullptr;
        checked_cuda_malloc(&d, bytes, ctx.stream);
        if (d == nullptr)
            return false;
        ctx.gpu_allocs.push_back(d);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d, src, bytes, cudaMemcpyHostToDevice, ctx.stream));
        *out = d;
        return true;
    };
    const void *gt = nullptr, *dt = nullptr, *gs = nullptr, *ds = nullptr;
    const bool ok = put(static_cast<const void*>(gu_tab.data()), gu_tab.size() * sizeof(void*), &gt) &&
                    put(static_cast<const void*>(dn_tab.data()), dn_tab.size() * sizeof(void*), &dt) &&
                    put(gu_s.data(), gu_s.size() * sizeof(float), &gs) &&
                    put(dn_s.data(), dn_s.size() * sizeof(float), &ds);
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(ctx.stream));
    if (!ok)
        return false;
    head.fp8_gate_up_tab = static_cast<const uint8_t* const*>(gt);
    head.fp8_down_tab = static_cast<const uint8_t* const*>(dt);
    head.fp8_gate_up_scales = static_cast<const float*>(gs);
    head.fp8_down_scales = static_cast<const float*>(ds);
    IMP_LOG_INFO("MTP FP8 experts: %d x (gate, up, down) [%lld x %lld] E4M3 on device, %.1f MiB", ne,
                 (long long)eff, (long long)hid, 3.0 * mat * ne / (1024.0 * 1024.0));
    return true;
}

// upload_mtp_weights: BF16->FP16 upload of the MTP head. Walks all 19 named MtpHead
// tensors; MoE expert tensors (3D [n_experts,...]) upload raw as 3D FP16, per-expert
// slicing is the forward kernel's concern. CRITICAL: Qwen3.5/3.6 SafeTensors RMSNorm gammas
// are deltas (actual gamma=1+W); without ctx.arch_norm_offset on norm tensors the MTP norms
// run at scale~0, zeroing output and locking the LM-head argmax to a noise token.

static bool upload_mtp_weights(MtpHead& head, const UploadCtx& ctx) {
    if (!head.loaded) return true;  // nothing to upload

    // Norm uploader: applies arch_norm_offset for Qwen3.5/3.6's `gamma = 1 + W`
    // convention. Non-norm tensors use the no-offset path.
    auto up_norm = [&](Tensor& t, const char* name) -> bool {
        if (t.data == nullptr || t.on_device)
            return true;
        if (!upload_weight(t, t.qtype, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs,
                           /*raw_quant=*/true, ctx.arch_norm_offset, name, kMtpKeyLayer)) {
            IMP_LOG_ERROR("Failed to upload MTP norm: %s", name);
            return false;
        }
        return true;
    };
    auto up = [&](Tensor& t, const char* name) -> bool {
        if (t.data == nullptr || t.on_device)
            return true;
        if (!upload_unquantized_weight(t, t.qtype, ctx.compute_dtype,
                                       ctx.stream, ctx.gpu_allocs, true, name, kMtpKeyLayer)) {
            IMP_LOG_ERROR("Failed to upload MTP tensor: %s", name);
            return false;
        }
        return true;
    };

    bool ok = true;
    // Norm weights — REQUIRE arch_norm_offset for Qwen3.5/3.6
    ok &= up_norm(head.pre_fc_norm_embedding,    "pre_fc_norm_embedding");
    ok &= up_norm(head.pre_fc_norm_hidden,       "pre_fc_norm_hidden");
    ok &= up_norm(head.input_layernorm,          "input_layernorm");
    ok &= up_norm(head.post_attention_layernorm, "post_attention_layernorm");
    ok &= up_norm(head.q_norm,                   "q_norm");
    ok &= up_norm(head.k_norm,                   "k_norm");
    ok &= up_norm(head.final_norm,               "final_norm");
    // Projection / MoE weights — no offset
    ok &= up(head.fc,                        "fc");
    ok &= up(head.q_proj,                   "q_proj");
    ok &= up(head.k_proj,                   "k_proj");
    ok &= up(head.v_proj,                   "v_proj");
    ok &= up(head.o_proj,                   "o_proj");
    ok &= up(head.router,                   "router");
    ok &= up(head.experts_gate_up_packed,   "experts_gate_up_packed");
    ok &= up(head.experts_down_packed,      "experts_down_packed");
    ok &= up(head.shared_expert_gate_proj,  "shared_expert_gate_proj");
    ok &= up(head.shared_expert_up_proj,    "shared_expert_up_proj");
    ok &= up(head.shared_expert_down_proj,  "shared_expert_down_proj");
    ok &= up(head.shared_expert_gate,       "shared_expert_gate");
    // Qwen4Exp layout: split fc, hyper-connection modules (norm WITHOUT offset: hc_grouped_rmsnorm
    // applies 1 + W itself), FP8 experts. The QSA indexer stays on the host (dense draft attention).
    ok &= up(head.fc_embedding, "fc_embedding");
    ok &= up(head.fc_hidden, "fc_hidden");
    for (MtpHyperConnection* hc : {&head.attn_hc, &head.mlp_hc, &head.final_mixer}) {
        ok &= up(hc->norm, "hc_norm");
        ok &= up(hc->mix_down, "hc_mix_down");
        ok &= up(hc->mix_up, "hc_mix_up");
        ok &= up(hc->block_inject, "hc_block_inject");
    }
    if (ok && head.layout == MtpLayout::Qwen4Exp)
        ok &= upload_mtp_fp8_experts(head, ctx);
    // Nemotron layout: per-expert 2-D weights instead of the packed pair above.
    // Left on the host these would be read as device pointers by the draft
    // forward, so they upload here with everything else.
    ok &= up(head.router_score_bias, "router_score_bias");
    for (size_t e = 0; e < head.experts_up.size() && ok; ++e)
        ok &= up(head.experts_up[e], "mtp_expert_up");
    for (size_t e = 0; e < head.experts_down.size() && ok; ++e)
        ok &= up(head.experts_down[e], "mtp_expert_down");

    // Restacks per-expert weights into one contiguous slab each so the decode GEMV can address
    // an expert by device-side id; without this the draft has to copy routing D2H and loop on
    // host, which is what makes the draft path uncapturable. Per-expert Tensors are repointed
    // INTO the slab; the originals are NOT freed (stay in gpu_allocs until teardown), so both
    // copies are resident for the process lifetime - mtp_upload_peak_bytes counts the slabs on
    // top of the head for exactly this reason.
    if (ok && !head.experts_up.empty() && head.experts_up[0].data != nullptr) {
        auto stack = [&](std::vector<Tensor>& parts, Tensor& out, const char* what) -> bool {
            const int64_t ne = static_cast<int64_t>(parts.size());
            const int64_t rows = parts[0].shape[0];
            const int64_t cols = parts[0].shape[1];
            const size_t per = static_cast<size_t>(rows) * cols * sizeof(half);
            void* slab = nullptr;
            checked_cuda_malloc(&slab, per * static_cast<size_t>(ne), ctx.stream);
            if (!slab) {
                IMP_LOG_ERROR("MTP %s: slab allocation failed (%zu MiB)", what,
                              per * static_cast<size_t>(ne) / (size_t{1024} * 1024));
                return false;
            }
            ctx.gpu_allocs.push_back(slab);
            for (int64_t e = 0; e < ne; ++e) {
                // A shape mismatch here would silently stride into the wrong
                // expert, so it fails loud instead.
                if (parts[e].shape[0] != rows || parts[e].shape[1] != cols || !parts[e].data) {
                    IMP_LOG_ERROR("MTP %s: expert %lld has shape [%lld,%lld], expected [%lld,%lld]", what,
                                  (long long)e, (long long)parts[e].shape[0], (long long)parts[e].shape[1],
                                  (long long)rows, (long long)cols);
                    return false;
                }
                char* dst = static_cast<char*>(slab) + static_cast<size_t>(e) * per;
                IMP_CUDA_CHECK_LOG(
                    cudaMemcpyAsync(dst, parts[e].data, per, cudaMemcpyDeviceToDevice, ctx.stream));
            }
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(ctx.stream));
            int64_t shape[4] = {ne, rows, cols, 0};
            out = Tensor(slab, QType::F16, 3, shape, /*on_device=*/true);
            // Repoint the per-expert views into the slab; the standalone
            // allocations stay owned by gpu_allocs and are freed at teardown.
            for (int64_t e = 0; e < ne; ++e)
                parts[e].data = static_cast<char*>(slab) + static_cast<size_t>(e) * per;
            return true;
        };
        ok &= stack(head.experts_up, head.experts_up_stacked, "experts_up");
        ok &= stack(head.experts_down, head.experts_down_stacked, "experts_down");
        if (ok)
            IMP_LOG_INFO("MTP experts restacked contiguously for device-side decode dispatch");
    }
    return ok;
}

bool upload_mtp_head(MtpHead& head, QType compute_dtype, float arch_norm_offset, cudaStream_t stream,
                     std::vector<void*>& gpu_allocs) {
    std::vector<HostRegistration> pinned;
    std::vector<PinnedBuffer> pinned_allocs;
    const UploadCtx ctx{compute_dtype, stream, gpu_allocs, pinned, pinned_allocs, arch_norm_offset};
    return upload_mtp_weights(head, ctx);
}

// GPTQ staging copy status: false + ERROR naming the buffer.
static bool gptq_h2d_ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess)
        return true;
    IMP_LOG_ERROR("GPTQ: H2D copy of %s failed: %s", what, cudaGetErrorString(e));
    return false;
}

// upload_gptq_weight: dequantizes a GPTQ- or AWQ-packed (awq_gemm) weight to FP16 on GPU. Uploads
// qweight/qzeros/scales/g_idx to temporary GPU buffers, runs the dequant kernel, frees the
// temporaries, and sets the output tensor to the resulting FP16 GPU weight.
static bool upload_gptq_weight(const TransformerLayer::GPTQWeight& gptq, Tensor& output, cudaStream_t stream,
                               std::vector<void*>& gpu_allocs) {
    if (!gptq.qweight.data || !gptq.scales.data)
        return false;
    if (gptq.bits != 4 || gptq.zero_offset < 0 || !gptq.qzeros.data) {
        IMP_LOG_ERROR("GPTQ: projection not validated at load (bits=%d zero_offset=%d, qzeros required)",
                      gptq.bits, gptq.zero_offset);
        return false;
    }

    // qweight: GPTQ [K/8, N], AWQ GEMM [K, N/8] (shapes checked at load); 8 nibbles per INT32.
    const int64_t d0 = gptq.qweight.shape[0], d1 = gptq.qweight.shape[1];
    const int K = static_cast<int>(gptq.awq_gemm ? d0 : d0 * 8);
    const int N = static_cast<int>(gptq.awq_gemm ? d1 * 8 : d1);

    // 1. Upload qweight to GPU
    size_t qw_bytes = static_cast<size_t>(d0) * d1 * sizeof(int32_t);
    int32_t* d_qweight = nullptr;
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_qweight), qw_bytes, stream) != cudaSuccess || !d_qweight) {
        IMP_LOG_ERROR("GPTQ: failed to allocate qweight (%zu bytes)", qw_bytes);
        return false;
    }
    // Staging status (copies + g_idx alloc); checked at step 5 so one path frees every temporary.
    bool staged_ok = gptq_h2d_ok(h2d_copy(d_qweight, gptq.qweight.data, qw_bytes, stream), "qweight");

    // 2. Upload qzeros to GPU (required, checked at load)
    int32_t* d_qzeros = nullptr;
    size_t qz_bytes = static_cast<size_t>(gptq.qzeros.shape[0]) * gptq.qzeros.shape[1] * sizeof(int32_t);
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_qzeros), qz_bytes, stream) != cudaSuccess ||
        !d_qzeros) {
        IMP_LOG_ERROR("GPTQ: failed to allocate qzeros");
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qweight, stream));
        return false;
    }
    staged_ok &= gptq_h2d_ok(h2d_copy(d_qzeros, gptq.qzeros.data, qz_bytes, stream), "qzeros");

    // 3. Upload scales to GPU
    size_t sc_bytes = static_cast<size_t>(gptq.scales.shape[0]) * gptq.scales.shape[1] * sizeof(half);
    half* d_scales = nullptr;
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_scales), sc_bytes, stream) != cudaSuccess || !d_scales) {
        IMP_LOG_ERROR("GPTQ: failed to allocate scales");
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qweight, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qzeros, stream));
        return false;
    }
    staged_ok &= gptq_h2d_ok(h2d_copy(d_scales, gptq.scales.data, sc_bytes, stream), "scales");

    // 4. Upload g_idx to GPU (optional, for desc_act reordering)
    int32_t* d_g_idx = nullptr;
    if (gptq.g_idx.data) {
        size_t gi_bytes = static_cast<size_t>(K) * sizeof(int32_t);
        if (checked_cuda_malloc(reinterpret_cast<void**>(&d_g_idx), gi_bytes, stream) != cudaSuccess || !d_g_idx) {
            // #2253: no sequential-group fallback; fails through the step-5 cleanup below.
            IMP_LOG_ERROR("GPTQ: failed to allocate g_idx (%zu bytes)", gi_bytes);
            staged_ok = false;
        } else {
            staged_ok &= gptq_h2d_ok(h2d_copy(d_g_idx, gptq.g_idx.data, gi_bytes, stream), "g_idx");
        }
    }
    // desc_act=true without g_idx is refused at load (gptq_refuses, #2253).

    // 5. Allocate FP16 output [N, K]
    size_t out_bytes = static_cast<size_t>(N) * K * sizeof(half);
    half* d_out = nullptr;
    if (!staged_ok ||
        checked_cuda_malloc(reinterpret_cast<void**>(&d_out), out_bytes, stream) != cudaSuccess || !d_out) {
        if (staged_ok)
            IMP_LOG_ERROR("GPTQ: failed to allocate output (%zu bytes)", out_bytes);
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qweight, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qzeros, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_scales, stream));
        if (d_g_idx)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_g_idx, stream));
        return false;
    }

    // 6. Run dequantization kernel
    dequant_packed4(gptq.awq_gemm, d_out, d_qweight, d_qzeros, d_scales, d_g_idx, N, K, gptq.group_size,
                    static_cast<gptq::ZeroFormat>(gptq.zero_offset), stream);

    // 7. Sync and free temporary GPU buffers
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qweight, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qzeros, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_scales, stream));
    if (d_g_idx)
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_g_idx, stream));

    // 8. Set output tensor
    int64_t out_shape[4] = {N, K, 0, 0};
    output = Tensor(d_out, QType::F16, 2, out_shape, true);
    gpu_allocs.push_back(d_out);

    return true;
}

// ---------------------------------------------------------------------------
// upload_layer_attention_weights: wq/wk/wv/wo + norms + biases for one layer
// ---------------------------------------------------------------------------
// MLA (DeepSeek-V2/V3, GLM) RMSNorms on the latent after kv_a_proj and q_a_proj: uploaded with NO
// offset (DeepSeek stores plain gamma, not the Qwen3.5/3.6 1+W delta). No-op for non-MLA layers.
static bool upload_mla_latent_norms(TransformerLayer& L, int i, const UploadCtx& ctx) {
    for (auto [norm, label] : {std::pair<Tensor*, const char*>{&L.kv_a_layernorm, "kv_a_layernorm"},
                               {&L.q_a_layernorm, "q_a_layernorm"}}) {
        if (norm->data && !norm->on_device &&
            !upload_weight(*norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true, 0.0f,
                           label, i)) {
            IMP_LOG_ERROR("Failed to upload %s for layer %d", label, i);
            return false;
        }
    }
    return true;
}

static bool upload_layer_attention_weights(TransformerLayer& L, int i, const UploadCtx& ctx) {
    // Attention weights — try regular upload first, fall back to GPTQ dequant
    UPLOAD_OR_FAIL(L.wq, L.wq.qtype, "wq", i, ctx);
    UPLOAD_OR_FAIL(L.wk, L.wk.qtype, "wk", i, ctx);
    UPLOAD_OR_FAIL(L.wv, L.wv.qtype, "wv", i, ctx);
    UPLOAD_OR_FAIL(L.wo, L.wo.qtype, "wo", i, ctx);
    // MLA (DeepSeek-V2/V3) latent-attention projections. UPLOAD_OR_FAIL no-ops when
    // tensor.data==nullptr (non-MLA layers) or already on device, so safe to call
    // unconditionally for every architecture. kv_a_layernorm uploads separately (QK-norm section).
    UPLOAD_OR_FAIL(L.kv_a_proj, L.kv_a_proj.qtype, "kv_a_proj", i, ctx);
    UPLOAD_OR_FAIL(L.kv_b_proj, L.kv_b_proj.qtype, "kv_b_proj", i, ctx);
    UPLOAD_OR_FAIL(L.q_a_proj, L.q_a_proj.qtype, "q_a_proj", i, ctx);

    // GPTQ fallback: if regular weight is missing but GPTQ tensors are present
    struct {
        Tensor& w;
        TransformerLayer::GPTQWeight& gptq;
        const char* name;
    } attn_gptq[] = {
        {L.wq, L.gptq_q, "q_proj"},
        {L.wk, L.gptq_k, "k_proj"},
        {L.wv, L.gptq_v, "v_proj"},
        {L.wo, L.gptq_o, "o_proj"},
    };
    for (auto& [w, gptq, name] : attn_gptq) {
        if (!w.on_device && gptq.qweight.data) {
            if (!upload_gptq_weight(gptq, w, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to dequant GPTQ %s for layer %d", name, i);
                return false;
            }
            IMP_LOG_DEBUG("GPTQ dequant %s layer %d -> [%lld, %lld] FP16", name, i, (long long)w.shape[0],
                          (long long)w.shape[1]);
        }
    }

    // Attention norm (typically F32/F16, unquantized). For Qwen3.5/3.6 SafeTensors the BF16
    // weight stores W (delta from 1.0); ctx.arch_norm_offset bakes the +1 in during BF16->FP16.
    // GGUF F32 norms already carry the offset and are unaffected.
    if (L.attn_norm.data && !L.attn_norm.on_device) {
        if (!upload_weight(L.attn_norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true,
                           ctx.arch_norm_offset, "attn_norm", i)) {
            IMP_LOG_ERROR("Failed to upload attn_norm for layer %d", i);
            return false;
        }
    }

    // QK-norm weights (Qwen3-style per-head RMSNorm, F32 [head_dim]).
    // Qwen3.5/3.6 also use the `1 + W` convention here.
    if (L.attn_q_norm.data && !L.attn_q_norm.on_device) {
        if (!upload_weight(L.attn_q_norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true,
                           ctx.arch_norm_offset, "attn_q_norm", i)) {
            IMP_LOG_ERROR("Failed to upload attn_q_norm for layer %d", i);
            return false;
        }
    }
    if (L.attn_k_norm.data && !L.attn_k_norm.on_device) {
        if (!upload_weight(L.attn_k_norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true,
                           ctx.arch_norm_offset, "attn_k_norm", i)) {
            IMP_LOG_ERROR("Failed to upload attn_k_norm for layer %d", i);
            return false;
        }
    }
    if (!upload_mla_latent_norms(L, i, ctx))
        return false;
    // Qwen4Exp gated residual: eight BF16 tensors per layer, uploaded with NO offset. hc_norm is
    // the (1 + W) delta like the other Qwen norms, but hc_grouped_rmsnorm applies the +1 itself.
    {
        struct HcSlot {
            Tensor* t;
            const char* name;
        };
        const HcSlot hc_slots[] = {
            {&L.hc_attn_norm, "hc_attn_norm"},     {&L.hc_attn_down, "hc_attn_down"},
            {&L.hc_attn_up, "hc_attn_up"},         {&L.hc_attn_inject, "hc_attn_inject"},
            {&L.hc_mlp_norm, "hc_mlp_norm"},       {&L.hc_mlp_down, "hc_mlp_down"},
            {&L.hc_mlp_up, "hc_mlp_up"},           {&L.hc_mlp_inject, "hc_mlp_inject"},
        };
        for (const auto& s : hc_slots) {
            if (s.t->data && !s.t->on_device) {
                if (!upload_weight(*s.t, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true, 0.0f,
                                   s.name, i)) {
                    IMP_LOG_ERROR("Failed to upload %s for layer %d", s.name, i);
                    return false;
                }
            }
        }
        // PLE (one layer) and QSA indexer (attention layers): BF16 tensors go up, no offset (the
        // norms apply 1 + W in the kernel). The I64 hash buffers and the F8 table scale stay on the host.
        const HcSlot ple_slots[] = {
            {&L.ple_key_proj, "ple_key_proj"},   {&L.ple_value_proj, "ple_value_proj"},
            {&L.ple_conv1d, "ple_conv1d"},       {&L.ple_norm_key, "ple_norm_key"},
            {&L.ple_norm_query, "ple_norm_query"}, {&L.ple_norm_conv, "ple_norm_conv"},
            {&L.qsa_index_qk, "qsa_index_qk"},     {&L.qsa_index_q_norm, "qsa_index_q_norm"},
            {&L.qsa_index_k_norm, "qsa_index_k_norm"},
        };
        for (const auto& s : ple_slots) {
            if (s.t->data && !s.t->on_device) {
                if (!upload_weight(*s.t, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true, 0.0f,
                                   s.name, i)) {
                    IMP_LOG_ERROR("Failed to upload %s for layer %d", s.name, i);
                    return false;
                }
            }
        }
    }

    // Attention biases (Qwen2-style Q/K/V biases, F32)
    struct NamedTensor {
        Tensor* t;
        const char* name;
    };
    const NamedTensor biases[] = {
        {&L.q_bias, "q_bias"},
        {&L.k_bias, "k_bias"},
        {&L.v_bias, "v_bias"},
        {&L.o_bias, "o_bias"},
        {&L.attn_sinks, "attn_sinks"},
        {&L.router_bias, "router_bias"},
        {&L.expert_gate_bias, "expert_gate_bias"},
        {&L.expert_up_bias, "expert_up_bias"},
        {&L.expert_down_bias, "expert_down_bias"},
    };
    for (const auto& [bias, name] : biases) {
        if (bias->data && !bias->on_device) {
            if (!upload_unquantized_weight(*bias, QType::NONE, ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs, true, name, i)) {
                IMP_LOG_ERROR("Failed to upload attention bias for layer %d", i);
                return false;
            }
        }
    }

    // Post-layer norms (Gemma-3/4)
    const NamedTensor post_norms[] = {
        {&L.post_attn_norm, "post_attn_norm"},
        {&L.post_ffn_norm, "post_ffn_norm"},
        {&L.post_attn_norm_bias, "post_attn_norm_bias"},
        {&L.post_ffn_norm_bias, "post_ffn_norm_bias"},
        {&L.ffn_pre_norm_2, "ffn_pre_norm_2"},
        {&L.ffn_post_norm_1, "ffn_post_norm_1"},
        {&L.ffn_post_norm_2, "ffn_post_norm_2"},
        {&L.ffn_gate_inp_scale, "ffn_gate_inp_scale"},
        {&L.layer_out_scale, "layer_out_scale"},
        {&L.expert_down_scale, "expert_down_scale"},
    };
    for (const auto& [norm, name] : post_norms) {
        if (norm->data && !norm->on_device) {
            if (!upload_unquantized_weight(*norm, QType::NONE, ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs, true, name, i)) {
                IMP_LOG_ERROR("Failed to upload post-layer norm for layer %d", i);
                return false;
            }
        }
    }

    // rope_freqs: upload as raw FP32 (NOT converted to FP16).
    // The RoPE kernel reads these as float* — FP16 conversion would corrupt them.
    if (L.rope_freqs.data && !L.rope_freqs.on_device && L.rope_freqs.qtype == QType::F32) {
        IMP_LOG_INFO("Layer %d: uploading rope_freqs as raw FP32 (%lld elements)", i,
                     (long long)L.rope_freqs.numel());
        size_t bytes = L.rope_freqs.nbytes();
        void* d_data = nullptr;
        IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_data, bytes, ctx.stream));
        if (!d_data) {
            IMP_LOG_ERROR("Failed to allocate GPU memory for rope_freqs layer %d", i);
            return false;
        }
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(d_data, L.rope_freqs.data, bytes, cudaMemcpyHostToDevice, ctx.stream));
        ctx.gpu_allocs.push_back(d_data);
        L.rope_freqs.data = d_data;
        L.rope_freqs.on_device = true;
    }

    return true;
}

// upload_layer_ffn_weights: w_gate/w_up/w_down + norms + MoE routing + shared experts for
// one layer.
static bool upload_layer_ffn_weights(TransformerLayer& L, int i, const UploadCtx& ctx) {
    // FFN weights (dense path)
    UPLOAD_OR_FAIL(L.w_gate, L.w_gate.qtype, "w_gate", i, ctx);
    UPLOAD_OR_FAIL(L.w_up, L.w_up.qtype, "w_up", i, ctx);
    UPLOAD_OR_FAIL(L.w_down, L.w_down.qtype, "w_down", i, ctx);

    // GPTQ fallback for FFN weights
    struct {
        Tensor& w;
        TransformerLayer::GPTQWeight& gptq;
        const char* name;
    } ffn_gptq[] = {
        {L.w_gate, L.gptq_gate, "gate_proj"},
        {L.w_up, L.gptq_up, "up_proj"},
        {L.w_down, L.gptq_down, "down_proj"},
    };
    for (auto& [w, gptq, name] : ffn_gptq) {
        if (!w.on_device && gptq.qweight.data) {
            if (!upload_gptq_weight(gptq, w, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to dequant GPTQ %s for layer %d", name, i);
                return false;
            }
            IMP_LOG_DEBUG("GPTQ dequant %s layer %d -> [%lld, %lld] FP16", name, i, (long long)w.shape[0],
                          (long long)w.shape[1]);
        }
    }

    // FFN norm (typically F32/F16, no quant). Qwen3.5/3.6 SafeTensors stores
    // it as delta `W` (actual gamma = 1 + W); ctx.arch_norm_offset adds the +1
    // during BF16→FP16 conversion. GGUF F32 norms unaffected.
    if (L.ffn_norm.data && !L.ffn_norm.on_device) {
        if (!upload_weight(L.ffn_norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true,
                           ctx.arch_norm_offset)) {
            IMP_LOG_ERROR("Failed to upload ffn_norm for layer %d", i);
            return false;
        }
    }

    // MoE gate (routing weights, typically F32/F16)
    UPLOAD_UNQUANT_OR_FAIL(L.moe_gate, "moe_gate", i, ctx);

    // Router bias (Nemotron MoE)
    UPLOAD_UNQUANT_OR_FAIL(L.moe_router_bias, "moe_router_bias", i, ctx);

    // Shared expert weights (Nemotron/DeepSeek style)
    if (L.w_up_shared.data && !L.w_up_shared.on_device) {
        UPLOAD_OR_FAIL(L.w_up_shared, L.w_up_shared.qtype, "w_up_shared", i, ctx);
    }
    if (L.w_down_shared.data && !L.w_down_shared.on_device) {
        UPLOAD_OR_FAIL(L.w_down_shared, L.w_down_shared.qtype, "w_down_shared", i, ctx);
    }
    if (L.w_gate_shared.data && !L.w_gate_shared.on_device) {
        UPLOAD_OR_FAIL(L.w_gate_shared, L.w_gate_shared.qtype, "w_gate_shared", i, ctx);
    }
    // Qwen3-Next / Qwen3.6 shared-expert input gate (FP32 [d_model]).
    if (L.shared_expert_gate_inp.data && !L.shared_expert_gate_inp.on_device) {
        UPLOAD_OR_FAIL(L.shared_expert_gate_inp, QType::F32, "shared_expert_gate_inp", i, ctx);
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_layer_ssm_weights: SSM weights for one layer (Mamba2/Nemotron-H)
// ---------------------------------------------------------------------------
static bool upload_layer_ssm_weights(TransformerLayer& L, int i, const UploadCtx& ctx) {
    // SSM weights (Mamba2)
    if (L.ssm_in.data && !L.ssm_in.on_device) {
        ReleasePoolScope scope(ReleasePool::GdnSources);
        UPLOAD_OR_FAIL(L.ssm_in, L.ssm_in.qtype, "ssm_in", i, ctx);
    }
    if (L.ssm_out.data && !L.ssm_out.on_device) {
        ReleasePoolScope scope(ReleasePool::GdnSources);
        UPLOAD_OR_FAIL(L.ssm_out, L.ssm_out.qtype, "ssm_out", i, ctx);
    }
    // SSM tensors that convert to compute_dtype (FP16): conv1d weights, norm
    struct NamedSsm {
        Tensor* t;
        const char* name;
    };
    const NamedSsm ssm_fp16[] = {
        {&L.ssm_conv1d_w, "ssm_conv1d_w"},
        {&L.ssm_conv1d_b, "ssm_conv1d_b"},
        {&L.ssm_norm_w, "ssm_norm_w"},
    };
    for (const auto& [t, name] : ssm_fp16) {
        if (t->data && !t->on_device) {
            if (!upload_unquantized_weight(*t, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs,
                                           true, name, i)) {
                IMP_LOG_ERROR("Failed to upload SSM tensor for layer %d", i);
                return false;
            }
        }
    }
    // SSM scalars (A_log, D, dt_bias) must end up FP32 on device: the GDN/Mamba scan kernels
    // read them as const float*. GGUF emits F32 directly; SafeTensors BF16-model checkpoints
    // emit BF16, and h2d-copying the bytes verbatim let the kernel reinterpret BF16 as F32
    // (wrong sign/exponent/mantissa), producing NaN in the first GDN layer - convert on host
    // first. ssm_a additionally needs -exp() (HF stores the raw pre-transform A_log; GGUF/imp's
    // kernel expects the post-transform value): verified GGUF[i] == -exp(HF_A_log[perm(i)]).
    for (Tensor* t : {&L.ssm_a, &L.ssm_d, &L.ssm_dt_b}) {
        if (!t->data || t->on_device)
            continue;
        const int64_t n_elem = t->numel();
        const size_t fp32_bytes = static_cast<size_t>(n_elem) * sizeof(float);
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, fp32_bytes, ctx.stream);
        if (!d_data) {
            IMP_LOG_ERROR("Failed to allocate GPU memory for SSM F32 tensor in layer %d", i);
            return false;
        }

        std::vector<float> h_fp32(static_cast<size_t>(n_elem));
        if (t->qtype == QType::F32 || t->qtype == QType::NONE) {
            std::memcpy(h_fp32.data(), t->data, fp32_bytes);
        } else if (t->qtype == QType::BF16) {
            const uint16_t* src = static_cast<const uint16_t*>(t->data);
            for (int64_t k = 0; k < n_elem; ++k) {
                uint32_t bits = static_cast<uint32_t>(src[k]) << 16;
                std::memcpy(&h_fp32[k], &bits, sizeof(float));
            }
        } else if (t->qtype == QType::F16) {
            const uint16_t* src = static_cast<const uint16_t*>(t->data);
            for (int64_t k = 0; k < n_elem; ++k) {
                h_fp32[k] = fp16_to_float(src[k]);
            }
        } else {
            IMP_LOG_ERROR("SSM scalar tensor has unexpected qtype %u (layer %d)",
                          std::to_underlying(t->qtype), i);
            return false;
        }

        // Deciding the HF-to-GGUF A_log transform (-exp()) from dtype alone doesn't work: an HF
        // checkpoint may store A_log as F32 too (Qwen3.5-4B does), and skipping the transform then
        // fed the scan a positive decay rate that grew the state to inf (#1282). Decide from
        // VALUES instead: -exp(x) is strictly negative for any real x, so any value >=0 can only be
        // raw HF A_log. dtype is kept as an OR since BF16/F16 storage is HF-only regardless of sign.
        const bool has_nonnegative = std::any_of(h_fp32.begin(), h_fp32.end(), [](float v) { return v >= 0.0f; });
        const bool is_ssm_a_hf = (t == &L.ssm_a) &&
                                 (t->qtype == QType::BF16 || t->qtype == QType::F16 || has_nonnegative);
        if (is_ssm_a_hf) {
            for (int64_t k = 0; k < n_elem; ++k) {
                h_fp32[k] = -std::exp(h_fp32[k]);
            }
            if (i == 0) {
                IMP_LOG_INFO(
                    "HF GDN A_log: applied -exp() transform to ssm_a (layer 0 first 4 values: %.4f %.4f %.4f "
                    "%.4f)",
                    h_fp32[0], h_fp32[1], h_fp32[2], h_fp32[3]);
            }
        }
        if (const cudaError_t e = h2d_copy(d_data, h_fp32.data(), fp32_bytes, ctx.stream); e != cudaSuccess) {
            IMP_LOG_ERROR("SSM F32 tensor H2D copy failed (layer %d): %s", i, cudaGetErrorString(e));
            ctx.gpu_allocs.push_back(d_data);  // owner frees it with the rest of the failed load
            return false;
        }

        ctx.gpu_allocs.push_back(d_data);
        t->data = d_data;
        t->qtype = QType::F32;
        t->on_device = true;
    }

    // GDN alpha/beta upload via gemm_dispatch like every quantized weight. Pre-dequanting to
    // FP16 on host (raw_quant=false) left L.gdn_*_qtype at Q8_0 after conversion, so
    // gemm_dispatch re-interpreted the FP16 bytes as Q8_0 blocks (~80x too-large values,
    // immediate state collapse). Upload raw Q8_0 instead so the qtype stays consistent with the
    // on-device bytes.
    if (L.gdn_gate.data && !L.gdn_gate.on_device) {
        ReleasePoolScope scope(ReleasePool::GdnSources);
        UPLOAD_OR_FAIL(L.gdn_gate, L.gdn_gate.qtype, "gdn_gate", i, ctx);
    }
    if (L.gdn_alpha.data && !L.gdn_alpha.on_device) {
        UPLOAD_OR_FAIL(L.gdn_alpha, L.gdn_alpha.qtype, "gdn_alpha", i, ctx);
    }
    if (L.gdn_beta.data && !L.gdn_beta.on_device) {
        UPLOAD_OR_FAIL(L.gdn_beta, L.gdn_beta.qtype, "gdn_beta", i, ctx);
    }

    // GDN input projection fusion (M=1 decode GEMV): tries the full 4-way pack (ssm_in+
    // gdn_gate+gdn_alpha+gdn_beta -> one weight) first, falls back to the alpha+beta-only 2-way
    // pack if ssm_in/gdn_gate aren't FP16/BF16 (raw-quant Q*_K, NVFP4 prequant). Originals stay
    // live for prefill's unchanged 4-call path; decode opts in via the executor at n==1.
    auto& a = L.ssm_in;
    auto& b = L.gdn_gate;
    auto& c = L.gdn_alpha;
    auto& d = L.gdn_beta;

    auto fp_uniform = [](QType q, QType expect) {
        return q == expect && (q == QType::F16 || q == QType::BF16);
    };

    bool four_way_ok =
        a.data && a.on_device && b.data && b.on_device && c.data && c.on_device && d.data && d.on_device &&
        a.ndim == 2 && b.ndim == 2 && c.ndim == 2 && d.ndim == 2 &&
        (a.qtype == QType::F16 || a.qtype == QType::BF16) && fp_uniform(b.qtype, a.qtype) &&
        fp_uniform(c.qtype, a.qtype) && fp_uniform(d.qtype, a.qtype) && a.shape[1] == b.shape[1] &&
        a.shape[1] == c.shape[1] && a.shape[1] == d.shape[1] && c.shape[0] == d.shape[0];

    if (four_way_ok) {
        int64_t conv_channels = a.shape[0];
        int64_t inner = b.shape[0];
        int64_t n_heads = c.shape[0];
        int64_t d_model = a.shape[1];
        int64_t total_out = conv_channels + inner + 2 * n_heads;
        size_t es = 2;  // F16 / BF16 = 2 bytes
        size_t total_bytes = static_cast<size_t>(total_out) * d_model * es;
        void* d_packed = nullptr;
        ReleasePoolScope pack_scope(ReleasePool::GdnPacks);  // Phase 4b frees it (default pool kept 1137 MiB)
        if (malloc_async_in_scope(&d_packed, total_bytes, ctx.stream) == cudaSuccess && d_packed) {
            ctx.gpu_allocs.push_back(d_packed);
            char* base = static_cast<char*>(d_packed);
            // Concat in N (rows): each weight is a contiguous [out, d_model] block,
            // so plain cudaMemcpyAsync into the packed buffer's row range is the
            // entire pack op.
            size_t bytes_a = static_cast<size_t>(conv_channels) * d_model * es;
            size_t bytes_b = static_cast<size_t>(inner) * d_model * es;
            size_t bytes_c = static_cast<size_t>(n_heads) * d_model * es;
            IMP_CUDA_CHECK_LOG(
                cudaMemcpyAsync(base, a.data, bytes_a, cudaMemcpyDeviceToDevice, ctx.stream));
            IMP_CUDA_CHECK_LOG(
                cudaMemcpyAsync(base + bytes_a, b.data, bytes_b, cudaMemcpyDeviceToDevice, ctx.stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(base + bytes_a + bytes_b, c.data, bytes_c,
                                                cudaMemcpyDeviceToDevice, ctx.stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(base + bytes_a + bytes_b + bytes_c, d.data, bytes_c,
                                                cudaMemcpyDeviceToDevice, ctx.stream));
            int64_t packed_shape[2] = {total_out, d_model};
            L.gdn_input_packed = Tensor(d_packed, a.qtype, 2, packed_shape, true);
            L.gdn_packed_conv_channels = static_cast<int>(conv_channels);
            L.gdn_packed_inner = static_cast<int>(inner);
            L.gdn_packed_n_heads = static_cast<int>(n_heads);
            IMP_LOG_DEBUG("  layer %d: gdn_input_packed [%lld, %lld] %.2f MiB", i, (long long)total_out,
                          (long long)d_model, total_bytes / (1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN(
                "layer %d: gdn_input_packed alloc failed (%zu bytes), falling back to 2-way / 4-call", i,
                total_bytes);
        }
    }

    // Step-1 fallback: alpha+beta only (decode 2-way fusion). Skipped if
    // gdn_input_packed already covers the same case.
    if (!L.gdn_input_packed.data && c.data && c.on_device && d.data && d.on_device && c.ndim == 2 &&
        d.ndim == 2 && c.qtype == d.qtype && (c.qtype == QType::F16 || c.qtype == QType::BF16) &&
        c.shape[0] == d.shape[0] && c.shape[1] == d.shape[1]) {
        int64_t d_model = c.shape[0];
        int64_t n_heads = c.shape[1];
        size_t es = 2;
        size_t row_bytes = static_cast<size_t>(n_heads) * es;
        size_t total_bytes = static_cast<size_t>(d_model) * 2 * row_bytes;
        void* d_packed = nullptr;
        if (cudaMallocAsync(&d_packed, total_bytes, ctx.stream) == cudaSuccess && d_packed) {
            ctx.gpu_allocs.push_back(d_packed);
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(d_packed, 2 * row_bytes, c.data, row_bytes, row_bytes,
                                                 d_model, cudaMemcpyDeviceToDevice, ctx.stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(static_cast<char*>(d_packed) + row_bytes, 2 * row_bytes,
                                                 d.data, row_bytes, row_bytes, d_model,
                                                 cudaMemcpyDeviceToDevice, ctx.stream));
            int64_t packed_shape[2] = {d_model, 2 * n_heads};
            L.gdn_alpha_beta_packed = Tensor(d_packed, c.qtype, 2, packed_shape, true);
            IMP_LOG_DEBUG("  layer %d: gdn_alpha_beta_packed [%lld, %lld] %.2f KiB (Step-1 fallback)", i,
                          (long long)d_model, (long long)(2 * n_heads), total_bytes / 1024.0);
        }
    }

    return true;
}

// Pass 1: non-expert per-layer weights. model.upload_host_layers_ (--gpu-layers) layers keep their
// verbatim matmuls on host, pinned when the whole set fits host RAM (layer_host_keep.h).
static bool upload_pass1_layers(Model& model, const UploadCtx& ctx) {
    const bool pin_ok = host_pin_ram_available(host_keep_bytes(model, &dequant_gpu_supported));
    auto pin = [&](size_t bytes) -> void* {
        PinnedBuffer buf = pin_ok ? PinnedBuffer::acquire(cuda_host_pinned_allocator(), bytes) : PinnedBuffer();
        void* p = buf.data();
        if (p)
            ctx.host_pinned_allocs.push_back(std::move(buf));
        return p;
    };
    return upload_pass1_keeping_host(model, &dequant_gpu_supported, pin, [&](TransformerLayer& L, int i) {
        return upload_layer_attention_weights(L, i, ctx) && upload_layer_ffn_weights(L, i, ctx) &&
               upload_layer_ssm_weights(L, i, ctx);
    });
}

// upload_expert_weights (Pass 2): handles packed 3D and per-expert 2D expert tensors.
// Phase 1 computes per-layer expert-tensor byte cost (packed 3-D plus per-expert 2-D
// NVFP4 llm-compressor tensors), filling layer_expert_bytes and returning total_expert_bytes.
static size_t compute_expert_layer_costs_(const std::vector<TransformerLayer>& layers, int n_layers,
                                          std::vector<size_t>& layer_expert_bytes) {
    size_t total_expert_bytes = 0;
    layer_expert_bytes.assign(n_layers, 0);
    for (int i = 0; i < n_layers; ++i) {
        const TransformerLayer& L = layers[i];
        auto add_packed = [&](const Tensor& p, QType qt) {
            if (!p.data || p.ndim < 3 || !dequant_gpu_supported(qt))
                return;
            size_t row_bytes = qtype_row_bytes(qt, p.shape[2]);
            size_t bytes = static_cast<size_t>(p.shape[0]) * p.shape[1] * row_bytes;
            layer_expert_bytes[i] += bytes;
            total_expert_bytes += bytes;
        };
        add_packed(L.expert_gate_packed, L.expert_gate_packed.qtype);
        add_packed(L.expert_up_packed, L.expert_up_packed.qtype);
        add_packed(L.expert_down_packed, L.expert_down_packed.qtype);

        // Also account for per-expert 2D tensors (e.g. NVFP4 llm-compressor format).
        // These are NOT packed 3D, so add_packed misses them.
        auto add_2d_expert = [&](const std::vector<Tensor>& vt) {
            for (const auto& t : vt) {
                if (!t.data || t.on_device || t.ndim < 2)
                    continue;
                layer_expert_bytes[i] += t.nbytes();
                total_expert_bytes += t.nbytes();
            }
        };
        add_2d_expert(L.expert_w_gate);
        add_2d_expert(L.expert_w_up);
        add_2d_expert(L.expert_w_down);
    }
    return total_expert_bytes;
}

// Phase 2: picks which MoE layers' experts stay on GPU vs go to host. Honors the VRAM
// reserve Engine passes (KV cache + workspaces + FP16 cache), WSL2/WDDM driver overhead
// (10% aggressive if all fit, else 30% conservative, or an explicit override), and
// moe.force_host_experts=N. Re-arms the g_cached_free_mem/g_total_allocated/g_vram_reserve
// trio so per-expert checked_cuda_malloc calls don't double-count the reserve.
static void decide_expert_layer_placement_(const std::vector<size_t>& layer_expert_bytes,
                                           size_t total_expert_bytes, size_t expert_reserve_bytes,
                                           int n_layers, std::vector<bool>& experts_upload_layer,
                                           bool is_nvfp4_prequant) {
    if (total_expert_bytes == 0)
        return;

    size_t free_mem = 0, total_mem = 0;
    // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
    (void)vram_budget_mem_get_info(&free_mem, &total_mem);

    // Auto-pick default: 10% (aggressive) if ALL experts fit with that overhead, else 30%
    // (conservative). Saves users from a silent large perf penalty on Qwen3-Coder-30B/
    // Qwen3.6-35B-class MoE models where the conservative default unnecessarily offloads
    // experts to host.
    int overhead_pct = process_diag_moe_expert_overhead_pct();
    if (overhead_pct < 0 || overhead_pct > 50) {
        overhead_pct = 30;
    } else if (overhead_pct == 10) {
        size_t aggressive_overhead = static_cast<size_t>(free_mem * 10 / 100);
        size_t aggressive_reserve = expert_reserve_bytes + aggressive_overhead;
        size_t aggressive_budget = (free_mem > aggressive_reserve) ? (free_mem - aggressive_reserve) : 0;
        if (aggressive_budget < total_expert_bytes) {
            overhead_pct = 30;
        } else {
            IMP_LOG_INFO(
                "Expert offload: all experts fit with 10%% overhead "
                "(%.2f GiB experts, %.2f GiB free) — picking aggressive.",
                total_expert_bytes / (1024.0 * 1024.0 * 1024.0), free_mem / (1024.0 * 1024.0 * 1024.0));
        }
    }
    // NVFP4-prequant experts have no working host-offload path: host-resident they stay
    // QType::INT8 packed (Phase-0 promote never runs on them) and the MoE fallback hands raw
    // packed bytes to cuBLAS -> CUBLAS_STATUS_NOT_SUPPORTED, repeated-token garbage + IMA. They
    // are mandatory on-device: drop the KV/workspace reserve so they all upload when they
    // physically fit, and size the KV pool from what's left (correctness over context length).
    if (is_nvfp4_prequant)
        overhead_pct = 0;
    size_t overhead = static_cast<size_t>(free_mem * overhead_pct / 100);
    size_t total_reserve = expert_reserve_bytes + overhead;
    size_t budget = (free_mem > total_reserve) ? (free_mem - total_reserve) : 0;
    int force_host_n = process_diag_moe_force_host_experts();
    if (force_host_n < 0)
        force_host_n = 0;

    // NVFP4 experts that do not all fit: every layer goes to the host and the VRAM the
    // partial upload would have taken becomes expert-cache slots. A resident layer buys a
    // 100% hit rate on 1/n_layers of the work for ~1.17 GiB; the same VRAM in the cache
    // buys ~27 slots/layer on EVERY layer. Measured 2026-09-22, Qwen3.8-Flash-Next: the
    // greedy partial upload put 13 layers on the card, left the cache 0 slots and the load
    // was refused outright; all-on-host with the freed VRAM serves 77 tok/s.
    if (is_nvfp4_prequant && budget < total_expert_bytes && force_host_n == 0) {
        int moe_layers = 0;
        for (int i = 0; i < n_layers; ++i)
            if (layer_expert_bytes[i] > 0)
                moe_layers++;
        IMP_LOG_INFO(
            "NVFP4 experts (%.2f GiB) do not fit the %.2f GiB budget: all %d MoE layer(s) stay "
            "host-resident so the expert cache gets the VRAM (a partial upload starves it).",
            total_expert_bytes / (1024.0 * 1024.0 * 1024.0), budget / (1024.0 * 1024.0 * 1024.0),
            moe_layers);
        return;  // experts_upload_layer stays all-false
    }

    if (budget >= total_expert_bytes && force_host_n == 0) {
        for (int i = 0; i < n_layers; ++i) {
            if (layer_expert_bytes[i] > 0)
                experts_upload_layer[i] = true;
        }
        IMP_LOG_INFO(
            "Expert weights: %.2f GiB -> uploading ALL to GPU "
            "(%.2f GiB free, %.2f GiB reserve)",
            total_expert_bytes / (1024.0 * 1024.0 * 1024.0), free_mem / (1024.0 * 1024.0 * 1024.0),
            expert_reserve_bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (force_host_n > 0) {
        // Debug: force last N MoE layers to host. Still respect budget for
        // the ones we do upload.
        std::vector<int> moe_layer_idxs;
        for (int i = 0; i < n_layers; ++i)
            if (layer_expert_bytes[i] > 0)
                moe_layer_idxs.push_back(i);
        int skip_from = std::max(0, (int)moe_layer_idxs.size() - force_host_n);
        size_t uploaded = 0;
        int n_uploaded = 0;
        for (int k = 0; k < skip_from; ++k) {
            int i = moe_layer_idxs[k];
            if (uploaded + layer_expert_bytes[i] <= budget) {
                experts_upload_layer[i] = true;
                uploaded += layer_expert_bytes[i];
                n_uploaded++;
            }
        }
        IMP_LOG_INFO(
            "Expert weights (IMP_FORCE_HOST_EXPERTS=%d): uploading %d/%zu MoE layers, %d forced to host",
            force_host_n, n_uploaded, moe_layer_idxs.size(), force_host_n);
    } else {
        // Partial upload: greedily upload layers until budget exhausted
        size_t uploaded = 0;
        int n_uploaded = 0, n_total_moe = 0;
        for (int i = 0; i < n_layers; ++i) {
            if (layer_expert_bytes[i] == 0)
                continue;
            n_total_moe++;
            if (uploaded + layer_expert_bytes[i] <= budget) {
                experts_upload_layer[i] = true;
                uploaded += layer_expert_bytes[i];
                n_uploaded++;
            }
        }
        IMP_LOG_INFO(
            "Expert weights: %.2f GiB total, uploading %d/%d MoE layers "
            "(%.2f GiB on GPU, %.2f GiB on host, %.2f GiB free, "
            "%.2f GiB reserve)",
            total_expert_bytes / (1024.0 * 1024.0 * 1024.0), n_uploaded, n_total_moe,
            uploaded / (1024.0 * 1024.0 * 1024.0),
            (total_expert_bytes - uploaded) / (1024.0 * 1024.0 * 1024.0),
            free_mem / (1024.0 * 1024.0 * 1024.0), expert_reserve_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    // Re-arm the cached free-memory window so per-expert checked_cuda_malloc
    // calls don't fall back to a sync cudaMemGetInfo on every tensor.
    g_cached_free_mem = free_mem;
    g_total_allocated = 0;
    g_vram_reserve = 0;  // budget already accounted for above
}

static bool upload_expert_weights(std::vector<TransformerLayer>& layers, int n_layers,
                                  size_t expert_reserve_bytes, const UploadCtx& ctx) {
    std::vector<size_t> layer_expert_bytes;
    size_t total_expert_bytes = compute_expert_layer_costs_(layers, n_layers, layer_expert_bytes);

    std::vector<bool> experts_upload_layer(n_layers, false);
    decide_expert_layer_placement_(layer_expert_bytes, total_expert_bytes, expert_reserve_bytes, n_layers,
                                   experts_upload_layer, ctx.is_nvfp4_prequant);

    // A host-resident NVFP4 placement is servable: the expert cache stages those experts per
    // layer and the fused kernels address its slot pool. Whether the cache will be big enough
    // cannot be decided here - it's sized in init_weights()' workspace pass, which runs after.
    // This only reports; the refusal lives in GraphExecutor::verify_host_expert_placement(),
    // once both the promotion and the real cache exist.
    // One decision for the whole model, before the first projection is pinned.
    g_pin_host_experts_fits = true;
    if (ctx.is_nvfp4_prequant) {
        size_t host_expert_bytes = 0;
        for (int i = 0; i < n_layers; ++i)
            if (!experts_upload_layer[i])
                host_expert_bytes += layer_expert_bytes[i];
        if (host_expert_bytes > 0 && !host_pin_ram_available(host_expert_bytes)) {
            g_pin_host_experts_fits = false;
            IMP_LOG_WARN(
                "Host NVFP4 experts: not pinning %.2f GiB, host RAM available is %.2f GiB (6 GiB "
                "headroom kept) — decode serves from the host LRU path, measured 2.5x slower. "
                "Free host RAM or raise the container memory limit.",
                host_expert_bytes / (1024.0 * 1024.0 * 1024.0),
                host_mem_available_bytes() / (1024.0 * 1024.0 * 1024.0));
        }
    }

    if (expert_placement_needs_host_path(ctx.is_nvfp4_prequant, layer_expert_bytes,
                                         experts_upload_layer)) {
        const int host_layers = expert_placement_host_layers(layer_expert_bytes, experts_upload_layer);
        size_t free_mem = 0, total_mem = 0;
        // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
        (void)vram_budget_mem_get_info(&free_mem, &total_mem);
        IMP_LOG_INFO(
            "NVFP4 experts: %d MoE layer(s) stay host-resident (%zu MiB of experts, %zu MiB free) — "
            "they will be served from the expert cache. Decode on those layers is materially "
            "slower than resident; the load is refused later if the cache cannot hold them.",
            host_layers, total_expert_bytes / (size_t{1024} * 1024), free_mem / (size_t{1024} * 1024));
    }

    HostExpertPinner pinner(ctx.host_pinned_allocs);  // joins on every return path

    // Upload expert weights for each layer
    for (int i = 0; i < n_layers; ++i) {
        TransformerLayer& L = layers[i];

        // MoE expert weights, two paths: (A) packed 3D *_exps - quantized types (Q6_K/Q8_0/Q4_0)
        // upload raw bytes and dequant on-the-fly in run_moe_ffn; F16/BF16/F32 dequant/upload and
        // slice into per-expert views. (B) per-expert 2D tensors upload individually (legacy
        // per-expert GGUF format).

        auto upload_packed_experts = [&](Tensor& packed, QType qtype, std::vector<Tensor>& expert_vec,
                                         const char* name) -> bool {
            if (!packed.data || packed.ndim < 3)
                return true;  // nothing to do
            if (packed.on_device)
                return true;  // already on GPU (e.g. from Gemma 4 fused split)

            // gpt-oss MXFP4 experts stay host-resident raw: the executor's pre_dequant phase converts
            // them to NVFP4 and registers the CUTLASS grouped path. Uploading raw MXFP4 here would
            // leave them in a format no MoE kernel consumes (NaN); the F16-slice fallback also
            // mis-strides MXFP4 bytes.
            if (ctx.mxfp4_experts_deferred && qtype == QType::MXFP4) {
                IMP_LOG_DEBUG("  %s: gpt-oss MXFP4 experts kept host-resident for NVFP4 convert", name);
                return true;
            }

            int n_experts = static_cast<int>(packed.shape[0]);
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];

            // Path A1: Quantized types -- upload raw bytes to GPU if they fit,
            // otherwise keep on host (mmap'd) with optional pinning for H2D.
            if (dequant_gpu_supported(qtype)) {
                size_t row_bytes = qtype_row_bytes(qtype, cols);
                size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
                size_t total_raw = static_cast<size_t>(n_experts) * expert_raw;

                if (experts_upload_layer[i]) {
                    // Suspend-to-RAM warm hit: restore the raw expert bytes from
                    // the host snapshot instead of re-copying from mmap.
                    char keybuf[96];
                    const char* key = (g_warm || g_upload_log) ? make_weight_key(keybuf, name, i) : nullptr;
                    if (key && g_warm &&
                        g_warm->try_restore(key, packed, ctx.stream, ctx.gpu_allocs, kWarmOps,
                                            g_upload_log)) {
                        IMP_LOG_DEBUG("  %s: %d experts restored from snapshot (%.2f MiB)", name,
                                      n_experts, total_raw / (1024.0 * 1024.0));
                        return true;
                    }
                    // Upload raw quantized bytes to GPU (respects VRAM reserve)
                    void* gpu_ptr = nullptr;
                    cudaError_t err = checked_cuda_malloc(&gpu_ptr, total_raw, ctx.stream);
                    if (err == cudaSuccess) {
                        cudaError_t cpy_err = h2d_copy(gpu_ptr, packed.data, total_raw, ctx.stream);
                        if (cpy_err != cudaSuccess) {
                            IMP_LOG_ERROR("  %s: h2d_copy failed: %s", name, cudaGetErrorString(cpy_err));
                            IMP_CUDA_CHECK_LOG(cudaFreeAsync(gpu_ptr, ctx.stream));
                            return false;
                        }
                        const QType src_qtype = packed.qtype;
                        const int64_t src_numel = packed.numel();
                        packed.data = gpu_ptr;
                        packed.on_device = true;
                        ctx.gpu_allocs.push_back(gpu_ptr);
                        if (key && g_upload_log) {
                            const void* const rec_allocs[] = {gpu_ptr};
                            g_upload_log->record(key, rec_allocs, 1, packed, src_qtype, src_numel,
                                                 total_raw);
                        }
                        IMP_LOG_DEBUG("  %s: %d experts uploaded to GPU (%.2f MiB)", name, n_experts,
                                      total_raw / (1024.0 * 1024.0));
                        return true;
                    }
                    // cudaMalloc failed — fall through to host path
                    IMP_LOG_WARN("  %s: cudaMalloc failed for %.2f MiB, falling back to host", name,
                                 total_raw / (1024.0 * 1024.0));
                }

                // Host path: pin memory for fast async DMA H2D during decode.
                if (is_wsl2()) {
                    // WSL2: cudaHostRegister fails on mmap'd memory. Instead,
                    // allocate fresh pinned memory and copy mmap'd data there.
                    // This enables true async DMA H2D (no per-token CPU memcpy).
                    PinnedBuffer pinned_buf =
                        PinnedBuffer::acquire(cuda_host_pinned_allocator(), total_raw);
                    if (!pinned_buf.empty()) {
                        memcpy(pinned_buf.data(), packed.data, total_raw);
                        packed.data = pinned_buf.data();
                        ctx.host_pinned_allocs.push_back(std::move(pinned_buf));
                        IMP_LOG_INFO("  %s: WSL2 pinned copy (%.2f MiB, DMA-ready)", name,
                                     total_raw / (1024.0 * 1024.0));
                    } else {
                        cudaGetLastError();  // clear sticky CUDA error state
                        IMP_LOG_INFO(
                            "  %s: WSL2 pinned alloc failed, falling back to "
                            "unpinned mmap (%.2f MiB)",
                            name, total_raw / (1024.0 * 1024.0));
                    }
                } else {
                    // T5b: the registration is owned, so an early return between
                    // here and teardown cannot leave a page-locked region behind
                    // (memory/host_pinned.h). It logs its own failure.
                    HostRegistration reg =
                        HostRegistration::acquire_read_only(packed.data, total_raw);
                    if (!reg.empty()) {
                        ctx.host_pinned.push_back(std::move(reg));
                        IMP_LOG_DEBUG("  %s: %d experts, raw %s pinned on host (%.2f MiB)", name, n_experts,
                                      qtype == QType::Q6_K   ? "Q6_K"
                                      : qtype == QType::Q8_0 ? "Q8_0"
                                                             : "Q4_0",
                                      total_raw / (1024.0 * 1024.0));
                    }
                }

                return true;
            }

            // Path A2: Unquantized (F16/BF16/F32) -- dequant to FP16, slice per-expert.
            int64_t flat_shape[4] = {static_cast<int64_t>(n_experts) * rows, cols, 0, 0};
            Tensor flat(packed.data, packed.qtype, 2, flat_shape, packed.on_device);

            if (!upload_weight(flat, qtype, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true, 0.0f,
                               name, i)) {
                IMP_LOG_ERROR("Failed to upload packed %s for layer %d", name, i);
                return false;
            }

            expert_vec.resize(n_experts);
            size_t expert_bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
            for (int e = 0; e < n_experts; e++) {
                char* ptr = static_cast<char*>(flat.data) + e * expert_bytes;
                int64_t eshape[4] = {rows, cols, 0, 0};
                expert_vec[e] = Tensor(ptr, QType::F16, 2, eshape, true);
            }
            packed = Tensor();
            return true;
        };

        if (!upload_packed_experts(L.expert_gate_packed, L.expert_gate_packed.qtype, L.expert_w_gate,
                                   "expert_gate_exps"))
            return false;

        // Gemma 4: splits the fused ffn_gate_up_exps [n_exp,2*n_ff_exp,d_model] (rows [0,n_ff_exp)
        // = gate, [n_ff_exp,2*n_ff_exp) = up) into separate gate/up packed tensors via cudaMemcpy2D
        // on GPU, then frees the fused buffer. Host-resident layers get the same split so MoE
        // dispatch finds expert_up_packed; without it a host-resident layer's null
        // expert_up_packed falls back to an uninitialized expert_w_up[eidx] (garbage output). Gated
        // on !experts_upload_layer[i]; upload-destined layers use the GPU split path instead.
        if (!experts_upload_layer[i] && L.expert_gate_packed.data && !L.expert_gate_packed.on_device &&
            L.expert_up_packed.data == nullptr && L.expert_gate_packed.ndim >= 3 &&
            (L.expert_gate_packed.shape[1] & 1) == 0 && dequant_gpu_supported(L.expert_gate_packed.qtype)) {
            int64_t n_exp = L.expert_gate_packed.shape[0];
            int64_t fused_rows = L.expert_gate_packed.shape[1];
            int64_t cols = L.expert_gate_packed.shape[2];
            int64_t half_rows = fused_rows / 2;
            size_t row_bytes = qtype_row_bytes(L.expert_gate_packed.qtype, cols);
            size_t half_raw = static_cast<size_t>(n_exp) * half_rows * row_bytes;
            size_t src_pitch = static_cast<size_t>(fused_rows) * row_bytes;
            size_t dst_pitch = static_cast<size_t>(half_rows) * row_bytes;

            // Allocate two pinned host buffers for the split halves.
            PinnedBuffer gate_pin = PinnedBuffer::acquire(cuda_host_pinned_allocator(), half_raw);
            PinnedBuffer up_pin = PinnedBuffer::acquire(cuda_host_pinned_allocator(), half_raw);
            if (gate_pin.empty() || up_pin.empty()) {
                // Whichever one succeeded releases itself on the way out — the
                // hand-written unwind is gone (T5a/T5b, memory/host_pinned.h).
                IMP_LOG_ERROR("Gemma 4: host split pinned alloc failed (layer %d, %.1f MiB each)", i,
                              half_raw / (1024.0 * 1024.0));
                return false;
            }
            void* gate_buf = gate_pin.data();
            void* up_buf = up_pin.data();

            const char* src_base = static_cast<const char*>(L.expert_gate_packed.data);
            for (int64_t e = 0; e < n_exp; ++e) {
                // gate half = rows [0, half_rows)
                memcpy(static_cast<char*>(gate_buf) + e * dst_pitch, src_base + e * src_pitch, dst_pitch);
                // up half = rows [half_rows, fused_rows)
                memcpy(static_cast<char*>(up_buf) + e * dst_pitch, src_base + e * src_pitch + dst_pitch,
                       dst_pitch);
            }

            int64_t split_shape[4] = {n_exp, half_rows, cols, 0};
            L.expert_gate_packed = Tensor(gate_buf, L.expert_gate_packed.qtype, 3, split_shape, false);
            L.expert_up_packed = Tensor(up_buf, L.expert_gate_packed.qtype, 3, split_shape, false);
            L.expert_up_packed.qtype = L.expert_gate_packed.qtype;
            ctx.host_pinned_allocs.push_back(std::move(gate_pin));
            ctx.host_pinned_allocs.push_back(std::move(up_pin));
            IMP_LOG_INFO("Gemma 4: host-split fused gate_up_exps layer %d (n_ff_exp=%ld, %.1f MiB each)", i,
                         (long)half_rows, half_raw / (1024.0 * 1024.0));
        }

        if (L.expert_gate_packed.data && L.expert_gate_packed.on_device &&
            L.expert_up_packed.data == nullptr && L.expert_gate_packed.ndim >= 3 &&
            (L.expert_gate_packed.shape[1] & 1) == 0 && dequant_gpu_supported(L.expert_gate_packed.qtype)) {
            int64_t n_exp = L.expert_gate_packed.shape[0];
            int64_t fused_rows = L.expert_gate_packed.shape[1];
            int64_t cols = L.expert_gate_packed.shape[2];
            int64_t half_rows = fused_rows / 2;
            size_t row_bytes = qtype_row_bytes(L.expert_gate_packed.qtype, cols);
            size_t half_raw = static_cast<size_t>(n_exp) * half_rows * row_bytes;

            // Memory-efficient split: allocates only ONE half-sized buffer for the up half, copies it
            // out, then reuses the fused buffer in-place for the gate half (rows already at the front).
            // Peak overhead 0.5x fused instead of 1.0x for a two-buffer split.
            void* up_buf = nullptr;
            cudaError_t e2 = checked_cuda_malloc(&up_buf, half_raw, ctx.stream);
            if (e2 != cudaSuccess) {
                IMP_LOG_ERROR("Gemma 4: cudaMalloc failed for fused expert split (layer %d, %.1f MiB)", i,
                              half_raw / (1024.0 * 1024.0));
                return false;
            }

            size_t dst_pitch = static_cast<size_t>(half_rows) * row_bytes;
            size_t src_pitch = static_cast<size_t>(fused_rows) * row_bytes;
            const char* src_base = static_cast<const char*>(L.expert_gate_packed.data);

            // Copy up half (rows [half_rows, fused_rows)) into the new up buffer.
            cudaError_t cp = cudaMemcpy2DAsync(up_buf, dst_pitch, src_base + dst_pitch, src_pitch, dst_pitch,
                                               n_exp, cudaMemcpyDeviceToDevice, ctx.stream);
            if (cp != cudaSuccess) {
                IMP_LOG_ERROR("Gemma 4: cudaMemcpy2DAsync failed (layer %d): %s", i, cudaGetErrorString(cp));
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(up_buf, ctx.stream));
                return false;
            }
            // Compacts the gate half in-place: row e (offset e*src_pitch) moves to e*dst_pitch. With
            // dst_pitch < src_pitch expert e+1's dst overlaps expert e's src region, so experts are
            // walked forward and copied one at a time rather than as a single overlapping 2D launch.
            for (int64_t e = 1; e < n_exp; ++e) {  // e=0 already at the right offset
                cudaError_t cp_e = cudaMemcpyAsync(const_cast<char*>(src_base) + e * dst_pitch,
                                                   src_base + e * src_pitch, dst_pitch,
                                                   cudaMemcpyDeviceToDevice, ctx.stream);
                if (cp_e != cudaSuccess) {
                    IMP_LOG_ERROR("Gemma 4: gate compact memcpy failed (layer %d, expert %ld): %s", i,
                                  (long)e, cudaGetErrorString(cp_e));
                    IMP_CUDA_CHECK_LOG(cudaFreeAsync(up_buf, ctx.stream));
                    return false;
                }
            }

            // The fused source buffer was just compacted IN PLACE — a weight
            // snapshot taken later would capture post-split bytes the resume
            // replay would split again. Suspend is unsupported for this model.
            if (ctx.model)
                ctx.model->mark_device_sources_mutated();

            // Reuse fused buffer (now compacted to half size) as gate_packed.
            int64_t split_shape[4] = {n_exp, half_rows, cols, 0};
            void* gate_buf = L.expert_gate_packed.data;
            L.expert_gate_packed = Tensor(gate_buf, L.expert_gate_packed.qtype, 3, split_shape, true);
            L.expert_up_packed = Tensor(up_buf, L.expert_gate_packed.qtype, 3, split_shape, true);
            L.expert_up_packed.qtype = L.expert_gate_packed.qtype;
            // gate_buf is already in ctx.gpu_allocs from the original upload.
            ctx.gpu_allocs.push_back(up_buf);
            IMP_LOG_INFO("Gemma 4: split fused gate_up_exps layer %d (n_ff_exp=%ld, %.1f MiB each)", i,
                         (long)half_rows, half_raw / (1024.0 * 1024.0));
        }

        if (!upload_packed_experts(L.expert_up_packed, L.expert_up_packed.qtype, L.expert_w_up,
                                   "expert_up_exps"))
            return false;
        if (!upload_packed_experts(L.expert_down_packed, L.expert_down_packed.qtype, L.expert_w_down,
                                   "expert_down_exps"))
            return false;

        // Path B: per-expert 2D tensors (from per-expert GGUF or llm-compressor NVFP4 format).
        // Respect the experts_upload_layer budget flag — skip if experts for this layer
        // don't fit in the remaining VRAM budget.
        if (experts_upload_layer[i]) {
            char ename[48];
            for (size_t e = 0; e < L.expert_w_gate.size(); ++e) {
                if (!L.expert_w_gate[e].data || L.expert_w_gate[e].on_device)
                    continue;
                snprintf(ename, sizeof(ename), "expert_w_gate.%zu", e);
                if (!upload_weight(L.expert_w_gate[e], L.expert_gate_packed.qtype, ctx.compute_dtype,
                                   ctx.stream, ctx.gpu_allocs, true, 0.0f, ename, i)) {
                    IMP_LOG_ERROR("Failed to upload expert_w_gate[%zu] for layer %d", e, i);
                    return false;
                }
            }
            for (size_t e = 0; e < L.expert_w_up.size(); ++e) {
                if (!L.expert_w_up[e].data || L.expert_w_up[e].on_device)
                    continue;
                snprintf(ename, sizeof(ename), "expert_w_up.%zu", e);
                if (!upload_weight(L.expert_w_up[e], L.expert_up_packed.qtype, ctx.compute_dtype, ctx.stream,
                                   ctx.gpu_allocs, true, 0.0f, ename, i)) {
                    IMP_LOG_ERROR("Failed to upload expert_w_up[%zu] for layer %d", e, i);
                    return false;
                }
            }
            for (size_t e = 0; e < L.expert_w_down.size(); ++e) {
                if (!L.expert_w_down[e].data || L.expert_w_down[e].on_device)
                    continue;
                snprintf(ename, sizeof(ename), "expert_w_down.%zu", e);
                if (!upload_weight(L.expert_w_down[e], L.expert_down_packed.qtype, ctx.compute_dtype,
                                   ctx.stream, ctx.gpu_allocs, true, 0.0f, ename, i)) {
                    IMP_LOG_ERROR("Failed to upload expert_w_down[%zu] for layer %d", e, i);
                    return false;
                }
            }
        } else {
            // Host-resident per-expert 2D BF16 tensors: converted BF16->FP16 in pinned host memory,
            // since the legacy MoE GEMM path's cuBLAS call rejects mixed FP16-activation x BF16-weight
            // (status 15). Only fires for unquantized per-expert BF16 (e.g. DeepSeek-V2 SafeTensors);
            // quantized types stay raw and go through dequant-on-the-fly instead.
            auto convert_host_bf16 = [&](std::vector<Tensor>& expert_vec) {
                for (Tensor& w : expert_vec) {
                    if (!w.data || w.on_device || w.qtype != QType::BF16)
                        continue;
                    int64_t n_elem = w.numel();
                    size_t bytes = static_cast<size_t>(n_elem) * sizeof(uint16_t);
                    PinnedBuffer pin = PinnedBuffer::acquire(cuda_host_pinned_allocator(), bytes);
                    if (pin.empty()) {
                        IMP_LOG_WARN(
                            "Host expert BF16→FP16: pinned alloc failed "
                            "(%.1f MiB) — expert stays BF16, GEMM will fail",
                            bytes / (1024.0 * 1024.0));
                        continue;
                    }
                    uint16_t* pinned = pin.as<uint16_t>();
                    const uint16_t* src = static_cast<const uint16_t*>(w.data);
                    for (int64_t k = 0; k < n_elem; ++k) {
                        uint32_t bits = static_cast<uint32_t>(src[k]) << 16;
                        float f;
                        std::memcpy(&f, &bits, sizeof(float));
                        pinned[k] = float_to_fp16(f);
                    }
                    w = Tensor(pinned, QType::F16, w.ndim, w.shape, /*on_device=*/false);
                    ctx.host_pinned_allocs.push_back(std::move(pin));
                }
            };
            convert_host_bf16(L.expert_w_gate);
            convert_host_bf16(L.expert_w_up);
            convert_host_bf16(L.expert_w_down);

            // Host-resident NVFP4 experts: copies mmap'd bytes into pinned host memory (same reason as
            // the packed-GGUF path - WSL2 can't page-lock an mmap in place, so cudaMemcpyAsync from it
            // stages synchronously in-driver at far lower bandwidth). ONE pinned buffer per projection,
            // not one per expert: pinning is syscall-heavy, and a per-expert buffer costs orders of
            // magnitude more time in cudaHostAlloc/cudaFreeHost than the transfer time it saves; a
            // per-projection buffer also lands the experts contiguously as a side benefit.
            auto pin_host_nvfp4_experts = [&](std::vector<Tensor>& expert_vec) {
                if (!ctx.is_nvfp4_prequant || expert_vec.empty())
                    return;
                size_t total = 0;
                for (const Tensor& w : expert_vec) {
                    if (!w.data || w.on_device || w.qtype == QType::BF16)
                        return;  // mixed placement: leave the whole projection alone
                    total += w.nbytes();
                }
                if (total == 0)
                    return;
                // Host-resident NVFP4 experts reach this line, and for them the pinned slab is
                // not the opt-in trade moe.pin_host_experts describes: it is what the device
                // expert cache maps. Without it decode falls to the host LRU path and CUDA
                // graphs stay off - measured 2026-09-22 on Qwen3.8-Flash-Next, real prose:
                // 30.3 tok/s on the host path against 74.5-79.6 with the device cache.
                // Pinning is skipped only when the host cannot spare the pages.
                if (!process_diag_moe_pin_host_experts() && !g_pin_host_experts_fits)
                    return;
                pinner.pin(expert_vec, total);  // joined when upload_expert_weights returns
            };
            pin_host_nvfp4_experts(L.expert_w_gate);
            pin_host_nvfp4_experts(L.expert_w_up);
            pin_host_nvfp4_experts(L.expert_w_down);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Model::upload_weights_gpu
// ---------------------------------------------------------------------------

bool Model::upload_weights_gpu(QType compute_dtype, cudaStream_t stream, size_t expert_reserve_bytes,
                               bool warm_cache, const std::string& warm_cache_dir) {
    if (gpu_weights_ready_) {
        IMP_LOG_WARN("Weights already uploaded to GPU");
        return true;
    }

    IMP_LOG_INFO("Uploading model weights to GPU (%d layers)...", n_layers());

    // Initialize pinned staging for fast H2D (especially on WSL2 where mmap can't be pinned).
    // StagingGuard provides RAII cleanup on all exit paths (including early return false).
    struct StagingGuard {
        PinnedStager stager;
        ~StagingGuard() {
            g_stager = nullptr;
            stager.destroy();
            g_cached_free_mem = 0;
            g_total_allocated = 0;
            g_vram_reserve = 0;
            g_upload_log = nullptr;
            g_warm = nullptr;
        }
    } staging_guard;

    // Suspend-to-RAM: always log uploads (metadata only); consume an armed
    // warm snapshot for this pass if one matches (memory/weight_snapshot.h).
    if (!upload_log_)
        upload_log_ = std::make_unique<WeightUploadLog>();
    g_upload_log = upload_log_.get();
    if (WeightSnapshot* warm = weight_snapshot_take_armed()) {
        if (warm->matches(*this)) {
            g_warm = warm;
            IMP_LOG_INFO("Warm resume: weight snapshot armed (%.2f GiB host)",
                         warm->total_bytes() / (1024.0 * 1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN("Armed weight snapshot does not match this model (arch/layers) — ignoring");
        }
    }

    // On-disk warm cache (memory/weight_cache_file.h): consulted only when no
    // suspend snapshot is armed. Owns its blobs for the duration of this pass.
    std::unique_ptr<WeightSnapshot> disk_cache;
    if (!g_warm && warm_cache && !source_path_.empty()) {
        disk_cache = weight_cache_load(weight_cache_path_for(source_path_, warm_cache_dir), *this);
        if (disk_cache)
            g_warm = disk_cache.get();
    }
    const bool fully_cold = (g_warm == nullptr);

    // Use Engine's computed reserve (workspace + KV cache + SSM state + safety)
    // directly — no additional margin needed here.
    g_vram_reserve = expert_reserve_bytes;

    if (staging_guard.stager.init()) {
        g_stager = &staging_guard.stager;
        IMP_LOG_INFO("Pinned staging enabled (%dx %.0f MiB ring)", PinnedStager::ring(),
                     PinnedStager::chunk_size() / (1024.0 * 1024.0));
    } else {
        IMP_LOG_WARN("Pinned staging alloc failed, using default H2D path");
    }

    // Cache VRAM state to avoid per-tensor cudaMemGetInfo calls
    {
        size_t free_mem = 0, total_mem = 0;
        // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
        (void)vram_budget_mem_get_info(&free_mem, &total_mem);
        g_cached_free_mem = free_mem;
        g_total_allocated = 0;
        IMP_LOG_DEBUG("VRAM at upload start: %.2f GiB free / %.2f GiB total",
                      free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));
    }
    g_vram_reserve = pass1_upload_reserve(upload_batch_fit_, expert_reserve_bytes);

    // Qwen3.5/3.6 SafeTensors stores block-norm gammas as deltas (gamma = 1+W).
    // GGUF stores the post-+1 values directly. We bake the +1 in during the
    // BF16→FP16 upload conversion; F32-source norms (GGUF) are unaffected.
    const float arch_norm_offset = (config_.arch == ModelArch::QWEN35 ||
                                    config_.arch == ModelArch::QWEN35_MOE ||
                                    config_.arch == ModelArch::QWEN36_MOE ||
                                    config_.arch == ModelArch::QWEN4_EXP ||
                                    config_.arch == ModelArch::QWEN3_NEXT)
                                       ? 1.0f
                                       : 0.0f;
    const bool mxfp4_experts_deferred = (config_.arch == ModelArch::GPT_OSS);
    UploadCtx ctx{compute_dtype,
                  stream,
                  gpu_allocations_,
                  host_pinned_,
                  host_pinned_allocs_,
                  arch_norm_offset,
                  mxfp4_experts_deferred,
                  config_.is_nvfp4_prequant,
                  this};

    // --- Embeddings, output norm, output projection ---
    if (!upload_embeddings_and_output(tok_emb_, out_norm_, out_proj_, ctx)) {
        return false;
    }
    if (!upload_lm_head_row_scales(out_proj_, out_norm_, out_proj_row_scales_, ctx))
        return false;

    // --- Encoder embedder extras (#836, nomic-bert) ---
    for (auto* t : {&tok_emb_norm_, &tok_emb_norm_bias_, &token_types_}) {
        if (t->data && !t->on_device) {
            if (!upload_unquantized_weight(*t, QType::NONE, ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload encoder embedding norm/type tensor");
                return false;
            }
        }
    }

    // --- Qwen4Exp final hyper-connection mixer (BF16, no offset: the kernel applies 1 + W) ---
    for (auto* t : {&hc_mixer_norm_, &hc_mixer_down_, &hc_mixer_up_}) {
        if (t->data && !t->on_device) {
            if (!upload_unquantized_weight(*t, QType::NONE, ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload hyper_connection_mixer tensor");
                return false;
            }
        }
    }

    // Two-pass upload: Pass 1 uploads all non-expert per-layer weights (attention, FFN, norms,
    // SSM, shared experts, routing), whose VRAM cost is hard to estimate up front. Pass 2 reads
    // actual remaining VRAM via cudaMemGetInfo after Pass 1 and greedily uploads expert layers
    // until the budget is exhausted.

    // --- Pass 1: Non-expert per-layer weights (expert weights are uploaded in Pass 2 below) ---
    if (!upload_pass1_layers(*this, ctx))
        return false;

    // Sync Pass 1 before measuring free VRAM for expert budget
    if (!sync_upload(stream, "pass-1"))
        return false;
    expert_reserve_bytes = fit_batch_after_pass1(upload_batch_fit_, expert_reserve_bytes);

    // Reset cached VRAM state — the dense upload pass consumed an unknown amount
    // of VRAM (including CUDA driver overhead, page tables, alignment).
    // Expert upload must use real cudaMemGetInfo for accurate budget enforcement.
    g_cached_free_mem = 0;
    g_total_allocated = 0;


    if (!upload_expert_weights(layers_, n_layers(), expert_reserve_bytes, ctx)) {
        return false;
    }

    // Upload NVFP4 pre-quantized scale tensors. Single scratch map keyed by
    // canonical slot name; replaced the per-layer NvFP4PreQuantWeight slots.
    if (config_.is_nvfp4_prequant) {
        int scale_count = 0;
        // Diagnostic (diagnostics.audit_nvfp4_scales): dumps per-slot weight_scale_2 (tensor-level
        // FP32 scalar) stats before upload, to bisect NVFP4 long-form bugs by comparing scale
        // ranges against a known-good model.
        const bool audit = imp::process_diag_audit_nvfp4_scales();
        float ws2_min = 1e30f, ws2_max = -1e30f, ws2_sum = 0.0f;
        int ws2_count = 0, ws2_zero = 0;
        std::vector<std::pair<std::string, float>> ws2_samples;
        // Same stats for input_scale (FP32 scalar per Linear, optional).
        float is_min = 1e30f, is_max = -1e30f, is_sum = 0.0f;
        int is_count = 0, is_zero = 0, is_present = 0;
        int is_scalar_count = 0, is_per_channel_count = 0;
        struct InputScaleSample {
            std::string name;
            int ndim;
            int64_t shape[4];
            size_t numel;
            float first_val;
        };
        std::vector<InputScaleSample> is_samples;
        if (audit) {
            ws2_samples.reserve(8);
            is_samples.reserve(8);
        }
        auto upload_scale = [&](Tensor& t) {
            if (!t.data || t.on_device || t.numel() == 0)
                return;
            void* d_ptr = scale_to_device(t.data, t.nbytes(), stream, gpu_allocations_);
            if (!d_ptr)
                return;  // stays on host
            t.data = d_ptr;
            t.on_device = true;
            scale_count++;
        };
        // NVFP4 MoE expert micro-scales: uploads one CONTIGUOUS slab per (layer, projection) so
        // experts[e].scales = base + e*e_ms. Otherwise each expert's weight_scale is a separate
        // cudaMallocAsync landing at non-adjacent addresses, forcing the decode-cache borrow to
        // allocate a contiguous copy and free the scattered originals at load time (#679 copy/free
        // dance). A single slab makes that borrow zero-copy; per-expert sub-blocks stay
        // byte-identical to individual uploads.
        {
            int moe_scale_slabs = 0;
            auto slab_proj_scales = [&](std::vector<Tensor>& experts, int layer, const char* kind) {
                const int N = static_cast<int>(experts.size());
                if (N == 0)
                    return;
                // Experts that stayed on host keep their micro-scales there: the expert cache stages both
                // halves of an NVFP4 expert into one slot, so a scale in VRAM would be the wrong address
                // space and wasted VRAM on a layer moved out precisely because it didn't fit. They are
                // still pinned, in ONE slab per projection, for the same per-allocation discipline as the
                // weights.
                if (!experts[0].on_device) {
                    size_t total = 0;
                    std::vector<Tensor*> host_scales;
                    host_scales.reserve(N);
                    for (int e = 0; e < N; ++e) {
                        std::string k = "L" + std::to_string(layer) + "." + kind + "." +
                                        std::to_string(e);
                        auto it = nvfp4_scratch_.find(k);
                        if (it == nvfp4_scratch_.end())
                            return;
                        Tensor& ws = it->second.weight_scale;
                        if (!ws.data || ws.on_device || ws.numel() == 0)
                            return;
                        total += ws.nbytes();
                        host_scales.push_back(&ws);
                    }
                    if (total == 0)
                        return;
                    // Same rule as the weights above: the device expert cache reads these
                    // through a device view, so an unpinned scale keeps the whole layer on
                    // the host LRU path. Skipped only when the host has no RAM to spare.
                    if (!process_diag_moe_pin_host_experts() && !g_pin_host_experts_fits)
                        return;  // stays on host, unpinned
                    PinnedBuffer pin = PinnedBuffer::acquire(cuda_host_pinned_allocator(), total, HostPinnedKind::Mapped);
                    if (pin.empty())
                        return;  // stays on mmap: slower, still correct
                    char* dst = static_cast<char*>(pin.data());
                    for (Tensor* ws : host_scales) {
                        const size_t b = ws->nbytes();
                        std::memcpy(dst, ws->data, b);
                        ws->data = dst;
                        dst += b;
                    }
                    host_pinned_allocs_.push_back(std::move(pin));
                    return;
                }
                std::vector<NvFP4PreQuantWeight*> grp(N, nullptr);
                size_t e_ms = 0;
                for (int e = 0; e < N; ++e) {
                    std::string k =
                        "L" + std::to_string(layer) + "." + kind + "." + std::to_string(e);
                    auto it = nvfp4_scratch_.find(k);
                    if (it == nvfp4_scratch_.end())
                        return;  // not NVFP4-prequant experts — leave to per-tensor upload
                    Tensor& ws = it->second.weight_scale;
                    if (!ws.data || ws.on_device || ws.numel() == 0)
                        return;
                    size_t b = ws.nbytes();
                    if (e == 0)
                        e_ms = b;
                    else if (b != e_ms)
                        return;  // ragged — bail, per-tensor upload still correct
                    grp[e] = &it->second;
                }
                if (e_ms == 0)
                    return;
                void* slab = nullptr;
                if (cudaMallocAsync(&slab, static_cast<size_t>(N) * e_ms, stream) != cudaSuccess)
                    return;
                gpu_allocations_.push_back(slab);  // single base alloc; ~Model frees it once
                moe_scale_slabs += upload_scale_slab(slab, grp, e_ms, stream, scale_count);
            };
            for (size_t i = 0; i < layers_.size(); ++i) {
                slab_proj_scales(layers_[i].expert_w_gate, static_cast<int>(i), "expert_w_gate");
                slab_proj_scales(layers_[i].expert_w_up, static_cast<int>(i), "expert_w_up");
                slab_proj_scales(layers_[i].expert_w_down, static_cast<int>(i), "expert_w_down");
            }
            if (moe_scale_slabs > 0)
                IMP_LOG_INFO("NVFP4 MoE expert micro-scales: %d contiguous (layer,proj) slabs "
                             "(zero-copy decode borrow, no load-time scale copy)",
                             moe_scale_slabs);
        }
        for (auto& [name, sc] : nvfp4_scratch_) {
            // Audit weight_scale_2 BEFORE upload (it's still a host pointer).
            if (audit && sc.weight_scale_2.data && !sc.weight_scale_2.on_device) {
                size_t n = sc.weight_scale_2.numel();
                const float* p = static_cast<const float*>(sc.weight_scale_2.data);
                for (size_t i = 0; i < n; ++i) {
                    float v = p[i];
                    if (v == 0.0f)
                        ws2_zero++;
                    if (v < ws2_min)
                        ws2_min = v;
                    if (v > ws2_max)
                        ws2_max = v;
                    ws2_sum += v;
                    ws2_count++;
                }
                if (ws2_samples.size() < 8) {
                    ws2_samples.emplace_back(name, p[0]);
                }
            }
            if (audit && sc.input_scale.data && !sc.input_scale.on_device) {
                is_present++;
                size_t n = sc.input_scale.numel();
                if (n <= 1)
                    is_scalar_count++;
                else
                    is_per_channel_count++;
                const float* p = static_cast<const float*>(sc.input_scale.data);
                for (size_t i = 0; i < n; ++i) {
                    float v = p[i];
                    if (v == 0.0f)
                        is_zero++;
                    if (v < is_min)
                        is_min = v;
                    if (v > is_max)
                        is_max = v;
                    is_sum += v;
                    is_count++;
                }
                if (is_samples.size() < 8) {
                    InputScaleSample s;
                    s.name = name;
                    s.ndim = sc.input_scale.ndim;
                    for (int d = 0; d < s.ndim && d < 4; ++d)
                        s.shape[d] = sc.input_scale.shape[d];
                    s.numel = n;
                    s.first_val = p[0];
                    is_samples.push_back(std::move(s));
                }
            }
            // Same rule as the slab path: a scale whose weight is host-resident stays on host, for
            // experts declined for ragged shapes or a bailed-out projection. Asking the weight itself
            // rather than tracking a parallel flag keeps the two in agreement (the failure mode behind
            // both #1384 and #1403).
            bool weight_is_host_expert = false;
            if (const auto ek = parse_expert_key(name); ek.valid && ek.layer < n_layers()) {
                const auto& L = layers_[ek.layer];
                const std::vector<Tensor>& vec = (ek.kind == NvFP4ExpertKey::Kind::Gate) ? L.expert_w_gate
                                                 : (ek.kind == NvFP4ExpertKey::Kind::Up) ? L.expert_w_up
                                                                                         : L.expert_w_down;
                if (ek.expert < static_cast<int>(vec.size()))
                    weight_is_host_expert = (vec[ek.expert].data != nullptr && !vec[ek.expert].on_device);
            }
            // A host expert's scale stays on host. The slab pass above already
            // pinned it per projection; if that pass declined (ragged shapes,
            // missing entry) it stays on the mmap, which is slower but correct.
            if (!weight_is_host_expert) {
                upload_scale(sc.weight_scale);
            }
            upload_scale(sc.weight_scale_2);
            // input_scale is loaded for diagnostics but never read by any GEMM kernel (see
            // executor_pre_dequant.cpp Phase 0). Only uploaded when audit mode is on, to avoid burning
            // VRAM on a tensor never used in production.
            if (audit) {
                upload_scale(sc.input_scale);
            }
        }
        if (audit && ws2_count > 0) {
            IMP_LOG_INFO(
                "NVFP4 audit: weight_scale_2 stats — count=%d zeros=%d "
                "min=%.6g max=%.6g mean=%.6g",
                ws2_count, ws2_zero, ws2_min, ws2_max, ws2_sum / ws2_count);
            for (auto& [n, v] : ws2_samples) {
                IMP_LOG_INFO("  sample: %s = %.6g", n.c_str(), v);
            }
        }
        if (audit) {
            if (is_count > 0) {
                IMP_LOG_INFO(
                    "NVFP4 audit: input_scale present in %d/%zu Linears "
                    "(scalar=%d per_channel=%d), "
                    "stats — count=%d zeros=%d min=%.6g max=%.6g mean=%.6g",
                    is_present, nvfp4_scratch_.size(),
                    is_scalar_count, is_per_channel_count,
                    is_count, is_zero, is_min, is_max, is_sum / is_count);
                for (auto& s : is_samples) {
                    char shape_str[64];
                    int off = 0;
                    off += snprintf(shape_str + off, sizeof(shape_str) - off, "[");
                    for (int d = 0; d < s.ndim; ++d) {
                        off += snprintf(shape_str + off, sizeof(shape_str) - off,
                                        "%s%lld", d > 0 ? "," : "", (long long)s.shape[d]);
                    }
                    snprintf(shape_str + off, sizeof(shape_str) - off, "]");
                    IMP_LOG_INFO(
                        "  sample: %s.input_scale ndim=%d shape=%s numel=%zu first=%.6g",
                        s.name.c_str(), s.ndim, shape_str, s.numel, s.first_val);
                }
            } else {
                IMP_LOG_INFO(
                    "NVFP4 audit: no input_scale tensors found "
                    "(model uses purely dynamic input act-quant)");
            }
        }
        if (scale_count > 0)
            IMP_LOG_INFO("NVFP4 prequant: uploaded %d scale tensors to GPU", scale_count);
    }

    // MTP head weights (DeepSeek-V3/Qwen3.6 sidecar, optional), Phase 2 of MTP wiring; the
    // consuming forward path is Phase 3+. Gates on VRAM here: if the upload fails, MTP is
    // disabled rather than failing the whole model load.
    if (mtp_.has_value() && mtp_->loaded) {
        // Refuse a head that does not fit BEFORE uploading any of it. See
        // mtp_upload_peak_bytes: a per-allocation refusal partway through
        // strands everything already uploaded for the life of the process.
        size_t mtp_free = 0, mtp_total = 0;
        const size_t mtp_need = mtp_upload_peak_bytes(*mtp_);
        const bool have_reading = vram_budget_mem_get_info(&mtp_free, &mtp_total);
        const size_t mtp_headroom = vram_allocator_headroom(mtp_total);
        if (have_reading && mtp_free < mtp_need + mtp_headroom) {
            IMP_LOG_WARN(
                "MTP head: needs %zu MiB plus %zu MiB allocator headroom, %zu MiB free. "
                "Skipping the head, spec-decode disabled.",
                mtp_need / (size_t{1024} * 1024), mtp_headroom / (size_t{1024} * 1024),
                mtp_free / (size_t{1024} * 1024));
            mtp_->loaded = false;
        }
    }
    if (mtp_.has_value() && mtp_->loaded) {
        size_t allocs_before = gpu_allocations_.size();
        size_t mtp_free_before = 0;
        // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
        (void)vram_budget_mem_get_info(&mtp_free_before, nullptr);
        if (upload_mtp_weights(*mtp_, ctx)) {
            // Report what the device actually gave up, not the size on disk:
            // the restacking copy makes the second number roughly 2.5x the
            // first, and the file size read like the whole cost.
            size_t mtp_free_after = 0;
            // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
            (void)vram_budget_mem_get_info(&mtp_free_after, nullptr);
            const size_t mtp_used = mtp_free_before > mtp_free_after ? mtp_free_before - mtp_free_after : 0;
            IMP_LOG_INFO(
                "MTP head: uploaded to GPU (%zu allocations, %zu MiB of device free, "
                "%.2f GiB on disk)",
                gpu_allocations_.size() - allocs_before, mtp_used / (size_t{1024} * 1024),
                static_cast<double>(mtp_->info.file_bytes) / (1024.0 * 1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN("MTP head: GPU upload failed — spec-decode disabled");
            mtp_->loaded = false;
        }
    }

    // Final sync
    if (!sync_upload(stream, "final"))
        return false;

    gpu_weights_ready_ = true;
    last_warm_hits_ = g_warm ? g_warm->hits() : 0;
    if (g_warm)
        IMP_LOG_INFO("Warm restore: %d uploads served from %s", g_warm->hits(),
                     disk_cache ? "the on-disk warm cache" : "the suspend snapshot");

    // Fully-cold load: persist the transformed records so the NEXT boot skips
    // the conversion work. Best-effort; skipped for models whose device
    // sources were mutated in place during upload (Gemma-4 fused split).
    if (warm_cache && fully_cold && !device_sources_mutated_ && !source_path_.empty())
        (void)weight_cache_write(*this, weight_cache_path_for(source_path_, warm_cache_dir));  // best-effort, logs

    IMP_LOG_INFO("All model weights uploaded to GPU (%zu allocations)", gpu_allocations_.size());
    return true;
}

#undef UPLOAD_OR_FAIL
#undef UPLOAD_OR_FAIL_RAW
#undef UPLOAD_UNQUANT_OR_FAIL

}  // namespace imp
