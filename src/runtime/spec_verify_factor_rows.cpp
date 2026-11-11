// Factored rows of the batched verify: the full verify's spare (speculative.factored_spare) and the bound
// verify's draft 1 at 2 drafts (speculative.mtp_batch_verify_factor_row). See spec_verify_factor_rows.h.

#include "runtime/spec_verify_factor_rows.h"

#include "compute/gdn_factor.cuh"
#include "compute/ssm.h"
#include "core/logging.h"
#include "memory/ssm_state.h"
#include "memory/vram_allocator.h"
#include "model/model.h"
#include "runtime/config.h"
#include "runtime/spec_batch_verify_state.h"

#include <algorithm>

namespace imp {

static_assert(bound_verify_factor_row_bytes(1, 1, 1, 128, 128, 0) ==
                  (gdn_factor_floats_per_head(128, 128) * sizeof(float)) + sizeof(int32_t),
              "bound_verify_factor_row_bytes prices the factor row layout of gdn_factor.cuh");

bool mtp_bound_verify_factor_row(const RuntimeConfig& cfg) {
    return cfg.speculative.mtp_batch_verify_factor_row &&
           mtp_bound_verify_depth(cfg.speculative.mtp_batch_verify_k) == 2;
}

size_t mtp_bound_verify_factor_row_plan_bytes(const RuntimeConfig& cfg, const Model* model,
                                              int max_batch_size, int slots) {
    if (!mtp_bound_verify_factor_row(cfg) || mtp_bound_verify_rows(cfg, model, max_batch_size) == 0 ||
        slots <= 0)
        return 0;
    const auto& mc = model->config();
    const int heads = mc.ssm_dt_rank;
    if (heads <= 0)
        return 0;
    int layers = 0;
    for (int i = 0; i < mc.n_layers; i++)
        layers += model->layer(i).ssm_in.data != nullptr ? 1 : 0;
    return bound_verify_factor_row_bytes(layers, slots, heads, mc.ssm_state_size, mc.ssm_inner_size / heads,
                                         mc.ssm_conv_channels());
}

bool ensure_factored_rows(RuntimeConfig& cfg, const ModelConfig& mc, SSMState* ssm, VRAMAllocator& alloc,
                          BatchVerifyState& bv, cudaStream_t stream) {
    if (!cfg.speculative.factored_spare || bv.d_fac != nullptr)
        return true;
    const int layers = ssm ? ssm->n_ssm_layers() : 0;
    const int slots = ssm ? ssm->max_sequences() : 0;
    const int heads = mc.ssm_dt_rank;
    const int hd = heads > 0 ? mc.ssm_inner_size / heads : 0;
    const int ss = mc.ssm_state_size;
    const int ch = mc.ssm_conv_channels();
    if (layers <= 0 || slots <= 0 || heads <= 0 || hd <= 0 || ss <= 0 || ch <= 0) {
        IMP_LOG_WARN("speculative.factored_spare ignored: no GDN state geometry");
        cfg.speculative.factored_spare = false;
        return true;
    }
    bv.fac_stride = gdn_factor_floats_per_head(ss, hd);
    bv.fac_layer_stride = static_cast<int64_t>(slots) * heads * bv.fac_stride;
    bv.tap_layer_stride = static_cast<int64_t>(slots) * ch;
    const size_t fac_bytes = static_cast<size_t>(layers) * bv.fac_layer_stride * sizeof(float);
    const size_t tap_bytes = static_cast<size_t>(layers) * bv.tap_layer_stride * sizeof(uint16_t);
    bv.d_fac = static_cast<float*>(alloc.allocate(fac_bytes, "spec_factored_rows"));
    bv.d_tap = alloc.allocate(tap_bytes, "spec_factored_taps");
    bv.d_pending = static_cast<int*>(
        alloc.allocate(static_cast<size_t>(slots) * sizeof(int), "spec_factored_pending"));
    bv.d_clear = static_cast<int*>(
        alloc.allocate(static_cast<size_t>(slots) * sizeof(int), "spec_factored_clear"));
    if (bv.d_fac == nullptr || bv.d_tap == nullptr || bv.d_pending == nullptr || bv.d_clear == nullptr) {
        IMP_LOG_WARN("speculative.factored_spare ignored: %.1f MiB of factored rows would not fit",
                     (fac_bytes + tap_bytes) / (1024.0 * 1024.0));
        free_factored_rows(alloc, bv);
        cfg.speculative.factored_spare = false;
        return true;
    }
    bv.h_pending = PinnedBuffer::acquire(cuda_host_pinned_allocator(), slots * sizeof(int32_t));
    bv.h_clear = PinnedBuffer::acquire(cuda_host_pinned_allocator(), slots * sizeof(int32_t));
    if (bv.h_pending.empty() || bv.h_clear.empty()) {
        IMP_LOG_WARN("speculative.factored_spare ignored: pinned slot staging unavailable");
        free_factored_rows(alloc, bv);
        cfg.speculative.factored_spare = false;
        return true;
    }
    // On the verify stream: a legacy-stream memset is not ordered before its kernels (#2275).
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(bv.d_fac, 0, fac_bytes, stream));
    IMP_LOG_INFO(
        "speculative.factored_spare: %d layers x %d slots, %.1f MiB of rows + %.1f MiB of conv "
        "taps (a spare slot pool would be %.1f MiB)",
        layers, slots, fac_bytes / (1024.0 * 1024.0), tap_bytes / (1024.0 * 1024.0),
        static_cast<double>(ssm->h_bytes()) * layers * slots / (1024.0 * 1024.0));
    return true;
}

void free_factored_rows(VRAMAllocator& alloc, BatchVerifyState& bv) {
    if (bv.d_fac)
        alloc.free(bv.d_fac);
    if (bv.d_tap)
        alloc.free(bv.d_tap);
    if (bv.d_pending)
        alloc.free(bv.d_pending);
    if (bv.d_clear)
        alloc.free(bv.d_clear);
    bv.d_fac = nullptr;
    bv.d_tap = nullptr;
    bv.d_pending = nullptr;
    bv.d_clear = nullptr;
    bv.pending.clear();
    bv.clear_slots.clear();
    bv.h_pending = PinnedBuffer{};
    bv.h_clear = PinnedBuffer{};
}

bool bound_verify_factor_row_alloc(const ModelConfig& mc, SSMState* ssm, VRAMAllocator& alloc,
                                   BatchVerifyState& bv, cudaStream_t stream) {
    if (bv.d_fac != nullptr)
        return true;
    const int layers = ssm ? ssm->n_ssm_layers() : 0;
    const int slots = ssm ? ssm->max_sequences() + ssm->n_reserved() : 0;
    const int heads = mc.ssm_dt_rank;
    const int hd = heads > 0 ? mc.ssm_inner_size / heads : 0;
    const int ss = mc.ssm_state_size;
    const int ch = mc.ssm_conv_channels();
    const bool dtype_ok = ssm && (ssm->h_dtype() == QType::F32 || ssm->h_dtype() == QType::BF16);
    if (layers <= 0 || slots <= 0 || heads <= 0 || hd <= 0 || ss <= 0 || ch <= 0 || !dtype_ok) {
        IMP_LOG_WARN("speculative.mtp_batch_verify_factor_row: no GDN state geometry for factor rows");
        return false;
    }
    bv.fac_stride = gdn_factor_floats_per_head(ss, hd);
    bv.fac_layer_stride = static_cast<int64_t>(slots) * heads * bv.fac_stride;
    bv.tap_layer_stride = static_cast<int64_t>(slots) * ch;
    const size_t fac_bytes = static_cast<size_t>(layers) * bv.fac_layer_stride * sizeof(float);
    const size_t tap_bytes = static_cast<size_t>(layers) * bv.tap_layer_stride * sizeof(uint16_t);
    const size_t list_bytes = static_cast<size_t>(slots) * sizeof(int32_t);
    bv.d_fac = static_cast<float*>(alloc.allocate(fac_bytes, "spec_bound_fac_rows"));
    bv.d_tap = alloc.allocate(tap_bytes, "spec_bound_fac_taps");
    bv.d_clear = static_cast<int*>(alloc.allocate(list_bytes, "spec_bound_fac_folds"));
    bv.h_clear = PinnedBuffer::acquire(cuda_host_pinned_allocator(), list_bytes);
    if (bv.d_fac == nullptr || bv.d_tap == nullptr || bv.d_clear == nullptr || bv.h_clear.empty()) {
        IMP_LOG_WARN(
            "speculative.mtp_batch_verify_factor_row: %.1f MiB of factor rows and taps would not fit",
            (fac_bytes + tap_bytes) / (1024.0 * 1024.0));
        free_factored_rows(alloc, bv);
        return false;
    }
    // On the verify stream: a legacy-stream memset is not ordered before its kernels (#2275).
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(bv.d_fac, 0, fac_bytes, stream));
    IMP_LOG_INFO(
        "speculative.mtp_batch_verify_factor_row: %d layers x %d slots, %.1f MiB of factor rows + %.1f "
        "MiB of conv taps (a second spare per row is %.1f MiB each)",
        layers, slots, fac_bytes / (1024.0 * 1024.0), tap_bytes / (1024.0 * 1024.0),
        static_cast<double>(ssm->slot_stride_bytes()) / (1024.0 * 1024.0));
    return true;
}

void bound_verify_factor_row_fold(const ModelConfig& mc, SSMState& ssm, BatchVerifyState& bv, const int* live,
                                  const int* fold, int n_groups, cudaStream_t stream) {
    if (!bv.factor_row || bv.d_fac == nullptr)
        return;
    // Pinned twin: the list stays untouched until the next verify, which syncs this stream first.
    auto* h_live = static_cast<int32_t*>(bv.h_clear.data());
    int n = 0;
    for (int g = 0; g < n_groups; g++)
        if (fold[g] != 0)
            h_live[n++] = live[g];
    if (n == 0)
        return;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(bv.d_clear, h_live, n * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    const int layers = ssm.n_ssm_layers();
    const int heads = mc.ssm_dt_rank;
    const auto slot_stride = static_cast<int64_t>(ssm.slot_stride_bytes());
    const int64_t layer_stride = layers > 1 ? static_cast<char*>(ssm.h_state(0, 1)) -
                                                  static_cast<char*>(ssm.h_state(0, 0))
                                            : 0;
    ssm_conv_tap_fold(ssm.conv_state(0, 0), slot_stride, layer_stride, bv.d_tap, bv.tap_layer_stride, layers,
                      mc.ssm_conv_channels(), mc.ssm_conv_kernel, bv.d_clear, bv.d_clear, n, stream);
    gdn_factor_fold(ssm.h_state(0, 0), ssm.h_dtype() == QType::BF16, slot_stride, layer_stride, bv.d_fac,
                    bv.fac_layer_stride, layers, bv.d_clear, bv.d_clear, n, heads, mc.ssm_state_size,
                    mc.ssm_inner_size / heads, bv.fac_stride, stream);
}

}  // namespace imp
