// Factor rows of the bound batched verify: its drafts before the last at 2+ drafts (#3088). See
// spec_verify_factor_rows.h.

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

static_assert(bound_verify_factor_row_bytes(1, 1, 1, 1, 128, 128, 0) ==
                  (gdn_factor_floats_per_head(128, 128) * sizeof(float)) + (3 * sizeof(int32_t)),
              "bound_verify_factor_row_bytes prices the factor row layout of gdn_factor.cuh");
static_assert(kMtpBoundVerifyMaxDepth - 1 <= kGdnFactorFoldMaxRows,
              "a partial accept folds up to depth - 1 rows in one gdn_factor_fold");

bool mtp_bound_verify_factor_row(const RuntimeConfig& cfg) { return cfg.speculative.mtp_batch_verify_k >= 2; }

size_t mtp_bound_verify_factor_row_plan_bytes(const RuntimeConfig& cfg, const Model* model,
                                              int max_batch_size) {
    const int groups = mtp_bound_verify_rows(cfg, model, max_batch_size);
    if (!mtp_bound_verify_factor_row(cfg) || groups <= 0)
        return 0;
    const auto& mc = model->config();
    const int heads = mc.ssm_dt_rank;
    if (heads <= 0)
        return 0;
    int layers = 0;
    for (int i = 0; i < mc.n_layers; i++)
        layers += model->layer(i).ssm_in.data != nullptr ? 1 : 0;
    const int levels = mtp_bound_verify_top_depth(cfg) - 1;
    return bound_verify_factor_row_bytes(layers, groups, levels, heads, mc.ssm_state_size,
                                         mc.ssm_inner_size / heads, mc.ssm_conv_channels());
}

void free_factored_rows(VRAMAllocator& alloc, BatchVerifyState& bv) {
    if (bv.d_fac)
        alloc.free(bv.d_fac);
    if (bv.d_tap)
        alloc.free(bv.d_tap);
    if (bv.d_clear)
        alloc.free(bv.d_clear);
    bv.d_fac = nullptr;
    bv.d_tap = nullptr;
    bv.d_clear = nullptr;
    bv.h_clear = PinnedBuffer{};
}

bool bound_verify_factor_row_alloc(const ModelConfig& mc, SSMState* ssm, VRAMAllocator& alloc,
                                   BatchVerifyState& bv, int groups, int levels, cudaStream_t stream) {
    if (bv.d_fac != nullptr)
        return true;
    const int layers = ssm ? ssm->n_ssm_layers() : 0;
    const int rows = groups * levels;
    const int heads = mc.ssm_dt_rank;
    const int hd = heads > 0 ? mc.ssm_inner_size / heads : 0;
    const int ss = mc.ssm_state_size;
    const int ch = mc.ssm_conv_channels();
    const bool dtype_ok = ssm && (ssm->h_dtype() == QType::F32 || ssm->h_dtype() == QType::BF16);
    if (layers <= 0 || groups <= 0 || levels <= 0 || levels > kGdnFactorFoldMaxRows || heads <= 0 ||
        hd <= 0 || ss <= 0 || ch <= 0 || !dtype_ok) {
        IMP_LOG_WARN("speculative.mtp_batch_verify_factor_row: no GDN state geometry for factor rows");
        return false;
    }
    bv.fac_stride = gdn_factor_floats_per_head(ss, hd);
    bv.fac_layer_stride = static_cast<int64_t>(rows) * heads * bv.fac_stride;
    bv.tap_layer_stride = static_cast<int64_t>(rows) * ch;
    const size_t fac_bytes = static_cast<size_t>(layers) * bv.fac_layer_stride * sizeof(float);
    const size_t tap_bytes = static_cast<size_t>(layers) * bv.tap_layer_stride * sizeof(uint16_t);
    const size_t list_bytes = 3 * static_cast<size_t>(groups) * sizeof(int32_t);
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
    bv.fac_levels = levels;
    IMP_LOG_INFO(
        "speculative.mtp_batch_verify_factor_row: %d layers x %d groups x %d draft(s), %.1f MiB of factor "
        "rows + %.1f MiB of conv taps (a second spare per row is %.1f MiB each)",
        layers, groups, levels, fac_bytes / (1024.0 * 1024.0), tap_bytes / (1024.0 * 1024.0),
        static_cast<double>(ssm->slot_stride_bytes()) / (1024.0 * 1024.0));
    return true;
}

void bound_verify_factor_row_fold(const ModelConfig& mc, SSMState& ssm, BatchVerifyState& bv, int levels,
                                  const int* live, const int* fold, int n_groups, cudaStream_t stream) {
    if (!bv.factor_row || bv.d_fac == nullptr || levels <= 0 || levels > bv.fac_levels)
        return;
    // Three lists of n (bound_verify_fold_lists). Pinned twin: untouched until the next verify, which syncs
    // this stream first.
    auto* h_list = static_cast<int32_t*>(bv.h_clear.data());
    const int n = bound_verify_fold_lists(levels, live, fold, n_groups, h_list);
    if (n == 0)
        return;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(bv.d_clear, h_list, 3ull * n * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    const int* d_live = bv.d_clear;
    const int* d_row = bv.d_clear + n;
    const int* d_count = bv.d_clear + (static_cast<ptrdiff_t>(n) * 2);
    const int layers = ssm.n_ssm_layers();
    const int heads = mc.ssm_dt_rank;
    const auto slot_stride = static_cast<int64_t>(ssm.slot_stride_bytes());
    const int64_t layer_stride = layers > 1 ? static_cast<char*>(ssm.h_state(0, 1)) -
                                                  static_cast<char*>(ssm.h_state(0, 0))
                                            : 0;
    ssm_conv_tap_fold(ssm.conv_state(0, 0), slot_stride, layer_stride, bv.d_tap, bv.tap_layer_stride, layers,
                      mc.ssm_conv_channels(), mc.ssm_conv_kernel, d_live, d_row, n, stream, d_count);
    gdn_factor_fold(ssm.h_state(0, 0), ssm.h_dtype() == QType::BF16, slot_stride, layer_stride, bv.d_fac,
                    bv.fac_layer_stride, layers, d_live, d_row, n, heads, mc.ssm_state_size,
                    mc.ssm_inner_size / heads, bv.fac_stride, stream, d_count);
}

}  // namespace imp
