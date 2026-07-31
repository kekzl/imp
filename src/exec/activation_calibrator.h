#pragma once

// ActivationCalibrator — per-input-channel activation magnitudes, collected
// from a live forward pass.
//
// Hooked into GraphExecutor::gemm_via_handle_, the single dispatch point every
// registered weight GEMM goes through. That placement is the point: it sees
// the activation each weight actually consumes, whatever tier or kernel the
// weight ends up on, so no call site has to be taught about calibration.
//
// Keyed by (layer, TensorKind), which is what an offline quantizer can map
// back onto checkpoint tensor names. Kinds that share an input (q/k/v off one
// norm; gate/up off another) simply record the same numbers — the consumer
// merges them.
//
// Off unless [calibration] enabled is set; the hook is a null-pointer test on
// the hot path. Collection is FP16-activations only, which is every dense
// prefill path.

#include "core/tensor.h"
#include "core/tensor_kind.h"
#include "memory/vram_allocator.h"
#include "quant/calibration_stats.h"

#include <cstdint>
#include <map>
#include <string>

namespace imp {

class ActivationCalibrator {
public:
    // `alloc` may be null in tests; then the collector allocates nothing and
    // stays empty rather than reaching for cudaMalloc behind the allocator's
    // back (docs/MEMORY_ARCHITECTURE.md A3 — every device allocation routes
    // through src/memory/).
    explicit ActivationCalibrator(VRAMAllocator* alloc) : alloc_(alloc) {}
    ~ActivationCalibrator();

    ActivationCalibrator(const ActivationCalibrator&) = delete;
    ActivationCalibrator& operator=(const ActivationCalibrator&) = delete;

    // Accumulates sum_over_rows |input[row][j]| into the (layer, kind) slot.
    // Silently ignores non-FP16 inputs and inputs that are not 2-D — those are
    // reported as missing entries by the consumer rather than as wrong numbers.
    void accumulate(int layer, TensorKind kind, const Tensor& input, cudaStream_t stream);

    // Copies the accumulators back and converts sums to per-channel means.
    // Empty when nothing was collected.
    CalibrationStats snapshot(const std::string& model_id) const;

    bool empty() const { return entries_.empty(); }
    size_t skipped_non_fp16() const { return skipped_non_fp16_; }

private:
    struct Entry {
        double* d_sum = nullptr;  // [K], FP64 so long corpora do not lose the tail
        int64_t K = 0;
        uint64_t rows = 0;
    };
    // key = layer * 256 + kind, both bounded well below that.
    std::map<uint32_t, Entry> entries_;
    size_t skipped_non_fp16_ = 0;
    VRAMAllocator* alloc_ = nullptr;
};

}  // namespace imp
