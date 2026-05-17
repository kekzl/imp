#include "graph/gemm_kernel_registry.h"

#include "compute/gemm.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "graph/executor.h"  // FP8CacheEntry
#include "quant/fp8_quant.h"

namespace imp {

// ---------------------------------------------------------------------------
// FP8 tier — R5 Slice 2.
//
// Adapter that wraps the existing FP8 prefill path from gemm_dispatch_impl
// (executor_kernels.cu, M>1 branch): quantize FP16 activations to FP8_E4M3
// with per-tensor scale, then run a FP8xFP8 cuBLASLt GEMM via the cached
// pre-quantized FP8 weight. Behavior is a verbatim wrap of the legacy call
// site — no algorithmic change.
//
// Strategy: tier=FP8, weight_qtype=F16 (the source qtype the engine
// observes for an FP8-cached weight; the FP8 conversion happened at load
// time and lives inside FP8CacheEntry), m_is_one=false. M=1 / decode is
// handled by GEMV / dp4a fast paths elsewhere — this adapter is prefill
// (M>1) only, mirroring the legacy branch guard
// (`fp8_cache != nullptr && input.shape[0] > 1`).
//
// Preconditions checked at the dispatch site (executor_kernels.cu): the
// weight has an entry in `WeightCaches::fp8`, the FP8 activation scratch
// (`fp8_act_buf`, `d_act_scale`, `d_fp8_block_maxes`, `d_fp8_absmax`) is
// allocated, and M > 1. If any are missing the dispatch site must NOT
// emit an FP8 strategy; the kernel returns PreconditionFail loud rather
// than silently falling back.
// ---------------------------------------------------------------------------
static GemmDispatchResult fp8_gemm_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "fp8_gemm_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "fp8_gemm_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "fp8_gemm_kernel: weight_payload is null");

    // FP8 prefill needs the activation quant scratch + the device-side scale.
    // Without them we cannot run; the dispatch site is responsible for not
    // emitting an FP8 strategy when these are unavailable. Refuse loud.
    if (args.fp8_act_buf == nullptr || args.d_act_scale == nullptr)
        return GemmDispatchResult::PreconditionFail;

    const FP8CacheEntry& entry = *static_cast<const FP8CacheEntry*>(args.weight_payload);
    const Tensor& input = *args.input;
    Tensor& output = *args.output;

    // Mirror executor_kernels.cu:2278-2282 verbatim — same activation
    // quant kernel, same cuBLASLt call, same alpha/beta semantics.
    Tensor fp8_act(args.fp8_act_buf, QType::FP8_E4M3, input.ndim, input.shape, /*on_device=*/true);
    quantize_fp16_to_fp8_e4m3(input, fp8_act, args.d_act_scale, args.stream, args.d_fp8_block_maxes,
                              args.d_fp8_absmax, args.fp8_max_grid);
    gemm_cublaslt(fp8_act, entry.weight, output, /*alpha=*/1.0f, args.beta, args.d_act_scale, entry.d_scale,
                  args.stream);
    return GemmDispatchResult::Ok;
}

// Static registration. Slice 2 registers only the (M>1) prefill strategy
// because the legacy FP8 branch only fires for M>1 (decode uses GEMV
// fast paths, not this dispatch).
namespace {
struct FP8Registration {
    FP8Registration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::FP8, QType::F16, /*m_is_one=*/false}, &fp8_gemm_kernel);
    }
};
static FP8Registration s_fp8_registration;
}  // namespace

}  // namespace imp
