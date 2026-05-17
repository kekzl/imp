#include "graph/gemm_kernel_registry.h"

#include "compute/gemm.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "graph/executor.h"  // FP8CacheEntry
#include "quant/dequant_gpu.h"
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

// ---------------------------------------------------------------------------
// FP8 cache-miss fallback — R5 Slice 8.1.
//
// Migrates the dequant→FP16 cuBLAS fallback that fires when the FP8 path is
// enabled (fp8_cache != nullptr, M>1, activation scratch present) but the
// specific weight is NOT in `WeightCaches::fp8`. The legacy branch lives at
// executor_kernels.cu:2287-2294 inside the `else if (fp8_cache != nullptr ...)`
// gate:
//
//   } else if (dequant_scratch != nullptr && dequant_gpu_supported(qtype)) {
//       dequant_gpu(weight.data, dequant_scratch, qtype, rows, cols, stream);
//       Tensor w_fp16(dequant_scratch, QType::F16, weight.ndim, weight.shape, true);
//       gemm(input, w_fp16, output, 1.0f, beta, stream);
//   } else {
//       gemm(input, weight, output, 1.0f, beta, stream);  // raw FP16/BF16
//   }
//
// The raw-fallback case (line 2294) is ALREADY covered by the FP16 strategy
// registered in Slice 1 (gemm_kernel_registry.cu) — when the dispatch site
// observes a cache-miss + FP16/BF16 weight, it falls through to the FP16
// strategy lookup just below the FP8 block. Slice 8.1 therefore migrates
// only the dequant fallback.
//
// Strategy key: (FP8, NONE, m_is_one=false). The qtype-agnostic NONE key is
// used because the handler doesn't switch on qtype — it just reads
// `weight.qtype`, checks `dequant_gpu_supported`, and calls `dequant_gpu`
// with the qtype. Registering one strategy per FP8-eligible weight qtype
// (Q4_K / Q5_K / Q6_K / Q8_0 / Q5_1 / ...) would multiply entries without
// changing dispatch behavior. The dispatch site emits NONE explicitly to
// signal "FP8-eligible, cache missed, try dequant fallback".
//
// Preconditions checked here:
//   - input != nullptr, output != nullptr, weight_payload != nullptr (weight Tensor)
//   - dequant_scratch != nullptr
//   - dequant_gpu_supported(weight.qtype)
// On any precondition fail the kernel returns PreconditionFail so the
// dispatch site falls through to the FP16 strategy (which handles raw
// FP16/BF16 weights) or eventually to legacy `gemm_dispatch_impl`.
// ---------------------------------------------------------------------------
static GemmDispatchResult fp8_cache_miss_dequant_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "fp8_cache_miss_dequant_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "fp8_cache_miss_dequant_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr,
              "fp8_cache_miss_dequant_kernel: weight_payload is null");

    if (args.dequant_scratch == nullptr)
        return GemmDispatchResult::PreconditionFail;

    const Tensor& weight = *static_cast<const Tensor*>(args.weight_payload);
    if (!dequant_gpu_supported(weight.qtype))
        return GemmDispatchResult::PreconditionFail;

    // Mirror executor_kernels.cu:2287-2292 verbatim — dequant the raw quant
    // bytes into the FP16 scratch, then wrap a Tensor view and call cuBLAS.
    const int rows = static_cast<int>(weight.shape[0]);
    const int cols = static_cast<int>(weight.shape[1]);
    dequant_gpu(weight.data, args.dequant_scratch, weight.qtype, rows, cols, args.stream);
    Tensor w_fp16(args.dequant_scratch, QType::F16, weight.ndim, weight.shape, /*on_device=*/true);
    gemm(*args.input, w_fp16, *args.output, /*alpha=*/1.0f, args.beta, args.stream);
    return GemmDispatchResult::Ok;
}

// Static registration. Slice 2 registers the (M>1, cache-hit) prefill
// strategy; Slice 8.1 adds the (M>1, cache-miss dequant fallback) sibling
// under the same FP8 tier. Decode (M==1) still has no FP8 strategy because
// decode uses GEMV fast paths upstream, not this dispatch.
namespace {
struct FP8Registration {
    FP8Registration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::FP8, QType::F16, /*m_is_one=*/false}, &fp8_gemm_kernel);
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::FP8, QType::NONE, /*m_is_one=*/false},
            &fp8_cache_miss_dequant_kernel);
    }
};
static FP8Registration s_fp8_registration;
}  // namespace

}  // namespace imp
