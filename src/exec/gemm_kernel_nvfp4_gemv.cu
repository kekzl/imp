#include "exec/gemm_kernel_registry.h"

#include "core/logging.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"  // NvFP4QuantResult

namespace imp {

// ---------------------------------------------------------------------------
// NVFP4 GEMV tier — R5 Slice 3.
//
// Adapter that wraps the existing NVFP4 GEMV (M==1 decode) path from
// gemm_dispatch_impl (executor_kernels.cu:2136-2139): call
// `gemv_nvfp4_kpar` with raw FP16 input/output pointers and the cached
// `NvFP4QuantResult` (packed FP4 data + FP8 micro-scales + FP32 tensor
// scale). Behavior is a verbatim wrap of the legacy call site — no
// algorithmic change.
//
// Strategy: tier=NVFP4, weight_qtype=F16 (the engine observes FP16 source
// qtype for an NVFP4-cached weight; the NVFP4 conversion happened at load
// time and lives inside the NvFP4QuantResult), m_is_one=true. Slice 3
// only covers decode (M==1); the M>1 GEMM path stays on legacy until
// Slice 4 (NVFP4 GEMM).
//
// Preconditions checked at the dispatch site (executor_kernels.cu): the
// `nvfp4_view` pointer is non-null (built either from the prequant Tensor
// sidecars OR from the `WeightCaches::nvfp4` cache) and `M == 1`. If
// either is missing the dispatch site must NOT emit this strategy; the
// kernel returns PreconditionFail loud rather than silently falling back.
// ---------------------------------------------------------------------------
static GemmDispatchResult nvfp4_gemv_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "nvfp4_gemv_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "nvfp4_gemv_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "nvfp4_gemv_kernel: weight_payload is null");

    const NvFP4QuantResult& res = *static_cast<const NvFP4QuantResult*>(args.weight_payload);
    const Tensor& input = *args.input;
    Tensor& output = *args.output;

    // Mirror executor_kernels.cu:2137-2139 verbatim — same kernel, same
    // pointer casts, same N/K parameters from the NvFP4QuantResult.
    gemv_nvfp4_kpar(res, reinterpret_cast<const half*>(input.data), reinterpret_cast<half*>(output.data),
                    static_cast<int>(res.N), static_cast<int>(res.K), args.stream);
    return GemmDispatchResult::Ok;
}

// Static registration. Slice 3 registers only the (M==1) decode GEMV
// strategy. The (M>1) NVFP4 GEMM path stays on legacy until Slice 4.
namespace {
struct NvFP4GemvRegistration {
    NvFP4GemvRegistration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::NVFP4, QType::F16, /*m_is_one=*/true}, &nvfp4_gemv_kernel);
    }
};
static NvFP4GemvRegistration s_nvfp4_gemv_registration;
}  // namespace

}  // namespace imp
