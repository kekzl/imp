#include "exec/gemm_kernel_registry.h"

#include "core/logging.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"  // NvFP4QuantResult

namespace imp {

// ---------------------------------------------------------------------------
// NVFP4 GEMM tier — R5 Slice 4.
//
// Adapter that wraps the existing NVFP4 GEMM (M>1 prefill, dequant→cuBLAS
// fallback path) from gemm_dispatch_impl (executor_kernels.cu:2186-2188):
// call `gemm_nvfp4(res, input, output, stream)` which dequantizes the
// NVFP4 weight to FP16 and runs cuBLAS GEMM. Behavior is a verbatim wrap
// of the legacy call site — no algorithmic change.
//
// Strategy: tier=NVFP4, weight_qtype=F16 (engine observes FP16 source qtype
// for an NVFP4-cached weight; the NVFP4 conversion happened at load time
// and lives inside the NvFP4QuantResult), m_is_one=false. Slice 4 only
// covers the dequant fallback GEMM path; the CUTLASS sm_120 native FP4
// GEMM (gemm_nvfp4_cutlass_sm120) is a separate tier — that's Slice 5
// territory under StorageTier::CUTLASS_NVFP4.
//
// Preconditions checked at the dispatch site (executor_kernels.cu): the
// `nvfp4_view` pointer is non-null (built either from the prequant Tensor
// sidecars OR from the `WeightCaches::nvfp4` cache) and `M > 1`. If
// either is missing the dispatch site must NOT emit this strategy; the
// kernel returns PreconditionFail loud rather than silently falling back.
// ---------------------------------------------------------------------------
static GemmDispatchResult nvfp4_gemm_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "nvfp4_gemm_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "nvfp4_gemm_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "nvfp4_gemm_kernel: weight_payload is null");
    IMP_CHECK(args.input->shape[0] > 1, "nvfp4_gemm_kernel: M must be > 1 (use GEMV strategy for M==1)");

    const NvFP4QuantResult& res = *static_cast<const NvFP4QuantResult*>(args.weight_payload);
    const Tensor& input = *args.input;
    Tensor& output = *args.output;

    // Mirror executor_kernels.cu:2187 verbatim — same dequant→cuBLAS GEMM
    // call, same arg order. `gemm_nvfp4` internally dequantizes the NVFP4
    // weight to FP16 and dispatches to cuBLAS — no workspace pointer
    // required on the args struct.
    gemm_nvfp4(res, input, output, args.stream);
    return GemmDispatchResult::Ok;
}

// Static registration. Slice 4 registers only the (M>1) prefill GEMM
// strategy. The (M==1) decode GEMV path was registered by Slice 3.
namespace {
struct NvFP4GemmRegistration {
    NvFP4GemmRegistration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::NVFP4, QType::F16, /*m_is_one=*/false}, &nvfp4_gemm_kernel);
    }
};
static NvFP4GemmRegistration s_nvfp4_gemm_registration;
}  // namespace

}  // namespace imp
