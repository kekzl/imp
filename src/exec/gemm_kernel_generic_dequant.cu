#include "exec/gemm_kernel_registry.h"

#include "compute/gemm.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "quant/dequant_gpu.h"

namespace imp {

// Generic dequant->cuBLAS catch-all: fires after every tier-specific dispatcher
// (MXFP4/NVFP4/CUTLASS/FP8/FP16 cache/GGUF small-M) returns NoMatch or PreconditionFail
// and the weight qtype is dequantable. Strategy key (FP16, NONE, m_is_one=false): NONE
// is qtype-agnostic, the handler reads weight.qtype and dispatches dequant_gpu itself.
// Preconditions: input/output/weight_payload/dequant_scratch non-null,
// dequant_gpu_supported(weight.qtype). Failure returns PreconditionFail, falling through
// to legacy gemm_dispatch_impl's final raw gemm() arm (FP16/BF16, no dequant).
static GemmDispatchResult generic_dequant_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "generic_dequant_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "generic_dequant_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr,
              "generic_dequant_kernel: weight_payload is null");

    if (args.dequant_scratch == nullptr)
        return GemmDispatchResult::PreconditionFail;

    const Tensor& weight = *static_cast<const Tensor*>(args.weight_payload);
    if (!dequant_gpu_supported(weight.qtype))
        return GemmDispatchResult::PreconditionFail;

    // Mirror executor_kernels.cu:2315-2319 verbatim — dequant the raw quant
    // bytes into the FP16 scratch, then wrap a Tensor view and call cuBLAS.
    const int rows = static_cast<int>(weight.shape[0]);
    const int cols = static_cast<int>(weight.shape[1]);
    dequant_gpu(weight.data, args.dequant_scratch, weight.qtype, rows, cols, args.stream);
    Tensor w_fp16(args.dequant_scratch, QType::F16, weight.ndim, weight.shape, /*on_device=*/true);
    gemm(*args.input, w_fp16, *args.output, /*alpha=*/1.0f, args.beta, args.stream);
    return GemmDispatchResult::Ok;
}

// Slice 8.5 adds one (FP16, NONE, m_is_one=false) catch-all entry. No M==1 GEMV catch-all
// needed: M==1 is already covered by the per-qtype GGUF strategies and the FP16-cache
// path; this catch-all is only for M>1 prefill dequant->cuBLAS last resort.
namespace {
struct GenericDequantRegistration {
    GenericDequantRegistration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::FP16, QType::NONE, /*m_is_one=*/false},
            &generic_dequant_kernel);
    }
};
static GenericDequantRegistration s_generic_dequant_registration;
}  // namespace

}  // namespace imp
