#include "exec/gemm_kernel_registry.h"

#include "compute/gemm.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "quant/dequant_gpu.h"

namespace imp {

// ---------------------------------------------------------------------------
// Generic dequant→cuBLAS catch-all — R5 Slice 8.5.
//
// Migrates the qtype-agnostic large-M fallback at the bottom of the legacy
// gemm_dispatch_impl chain (executor_kernels.cu:2313-2319):
//
//   } else if (dequant_scratch != nullptr && dequant_gpu_supported(qtype)) {
//       int rows = static_cast<int>(weight.shape[0]);
//       int cols = static_cast<int>(weight.shape[1]);
//       dequant_gpu(weight.data, dequant_scratch, qtype, rows, cols, stream);
//       Tensor w_fp16(dequant_scratch, QType::F16, weight.ndim, weight.shape, true);
//       gemm(input, w_fp16, output, 1.0f, beta, stream);
//   }
//
// This branch is the qtype-agnostic generic dequant→FP16→cuBLAS GEMM path.
// It fires after every tier-specific dispatcher (MXFP4 / NVFP4 / CUTLASS /
// FP8 cache / FP16 cache / GGUF small-M) has either NoMatched or
// PreconditionFailed and the weight qtype is dequantable. The handler
// performs the same `dequant_gpu` + `gemm` sequence as the legacy branch —
// bit-identical behavior.
//
// Strategy key: (FP16, NONE, m_is_one=false). The FP16 tier is reused
// because the GEMM step itself runs on FP16 inputs (the dequant target);
// the qtype-agnostic NONE key mirrors the Slice 8.1 FP8 cache-miss
// adapter — the handler reads `weight.qtype` off the Tensor and switches
// internally via `dequant_gpu_supported` + `dequant_gpu`. Registering one
// strategy per dequant-eligible qtype (Q4_K / Q5_K / Q6_K / Q8_0 / ...)
// would multiply entries without changing dispatch behavior. The dispatch
// site emits NONE explicitly to signal "fallback: dequant any supported
// qtype to FP16 then cuBLAS".
//
// Preconditions checked here:
//   - input != nullptr, output != nullptr, weight_payload != nullptr
//   - dequant_scratch != nullptr
//   - dequant_gpu_supported(weight.qtype)
// On any precondition fail the kernel returns PreconditionFail so the
// dispatch site falls through to legacy `gemm_dispatch_impl` (which has
// a final raw `gemm()` arm at line 2322 for the FP16/BF16-no-dequant
// case).
// ---------------------------------------------------------------------------
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

// Static registration. Slice 8.5 adds one (FP16, NONE, m_is_one=false)
// entry — the qtype-agnostic generic-dequant catch-all. M==1 GEMV-style
// catch-all is unnecessary because the M==1 path is already covered by
// the per-qtype Slice 7 GGUF strategies and the FP16-cache path; the
// catch-all is needed only for M>1 prefill where dequant→cuBLAS is the
// last resort.
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
