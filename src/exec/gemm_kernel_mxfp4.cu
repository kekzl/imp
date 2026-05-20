#include "exec/gemm_kernel_registry.h"

#include "compute/gemm.h"                       // gemm() FP16 cuBLAS wrapper
#include "compute/gemm_cutlass_mxfp4_sm120.h"  // CutlassMxFP4Weight, dequant_mxfp4_to_fp16
#include "compute/hadamard.h"                   // hadamard_block_size_valid, hadamard_transform_fp16
#include "core/logging.h"
#include "core/tensor.h"
#include "quant/mxfp4_gemm.h"                   // gemv_mxfp4_kpar

namespace imp {

// ---------------------------------------------------------------------------
// MXFP4 tiers — R5 Slice 6.
//
// Migrate the legacy MXFP4 GGUF dispatch (executor_kernels.cu:2085-2113) to
// the GemmKernel registry. Two distinct backends mirror the Slice 3/4 split
// for NVFP4 — GEMV for M==1 decode and dequant→cuBLAS for M>1 prefill — so
// we register two strategies under StorageTier::MXFP4:
//
//   {MXFP4, F16, m_is_one=true}  → gemv_mxfp4_kpar (with optional Hadamard
//                                  online rotation when hadamard_bs > 0).
//   {MXFP4, F16, m_is_one=false} → dequant_mxfp4_to_fp16 → cuBLAS gemm.
//
// The third candidate — the dual-cache MXFP4 CUTLASS branch at
// executor_kernels.cu:2149-2174 — is NOT migrated here. That branch lives
// *inside* the NVFP4 path, requires the same weight.data to be present in
// both `cutlass_nvfp4` and `cutlass_mxfp4` caches, and carries a `s_logged`
// QW7 instrumentation probe documenting that production firings are still
// under investigation. Migrating it would require a multi-tier strategy key
// (NVFP4 cache hit + MXFP4 cache hit) that doesn't fit the (tier, qtype,
// m_is_one) trio, and it stays on legacy until that branch is either
// retired or earns a first-class strategy slot.
//
// Both kernels gate on `linear_scales != nullptr` — same precondition as
// the legacy dispatch site (executor_kernels.cu:2087). When the gate fails
// we return PreconditionFail so the dispatch site falls through to the
// legacy switch.
// ---------------------------------------------------------------------------

static GemmDispatchResult mxfp4_gemv_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "mxfp4_gemv_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "mxfp4_gemv_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "mxfp4_gemv_kernel: weight_payload is null");
    IMP_CHECK(args.input->qtype == QType::F16, "mxfp4_gemv_kernel: input qtype must be F16");

    const CutlassMxFP4Weight& W = *static_cast<const CutlassMxFP4Weight*>(args.weight_payload);
    if (W.linear_scales == nullptr) {
        // Legacy: native MXFP4 GEMV requires linear_scales (sequential UE8M0
        // layout). If absent the dispatch site must fall back to legacy.
        return GemmDispatchResult::PreconditionFail;
    }

    // Mirror executor_kernels.cu:2088-2100 verbatim. Apply Hadamard online
    // rotation in-place on the FP16 input when hadamard_bs is set, then run
    // the kpar GEMV. `input.data` is a const half* but the legacy code does
    // an in-place rewrite — we preserve that semantic via a const_cast.
    const int K = static_cast<int>(W.K);
    if (W.hadamard_bs > 0 && hadamard_block_size_valid(W.hadamard_bs)) {
        hadamard_transform_fp16(reinterpret_cast<const half*>(args.input->data),
                                reinterpret_cast<half*>(const_cast<void*>(args.input->data)),
                                /*M=*/1, K, W.hadamard_bs, args.stream);
    }
    gemv_mxfp4_kpar(W, reinterpret_cast<const half*>(args.input->data),
                    reinterpret_cast<half*>(args.output->data), static_cast<int>(W.N),
                    K, args.stream);
    return GemmDispatchResult::Ok;
}

static GemmDispatchResult mxfp4_gemm_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "mxfp4_gemm_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "mxfp4_gemm_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "mxfp4_gemm_kernel: weight_payload is null");
    IMP_CHECK(args.input->qtype == QType::F16, "mxfp4_gemm_kernel: input qtype must be F16");
    IMP_CHECK(args.input->shape[0] > 1, "mxfp4_gemm_kernel: M must be > 1 (use GEMV strategy for M==1)");

    const CutlassMxFP4Weight& W = *static_cast<const CutlassMxFP4Weight*>(args.weight_payload);
    if (W.linear_scales == nullptr) {
        return GemmDispatchResult::PreconditionFail;
    }
    if (args.dequant_scratch == nullptr) {
        // Legacy prefill path (executor_kernels.cu:2101) requires the FP16
        // dequant scratch — without it the legacy switch silently skipped
        // this branch. We surface the same fall-through via PreconditionFail.
        return GemmDispatchResult::PreconditionFail;
    }

    // Mirror executor_kernels.cu:2102-2110: dequant MXFP4 → FP16 scratch,
    // wrap as a borrowed Tensor, run cuBLAS FP16 GEMM with the caller's beta.
    const int N = static_cast<int>(W.N);
    const int K = static_cast<int>(W.K);
    dequant_mxfp4_to_fp16(W.data, N, K, args.dequant_scratch, args.stream);
    int64_t w_shape[2] = {N, K};
    Tensor w_fp16(args.dequant_scratch, QType::F16, 2, w_shape, /*on_device=*/true);
    gemm(*args.input, w_fp16, *args.output, /*alpha=*/1.0f, args.beta, args.stream);
    return GemmDispatchResult::Ok;
}

// Static registration. Two strategies under StorageTier::MXFP4 — one for
// each side of the m_is_one axis. The dual-cache CUTLASS MXFP4 branch
// (executor_kernels.cu:2149-2174) stays on legacy (see file header).
namespace {
struct MxFP4Registration {
    MxFP4Registration() {
        auto& reg = GemmKernelRegistry::instance();
        reg.register_kernel(GemmStrategy{StorageTier::MXFP4, QType::F16, /*m_is_one=*/true},
                            &mxfp4_gemv_kernel);
        reg.register_kernel(GemmStrategy{StorageTier::MXFP4, QType::F16, /*m_is_one=*/false},
                            &mxfp4_gemm_kernel);
    }
};
static MxFP4Registration s_mxfp4_registration;
}  // namespace

}  // namespace imp
