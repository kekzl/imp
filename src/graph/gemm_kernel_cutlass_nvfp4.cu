#include "graph/gemm_kernel_registry.h"

#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight, gemm_nvfp4_cutlass_sm120, quantize_fp16_to_nvfp4_cutlass
#include "core/logging.h"
#include "core/tensor.h"

namespace imp {

// ---------------------------------------------------------------------------
// CUTLASS_NVFP4 tier — R5 Slice 5.
//
// Adapter that wraps the existing CUTLASS sm_120 block-scaled NVFP4 GEMM
// from gemm_dispatch_impl (executor_kernels.cu:2147-2183): quantize the FP16
// activation into the pre-allocated CUTLASS NVFP4 scratch (`cutlass_act_data`
// + `cutlass_act_sf`), then run `gemm_nvfp4_cutlass_sm120` against the
// pre-converted CUTLASS weight payload. This is the *preferred* NVFP4 GEMM
// path — used when the loader has populated `cutlass_nvfp4_cache` AND the
// workspace pointers are present; otherwise Slice 4's dequant fallback
// (gemm_nvfp4) takes over.
//
// Strategy: tier=CUTLASS_NVFP4, weight_qtype=F16 (engine observes FP16 source
// qtype for an NVFP4-cached weight; the CUTLASS conversion happened at load
// time and lives in the CutlassNvFP4Weight), m_is_one=false. The (M==1)
// decode case never reaches the CUTLASS path in legacy — it goes to the
// faster gemv_nvfp4_kpar (Slice 3) — so we only register the prefill slot.
//
// Preconditions checked here (and at the dispatch site for fast fail):
// - input/output/weight_payload non-null
// - input qtype == F16
// - cutlass_act_data + cutlass_act_sf non-null (mandatory; the FP16
//   activation has to land somewhere). cutlass_workspace is *optional* —
//   gemm_nvfp4_cutlass_sm120 has its own static-fallback workspace alloc.
// If the activation scratch is missing we return PreconditionFail so the
// caller can fall back to the Slice 4 dequant path. Same drop-through
// applies if gemm_nvfp4_cutlass_sm120 itself returns false (CUTLASS
// can_implement rejected the dims) — mirrors the legacy `if (ok) return;`.
// ---------------------------------------------------------------------------
static GemmDispatchResult cutlass_nvfp4_gemm_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "cutlass_nvfp4_gemm_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "cutlass_nvfp4_gemm_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "cutlass_nvfp4_gemm_kernel: weight_payload is null");
    IMP_CHECK(args.input->qtype == QType::F16, "cutlass_nvfp4_gemm_kernel: input qtype must be F16");

    // Workspace precondition. Loud-but-soft: return PreconditionFail so the
    // dispatch site can fall back to Slice 4's dequant kernel (or legacy).
    // Only the activation scratch (act_data + act_sf) is mandatory — the GEMM
    // workspace is optional because gemm_nvfp4_cutlass_sm120 has its own
    // static-fallback path when the caller-supplied workspace is too small or
    // null. This matches the legacy guard at executor_kernels.cu:2140 which
    // only checks `cutlass_act_data != nullptr`.
    if (args.cutlass_act_data == nullptr || args.cutlass_act_sf == nullptr) {
        return GemmDispatchResult::PreconditionFail;
    }

    const CutlassNvFP4Weight& payload =
        *static_cast<const CutlassNvFP4Weight*>(args.weight_payload);

    const int M = static_cast<int>(args.input->shape[0]);
    const int K = static_cast<int>(args.input->shape[1]);
    const int N = static_cast<int>(payload.N);

    // Mirror executor_kernels.cu:2147 + 2179-2181 verbatim — same activation
    // quantization step, same CUTLASS GEMM call, same arg order.
    quantize_fp16_to_nvfp4_cutlass(args.input->data, args.cutlass_act_data, args.cutlass_act_sf, M, K,
                                   args.stream);
    bool ok = gemm_nvfp4_cutlass_sm120(args.cutlass_act_data, args.cutlass_act_sf, payload,
                                       args.output->data, M, N, K, args.cutlass_workspace,
                                       args.cutlass_workspace_size, args.stream);
    if (!ok) {
        // Legacy behaviour: failed CUTLASS run drops through to dequant
        // fallback. Signal PreconditionFail so the caller invokes Slice 4.
        // gemm_nvfp4_cutlass_sm120 already logs IMP_LOG_WARN/ERROR on the
        // can_implement / initialize / run failures internally — no need to
        // double-log here.
        return GemmDispatchResult::PreconditionFail;
    }
    return GemmDispatchResult::Ok;
}

// Static registration. Slice 5 registers only the (M>1) prefill slot — the
// (M==1) decode case always picks the gemv_nvfp4_kpar fast path (Slice 3).
namespace {
struct CutlassNvFP4Registration {
    CutlassNvFP4Registration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::CUTLASS_NVFP4, QType::F16, /*m_is_one=*/false},
            &cutlass_nvfp4_gemm_kernel);
    }
};
static CutlassNvFP4Registration s_cutlass_nvfp4_registration;
}  // namespace

}  // namespace imp
