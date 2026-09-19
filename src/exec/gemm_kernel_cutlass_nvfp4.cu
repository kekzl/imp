#include "exec/gemm_kernel_registry.h"

#include "compute/gemm_cutlass_mxfp4_sm120.h"  // CutlassMxFP4Weight, gemm_mxfp4_cutlass_sm120 (QW7 dual-cache path)
#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight, gemm_nvfp4_cutlass_sm120, quantize_fp16_to_nvfp4_cutlass
#include "core/logging.h"
#include "core/tensor.h"

namespace imp {

// Adapter: quantizes the FP16 activation into cutlass_act_data/_sf, then runs
// gemm_nvfp4_cutlass_sm120 against the pre-converted CUTLASS weight. Preferred NVFP4
// GEMM path when cutlass_nvfp4_cache + workspace pointers are populated; else the
// dequant fallback (gemm_nvfp4) takes over. Decode (M==1) never reaches here (uses
// gemv_nvfp4_kpar); only the prefill strategy is registered.
// Dual-cache MXFP4 (--mxfp4-prefill): if weight.data also hits cutlass_mxfp4, try MXFP4
// CUTLASS via the forwarded mxfp4_payload/mxfp4_act_sf/mxfp4_workspace before falling
// back to NVFP4 CUTLASS; nullptr mxfp4_payload = no dual-cache hit.
// Preconditions: input/output/weight_payload non-null, input qtype F16,
// cutlass_act_data+cutlass_act_sf mandatory (cutlass_workspace optional). Any
// precondition or can_implement failure returns PreconditionFail (caller falls back).
static GemmDispatchResult cutlass_nvfp4_gemm_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "cutlass_nvfp4_gemm_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "cutlass_nvfp4_gemm_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "cutlass_nvfp4_gemm_kernel: weight_payload is null");
    IMP_CHECK(args.input->qtype == QType::F16, "cutlass_nvfp4_gemm_kernel: input qtype must be F16");

    // Workspace precondition: only act_data+act_sf are mandatory. cutlass_workspace may be
    // null (allocate_auxiliary_buffers sizes it at the max shape, 0 allocates nothing).
    // A refusal here or from the GEMM itself returns PreconditionFail -> dequant fallback.
    if (args.cutlass_act_data == nullptr || args.cutlass_act_sf == nullptr) {
        return GemmDispatchResult::PreconditionFail;
    }

    const CutlassNvFP4Weight& payload =
        *static_cast<const CutlassNvFP4Weight*>(args.weight_payload);

    const int M = static_cast<int>(args.input->shape[0]);
    const int K = static_cast<int>(args.input->shape[1]);
    const int N = static_cast<int>(payload.N);

    // QW7 dual-cache MXFP4 branch: fires when the dispatch found a cutlass_mxfp4 entry for
    // this weight.data, the MXFP4 activation scratch is allocated, and K % 32 == 0 (UE8M0
    // SfAtom group). Reuses the NVFP4 activation buffer (same packed E2M1 layout); only the
    // scale factor layout differs (UE8M0 per 32 vs NVFP4's UE4M3 per 16).
    if (args.mxfp4_payload != nullptr && args.mxfp4_act_sf != nullptr && K % 32 == 0) {
        const CutlassMxFP4Weight& mx_payload =
            *static_cast<const CutlassMxFP4Weight*>(args.mxfp4_payload);
        quantize_fp16_to_mxfp4_cutlass(args.input->data, args.cutlass_act_data, args.mxfp4_act_sf, M, K,
                                       args.stream);
        bool mx_ok = gemm_mxfp4_cutlass_sm120(args.cutlass_act_data, args.mxfp4_act_sf, mx_payload,
                                              args.output->data, M, N, K, args.mxfp4_workspace,
                                              args.mxfp4_workspace_size, args.stream);
        if (mx_ok)
            return GemmDispatchResult::Ok;
        // MXFP4 CUTLASS rejected the dims: fall through to NVFP4 CUTLASS below. The activation
        // buffer is reusable (quantize_fp16_to_nvfp4_cutlass overwrites it; separate scale buffer).
    }

    // act_prequantized: scratch already holds quantize(input) from a prior dispatch on the
    // same input (QKV / gate-up dedupe); skip re-quantizing. Except when mxfp4_payload is
    // set: the dual-cache branch above may have clobbered the scratch with MXFP4 scale layout.
    if (!args.act_prequantized || args.mxfp4_payload != nullptr)
        quantize_fp16_to_nvfp4_cutlass(args.input->data, args.cutlass_act_data, args.cutlass_act_sf, M, K,
                                       args.stream);
    // The CUTLASS NVFP4 epilogue is built with beta=0 literal; a beta=1 call would overwrite
    // the residual instead of adding, producing silent garbage. Refuse (caller falls to the
    // dequant path, which honours beta). Unreachable today (o_proj/down_proj exclude this
    // tier) but kept because that is one edit away from not being true (#1547).
    if (args.beta != 0.0f)
        return GemmDispatchResult::PreconditionFail;
    bool ok = gemm_nvfp4_cutlass_sm120(args.cutlass_act_data, args.cutlass_act_sf, payload,
                                       args.output->data, M, N, K, args.cutlass_workspace,
                                       args.cutlass_workspace_size, args.stream);
    if (!ok) {
        // Failed CUTLASS run drops through to the dequant fallback: return PreconditionFail.
        // gemm_nvfp4_cutlass_sm120 already logs WARN/ERROR on failure internally; no double-log.
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
