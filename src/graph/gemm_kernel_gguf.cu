#include "graph/gemm_kernel_registry.h"

#include "compute/gemm.h"        // quantize_fp16_to_q8_1, block_q8_1
#include "compute/ggml_mmvq.h"   // ggml_mmvq_q4k / q5k / q5_1 / q8_0
#include "core/logging.h"
#include "core/tensor.h"
#include "graph/executor_kernels.h"  // is_dp4a_qtype, dispatch_dp4a_gemv, block_q8_1
#include "runtime/config.h"

#include <cuda_fp16.h>

namespace imp {

// ---------------------------------------------------------------------------
// GGUF small-M tier — R5 Slice 7.
//
// Migrates the legacy mmvq + dp4a small-M GGUF branches (gemm_dispatch_impl
// at executor_kernels.cu:2216-2251) to the GemmKernel registry. The legacy
// dispatch evaluates two parallel gates at M==1:
//
//   use_mmvq = gemma4.force_mmvq && qtype ∈ {Q4_K, Q5_K, Q5_1, Q8_0}
//              && weight.shape[1] % 32 == 0 && !no_mmvq && !no_mmvq_q8_0(Q8_0)
//              && !prefer_fp16_cache && input.qtype == F16 && !fp32_output
//
//   use_dp4a = !no_dp4a_gemv && qtype ∈ {Q6_K, Q8_0, Q4_0, Q4_K, Q5_K, Q2_K, Q3_K}
//              && q8_1_buf && d8_buf && !prefer_fp16_cache && input.qtype == F16
//              && !fp32_output && input.shape[0] == 1
//
// `use_mmvq` wins when both are eligible (legacy if/else if order). The two
// backends overlap on Q4_K / Q5_K / Q8_0; the dispatch site emits a single
// (FP16, <qtype>, m_is_one=true) strategy key per qtype and the registered
// handler internally re-evaluates `force_mmvq`/`no_mmvq`/etc. and picks the
// backend — same conditions, same precedence as legacy. PreconditionFail
// when neither backend can run (e.g. dp4a scratch missing, mmvq disabled),
// surfacing the same fall-through to legacy that the legacy `else if` chain
// produces today.
//
// Scope decision — Option 2 trimmed (per-qtype, M==1 only), with internal
// backend selection inside each handler.
//
// Why this shape:
//   - Option 1 (one strategy + giant switch) buries the qtype axis inside
//     the kernel handler — defeats the registry's "qtype key resolves to
//     handler" contract.
//   - Option 3 (one strategy per backend with internal qtype switch)
//     collides on strategy keys because mmvq and dp4a overlap on three
//     qtypes — no clean axis to disambiguate the two backends without
//     overloading m_is_one or extending the StorageTier / GemmStrategy
//     types (constrained by Slice 7 charter).
//   - Option 2 (one strategy per qtype × m_is_one) maps cleanly onto the
//     registry: the dispatch site emits exactly the qtype the weight has,
//     the registry resolves to that qtype's handler, and the handler picks
//     the backend based on the SAME RuntimeConfig fields the legacy switch
//     reads. The "which backend wins" decision is local to the handler —
//     mirroring how legacy makes it local to the dispatch impl. The cross-
//     axis maintainability win still holds because each handler is small
//     (~30 lines) and adding a new qtype is a single new registration.
//
// Strategy key set (8 entries, M==1 only):
//   {FP16, Q4_K, true}, {FP16, Q5_K, true}, {FP16, Q5_1, true},
//   {FP16, Q8_0, true}, {FP16, Q6_K, true}, {FP16, Q4_0, true},
//   {FP16, Q2_K, true}, {FP16, Q3_K, true}
//
// StorageTier::FP16 = the INPUT tier (activations are FP16). GGUF weights
// live raw in their qtype without a dedicated StorageTier — using FP16 as
// the "input tier" matches Slice 1-6 semantics (where the tier name often
// reflects what the engine observes for the weight, not its packed format).
//
// Out of scope (stays on legacy for now):
//   - Q4_1 quant_gemm_int4 (also a small-M path but neither mmvq nor dp4a).
//   - M>1 dequant+cuBLAS fallback (the "large-M path" in the slice 7 prompt).
//   - Fused Q6_K/Q8_0 GEMV (the `else if` fallback below dp4a in legacy).
//   - prefer_fp16_cache decision (stays on dispatch site — strictly upstream).
//   - FP32-output paths (write directly to half*, so legacy stays in charge).
// All of these continue through `gemm_dispatch_impl` when the registry
// returns NoMatch / PreconditionFail. Slice 8 retires the legacy switch.
// ---------------------------------------------------------------------------

// Forward declarations — single-global mmvq scratch lives in
// executor_kernels.cu. Calling it via prototype keeps the invariant
// without widening the public header surface in this slice.
void mmvq_scratch_get_or_grow(size_t need, void** out_buf, size_t* out_size);

// ---------------------------------------------------------------------------
// Backend helpers — small, qtype-parameterised. Both backends are M==1 only
// (legacy gate at executor_kernels.cu:2221).
// ---------------------------------------------------------------------------

// mmvq backend: Q8_1-quantize activations into the file-scope mmvq scratch,
// then run the ggml-compatible GEMV. Mirrors legacy lines 2230-2245.
template <void (*Launcher)(const void*, const half*, half*, int, int, int, void*, size_t, cudaStream_t)>
static void run_mmvq_backend(const GemmKernelArgs& args, const Tensor& weight) {
    const int M = static_cast<int>(args.input->shape[0]);
    const int N = static_cast<int>(weight.shape[0]);
    const int K = static_cast<int>(weight.shape[1]);
    size_t q8_need = static_cast<size_t>(M) * ((K + 31) / 32) * 36;
    void* s_mmvq_scratch = nullptr;
    size_t s_mmvq_scratch_size = 0;
    mmvq_scratch_get_or_grow(q8_need, &s_mmvq_scratch, &s_mmvq_scratch_size);
    Launcher(weight.data, static_cast<const half*>(args.input->data),
             static_cast<half*>(args.output->data), M, N, K, s_mmvq_scratch, s_mmvq_scratch_size,
             args.stream);
}

// dp4a backend: caller-provided q8_1_buf + d8_buf (sized by engine init via
// QuantScratch). Mirrors legacy lines 2246-2251 verbatim.
static void run_dp4a_backend(const GemmKernelArgs& args, const Tensor& weight, QType qtype) {
    const int N = static_cast<int>(weight.shape[0]);
    const int K = static_cast<int>(weight.shape[1]);
    quantize_fp16_to_q8_1(static_cast<const half*>(args.input->data),
                          static_cast<block_q8_1*>(args.q8_1_buf), args.d8_buf, K, args.stream);
    dispatch_dp4a_gemv(qtype, weight.data, static_cast<const block_q8_1*>(args.q8_1_buf), args.d8_buf,
                       static_cast<half*>(args.output->data), N, K, args.stream);
}

// Per-qtype dispatcher — internally picks mmvq vs dp4a based on RuntimeConfig
// + workspace availability. `mmvq_eligible` flags qtypes mmvq supports; the
// dp4a check uses is_dp4a_qtype(). When neither backend matches, returns
// PreconditionFail so the dispatch site falls back to the legacy switch.
template <void (*MmvqLauncher)(const void*, const half*, half*, int, int, int, void*, size_t, cudaStream_t)>
static GemmDispatchResult run_gguf_smallm(const GemmKernelArgs& args, QType qtype, bool mmvq_eligible) {
    IMP_CHECK(args.input != nullptr, "gguf_smallm: input is null");
    IMP_CHECK(args.output != nullptr, "gguf_smallm: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "gguf_smallm: weight_payload is null");
    const Tensor& weight = *static_cast<const Tensor*>(args.weight_payload);
    IMP_CHECK(weight.qtype == qtype, "gguf_smallm: weight qtype mismatch");

    // The dispatch site already filtered on `input.qtype==F16`, `M==1`, and
    // `output.qtype != F32`. The kernel re-checks the runtime-config gates
    // that legacy evaluates at the call site:
    const auto& rcfg = RuntimeConfig::current();
    const int K = static_cast<int>(weight.shape[1]);

    // mmvq has stricter eligibility (legacy line 2216-2220): force_mmvq set,
    // qtype mmvq supports, K%32==0, no_mmvq off, no_mmvq_q8_0 off for Q8_0.
    const bool no_mmvq_q8_0 = rcfg.gemm.no_mmvq_q8_0;
    const bool no_mmvq_all = rcfg.gemm.no_mmvq;
    const bool use_mmvq = rcfg.gemma4.force_mmvq && mmvq_eligible && (K % 32 == 0) && !no_mmvq_all &&
                          !(no_mmvq_q8_0 && qtype == QType::Q8_0);

    // dp4a eligibility (legacy line 2221-2222): scratch present + is_dp4a_qtype.
    const bool no_dp4a_gemv = rcfg.gemm.no_dp4a_gemv;
    const bool use_dp4a = !no_dp4a_gemv && args.q8_1_buf != nullptr && args.d8_buf != nullptr &&
                          is_dp4a_qtype(qtype);

    if (use_mmvq) {
        run_mmvq_backend<MmvqLauncher>(args, weight);
        return GemmDispatchResult::Ok;
    }
    if (use_dp4a) {
        run_dp4a_backend(args, weight, qtype);
        return GemmDispatchResult::Ok;
    }
    // Neither backend matches — fall back to legacy (e.g. fused Q6_K/Q8_0
    // GEMV at executor_kernels.cu:2263-2272, dequant+cuBLAS, etc.).
    return GemmDispatchResult::PreconditionFail;
}

// A no-op mmvq launcher used by qtypes mmvq doesn't support (Q6_K, Q4_0,
// Q2_K, Q3_K). The template parameter must compile but is never invoked at
// runtime because `mmvq_eligible=false` for these qtypes.
static void no_op_mmvq_launcher(const void*, const half*, half*, int, int, int, void*, size_t,
                                cudaStream_t) {
    // Intentionally empty — dispatched only if mmvq_eligible=true for the
    // qtype, which is false for the qtypes that bind to this launcher.
}

// ---------------------------------------------------------------------------
// Per-qtype handler functions (8 total, one per supported qtype).
// ---------------------------------------------------------------------------

static GemmDispatchResult gguf_q4k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&ggml_mmvq_q4k>(args, QType::Q4_K, /*mmvq_eligible=*/true);
}

static GemmDispatchResult gguf_q5k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&ggml_mmvq_q5k>(args, QType::Q5_K, /*mmvq_eligible=*/true);
}

static GemmDispatchResult gguf_q5_1_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&ggml_mmvq_q5_1>(args, QType::Q5_1, /*mmvq_eligible=*/true);
}

static GemmDispatchResult gguf_q8_0_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&ggml_mmvq_q8_0>(args, QType::Q8_0, /*mmvq_eligible=*/true);
}

// dp4a-only qtypes (no mmvq backend).
static GemmDispatchResult gguf_q6k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&no_op_mmvq_launcher>(args, QType::Q6_K, /*mmvq_eligible=*/false);
}

static GemmDispatchResult gguf_q4_0_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&no_op_mmvq_launcher>(args, QType::Q4_0, /*mmvq_eligible=*/false);
}

static GemmDispatchResult gguf_q2k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&no_op_mmvq_launcher>(args, QType::Q2_K, /*mmvq_eligible=*/false);
}

static GemmDispatchResult gguf_q3k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&no_op_mmvq_launcher>(args, QType::Q3_K, /*mmvq_eligible=*/false);
}

namespace {
struct GgufRegistration {
    GgufRegistration() {
        auto& reg = GemmKernelRegistry::instance();
        // Eight strategies — every small-M GGUF qtype reachable via mmvq or
        // dp4a. All under (StorageTier::FP16, <qtype>, m_is_one=true). The
        // qtype axis is the discriminator; the handler internally picks the
        // backend.
        reg.register_kernel(GemmStrategy{StorageTier::FP16, QType::Q4_K, /*m_is_one=*/true},
                            &gguf_q4k_kernel);
        reg.register_kernel(GemmStrategy{StorageTier::FP16, QType::Q5_K, /*m_is_one=*/true},
                            &gguf_q5k_kernel);
        reg.register_kernel(GemmStrategy{StorageTier::FP16, QType::Q5_1, /*m_is_one=*/true},
                            &gguf_q5_1_kernel);
        reg.register_kernel(GemmStrategy{StorageTier::FP16, QType::Q8_0, /*m_is_one=*/true},
                            &gguf_q8_0_kernel);
        reg.register_kernel(GemmStrategy{StorageTier::FP16, QType::Q6_K, /*m_is_one=*/true},
                            &gguf_q6k_kernel);
        reg.register_kernel(GemmStrategy{StorageTier::FP16, QType::Q4_0, /*m_is_one=*/true},
                            &gguf_q4_0_kernel);
        reg.register_kernel(GemmStrategy{StorageTier::FP16, QType::Q2_K, /*m_is_one=*/true},
                            &gguf_q2k_kernel);
        reg.register_kernel(GemmStrategy{StorageTier::FP16, QType::Q3_K, /*m_is_one=*/true},
                            &gguf_q3k_kernel);
    }
};
static GgufRegistration s_gguf_registration;
}  // namespace

}  // namespace imp
