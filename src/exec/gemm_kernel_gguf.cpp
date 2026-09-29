#include "exec/gemm_kernel_registry.h"

#include "compute/gemm.h"        // quantize_fp16_to_q8_1, block_q8_1
#include "compute/ggml_mmvq.h"   // ggml_mmvq_q4k / q5k / q5_1 / q8_0
#include "core/logging.h"
#include "core/tensor.h"
#include "exec/executor_kernels.h"  // is_dp4a_qtype, dispatch_dp4a_gemv, block_q8_1
#include "exec/gemm_scratch.h"  // mmvq_scratch_get_or_grow (Slice 8.6 hoist)

#include <cuda_fp16.h>

namespace imp {

// GGUF small-M tier (M==1): per-qtype strategy {FP16, qtype, true}, handler internally
// picks mmvq -> dp4a -> fused gemv (Q6_K/Q8_0 only) via RuntimeConfig, mirroring legacy
// gemm_dispatch_impl. One strategy per qtype (not per backend) because mmvq and dp4a
// overlap on Q4_K/Q5_K/Q8_0 and would collide on a shared strategy key otherwise.
//
// use_mmvq = force_mmvq && qtype in {Q4_K,Q5_K,Q5_1,Q8_0} && K%32==0 && !no_mmvq &&
//            !no_mmvq_q8_0(Q8_0) && !prefer_fp16_cache && input==F16 && !fp32_output
// use_dp4a = !no_dp4a_gemv && qtype in {Q6_K,Q8_0,Q4_0,Q4_K,Q5_K,Q2_K,Q3_K} && q8_1_buf
//            && d8_buf && !prefer_fp16_cache && input==F16 && !fp32_output && M==1
// mmvq wins when both eligible (legacy if/else-if order).
//
// Fused gemv (Q6_K/Q8_0 only) fires when neither mmvq nor dp4a matches and
// dequant_scratch != nullptr (engine-ready sentinel; the fused kernel never reads the
// scratch). Strategy keys: 8 qtypes total (Q4_K,Q5_K,Q5_1,Q8_0,Q6_K,Q4_0,Q2_K,Q3_K),
// M==1 only. StorageTier::FP16 names the INPUT tier, not the weight's packed format.
//
// Out of scope (stays on legacy): Q4_1 (quant_gemm_int4), M>1 dequant+cuBLAS fallback,
// prefer_fp16_cache decision, FP32-output paths.

// mmvq_scratch_get_or_grow lives in exec/gemm_scratch.h (TU hoist). Engine init must call
// prewarm_mmvq_scratch from executor_workspace_buffers.cu with the model's largest dims
// before the hot path fires.

// Backend helpers, qtype-parameterised. Both backends are M==1 only (legacy gate at
// executor_kernels.cu:2221).

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

// Fused-gemv backend: runs gemv_q6k / gemv_q8_0 (compute/gemm.cu), signature
// (W, x, y, N, K, stream). Dequantizes inside the GEMV; no scratch buffer used.
template <void (*Launcher)(const void*, const half*, half*, int, int, cudaStream_t)>
static void run_fused_gemv_backend(const GemmKernelArgs& args, const Tensor& weight) {
    const int N = static_cast<int>(weight.shape[0]);
    const int K = static_cast<int>(weight.shape[1]);
    Launcher(weight.data, static_cast<const half*>(args.input->data),
             static_cast<half*>(args.output->data), N, K, args.stream);
}

// Per-qtype dispatcher: picks mmvq -> dp4a -> fused gemv based on RuntimeConfig +
// workspace availability. FusedGemvLauncher is real only for Q6_K/gemv_q6k and
// Q8_0/gemv_q8_0; other qtypes bind no_op_fused_gemv_launcher (third branch compiled out).
// No match returns PreconditionFail, falling back to the legacy switch.
template <void (*MmvqLauncher)(const void*, const half*, half*, int, int, int, void*, size_t, cudaStream_t),
          void (*FusedGemvLauncher)(const void*, const half*, half*, int, int, cudaStream_t)>
static GemmDispatchResult run_gguf_smallm(const GemmKernelArgs& args, QType qtype, bool mmvq_eligible,
                                          bool fused_gemv_eligible) {
    IMP_CHECK(args.input != nullptr, "gguf_smallm: input is null");
    IMP_CHECK(args.output != nullptr, "gguf_smallm: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "gguf_smallm: weight_payload is null");
    const Tensor& weight = *static_cast<const Tensor*>(args.weight_payload);
    IMP_CHECK(weight.qtype == qtype, "gguf_smallm: weight qtype mismatch");

    // Dispatch site already filtered input.qtype==F16, M==1, output.qtype!=F32. The kernel
    // re-checks the runtime-config gates legacy evaluates at the call site (via
    // GemmKernelArgs from GemmContext::make, not RuntimeConfig::current()).
    const int K = static_cast<int>(weight.shape[1]);

    // mmvq eligibility (legacy line 2216-2220): force_mmvq set, qtype mmvq-supported,
    // K%32==0, no_mmvq off, no_mmvq_q8_0 off for Q8_0. force_mmvq is the per-model override
    // from ModelConfig::Overrides::Gemma4::force_mmvq.
    const bool use_mmvq = args.force_mmvq && mmvq_eligible && (K % 32 == 0) && !args.no_mmvq &&
                          !(args.no_mmvq_q8_0 && qtype == QType::Q8_0);

    // dp4a eligibility (legacy line 2221-2222): scratch present + is_dp4a_qtype.
    const bool use_dp4a = !args.no_dp4a_gemv && args.q8_1_buf != nullptr && args.d8_buf != nullptr &&
                          is_dp4a_qtype(qtype);

    // Fused-gemv fallback (legacy lines 2267-2276): only Q6_K/Q8_0 have a fused kernel.
    // dequant_scratch != nullptr is the engine-ready sentinel matching legacy; the fused
    // kernel itself doesn't read the scratch.
    const bool use_fused_gemv = fused_gemv_eligible && args.dequant_scratch != nullptr;

    if (use_mmvq) {
        run_mmvq_backend<MmvqLauncher>(args, weight);
        return GemmDispatchResult::Ok;
    }
    if (use_dp4a) {
        run_dp4a_backend(args, weight, qtype);
        return GemmDispatchResult::Ok;
    }
    if (use_fused_gemv) {
        run_fused_gemv_backend<FusedGemvLauncher>(args, weight);
        return GemmDispatchResult::Ok;
    }
    // No backend matches — fall back to legacy (e.g. dequant+cuBLAS for
    // qtypes without a fused-gemv kernel, or Q6_K/Q8_0 when the engine
    // hasn't allocated the dequant_scratch sentinel).
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

// Same idea for the fused-gemv slot: only Q6_K / Q8_0 bind to real kernels
// in Slice 8.2; the other six qtypes bind to this no-op (the third branch
// is guarded by `fused_gemv_eligible=false`, never invoked at runtime).
static void no_op_fused_gemv_launcher(const void*, const half*, half*, int, int, cudaStream_t) {
    // Intentionally empty — see no_op_mmvq_launcher.
}

// ---------------------------------------------------------------------------
// Per-qtype handler functions (8 total, one per supported qtype).
// ---------------------------------------------------------------------------

static GemmDispatchResult gguf_q4k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&ggml_mmvq_q4k, &no_op_fused_gemv_launcher>(
        args, QType::Q4_K, /*mmvq_eligible=*/true, /*fused_gemv_eligible=*/false);
}

static GemmDispatchResult gguf_q5k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&ggml_mmvq_q5k, &no_op_fused_gemv_launcher>(
        args, QType::Q5_K, /*mmvq_eligible=*/true, /*fused_gemv_eligible=*/false);
}

static GemmDispatchResult gguf_q5_1_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&ggml_mmvq_q5_1, &no_op_fused_gemv_launcher>(
        args, QType::Q5_1, /*mmvq_eligible=*/true, /*fused_gemv_eligible=*/false);
}

// Slice 8.2: Q8_0 gains a fused-gemv fallback (`gemv_q8_0` in compute/gemm.cu).
static GemmDispatchResult gguf_q8_0_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&ggml_mmvq_q8_0, &gemv_q8_0>(
        args, QType::Q8_0, /*mmvq_eligible=*/true, /*fused_gemv_eligible=*/true);
}

// dp4a-only qtypes (no mmvq backend).
// Slice 8.2: Q6_K gains a fused-gemv fallback (`gemv_q6k`).
static GemmDispatchResult gguf_q6k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&no_op_mmvq_launcher, &gemv_q6k>(
        args, QType::Q6_K, /*mmvq_eligible=*/false, /*fused_gemv_eligible=*/true);
}

static GemmDispatchResult gguf_q4_0_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&no_op_mmvq_launcher, &no_op_fused_gemv_launcher>(
        args, QType::Q4_0, /*mmvq_eligible=*/false, /*fused_gemv_eligible=*/false);
}

static GemmDispatchResult gguf_q2k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&no_op_mmvq_launcher, &no_op_fused_gemv_launcher>(
        args, QType::Q2_K, /*mmvq_eligible=*/false, /*fused_gemv_eligible=*/false);
}

static GemmDispatchResult gguf_q3k_kernel(const GemmKernelArgs& args) {
    return run_gguf_smallm<&no_op_mmvq_launcher, &no_op_fused_gemv_launcher>(
        args, QType::Q3_K, /*mmvq_eligible=*/false, /*fused_gemv_eligible=*/false);
}

namespace {
struct GgufRegistration {
    GgufRegistration() {
        auto& reg = GemmKernelRegistry::instance();
        // Eight strategies, every small-M GGUF qtype reachable via mmvq or dp4a, all under
        // (StorageTier::FP16, <qtype>, m_is_one=true). qtype is the discriminator; the handler
        // picks the backend internally.
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
