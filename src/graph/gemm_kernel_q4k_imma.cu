// =============================================================================
// gemm_kernel_q4k_imma.cu — Phase 2C dispatch handler for Q4_K_M INT8 IMMA
// =============================================================================
//
// Routes Q4_K_M dense GEMM (M ≥ 1024, dense, non-MoE) through the Phase 2B
// production tile kernel via the mmq_q4k_imma_gemm high-level entry. Default
// off; the dispatch site checks `gemm.q4k_imma_enabled` before emitting the
// {FP16, Q4_K, m_is_one=false} strategy key this handler is registered for.
//
// Eligibility (re-checked here):
//   - weight.qtype == Q4_K
//   - input.qtype == FP16
//   - M ≥ 64 AND M % 64 == 0   (BLOCK_M; the dispatch site gates on M ≥ 1024)
//   - N ≥ 32 AND N % 32 == 0   (BLOCK_N)
//   - K % 32 == 0               (BLOCK_K — one m16n8k32 MMA per sub-block)
//
// On any failure the handler returns PreconditionFail and the dispatch site
// falls through to the generic dequant catch-all (Slice 8.5) which performs
// dequant→FP16 cuBLAS — bit-identical to the pre-Phase-2C behaviour.
//
// Reference: docs/plans/q4k_imma_design_2026_05_17.md,
// docs/superpowers/plans/2026-05-18-q4k-imma-phase2b-ceiling.md.

#include "graph/gemm_kernel_registry.h"

#include "compute/mmq_q4k_imma_tile.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "model/model_config.h"

#include <cuda_fp16.h>

namespace imp {

static GemmDispatchResult q4k_imma_kernel(const GemmKernelArgs& args) {
    // Soft preconditions — null args (e.g. from a registry-coverage probe test)
    // must return PreconditionFail rather than FATAL, so the registry-lookup
    // tests can route arbitrary strategy keys without instantiating real
    // tensors.
    if (args.input == nullptr || args.output == nullptr || args.weight_payload == nullptr)
        return GemmDispatchResult::PreconditionFail;

    const Tensor& weight = *static_cast<const Tensor*>(args.weight_payload);
    if (weight.qtype != QType::Q4_K) return GemmDispatchResult::PreconditionFail;
    if (args.input->qtype != QType::F16) return GemmDispatchResult::PreconditionFail;
    if (args.output->qtype != QType::F16) return GemmDispatchResult::PreconditionFail;
    if (args.beta != 0.0f) return GemmDispatchResult::PreconditionFail;  // residual fuse: dequant path handles it

    const int M = static_cast<int>(args.input->shape[0]);
    const int K = static_cast<int>(args.input->shape[1]);
    const int N = static_cast<int>(weight.shape[0]);

    // Kernel grid constraints from mmq_q4k_imma_tile (BLOCK_M=64, BLOCK_N=32,
    // BLOCK_K=32). M < 64 or non-aligned dims fall through to dequant.
    if (M < 64 || N < 32 || K < 32) return GemmDispatchResult::PreconditionFail;
    if (M % 64 != 0 || N % 32 != 0 || K % 32 != 0) return GemmDispatchResult::PreconditionFail;

    const __half* X = static_cast<const __half*>(args.input->data);
    __half* Y = static_cast<__half*>(args.output->data);

    if (!mmq_q4k_imma_gemm(weight.data, X, Y, M, N, K, args.stream)) {
        // Shape/cache failure inside the entry — fall back.
        return GemmDispatchResult::PreconditionFail;
    }
    return GemmDispatchResult::Ok;
}

namespace {
struct Q4kImmaRegistration {
    Q4kImmaRegistration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::FP16, QType::Q4_K, /*m_is_one=*/false},
            &q4k_imma_kernel);
    }
};
static Q4kImmaRegistration s_q4k_imma_registration;
}  // namespace

}  // namespace imp
