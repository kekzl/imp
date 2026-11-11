#pragma once

// Factored rows of the batched verify (docs/plans/2026-09-12-factored-verify-spare.md): the full verify's
// spare (speculative.factored_spare) and the bound verify's drafts before the last at 2+ drafts
// (speculative.mtp_batch_verify_factor_row), buffers and the accept-time fold.

#include <cuda_runtime.h>

namespace imp {

struct BatchVerifyState;
struct ModelConfig;
struct RuntimeConfig;
class SSMState;
class VRAMAllocator;

// Factored spare pools, sized over the LIVE slots only: rows are addressed by the slot ids the scheduler
// hands out. A failure clears cfg.speculative.factored_spare (slot-swapping) rather than failing the verify.
[[nodiscard]] bool ensure_factored_rows(RuntimeConfig& cfg, const ModelConfig& mc, SSMState* ssm,
                                        VRAMAllocator& alloc, BatchVerifyState& bv, cudaStream_t stream);
void free_factored_rows(VRAMAllocator& alloc, BatchVerifyState& bv);

// Factor rows and conv taps per (verify group, level) and the fold lists (bv.d_fac, d_tap, d_clear), levels =
// drafts - 1. false = nothing allocated (geometry, state dtype or VRAM): the caller verifies 1 draft.
[[nodiscard]] bool bound_verify_factor_row_alloc(const ModelConfig& mc, SSMState* ssm, VRAMAllocator& alloc,
                                                 BatchVerifyState& bv, int groups, int levels,
                                                 cudaStream_t stream);

// Group g folds its first fold[g] of the step's `levels` rows and taps (the matched drafts, 0 = none) into
// live[g], which holds the 1-row state. Queued on `stream` behind the forward that wrote them.
void bound_verify_factor_row_fold(const ModelConfig& mc, SSMState& ssm, BatchVerifyState& bv, int levels,
                                  const int* live, const int* fold, int n_groups, cudaStream_t stream);

}  // namespace imp
