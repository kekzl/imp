#pragma once

// Factored rows of the batched verify (docs/plans/2026-09-12-factored-verify-spare.md): the full verify's
// spare (speculative.factored_spare) and the bound verify's draft 1 at 2 drafts
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

// Factor rows, conv taps and the fold list over every recurrent slot (bv.d_fac, d_tap, d_clear).
// false = nothing allocated (geometry, state dtype or VRAM): the caller verifies 1 draft.
[[nodiscard]] bool bound_verify_factor_row_alloc(const ModelConfig& mc, SSMState* ssm, VRAMAllocator& alloc,
                                                 BatchVerifyState& bv, cudaStream_t stream);

// Draft 1 of the groups with fold[g] != 0: live[g] holds the 1-row state and names the row and tap the
// forward just wrote. Queued on `stream` behind that forward; the live slot then holds the 2-row state.
void bound_verify_factor_row_fold(const ModelConfig& mc, SSMState& ssm, BatchVerifyState& bv, const int* live,
                                  const int* fold, int n_groups, cudaStream_t stream);

}  // namespace imp
