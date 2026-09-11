#pragma once
// Sampler-side stop mask (think_logic::stop_mask_active): every id
// Engine::is_stop_token() accepts, minus the think markers the budget forces.
// On the steps should_stop would suppress a stop token, this list replaces the
// banned list in the sampler, so the stop never enters the context
// (docs/audit/AUDIT_qwen38_nvfp4.md, P3).
#include "memory/vram_allocator.h"
#include "memory/vram_owned.h"
#include "runtime/request.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace imp {

// One engine-owned device copy of a small id list, uploaded at first use.
// Returns nullptr for an empty list or a failed upload; callers mask nothing.
const int32_t* upload_id_list_once(VramOwned<int32_t>& slot, const std::vector<int32_t>& ids,
                                   VRAMAllocator& alloc, const char* tag, cudaStream_t stream);

struct StopMask {
    std::vector<int32_t> ids;  // host list: InferenceState::banned_tokens on masked steps
    VramOwned<int32_t> d_ids;  // graph paths: InferenceState::d_stop_mask_tokens

    // ids = banned + eos + stops, sorted, unique, without `exclude` or negatives.
    void build(const std::vector<int32_t>& banned, const std::vector<int32_t>& eos,
               const std::vector<int32_t>& stops, const std::vector<int32_t>& exclude);
    const int32_t* device(VRAMAllocator& alloc, cudaStream_t stream) {
        return upload_id_list_once(d_ids, ids, alloc, "stop_mask_tokens", stream);
    }
    int n() const { return static_cast<int>(ids.size()); }
};

// The per-request decision every sampling site shares. `think_end_id` < 0
// means no budget can force the close, so the in-think half stays off.
bool stop_mask_active(const Request& req, int32_t think_end_id);

}  // namespace imp
