#include "runtime/stop_mask.h"
#include "runtime/think_stop_logic.h"
#include "core/logging.h"

#include <algorithm>

namespace imp {

const int32_t* upload_id_list_once(VramOwned<int32_t>& slot, const std::vector<int32_t>& ids,
                                   VRAMAllocator& alloc, const char* tag, cudaStream_t stream) {
    if (ids.empty())
        return nullptr;
    if (slot)
        return slot.get();
    VramOwned<int32_t> buf(alloc, ids.size(), tag);
    if (!buf) {
        IMP_LOG_WARN("%s: device upload failed (%zu B), masking disabled", tag, buf.bytes());
        return nullptr;
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(buf.get(), ids.data(), buf.bytes(), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));  // once, at first use
    slot = std::move(buf);
    return slot.get();
}

void StopMask::build(const std::vector<int32_t>& banned, const std::vector<int32_t>& eos,
                     const std::vector<int32_t>& stops, const std::vector<int32_t>& exclude) {
    ids = banned;
    ids.insert(ids.end(), eos.begin(), eos.end());
    ids.insert(ids.end(), stops.begin(), stops.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::erase_if(ids, [&](int32_t id) {
        return id < 0 || std::find(exclude.begin(), exclude.end(), id) != exclude.end();
    });
    IMP_LOG_INFO("Stop mask: %zu ids masked while a stop would be suppressed", ids.size());
}

bool stop_mask_active(const Request& req, int32_t think_end_id) {
    const bool budget_can_close = req.think_budget > 0.0f && think_end_id >= 0;
    return think_logic::stop_mask_active(req.in_think_block, budget_can_close, req.think_exit_idx,
                                         static_cast<int>(req.output_tokens.size()), req.content_after_think,
                                         req.ignore_eos);
}

}  // namespace imp
