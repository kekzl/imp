#include "model/expert_pack.h"

#include "core/logging.h"

#include <algorithm>

namespace imp {

bool expert_pack_shape(const std::vector<Tensor>& experts, int64_t out_shape[3]) {
    if (experts.size() < 2)
        return false;
    const Tensor& e0 = experts[0];
    for (const Tensor& e : experts) {
        if (e.data == nullptr || !e.on_device || e.qtype != QType::F16 || e.ndim != 2 ||
            e.shape[0] != e0.shape[0] || e.shape[1] != e0.shape[1] || e.shape[0] <= 0 || e.shape[1] <= 0)
            return false;
    }
    out_shape[0] = static_cast<int64_t>(experts.size());
    out_shape[1] = e0.shape[0];
    out_shape[2] = e0.shape[1];
    return true;
}

bool pack_device_experts(std::vector<Tensor>& experts, Tensor& packed, std::vector<void*>& allocs,
                         const ExpertPackAlloc& alloc, const ExpertPackRelease& release,
                         cudaStream_t stream) {
    int64_t shape[3] = {};
    auto owned = [&](const Tensor& t) { return std::ranges::find(allocs, t.data) != allocs.end(); };
    if (packed.data != nullptr || !expert_pack_shape(experts, shape) || !std::ranges::all_of(experts, owned))
        return false;
    const size_t expert_bytes = static_cast<size_t>(shape[1]) * shape[2] * sizeof(uint16_t);
    const size_t total = expert_bytes * static_cast<size_t>(shape[0]);
    char* buf = static_cast<char*>(alloc(total));
    if (buf == nullptr)
        return false;
    for (size_t e = 0; e < experts.size(); ++e) {
        const cudaError_t err = cudaMemcpyAsync(buf + e * expert_bytes, experts[e].data, expert_bytes,
                                                cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("expert pack: copy of expert %zu failed (%s), keeping per-expert tensors", e,
                         cudaGetErrorString(err));
            release(buf, total);
            return false;
        }
    }
    for (size_t e = 0; e < experts.size(); ++e) {
        release(experts[e].data, expert_bytes);
        const int64_t eshape[4] = {shape[1], shape[2], 0, 0};
        experts[e] = Tensor(buf + e * expert_bytes, QType::F16, 2, eshape, true);
    }
    packed = Tensor(buf, QType::F16, 3, shape, true);
    allocs.push_back(buf);
    return true;
}

}  // namespace imp
