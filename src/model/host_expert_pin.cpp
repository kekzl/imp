#include "model/host_expert_pin.h"

#include "core/logging.h"

#include <cstring>
#include <utility>

namespace imp {

HostExpertPinner::~HostExpertPinner() {
    pool_.reset();
    for (PinnedBuffer& b : done_)
        sink_.push_back(std::move(b));
}

void HostExpertPinner::pin(std::vector<Tensor>& expert_vec, size_t total) {
    if (!pool_)
        pool_.emplace(kThreads);
    pool_->submit([this, &expert_vec, total] {
        PinnedBuffer pin = PinnedBuffer::acquire(cuda_host_pinned_allocator(), total, HostPinnedKind::Mapped);
        if (pin.empty()) {
            // Correct but slow: the experts stay on the mmap and every transfer pays the
            // in-driver staging cost.
            IMP_LOG_WARN(
                "Host NVFP4 experts: pinned alloc failed (%.2f MiB) - staying on mmap, "
                "H2D will be ~3x slower",
                total / (1024.0 * 1024.0));
            return;
        }
        char* dst = static_cast<char*>(pin.data());
        for (Tensor& w : expert_vec) {
            const size_t bytes = w.nbytes();
            std::memcpy(dst, w.data, bytes);
            w.data = dst;
            dst += bytes;
        }
        std::lock_guard<std::mutex> lk(mu_);
        done_.push_back(std::move(pin));
    });
}

}  // namespace imp
