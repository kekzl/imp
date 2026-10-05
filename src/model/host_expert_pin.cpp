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
        // Copy from the mmap first, register after (roadmap row 68): the driver pins written pages
        // 20x faster than cudaHostAlloc produces fresh ones on WSL2.
        PinnedBuffer pin = PinnedBuffer::acquire_filled(total, HostPinnedKind::Mapped,
                                                        [&expert_vec](void* host) {
                                                            char* dst = static_cast<char*>(host);
                                                            for (const Tensor& w : expert_vec) {
                                                                std::memcpy(dst, w.data, w.nbytes());
                                                                dst += w.nbytes();
                                                            }
                                                        });
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
            w.data = dst;
            dst += w.nbytes();
        }
        std::lock_guard<std::mutex> lk(mu_);
        done_.push_back(std::move(pin));
    });
}

}  // namespace imp
