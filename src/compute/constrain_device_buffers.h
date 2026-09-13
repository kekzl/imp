#pragma once

// Device buffers all four constrainers need (audit finding F-18). Kept separate from
// constrain_common.h: that header carries the JSON category classifier and only compiles once
// the includer has CAT_* flags, so grammar/regex (no categories) could not include it.

#include <cstdint>
#include <cuda_runtime.h>
#include <utility>

#include "core/logging.h"

namespace imp {

// Replaces four constrainers' separately-managed d_token_allow_/_categories_/_mask_ buffers
// (four lifetimes, four I1 allocation-allowlist entries) with one shared storage; mask LOGIC
// stays in the four classes (genuinely different grammars).
// INVARIANT (#1104): allocation happens at init and fails loudly - a constrainer that cannot
// mask must never look initialised. (Previously one class allocated lazily inside apply_mask()
// and a mid-decode failure returned without masking, answering prose where JSON was promised.)
class ConstrainDeviceBuffers {
public:
    ConstrainDeviceBuffers() = default;
    ~ConstrainDeviceBuffers() { free(); }

    ConstrainDeviceBuffers(const ConstrainDeviceBuffers&) = delete;
    ConstrainDeviceBuffers& operator=(const ConstrainDeviceBuffers&) = delete;

    ConstrainDeviceBuffers(ConstrainDeviceBuffers&& o) noexcept { swap(o); }
    ConstrainDeviceBuffers& operator=(ConstrainDeviceBuffers&& o) noexcept {
        if (this != &o) {
            free();
            swap(o);
        }
        return *this;
    }

    // Per-token allow list, one byte per vocabulary entry.
    [[nodiscard]] bool alloc_token_allow(const char* owner, int vocab_size) {
        cudaError_t err = cudaMalloc(&d_token_allow_, static_cast<size_t>(vocab_size));
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("%s: failed to allocate the %d-token allow list: %s", owner, vocab_size,
                          cudaGetErrorString(err));
            d_token_allow_ = nullptr;  // three of the four did this; the fourth did not
            return false;
        }
        return true;
    }

    // Per-token category bitmask, uploaded once and never mutated.
    [[nodiscard]] bool alloc_categories(const char* owner, const uint16_t* host, int vocab_size) {
        const size_t bytes = static_cast<size_t>(vocab_size) * sizeof(uint16_t);
        cudaError_t err = cudaMalloc(&d_token_categories_, bytes);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("%s: failed to allocate device categories: %s", owner, cudaGetErrorString(err));
            d_token_categories_ = nullptr;
            return false;
        }
        err = cudaMemcpy(d_token_categories_, host, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("%s: failed to copy categories to device: %s", owner, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // Single-word "which categories are allowed right now", rewritten per step.
    [[nodiscard]] bool alloc_allowed_mask(const char* owner) {
        cudaError_t err = cudaMalloc(&d_allowed_mask_, sizeof(uint16_t));
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("%s: failed to allocate mask buffer: %s", owner, cudaGetErrorString(err));
            d_allowed_mask_ = nullptr;
            return false;
        }
        return true;
    }

    void free() {
        if (d_token_categories_) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_token_categories_));
            d_token_categories_ = nullptr;
        }
        if (d_token_allow_) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_token_allow_));
            d_token_allow_ = nullptr;
        }
        if (d_allowed_mask_) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_allowed_mask_));
            d_allowed_mask_ = nullptr;
        }
    }

    uint8_t* token_allow() const { return d_token_allow_; }
    uint16_t* categories() const { return d_token_categories_; }
    uint16_t* allowed_mask() const { return d_allowed_mask_; }

    bool has_token_allow() const { return d_token_allow_ != nullptr; }

private:
    void swap(ConstrainDeviceBuffers& o) noexcept {
        std::swap(d_token_allow_, o.d_token_allow_);
        std::swap(d_token_categories_, o.d_token_categories_);
        std::swap(d_allowed_mask_, o.d_allowed_mask_);
    }

    uint8_t* d_token_allow_ = nullptr;
    uint16_t* d_token_categories_ = nullptr;
    uint16_t* d_allowed_mask_ = nullptr;
};

}  // namespace imp
