#pragma once

// Device buffers all four constrainers need (audit finding F-18). Kept separate from
// constrain_common.h: that header carries the JSON category classifier and only compiles once
// the includer has CAT_* flags, so grammar/regex (no categories) could not include it.

#include <cstdint>
#include <cuda_runtime.h>
#include <utility>

#include "core/logging.h"
#include "memory/device_alloc_all.h"
#include "memory/device_owned.h"

namespace imp {

// Replaces four constrainers' separately-managed d_token_allow_/_categories_/_mask_ buffers
// (four lifetimes, four I1 allocation-allowlist entries) with one shared storage; mask LOGIC
// stays in the four classes (genuinely different grammars).
// INVARIANT (#1104): allocation happens at init and fails loudly - a constrainer that cannot
// mask must never look initialised. (Previously one class allocated lazily inside apply_mask()
// and a mid-decode failure returned without masking, answering prose where JSON was promised.)
class ConstrainDeviceBuffers {
public:
    // Move-only through its three owners; each releases once, in ~ConstrainDeviceBuffers or free().
    ConstrainDeviceBuffers() = default;

    // Per-token allow list, one byte per vocabulary entry.
    [[nodiscard]] bool alloc_token_allow(const char* owner, int vocab_size) {
        uint8_t* p = nullptr;
        cudaError_t err = device_alloc_all(dev_req(p, static_cast<size_t>(vocab_size)));
        d_token_allow_.adopt(p, static_cast<size_t>(vocab_size), nullptr);  // null p: empty
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("%s: failed to allocate the %d-token allow list: %s", owner, vocab_size,
                          cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // Per-token category bitmask, uploaded once and never mutated.
    [[nodiscard]] bool alloc_categories(const char* owner, const uint16_t* host, int vocab_size) {
        const size_t bytes = static_cast<size_t>(vocab_size) * sizeof(uint16_t);
        uint16_t* p = nullptr;
        cudaError_t err = device_alloc_all(dev_req(p, bytes));
        d_token_categories_.adopt(p, bytes, nullptr);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("%s: failed to allocate device categories: %s", owner, cudaGetErrorString(err));
            return false;
        }
        err = cudaMemcpy(d_token_categories_.get(), host, bytes, cudaMemcpyHostToDevice);
        // Pageable H2D may return before the DMA lands; drain before a mask kernel reads it (#2275).
        if (err == cudaSuccess)
            err = cudaStreamSynchronize(nullptr);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("%s: failed to copy categories to device: %s", owner, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // Single-word "which categories are allowed right now", rewritten per step.
    [[nodiscard]] bool alloc_allowed_mask(const char* owner) {
        uint16_t* p = nullptr;
        cudaError_t err = device_alloc_all(dev_req(p, sizeof(uint16_t)));
        d_allowed_mask_.adopt(p, sizeof(uint16_t), nullptr);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("%s: failed to allocate mask buffer: %s", owner, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    void free() {
        d_token_categories_.reset();
        d_token_allow_.reset();
        d_allowed_mask_.reset();
    }

    uint8_t* token_allow() const { return d_token_allow_.get(); }
    uint16_t* categories() const { return d_token_categories_.get(); }
    uint16_t* allowed_mask() const { return d_allowed_mask_.get(); }

    bool has_token_allow() const { return static_cast<bool>(d_token_allow_); }

private:
    DeviceOwned<uint8_t> d_token_allow_;
    DeviceOwned<uint16_t> d_token_categories_;
    DeviceOwned<uint16_t> d_allowed_mask_;
};

}  // namespace imp
