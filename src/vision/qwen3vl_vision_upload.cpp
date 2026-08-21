#include "vision/qwen3vl_vision_upload.h"

#include "core/fp_bits.h"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include "vision/qwen3vl_vision_load.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace imp {

namespace {

// F16 is already the target and F32 is a plain narrowing; anything else means
// this checkpoint is not what the loader thinks. `out` is a caller-owned
// staging buffer reused across tensors, so it stays an out-parameter; only the
// failure path moves into the return type.
std::expected<void, std::string> to_fp16(const Tensor& t, std::vector<half>& out) {
    const int64_t n = t.numel();
    out.resize(static_cast<size_t>(n));
    switch (t.qtype) {
        case QType::F16:
            std::memcpy(out.data(), t.data, static_cast<size_t>(n) * sizeof(half));
            return {};
        case QType::BF16: {
            const uint16_t* src = static_cast<const uint16_t*>(t.data);
            for (int64_t i = 0; i < n; ++i)
                out[static_cast<size_t>(i)] = __float2half(bf16_to_float(src[i]));
            return {};
        }
        case QType::F32: {
            const float* src = static_cast<const float*>(t.data);
            for (int64_t i = 0; i < n; ++i)
                out[static_cast<size_t>(i)] = __float2half(src[i]);
            return {};
        }
        default:
            return std::unexpected(std::string("vision tower tensor has unsupported dtype ") +
                                   qtype_name(t.qtype));
    }
}

}  // namespace

std::expected<size_t, std::string> qwen3vl_upload_vision_tower(VisionModel& model) {
    // Collected first, applied last: on any failure every Tensor still points at
    // the host mapping, which is the only state the caller can safely drop the
    // tower from. The arena takes are not rewound on that path — it is a bump
    // allocator — but a failed tower upload ends vision for this engine anyway,
    // and the bytes go back when the arena closes.
    struct Pending {
        Tensor* slot;
        void* device;
        size_t bytes;
    };
    std::vector<Pending> pending;
    std::vector<half> staging;
    size_t total = 0;
    std::string err;
    bool ok = true;

    qwen3vl_visit_vision_tensors(model, [&](Tensor& t, const std::string& what) {
        if (!ok)
            return;
        if (t.data == nullptr || t.on_device) {
            err = "vision tower slot '" + what + "' is not a host tensor";
            ok = false;
            return;
        }
        if (auto converted = to_fp16(t, staging); !converted) {
            err = converted.error() + " (" + what + ")";
            ok = false;
            return;
        }
        const size_t bytes = staging.size() * sizeof(half);
        auto slab = engine_arena().take_bytes(bytes);
        if (slab.empty()) {
            err = "engine arena exhausted uploading vision tower tensor '" + what +
                  "': the arena was reserved without this tower";
            ok = false;
            return;
        }
        void* d = slab.data();
        if (cudaMemcpy(d, staging.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "upload failed for vision tower tensor '" + what + "'";
            ok = false;
            return;
        }
        pending.push_back({&t, d, bytes});
        total += bytes;
    });

    if (!ok)
        return std::unexpected(err);

    for (const auto& p : pending) {
        p.slot->data = p.device;
        p.slot->qtype = QType::F16;
        p.slot->on_device = true;
        p.slot->compute_strides();
    }
    IMP_LOG_INFO("Vision tower: %zu tensors uploaded, %.1f MiB", pending.size(),
                 static_cast<double>(total) / (1024.0 * 1024.0));
    return total;
}

void qwen3vl_release_vision_tower(VisionModel& model) {
    qwen3vl_visit_vision_tensors(model, [](Tensor& t, const std::string&) {
        if (t.on_device) {
            t.data = nullptr;
            t.on_device = false;
        }
    });
}

}  // namespace imp
