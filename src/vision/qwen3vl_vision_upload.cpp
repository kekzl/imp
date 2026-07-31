#include "vision/qwen3vl_vision_upload.h"

#include "core/logging.h"
#include "memory/vram_allocator.h"
#include "vision/qwen3vl_vision_load.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace imp {

namespace {

// BF16 is the top 16 bits of the FP32 pattern, so widening is a shift — no
// table, no rounding decision. F16 is already the target and F32 is a plain
// narrowing; anything else means this checkpoint is not what the loader thinks.
bool to_fp16(const Tensor& t, std::vector<half>& out, std::string& err) {
    const int64_t n = t.numel();
    out.resize(static_cast<size_t>(n));
    switch (t.qtype) {
        case QType::F16:
            std::memcpy(out.data(), t.data, static_cast<size_t>(n) * sizeof(half));
            return true;
        case QType::BF16: {
            const uint16_t* src = static_cast<const uint16_t*>(t.data);
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(f));
                out[static_cast<size_t>(i)] = __float2half(f);
            }
            return true;
        }
        case QType::F32: {
            const float* src = static_cast<const float*>(t.data);
            for (int64_t i = 0; i < n; ++i)
                out[static_cast<size_t>(i)] = __float2half(src[i]);
            return true;
        }
        default:
            err = std::string("vision tower tensor has unsupported dtype ") + qtype_name(t.qtype);
            return false;
    }
}

}  // namespace

bool qwen3vl_upload_vision_tower(VisionModel& model, VRAMAllocator* alloc, size_t& bytes_out,
                                 std::string& err) {
    if (!alloc) {
        err = "no VRAM allocator for the vision tower";
        return false;
    }

    // Collected first, applied last: on any failure the already-allocated blocks
    // are released and every Tensor still points at the host mapping, which is
    // the only state the caller can safely drop the tower from.
    struct Pending {
        Tensor* slot;
        void* device;
        size_t bytes;
    };
    std::vector<Pending> pending;
    std::vector<half> staging;
    size_t total = 0;
    bool ok = true;

    qwen3vl_visit_vision_tensors(model, [&](Tensor& t, const std::string& what) {
        if (!ok)
            return;
        if (t.data == nullptr || t.on_device) {
            err = "vision tower slot '" + what + "' is not a host tensor";
            ok = false;
            return;
        }
        if (!to_fp16(t, staging, err)) {
            err += " (" + what + ")";
            ok = false;
            return;
        }
        const size_t bytes = staging.size() * sizeof(half);
        void* d = alloc->allocate(bytes, "vision_tower");
        if (!d) {
            err = "out of VRAM uploading vision tower tensor '" + what + "'";
            ok = false;
            return;
        }
        if (cudaMemcpy(d, staging.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            alloc->free(d);
            err = "upload failed for vision tower tensor '" + what + "'";
            ok = false;
            return;
        }
        pending.push_back({&t, d, bytes});
        total += bytes;
    });

    if (!ok) {
        for (const auto& p : pending)
            alloc->free(p.device);
        return false;
    }

    for (const auto& p : pending) {
        p.slot->data = p.device;
        p.slot->qtype = QType::F16;
        p.slot->on_device = true;
        p.slot->compute_strides();
        model.gpu_allocs.push_back(p.device);
    }
    model.allocator = alloc;
    bytes_out = total;
    IMP_LOG_INFO("Vision tower: %zu tensors uploaded, %.1f MiB", pending.size(),
                 static_cast<double>(total) / (1024.0 * 1024.0));
    return true;
}

}  // namespace imp
