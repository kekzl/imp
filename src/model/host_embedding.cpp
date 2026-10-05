#include "model/host_embedding.h"

#include "core/fp_bits.h"
#include "core/logging.h"
#include "model/weight_upload_traits.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace imp {

namespace {

// The device upload's rounding (stage_bf16 / F32 staging), so both placements hold the same bits.
uint16_t to_half_bits(float f) { return wupload::float_to_fp16(f); }

void convert_range(const void* src, QType q, size_t lo, size_t hi, uint16_t* dst) {
    if (q == QType::F16) {
        std::memcpy(dst + lo, static_cast<const uint16_t*>(src) + lo, (hi - lo) * sizeof(uint16_t));
    } else if (q == QType::BF16) {
        const auto* s = static_cast<const uint16_t*>(src);
        for (size_t i = lo; i < hi; ++i)
            dst[i] = to_half_bits(bf16_to_float(s[i]));
    } else {
        const auto* s = static_cast<const float*>(src);
        for (size_t i = lo; i < hi; ++i)
            dst[i] = to_half_bits(s[i]);
    }
}

}  // namespace

bool host_embedding_supported(QType q) {
    return q == QType::F32 || q == QType::F16 || q == QType::BF16 || q == QType::Q8_0 || q == QType::Q6_K;
}

void embedding_to_fp16(const void* src, QType q, size_t n, uint16_t* dst, unsigned threads) {
    threads = std::max(1u, threads);
    const size_t chunk = (n + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads && t * chunk < n; ++t)
        pool.emplace_back(convert_range, src, q, t * chunk, std::min(n, (t + 1) * chunk), dst);
    convert_range(src, q, 0, std::min(n, chunk), dst);
    for (auto& th : pool)
        th.join();
}

bool place_embedding_on_host(Tensor& emb, std::vector<PinnedBuffer>& pins) {
    if (!emb.data || emb.on_device || emb.ndim != 2 || !host_embedding_supported(emb.qtype))
        return false;
    const bool raw = emb.qtype == QType::Q8_0 || emb.qtype == QType::Q6_K;
    const size_t n = static_cast<size_t>(emb.shape[0]) * static_cast<size_t>(emb.shape[1]);
    const size_t bytes = raw ? qtype_row_bytes(emb.qtype, emb.shape[1]) * static_cast<size_t>(emb.shape[0])
                             : n * sizeof(uint16_t);
    PinnedBuffer buf = PinnedBuffer::acquire_filled(bytes, HostPinnedKind::Mapped, [&](void* host) {
        if (raw)
            std::memcpy(host, emb.data, bytes);
        else
            embedding_to_fp16(emb.data, emb.qtype, n, static_cast<uint16_t*>(host),
                              std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
    });
    if (!buf || !buf.device()) {
        IMP_LOG_WARN("token embedding: %.1f MiB mapped pinned allocation failed, keeping it in VRAM",
                     bytes / (1024.0 * 1024.0));
        return false;
    }
    emb.data = buf.device();
    if (!raw)
        emb.qtype = QType::F16;
    emb.on_device = true;
    emb.compute_strides();
    IMP_LOG_INFO("token embedding: [%lld x %lld] %s in mapped pinned host memory (%.1f MiB off the device)",
                 static_cast<long long>(emb.shape[0]), static_cast<long long>(emb.shape[1]),
                 qtype_name(emb.qtype), bytes / (1024.0 * 1024.0));
    pins.push_back(std::move(buf));
    return true;
}

}  // namespace imp
