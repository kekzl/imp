#pragma once

#include <cstddef>

namespace imp {

// The FP16 dequant workspace of gemm_nvfp4's M>1 fallback, owned by the caller (the executor) and
// passed per call (audit D-9): no process global holds a buffer or an executor pointer. provide(ctx)
// is a deferred take (lazy arena): it runs once, on the host, before a kernel reads the buffer.
struct Nvfp4DequantWorkspace {
    void* buf = nullptr;
    size_t size = 0;  // bytes buf covers, or will cover once provide runs
    void* (*provide)(void* ctx) = nullptr;
    void* ctx = nullptr;
};

// The workspace buffer when it covers `needed` bytes, else null. Runs a pending provide once; a
// provide that returns null leaves no workspace (size 0).
[[nodiscard]] inline void* nvfp4_dequant_ws_take(Nvfp4DequantWorkspace* ws, size_t needed) {
    if (ws == nullptr || needed > ws->size)
        return nullptr;
    if (ws->provide != nullptr) {
        ws->buf = ws->provide(ws->ctx);
        ws->provide = nullptr;
        if (ws->buf == nullptr)
            ws->size = 0;
    }
    return ws->buf;
}

}  // namespace imp
