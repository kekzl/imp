// Engine-arena scratch a model may never use: taken at first use when the arena is lazy (it commits
// as takes reach it), at init otherwise. docs/VRAM_AUDIT.md "Insurance buffers taken at first use".

#include "exec/executor.h"

#include "core/logging.h"
#include "exec/executor_helpers.h"
#include "memory/engine_arena.h"
#include "quant/nvfp4_gemm.h"

namespace imp {

// The gemm_nvfp4 M>1 fallback workspace of covered_bytes: deferred, or from the arena, or (arena
// closed or short: bare tests) a direct allocation. False: none.
bool GraphExecutor::acquire_nvfp4_dequant_ws_(size_t covered_bytes) {
    nvfp4_dequant_ws_size_ = covered_bytes;
    if (engine_arena().lazy() && engine_arena().remaining() >= covered_bytes) {
        nvfp4_dequant_ws_deferred_ = true;
        set_nvfp4_dequant_workspace_provider(&GraphExecutor::provide_nvfp4_dequant_ws_, this, covered_bytes);
        IMP_LOG_INFO(
            "gemm_nvfp4 dequant workspace: %.2f MiB reserved in the engine arena, taken at the "
            "first M>1 fallback",
            covered_bytes / (1024.0 * 1024.0));
        return true;
    }
    // No graph epoch bump stands at these setters today: their address report is dropped.
    (void)nvfp4_dequant_ws_buf_.set_view(engine_arena().take_bytes(covered_bytes));
    if (!nvfp4_dequant_ws_buf_) {
        (void)nvfp4_dequant_ws_buf_.set_owned(
            DeviceOwned<void>::allocate(vram_alloc_, covered_bytes, "nvfp4_dequant"));
        if (nvfp4_dequant_ws_buf_)
            IMP_LOG_WARN(
                "nvfp4_dequant: engine arena could not supply %.2f MiB (remaining %.2f MiB) — "
                "fell back to a direct allocation",
                covered_bytes / (1024.0 * 1024.0), engine_arena().remaining() / (1024.0 * 1024.0));
    }
    if (!nvfp4_dequant_ws_buf_)
        return false;
    set_nvfp4_dequant_workspace(nvfp4_dequant_ws_buf_.get(), covered_bytes);
    return true;
}

void GraphExecutor::release_nvfp4_dequant_ws_() {
    if (!nvfp4_dequant_ws_buf_ && !nvfp4_dequant_ws_deferred_)
        return;
    set_nvfp4_dequant_workspace(nullptr, 0);  // clears a provider still pointing at this executor
    // I5: releases a private allocation, drops an arena view.
    (void)nvfp4_dequant_ws_buf_.reset();
    nvfp4_dequant_ws_size_ = 0;
    nvfp4_dequant_ws_deferred_ = false;
}

bool GraphExecutor::take_chunk_capture_scratch_() {
    if (chunk_capture_k_)
        return true;
    StableSpan<std::byte> k_slab = engine_arena().take_bytes(chunk_capture_bytes_);
    StableSpan<std::byte> v_slab = engine_arena().take_bytes(chunk_capture_bytes_);
    if (k_slab.empty() || v_slab.empty()) {
        IMP_LOG_WARN(
            "chunk-capture scratch unavailable from the T2 arena (2x %.1f MiB, %.1f MiB "
            "free) — captured verify disabled",
            chunk_capture_bytes_ / (1024.0 * 1024.0), engine_arena().remaining() / (1024.0 * 1024.0));
        chunk_capture_ctx_ = 0;
        return false;
    }
    chunk_capture_k_ = k_slab.as<half>();
    chunk_capture_v_ = v_slab.as<half>();
    IMP_LOG_INFO("chunk-capture K/V scratch: 2x %.1f MiB taken from the engine arena",
                 chunk_capture_bytes_ / (1024.0 * 1024.0));
    return true;
}

void* GraphExecutor::provide_nvfp4_dequant_ws_(void* self) {
    auto* ex = static_cast<GraphExecutor*>(self);
    (void)ex->nvfp4_dequant_ws_buf_.set_view(engine_arena().take_bytes(ex->nvfp4_dequant_ws_size_));
    if (ex->nvfp4_dequant_ws_buf_)
        IMP_LOG_INFO("gemm_nvfp4 dequant workspace: %.2f MiB taken from the engine arena at first use",
                     ex->nvfp4_dequant_ws_size_ / (1024.0 * 1024.0));
    else
        IMP_LOG_WARN("gemm_nvfp4 dequant workspace: the engine arena could not supply %.2f MiB at first use",
                     ex->nvfp4_dequant_ws_size_ / (1024.0 * 1024.0));
    return ex->nvfp4_dequant_ws_buf_.get();
}

}  // namespace imp
