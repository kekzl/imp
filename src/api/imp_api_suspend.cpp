// Suspend-to-RAM C API (weight snapshot + GPU release). Split out of
// imp_api.cpp as its own logical unit — see include/imp/imp.h for the
// documented suspend/resume flow and src/memory/weight_snapshot.h for the
// snapshot machinery.

#include "api/imp_internal.h"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "memory/mem_account.h"  // trim_device_mempool

#include <cuda_runtime.h>

#include <exception>
#include <new>

ImpError imp_weights_snapshot_capture(ImpModel model, size_t host_ram_headroom_mb,
                                      ImpWeightSnapshot* out_snap) {
    if (!model || !model->model || !out_snap)
        return IMP_ERROR_INVALID_ARG;
    *out_snap = nullptr;
    try {
        auto snap = imp::WeightSnapshot::capture(*model->model,
                                                 host_ram_headroom_mb * (1024ull * 1024ull));
        auto handle = new (std::nothrow) ImpWeightSnapshot_T();
        if (!handle)
            return IMP_ERROR_OUT_OF_MEMORY;
        handle->snap = std::move(snap);
        *out_snap = handle;
        return IMP_SUCCESS;
    } catch (const imp::SnapshotUnsupportedError& e) {
        IMP_LOG_ERROR("imp_weights_snapshot_capture: %s", e.what());
        return IMP_ERROR_UNSUPPORTED;
    } catch (const imp::SnapshotHostOomError& e) {
        IMP_LOG_ERROR("imp_weights_snapshot_capture: %s", e.what());
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_weights_snapshot_capture: %s", e.what());
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_weights_snapshot_arm(ImpWeightSnapshot snap) {
    if (!snap || !snap->snap)
        return IMP_ERROR_INVALID_ARG;
    imp::weight_snapshot_arm(snap->snap.get());
    return IMP_SUCCESS;
}

void imp_weights_snapshot_free(ImpWeightSnapshot snap) {
    if (!snap)
        return;
    if (snap->snap)
        imp::weight_snapshot_disarm(snap->snap.get());
    delete snap;
}

size_t imp_weights_snapshot_bytes(ImpWeightSnapshot snap) {
    return (snap && snap->snap) ? snap->snap->total_bytes() : 0;
}

int imp_weights_snapshot_hits(ImpWeightSnapshot snap) {
    return (snap && snap->snap) ? snap->snap->hits() : 0;
}

ImpError imp_gpu_release(int device_reset) {
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        IMP_LOG_WARN("imp_gpu_release: device sync failed (%s) — continuing", cudaGetErrorString(sync));
        (void)cudaGetLastError();
    }
    imp::trim_device_mempool();
    if (device_reset) {
        // Free + re-arm every lazily-created module-static CUDA resource
        // (cuBLAS handles, workspaces, scratch) while the context is still
        // valid — their `if (!ptr)` guards would otherwise hand out dangling
        // handles to the next engine after the reset.
        imp::reset_static_cuda_state();
        cudaError_t r = cudaDeviceReset();
        if (r != cudaSuccess) {
            IMP_LOG_ERROR("imp_gpu_release: cudaDeviceReset failed: %s", cudaGetErrorString(r));
            return IMP_ERROR_CUDA;
        }
        IMP_LOG_INFO("imp_gpu_release: CUDA primary context reset — process holds no GPU resources");
    }
    return IMP_SUCCESS;
}
