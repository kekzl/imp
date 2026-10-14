// Control vectors (roadmap row 114): load and activate. One vector + scale is active
// engine-wide (executor pointer, captured graphs), switched between requests like LoRA.

#include "runtime/engine.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "lora/control_vector.h"

#include <cuda_runtime.h>

#include <bit>
#include <cmath>

namespace imp {

int Engine::control_vector_load(const std::string& path) {
    if (!model_)
        return 0;
    // The gated-residual stream (Qwen4Exp hc_hidden_) is not the d_model hidden state the
    // vector was trained against.
    if (model_->profile().gated_residual) {
        IMP_LOG_ERROR("control vector: %s refused, gated-residual models are not supported", path.c_str());
        return 0;
    }
    ControlVectorHost host;
    std::string why;
    if (!load_control_vector_file(path, model_->config().d_model, model_->n_layers(), host, &why)) {
        IMP_LOG_ERROR("%s", why.c_str());
        return 0;
    }
    auto dev = std::make_unique<ControlVectorDevice>();
    dev->dirs = VramOwned<float>(vram_alloc_, host.dirs.size(), "control_vector");
    if (!dev->dirs) {
        IMP_LOG_ERROR("control vector: %s refused, %zu bytes not allocatable", path.c_str(),
                      host.dirs.size() * sizeof(float));
        return 0;
    }
    // On the decode stream, which reads it; synced before the host buffer goes away.
    cudaStream_t s = decode_stream();
    if (cudaMemcpyAsync(dev->dirs.get(), host.dirs.data(), host.dirs.size() * sizeof(float),
                        cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess) {
        IMP_LOG_ERROR("control vector: %s upload failed", path.c_str());
        return 0;
    }
    dev->layer_mask = std::move(host.layer_mask);
    dev->d_model = host.d_model;
    int n_dirs = 0;
    for (uint8_t m : dev->layer_mask)
        n_dirs += m;
    control_vectors_.push_back(std::move(dev));
    IMP_LOG_INFO("control vector: %s loaded as id %zu (%d layers)", path.c_str(), control_vectors_.size(),
                 n_dirs);
    return static_cast<int>(control_vectors_.size());  // 1-based id
}

bool Engine::control_vector_set(int id, float scale) {
    if (id < 0 || id > static_cast<int>(control_vectors_.size()) || !std::isfinite(scale))
        return false;
    if (id > 0 && !control_vectors_[id - 1])
        return false;  // unloaded slot
    if (id == 0)
        scale = 0.0f;
    if (id == active_cvec_ && scale == active_cvec_scale_)
        return true;
    active_cvec_ = id;
    active_cvec_scale_ = scale;
    executor_->set_control_vector(id == 0 ? nullptr : control_vectors_[id - 1].get(), scale);
    // Captured graphs hold the add launches (or their absence) with the old pointer and scale.
    invalidate_graphs();
    prefill_graph_runner_.invalidate();
    for (auto& g : decode_graph_pool_)
        g.invalidate();
    IMP_LOG_INFO("control vector: active -> %d (scale %.3f)", id, scale);
    return true;
}

size_t Engine::control_vector_salt_() const {
    if (active_cvec_ == 0)
        return 0;
    return (static_cast<size_t>(active_cvec_) * 0xC2B2AE3D27D4EB4FULL) ^
           (static_cast<size_t>(std::bit_cast<uint32_t>(active_cvec_scale_)) * 0x165667B19E3779F9ULL);
}

}  // namespace imp
