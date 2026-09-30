#pragma once
// --gpu-layers Pass 1 (#2298): the dense matmuls of a planned host layer keep their GGUF bytes
// on host, LayerOffloadManager streams them per forward. No device calls: tests use a stub pin.

#include "core/logging.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "memory/layer_offload.h"
#include "model/model.h"

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace imp {

// Weights Pass 1 may leave on host. Norms, biases and everything else upload as usual.
template <class Layer, class Fn>
void for_each_streamable_matmul(Layer& L, Fn&& fn) {
    for (auto* t : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down})
        fn(*t);
}

// Dense per-layer weights pre_dequant caches (phases 2 and 3). Host-streamed ones are skipped:
// their data is a host pointer outside ensure_layer, and a slot pointer inside it.
template <class ModelT, class Fn>
void for_each_device_dense_weight(const ModelT& model, int n_layers, Fn&& fn) {
    auto visit = [&](const Tensor& w) {
        if (w.data && !w.on_device)
            return;
        fn(w, w.qtype);
    };
    for (int i = 0; i < n_layers; i++) {
        const auto& L = model.layer(i);
        for (const Tensor* w : {&L.wq, &L.wk, &L.wv, &L.wo})
            visit(*w);
    }
    for (int i = 0; i < n_layers; i++) {
        const auto& L = model.layer(i);
        for (const Tensor* w : {&L.ssm_in, &L.ssm_out, &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                &L.w_gate, &L.w_up, &L.w_down})
            visit(*w);
    }
}

// pre_dequant phase 1 FP16-cache source: device-resident, GPU-dequantable or native FP8.
[[nodiscard]] inline bool fp16_cache_source(const Tensor& w, QType qtype, bool dequant_supported) {
    return w.data != nullptr && w.on_device && (dequant_supported || qtype == QType::FP8_E4M3);
}

// Upload stores w's bytes unchanged (raw GGUF rows, qtype and [N, K] kept: stage_rows_raw), so a
// slot holding the host bytes is what the executor would have read. MXFP4 is repacked, F16/F32 converted.
[[nodiscard]] inline bool upload_keeps_bytes_verbatim(const Tensor& w, bool dequant_supported) {
    return w.data != nullptr && !w.on_device && w.ndim == 2 && w.shape[0] > 0 && w.shape[1] > 0 &&
           dequant_supported && w.qtype != QType::MXFP4;
}

// nullptr: dense attention model, Pass 1 may keep planned layers on host. Otherwise the reason
// --gpu-layers cannot stream this model's layers from host.
[[nodiscard]] inline const char* dense_layer_streaming_refusal(const Model& m) {
    if (m.config().n_experts > 0)
        return "MoE model (experts are placed by the expert budget, not --gpu-layers)";
    if (m.config().is_nvfp4_prequant)
        return "NVFP4-prequant checkpoint (pre_dequant promotes its weights on device)";
    if (m.profile().is_encoder)
        return "encoder model (no decoder layer loop)";
    for (int i = 0; i < m.n_layers(); i++) {
        const auto& L = m.layer(i);
        if (L.ssm_in.data || L.gdn_gate.data)
            return "recurrent (SSM/GDN) layers";
    }
    return nullptr;
}

// Never null: the refusal, or why a dense model's upload still left every planned layer on device.
[[nodiscard]] inline const char* layer_streaming_failure(const Model& m) {
    const char* why = dense_layer_streaming_refusal(m);
    return why ? why : "dense model, cause logged above";
}

// Model::upload_host_layers_ for --gpu-layers: the plan's host layers, empty when the plan is
// inactive (gpu_layers < 0 or >= n_layers) or the model cannot stream dense layers.
[[nodiscard]] inline std::vector<bool> gpu_layers_host_plan(const Model& m, int gpu_layers) {
    const LayerOffloadPlan plan = plan_layer_offload(m, gpu_layers);
    if (!plan.active || dense_layer_streaming_refusal(m) != nullptr)
        return {};
    return plan.offloaded;
}

// Host bytes Pass 1 will keep for host_layers (sizes the pinned-RAM check up front).
[[nodiscard]] inline size_t host_keep_bytes(const Model& m, bool (*dequant_supported)(QType)) {
    size_t bytes = 0;
    for (int i = 0; i < m.n_layers() && i < static_cast<int>(m.upload_host_layers_.size()); i++) {
        if (!m.upload_host_layers_[i])
            continue;
        for_each_streamable_matmul(m.layer(i), [&](const Tensor& w) {
            if (upload_keeps_bytes_verbatim(w, dequant_supported(w.qtype)))
                bytes += static_cast<size_t>(w.shape[0]) * qtype_row_bytes(w.qtype, w.shape[1]);
        });
    }
    return bytes;
}

// Matmuls Pass 1 detached from one layer: restore() puts them back after the layer's uploads
// ran, so the upload calls saw data == nullptr and skipped them.
struct HostKeptLayer {
    std::vector<std::pair<Tensor*, Tensor>> kept;
    size_t bytes = 0;
    int pinned = 0;  // of kept.size(); the rest stay on the mmap'd checkpoint (pageable)

    HostKeptLayer() = default;
    HostKeptLayer(const HostKeptLayer&) = delete;
    HostKeptLayer& operator=(const HostKeptLayer&) = delete;
    ~HostKeptLayer() { restore(); }  // an upload failure mid-layer still leaves the layer whole

    void restore() {
        for (auto& [slot, t] : kept)
            *slot = t;
        kept.clear();
    }
};

// Detaches layer L's verbatim matmuls. pin(bytes) -> pinned host pointer or nullptr; a pinned
// copy makes the manager's H2D async, nullptr keeps the checkpoint pointer.
template <class Layer, class Pin>
void detach_host_matmuls(Layer& L, bool (*dequant_supported)(QType), Pin&& pin, HostKeptLayer& out) {
    for_each_streamable_matmul(L, [&](Tensor& w) {
        if (!upload_keeps_bytes_verbatim(w, dequant_supported(w.qtype)))
            return;
        const size_t bytes = static_cast<size_t>(w.shape[0]) * qtype_row_bytes(w.qtype, w.shape[1]);
        Tensor host = w;
        if (void* p = pin(bytes)) {
            std::memcpy(p, w.data, bytes);
            host.data = p;
            out.pinned++;
        }
        out.kept.emplace_back(&w, host);
        out.bytes += bytes;
        w = Tensor();
    });
}

// Pass 1 driver: per layer, detach the upload_host_layers_ matmuls (layer_host_keep plan), run
// upload(L, i) on the rest, restore. pin(bytes) as in detach_host_matmuls.
template <class Pin, class Upload>
[[nodiscard]] bool upload_pass1_keeping_host(Model& m, bool (*dequant_supported)(QType), Pin&& pin,
                                             Upload&& upload) {
    int n_kept = 0, n_pinned = 0;
    size_t bytes = 0;
    for (int i = 0; i < m.n_layers(); ++i) {
        auto& L = m.layer(i);
        HostKeptLayer kept;  // restores the detached matmuls on every exit
        if (i < static_cast<int>(m.upload_host_layers_.size()) && m.upload_host_layers_[i])
            detach_host_matmuls(L, dequant_supported, pin, kept);
        if (!upload(L, i))
            return false;
        n_kept += static_cast<int>(kept.kept.size());
        n_pinned += kept.pinned;
        bytes += kept.bytes;
    }
    if (n_kept > 0)
        IMP_LOG_INFO("--gpu-layers: %d matmul weights (%.2f MiB) stay on host, %d pinned, %d pageable", n_kept,
                     bytes / (1024.0 * 1024.0), n_pinned, n_kept - n_pinned);
    return true;
}

}  // namespace imp
