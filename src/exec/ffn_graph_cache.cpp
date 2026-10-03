#include "exec/ffn_graph_cache.h"

#include "compute/gemm.h"
#include "core/logging.h"

#include <exception>

namespace imp {

int FfnGraphCache::bucket_rows(int n) {
    auto up = [n](int step) { return (n + step - 1) / step * step; };
    if (n <= 1)
        return 0;
    if (n <= 64)
        return up(16);
    if (n <= 1024)
        return up(64);
    if (n <= kMaxRows)
        return up(256);
    return 0;
}

size_t FfnGraphCache::graphs() const {
    size_t n = 0;
    for (const auto& [k, e] : entries_)
        n += e.exec != nullptr;
    return n;
}

void FfnGraphCache::clear() {
    for (auto& [k, e] : entries_)
        if (e.exec)
            IMP_CUDA_CHECK_LOG(cudaGraphExecDestroy(e.exec));
    entries_.clear();
}

void FfnGraphCache::disable_(const char* what, cudaError_t err) {
    IMP_LOG_WARN("FFN prefill graphs disabled for this process: %s failed (%s); FFN runs eager", what,
                 cudaGetErrorString(err));
    (void)cudaGetLastError();
    disabled_ = true;
    clear();
}

void FfnGraphCache::evict_oldest_bucket_() {
    // Every layer of one row count goes together: a bucket is only worth anything complete.
    std::map<int, uint64_t> newest;
    for (const auto& [k, e] : entries_)
        newest[k.first] = std::max(newest[k.first], e.last_use);
    if (newest.size() < static_cast<size_t>(kMaxRowBuckets))
        return;
    int victim = newest.begin()->first;
    for (const auto& [rows, t] : newest)
        if (t < newest[victim])
            victim = rows;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->first.first == victim) {
            if (it->second.exec)
                IMP_CUDA_CHECK_LOG(cudaGraphExecDestroy(it->second.exec));
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

void FfnGraphCache::run(int layer, int rows, uint64_t generation, cudaStream_t stream,
                        const std::function<void()>& fn) {
    if (disabled_ || rows <= 0) {
        fn();
        return;
    }
    if (generation != generation_) {
        clear();
        generation_ = generation;
    }
    const auto key = std::make_pair(rows, layer);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        if (entries_.lower_bound({rows, -1}) == entries_.lower_bound({rows + 1, -1}))
            evict_oldest_bucket_();
        it = entries_.emplace(key, Entry{}).first;
    }
    Entry& e = it->second;
    e.last_use = ++clock_;
    if (e.exec) {
        const cudaError_t err = cudaGraphLaunch(e.exec, stream);
        if (err == cudaSuccess) {
            ++replays_;
            return;
        }
        disable_("cudaGraphLaunch", err);
        fn();
        return;
    }
    // First sighting eager: warms cuBLASLt's algo cache and does any lazy allocation outside capture.
    if (++e.sightings < 2) {
        fn();
        return;
    }
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        disable_("cudaStreamBeginCapture", err);
        fn();
        return;
    }
    cudaGraph_t graph = nullptr;
    gemm_set_lt_capture_allowed(true);
    try {
        fn();
    } catch (const std::exception& ex) {
        // A path that refuses capture (moe_host_args_capture_guard, #2549): nothing ran, run eager.
        // A real error throws again from the eager run.
        gemm_set_lt_capture_allowed(false);
        (void)cudaStreamEndCapture(stream, &graph);
        if (graph)
            IMP_CUDA_CHECK_LOG(cudaGraphDestroy(graph));
        IMP_LOG_WARN("FFN prefill graph capture refused at layer %d, %d rows: %s", layer, rows, ex.what());
        disable_("capture", cudaErrorStreamCaptureUnsupported);
        fn();
        return;
    }
    gemm_set_lt_capture_allowed(false);
    err = cudaStreamEndCapture(stream, &graph);
    if (err == cudaSuccess && graph != nullptr)
        err = cudaGraphInstantiate(&e.exec, graph, 0);
    if (graph)
        IMP_CUDA_CHECK_LOG(cudaGraphDestroy(graph));
    if (err != cudaSuccess || e.exec == nullptr) {
        e.exec = nullptr;
        disable_("capture", err);
        fn();  // the captured work never ran
        return;
    }
    err = cudaGraphLaunch(e.exec, stream);
    if (err != cudaSuccess) {
        disable_("cudaGraphLaunch", err);
        fn();
        return;
    }
    if (graphs() == 1)
        IMP_LOG_INFO("FFN prefill graphs ACTIVE: first capture at %d rows (layer %d)", rows, layer);
}

}  // namespace imp
