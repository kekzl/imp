#pragma once

#include <cuda_runtime.h>
#include <utility>

namespace imp {

// RAII wrapper for cudaStream_t.
// Owns the stream and destroys it on scope exit.
class CudaStream {
public:
    CudaStream() = default;

    // Create a stream with the given flags. Returns false from create() on failure.
    [[nodiscard]] bool create(unsigned int flags = cudaStreamNonBlocking) {
        if (stream_)
            cudaStreamDestroy(stream_);
        cudaError_t err = cudaStreamCreateWithFlags(&stream_, flags);
        if (err != cudaSuccess) {
            stream_ = nullptr;
            return false;
        }
        return true;
    }

    ~CudaStream() {
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& o) noexcept : stream_(std::exchange(o.stream_, nullptr)) {}
    CudaStream& operator=(CudaStream&& o) noexcept {
        if (this != &o) {
            if (stream_)
                cudaStreamDestroy(stream_);
            stream_ = std::exchange(o.stream_, nullptr);
        }
        return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }
    explicit operator bool() const noexcept { return stream_ != nullptr; }

    // Release ownership without destroying.
    cudaStream_t release() noexcept { return std::exchange(stream_, nullptr); }

private:
    cudaStream_t stream_ = nullptr;
};

// RAII wrapper for cudaEvent_t.
// Lazily created on first use or explicitly via create().
class CudaEvent {
public:
    CudaEvent() = default;

    [[nodiscard]] bool create(unsigned int flags = cudaEventDisableTiming) {
        if (event_)
            cudaEventDestroy(event_);
        cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
        if (err != cudaSuccess) {
            event_ = nullptr;
            return false;
        }
        return true;
    }

    ~CudaEvent() {
        if (event_)
            cudaEventDestroy(event_);
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& o) noexcept : event_(std::exchange(o.event_, nullptr)) {}
    CudaEvent& operator=(CudaEvent&& o) noexcept {
        if (this != &o) {
            if (event_)
                cudaEventDestroy(event_);
            event_ = std::exchange(o.event_, nullptr);
        }
        return *this;
    }

    cudaEvent_t get() const noexcept { return event_; }
    operator cudaEvent_t() const noexcept { return event_; }
    explicit operator bool() const noexcept { return event_ != nullptr; }

    // Convenience: ensure created, then record on stream.
    bool record(cudaStream_t stream) {
        if (!event_ && !create())
            return false;
        return cudaEventRecord(event_, stream) == cudaSuccess;
    }

    // Release ownership without destroying.
    cudaEvent_t release() noexcept { return std::exchange(event_, nullptr); }

private:
    cudaEvent_t event_ = nullptr;
};

// RAII wrapper for cudaGraph_t. Graphs come out of cudaStreamEndCapture /
// capture APIs as raw handles — adopt them via reset(raw).
class CudaGraph {
public:
    CudaGraph() = default;

    ~CudaGraph() {
        if (graph_)
            cudaGraphDestroy(graph_);
    }

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;

    CudaGraph(CudaGraph&& o) noexcept : graph_(std::exchange(o.graph_, nullptr)) {}
    CudaGraph& operator=(CudaGraph&& o) noexcept {
        if (this != &o) {
            if (graph_)
                cudaGraphDestroy(graph_);
            graph_ = std::exchange(o.graph_, nullptr);
        }
        return *this;
    }

    // Destroy the held graph (if any) and adopt `g`.
    void reset(cudaGraph_t g = nullptr) noexcept {
        if (graph_)
            cudaGraphDestroy(graph_);
        graph_ = g;
    }

    cudaGraph_t get() const noexcept { return graph_; }
    operator cudaGraph_t() const noexcept { return graph_; }
    explicit operator bool() const noexcept { return graph_ != nullptr; }

    // Release ownership without destroying.
    cudaGraph_t release() noexcept { return std::exchange(graph_, nullptr); }

private:
    cudaGraph_t graph_ = nullptr;
};

// RAII wrapper for cudaGraphExec_t (instantiated executable graph).
class CudaGraphExec {
public:
    CudaGraphExec() = default;

    ~CudaGraphExec() {
        if (exec_)
            cudaGraphExecDestroy(exec_);
    }

    CudaGraphExec(const CudaGraphExec&) = delete;
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;

    CudaGraphExec(CudaGraphExec&& o) noexcept : exec_(std::exchange(o.exec_, nullptr)) {}
    CudaGraphExec& operator=(CudaGraphExec&& o) noexcept {
        if (this != &o) {
            if (exec_)
                cudaGraphExecDestroy(exec_);
            exec_ = std::exchange(o.exec_, nullptr);
        }
        return *this;
    }

    // Destroy the held exec (if any) and adopt `e`.
    void reset(cudaGraphExec_t e = nullptr) noexcept {
        if (exec_)
            cudaGraphExecDestroy(exec_);
        exec_ = e;
    }

    cudaGraphExec_t get() const noexcept { return exec_; }
    operator cudaGraphExec_t() const noexcept { return exec_; }
    explicit operator bool() const noexcept { return exec_ != nullptr; }

    // Release ownership without destroying.
    cudaGraphExec_t release() noexcept { return std::exchange(exec_, nullptr); }

private:
    cudaGraphExec_t exec_ = nullptr;
};

}  // namespace imp
