#pragma once

#include "model/model.h"
#include "runtime/scheduler.h"
#include "runtime/request.h"
#include "runtime/batch.h"
#include "runtime/green_ctx.h"
#include "runtime/cuda_graph.h"
#include "runtime/speculative.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/ssm_state.h"
#include "memory/layer_offload.h"
#include "graph/executor.h"
#include <memory>
#include <string>
#include <cuda_runtime.h>

namespace imp {

struct EngineConfig {
    int max_batch_size = 32;
    int max_seq_len = 4096;
    int kv_cache_max_blocks = 0;  // 0 = auto
    bool use_green_contexts = false;
    bool use_cuda_graphs = false;
    bool use_pdl = false;
    DType compute_dtype = DType::FP16;

    // Default sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;

    // SSM state dtype: FP32 (default) or FP16 for ~50% VRAM savings on h_state
    DType ssm_state_dtype = DType::FP32;

    // Layer offloading: number of layers to keep on GPU (-1 = all on GPU, 0 = all offloaded)
    int gpu_layers = -1;

    // Speculative decoding
    bool enable_speculative = false;
    std::string draft_model_path;
    int spec_k = 4;
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    bool init(std::shared_ptr<Model> model, const EngineConfig& config);

    // Run one step of inference (prefill or decode depending on scheduler)
    bool step();

    // High-level generate with sampling parameters
    std::string generate(const std::string& prompt, int max_tokens,
                         float temperature = 1.0f, float top_p = 1.0f,
                         int top_k = 0, int seed = -1);

    void add_request(std::shared_ptr<Request> req);

    // Accessors for C API
    Scheduler* scheduler() { return scheduler_.get(); }
    KVCacheManager* kv_manager() { return kv_manager_.get(); }
    KVCache* kv_cache() { return kv_cache_raw_; }
    Model* model() { return model_.get(); }

private:
    std::shared_ptr<Model> model_;
    EngineConfig config_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<KVCacheManager> kv_manager_;
    KVCache* kv_cache_raw_ = nullptr;  // Non-owning pointer (owned by kv_manager_)
    std::unique_ptr<GraphExecutor> executor_;
    GreenContextManager green_ctx_;
    cudaStream_t stream_ = nullptr;
    int next_request_id_ = 0;

    // Pre-allocated GPU batch pool for decode (stable pointers for CUDA Graphs)
    GPUBatchPool decode_batch_pool_;

    // CUDA Graph support for decode iterations.
    CudaGraphRunner decode_graph_runner_;
    int last_decode_batch_size_ = -1;
    int last_decode_max_blocks_ = -1;

    // SSM state (Mamba2 hybrid models)
    std::unique_ptr<SSMState> ssm_state_;

    // Layer weight offloading
    std::unique_ptr<LayerOffloadManager> offload_mgr_;

    // Speculative decoding
    std::shared_ptr<Model> draft_model_;
    std::unique_ptr<KVCacheManager> draft_kv_manager_;
    std::unique_ptr<SpeculativeDecoder> spec_decoder_;

    // Stream helpers: return green context stream or default stream
    cudaStream_t prefill_stream() const;
    cudaStream_t decode_stream() const;

    // Initialize speculative decoding (called from init() if configured)
    bool init_speculative();
};

} // namespace imp
