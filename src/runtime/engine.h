#pragma once

#include "model/model.h"
#include "model/chat_template.h"
#include "runtime/scheduler.h"
#include "runtime/request.h"
#include "runtime/batch.h"
#include "runtime/green_ctx.h"
#include "runtime/cuda_graph.h"
#include "runtime/speculative.h"
#include "vision/vision_model.h"
#include "vision/vision_encoder.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/ssm_state.h"
#include "memory/layer_offload.h"
#include "graph/executor.h"
#include "compute/json_constrain.h"
#include <memory>
#include <string>
#include <cuda_runtime.h>

namespace imp {

struct EngineConfig {
    int max_batch_size = 32;
    int max_seq_len = 4096;
    int kv_cache_max_blocks = 0;  // 0 = auto
    bool use_green_contexts = false;
    float green_ctx_prefill_ratio = 0.8f;
    bool use_cuda_graphs = false;
    bool use_pdl = true;
    DType compute_dtype = DType::FP16;

    // Default sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;

    // KV cache dtype: FP16 (default) or FP8_E4M3 for ~50% KV VRAM savings
    DType kv_cache_dtype = DType::FP16;

    // SSM state dtype: FP32 (default) or FP16 for ~50% VRAM savings on h_state
    DType ssm_state_dtype = DType::FP32;

    // VRAM budget: max GPU memory to use (MiB), 0 = use all available
    size_t vram_budget_mb = 0;

    // Layer offloading: number of layers to keep on GPU (-1 = all on GPU, 0 = all offloaded)
    int gpu_layers = -1;

    // Chunked prefill
    int prefill_chunk_size = 0;  // 0 = no chunking

    // FP8 prefill weight cache: uses FP8 E4M3 instead of FP16 for ~2x prefill throughput
    bool use_fp8_prefill = false;

    // NVFP4 decode weight cache: -1=auto, 0=off, 1=additive (FP16+NVFP4), 2=NVFP4 only
    int use_nvfp4_decode = -1;

    // Speculative decoding
    bool enable_speculative = false;
    std::string draft_model_path;
    int spec_k = 4;

    // Vision (multimodal)
    std::string mmproj_path;  // path to mmproj GGUF, empty = text-only
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
                         int top_k = 0, int seed = -1,
                         bool apply_chat_template = true,
                         float min_p = 0.0f,
                         float repetition_penalty = 1.0f,
                         float frequency_penalty = 0.0f,
                         float presence_penalty = 0.0f);

    void add_request(std::shared_ptr<Request> req);

    // Set draft model for speculative decoding after init.
    // Can only be called once, before any generate/decode_step calls.
    bool set_draft_model(const std::string& path, int spec_k = 4);

    // Reset SSM state for a sequence (call on context_reset for hybrid models)
    void reset_ssm_state(int seq_id);

    // Vision: set image for next generation. Returns false if no mmproj loaded.
    bool set_image(const std::string& path);
    bool set_image_from_memory(const uint8_t* data, size_t len);
    void clear_image();
    bool has_vision() const { return vision_encoder_ != nullptr; }
    bool has_vision_input() const { return has_vision_input_; }

    // Accessors for C API
    Scheduler* scheduler() { return scheduler_.get(); }
    KVCacheManager* kv_manager() { return kv_manager_.get(); }
    KVCache* kv_cache() { return kv_cache_raw_; }
    Model* model() { return model_.get(); }
    const ChatTemplate& chat_template() const { return chat_template_; }

private:
    std::shared_ptr<Model> model_;
    EngineConfig config_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<KVCacheManager> kv_manager_;
    KVCache* kv_cache_raw_ = nullptr;  // Non-owning pointer (owned by kv_manager_)
    std::unique_ptr<GraphExecutor> executor_;
    GreenContextManager green_ctx_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t prefill_done_ = nullptr;
    cudaEvent_t decode_done_ = nullptr;
    int next_request_id_ = 0;

    // Pre-allocated GPU batch pool for decode (stable pointers for CUDA Graphs)
    GPUBatchPool decode_batch_pool_;

    // CUDA Graph support for decode iterations.
    CudaGraphRunner decode_graph_runner_;
    int last_decode_batch_size_ = -1;
    int last_decode_max_blocks_ = -1;

    // Pre-allocated block table padding buffer (reused across decode steps)
    std::vector<int> padded_block_table_;

    // SSM state (Mamba2 hybrid models)
    std::unique_ptr<SSMState> ssm_state_;

    // Layer weight offloading
    std::unique_ptr<LayerOffloadManager> offload_mgr_;

    // True when MoE expert weights are on host (not graph-capturable)
    bool experts_on_host_ = false;

    // FP16 weight dequant completed (set in init after pre_dequant_weights)
    bool dequant_done_ = false;

    // Chat template for formatting prompts
    ChatTemplate chat_template_;

    // Vision encoder (multimodal)
    std::unique_ptr<VisionModel> vision_model_;
    std::unique_ptr<VisionEncoder> vision_encoder_;
    half* d_vision_embeddings_ = nullptr;  // [num_image_tokens, d_model] on device
    bool has_vision_input_ = false;        // true when an image is set for next generation
    int32_t vision_soft_token_id_ = -1;    // <image_soft_token> token ID
    int32_t vision_boi_id_ = -1;           // <start_of_image>
    int32_t vision_eoi_id_ = -1;           // <end_of_image>

    // Speculative decoding
    std::shared_ptr<Model> draft_model_;
    std::unique_ptr<KVCacheManager> draft_kv_manager_;
    std::unique_ptr<SpeculativeDecoder> spec_decoder_;

    // Device buffer for penalty token history (reused across steps)
    int32_t* d_penalty_tokens_ = nullptr;
    size_t d_penalty_tokens_capacity_ = 0;  // current allocation capacity in tokens

    // JSON constrainer (lazily initialized on first json_mode request)
    std::unique_ptr<JsonConstrainer> json_constrainer_;

    // Pinned host buffer for graph-captured greedy sampling results.
    // When sampling is included in the CUDA graph, the argmax kernel writes
    // to d_sample_result_ (in executor) and a D2H memcpy copies to this
    // pinned buffer — all inside the graph. After replay, just sync + read.
    int32_t* h_sample_pinned_ = nullptr;
    bool graph_includes_sampling_ = false;  // true when graph was captured with sampling

    // Async conditional graph loop: runs the entire decode autonomously on GPU.
    // Tokens are polled from a ring buffer and delivered one per step() call.
    CudaGraphConditionalRunner async_graph_runner_;
    std::shared_ptr<Request> async_graph_req_;
    int* async_d_block_tables_ = nullptr;  // device memory for async graph loop
    std::vector<int32_t> async_pending_tokens_;  // polled but not yet delivered
    int async_pending_cursor_ = 0;

    // Stream helpers: return green context stream or default stream
    cudaStream_t prefill_stream() const;
    cudaStream_t decode_stream() const;

    // Returns effective free VRAM, capped by vram_budget_mb if set
    size_t effective_free_vram() const;

    // Initialize speculative decoding (called from init() if configured)
    bool init_speculative();

    // GPU-autonomous decode loop using conditional CUDA graph.
    // Returns generated tokens (empty if graph setup failed → caller falls back).
    std::vector<int32_t> try_graph_loop_decode(
        std::shared_ptr<Request> req, int32_t first_token, cudaStream_t stream);

    // Launch async conditional graph loop for single-sequence decode.
    // Returns true if successfully launched; subsequent step() calls poll from ring buffer.
    bool try_launch_async_graph_loop(std::shared_ptr<Request> req,
                                     int32_t first_token, cudaStream_t stream);
};

} // namespace imp
