#pragma once

#include "memory/host_pinned.h"
#include "model/hf_config_loader.h"
#include "model/model_config.h"
#include "model/model_profile.h"
#include "model/mtp_head.h"
#include "model/tokenizer.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace imp {

class WeightUploadLog;
// Vision tower of a multimodal checkpoint; held by Model, not the pipeline, since its
// weights are views into the same mapping and share the model's lifetime. Forward-declared
// so model/ has no header dependency on vision/.
struct VisionModel;

class Model {
public:
    // Defined in model.cpp: the unique_ptr<WeightUploadLog> member needs the
    // complete type for destruction, and weight_snapshot.h stays out of this
    // widely-included header.
    Model();
    ~Model();

    const ModelConfig& config() const { return config_; }

    // Architecture-derived facts (D1). Built once via build_profile() after the
    // layers are loaded, before the engine-init resolvers read them. Read-only
    // everywhere else.
    const ModelProfile& profile() const { return profile_; }
    void build_profile() { profile_ = derive_model_profile(*this, config_); }

    // Sampling/EOS defaults shipped by the model author in
    // generation_config.json (SafeTensors only; empty for GGUF). Sentinel
    // values (<0) mean the field was not present.
    const HFConfigLoader::GenerationConfig& generation_config() const { return generation_config_; }
    // C++23 deducing this: one overload serves const and non-const callers.
    template <typename Self>
    auto&& layer(this Self&& self, int i) {
        return self.layers_[i];
    }
    const Tensor& token_embedding() const { return tok_emb_; }
    const Tensor& output_norm() const { return out_norm_; }
    const Tensor& output_proj() const { return out_proj_; }
    int n_layers() const { return static_cast<int>(layers_.size()); }

    Tokenizer* tokenizer() const { return tokenizer_.get(); }
    void set_tokenizer(std::unique_ptr<Tokenizer> tok) { tokenizer_ = std::move(tok); }

    // Uploads mmap'd weights to GPU, dequantizing per dtype (Q4_0 unpacks to nibbles+scales on
    // GPU, Q8_0 dequants to FP16, F16/BF16 direct, F32 converts to compute_dtype). warm_cache
    // consults/writes the on-disk cache; skipped when a suspend-to-RAM snapshot is armed.
    bool upload_weights_gpu(QType compute_dtype = QType::F16, cudaStream_t stream = nullptr,
                            size_t expert_reserve_bytes = 1ULL << 30, bool warm_cache = false,
                            const std::string& warm_cache_dir = {});

    bool gpu_weights_ready() const { return gpu_weights_ready_; }

    // True once pre-dequant consumed source weights unrecoverably: freed via
    // release_gpu_allocation() (dangling .data) or destructively transformed in place
    // (mark_sources_consumed()). A second engine on this handle would read freed/transformed
    // memory (#830); reload for a fresh engine. Intact sources (e.g. dense Q8_0) support
    // create/free/create.
    void mark_sources_consumed() { sources_consumed_ = true; }
    bool sources_consumed() const { return sources_consumed_; }

    // True once device weight buffers were transformed in place after upload (MXFP4
    // compaction, native-MXFP4 GGUF unpack, Gemma-4 fused expert split). WeightSnapshot::capture
    // refuses these (suspend unsupported, v1). Distinct from sources_consumed_: a dropped
    // source only evicts its log record, it doesn't poison the rest of the snapshot.
    void mark_device_sources_mutated() { device_sources_mutated_ = true; }
    bool device_sources_mutated() const { return device_sources_mutated_; }

    // Path the loader was invoked with; empty for synthetic models.
    const std::string& source_path() const { return source_path_; }

    // Upload log for suspend-to-RAM (memory/weight_snapshot.h). Created by
    // upload_weights_gpu; null before the first upload.
    WeightUploadLog* upload_log() const { return upload_log_.get(); }

    // Uploads restored from a warm source (suspend snapshot or disk cache)
    // during the last upload_weights_gpu pass. 0 for a fully-cold load.
    int last_warm_hits() const { return last_warm_hits_; }

    // Remove a pointer from gpu_allocations_ tracking (does not free).
    void release_gpu_allocation(void* ptr);

    // Return true if `ptr` is a BASE pointer tracked in gpu_allocations_
    // (1:1 with a cudaMallocAsync), false if it's an offset into a shared
    // allocation. Used by Phase-4b drop-source to gate cudaFreeAsync.
    bool is_base_gpu_allocation(void* ptr) const;

    // Estimate total raw bytes for all expert packed tensors (for VRAM budget decisions).
    size_t estimate_expert_bytes() const;

    ModelConfig config_;
    ModelProfile profile_;
    HFConfigLoader::GenerationConfig generation_config_;
    Tensor tok_emb_, out_norm_, out_proj_;
    // Encoder embedder extras (#836, nomic-bert): post-embedding LayerNorm
    // (weight+bias) and the token-type embedding table [n_types, d_model]
    // (row 0 is added to every text token's embedding).
    Tensor tok_emb_norm_, tok_emb_norm_bias_, token_types_;
    // (qtype mirrors removed in Stage G — read tok_emb_.qtype directly.)
    TensorID out_proj_id = kInvalidTensorID;  // registry handle for LM head (Task 3.5)
    TensorID tok_emb_id = kInvalidTensorID;   // registry handle for token embedding
    std::vector<TransformerLayer> layers_;
    std::unique_ptr<Tokenizer> tokenizer_;

    // MTP head storage, populated when model_mtp.safetensors sits next to the main weights
    // (DeepSeek-V3 family, e.g. Qwen3.6). Phase 1.A: detection only. Phase 1.B: weights
    // loaded. Phase 2+: forward+verify wiring.
    std::optional<MtpHead> mtp_;

    // True when the checkpoint carries an MTP head this load did NOT take, because
    // speculative.mtp_k defaults to 0 (#1537). Lets /health report a documented +8..+22%
    // decode sitting switched off, instead of requiring a startup-log grep.
    bool mtp_head_available_unloaded_ = false;

    // Load-time scratch for NVFP4 prequant scale tensors, keyed "L{idx}.{slot}" (dense),
    // "L{idx}.expert_w_{kind}.{e}" (per-expert), "out_proj" (LM head). Cleared after Phase 0's
    // promote() copies scale pointers onto each weight's .scales/.tensor_scale. Empty for GGUF
    // and non-NVFP4 SafeTensors.
    std::unordered_map<std::string, NvFP4PreQuantWeight> nvfp4_scratch_;

    // Drops resident pages of weight-file mappings without unmapping: loaders map with
    // MAP_POPULATE+MADV_WILLNEED for the load, pointless after upload since .data becomes the
    // device pointer. MADV_DONTNEED not munmap: a host-resident/offloaded weight still holds a
    // pointer INTO the mapping and there's no iterator to prove otherwise field by field.
    size_t release_weight_pages();

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::vector<std::pair<void*, size_t>> split_mmaps_;  // additional shard mmaps

    bool gpu_weights_ready_ = false;
    bool sources_consumed_ = false;         // Phase-4b freed source tensors (#830)
    bool device_sources_mutated_ = false;   // in-place transformed device sources
    std::unique_ptr<WeightUploadLog> upload_log_;
    // Null unless the checkpoint carries one. Populated in two steps by the
    // parties that own each half: the config loader fills `->config` from
    // config.json, weight_map fills the tensors from the shard map.
    std::unique_ptr<VisionModel> vision_tower;
    int last_warm_hits_ = 0;
    // Path the loader was invoked with (GGUF file or SafeTensors directory).
    // Used to derive the on-disk warm-cache location and its fingerprint
    // (memory/weight_cache_file.h). Empty for synthetically-built models.
    std::string source_path_;
    std::vector<void*> gpu_allocations_;
    // mmap regions page-locked via cudaHostRegister. Owners (T5b): the
    // unregister loop that used to live in ~Model is gone.
    std::vector<HostRegistration> host_pinned_;
    // Pinned expert buffers for the WSL2 DMA path. Owners since T5b, so the
    // teardown loop that used to cudaFreeHost each entry is gone
    // (memory/host_pinned.h).
    std::vector<PinnedBuffer> host_pinned_allocs_;
    // Heap-allocated buffers used to hold permuted weight copies (e.g. Qwen3.5/3.6
    // GDN head reordering grouped→tiled). Freed with std::free() in destructor.
    std::vector<void*> host_owned_buffers_;
};

}  // namespace imp
