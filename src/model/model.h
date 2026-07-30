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

    // Upload mmap'd weights to GPU, dequantizing as needed.
    // For Q4_0: splits block format into packed nibbles + scales on GPU.
    // For Q8_0: dequantizes to FP16 on GPU.
    // For F16/BF16: direct upload.
    // For F32: converts to compute_dtype and uploads.
    // warm_cache: consult/write the on-disk warm weight cache next to the
    // model (memory/weight_cache_file.h) — skipped when a suspend-to-RAM
    // snapshot is armed (that takes precedence).
    bool upload_weights_gpu(QType compute_dtype = QType::F16, cudaStream_t stream = nullptr,
                            size_t expert_reserve_bytes = 1ULL << 30, bool warm_cache = false,
                            const std::string& warm_cache_dir = {});

    bool gpu_weights_ready() const { return gpu_weights_ready_; }

    // True once the executor's pre-dequant consumed source weights in a way a
    // rebuild cannot survive:
    //  - freed source tensors to reclaim VRAM (Phase-3 MoE expert-source drop,
    //    Phase-4b redundant-source drop — both via release_gpu_allocation(),
    //    which sets this): their .data pointers are dangling afterwards;
    //  - destructively converted sources in place (Phase-3c MXFP4 unpack
    //    compacts the GGUF raw blocks within the SAME buffer) or re-pointed
    //    model tensors at executor-owned cache memory (Phase-3c gdn_alpha/beta
    //    FP16 replace) — sets this explicitly via mark_sources_consumed().
    // Either way a SECOND engine on the same model handle would read freed or
    // already-transformed memory and poison the CUDA context with an illegal
    // access (#830) — Engine::init rejects it up front. Reload the model for a
    // fresh engine. Models whose sources stay intact (e.g. dense Q8_0) leave
    // this false and support create/free/create on one handle.
    void mark_sources_consumed() { sources_consumed_ = true; }
    bool sources_consumed() const { return sources_consumed_; }

    // True once device weight buffers were transformed IN PLACE after upload
    // (Phase-3c MXFP4 compaction, native-MXFP4 GGUF unpack, Gemma-4 fused
    // expert split). A weight snapshot taken afterwards would capture the
    // transformed bytes and the resume replay would re-transform them —
    // WeightSnapshot::capture refuses these models (suspend unsupported, v1).
    // Distinct from sources_consumed_: a DROPPED source only evicts its log
    // record (that tensor re-uploads cold at resume), it does not poison the
    // rest of the snapshot.
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

    // MTP head storage, populated when `model_mtp.safetensors` is present
    // next to the main weights (DeepSeek-V3-family models, e.g. Qwen3.6).
    // Phase 1.A: detection only (info populated, .loaded=false).
    // Phase 1.B: actual weights loaded into named Tensor fields.
    // Phase 2+: forward+verify wiring (see docs/superpowers/specs/2026-05-14-mtp-wiring-design.md).
    std::optional<MtpHead> mtp_;

    // Load-time scratch for NVFP4 prequant scale tensors.
    // Keys:
    //   "L{idx}.{slot}"          per-layer dense (e.g. "L5.wq", "L5.w_gate_shared")
    //   "L{idx}.expert_w_{kind}.{e}"  per-expert (e.g. "L5.expert_w_gate.7")
    //   "out_proj"               LM head
    // Populated by safetensors_loader → weight_map.cpp on the SafeTensors
    // NVFP4-prequant load path. Cleared after executor_pre_dequant.cu's
    // Phase 0 promote() copies the device-side scale pointers and the FP32
    // tensor scalar onto each main weight tensor's .scales / .tensor_scale
    // sidecars. Empty for GGUF and non-NVFP4 SafeTensors models.
    std::unordered_map<std::string, NvFP4PreQuantWeight> nvfp4_scratch_;

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::vector<std::pair<void*, size_t>> split_mmaps_;  // additional shard mmaps

    bool gpu_weights_ready_ = false;
    bool sources_consumed_ = false;         // Phase-4b freed source tensors (#830)
    bool device_sources_mutated_ = false;   // in-place transformed device sources
    std::unique_ptr<WeightUploadLog> upload_log_;
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
