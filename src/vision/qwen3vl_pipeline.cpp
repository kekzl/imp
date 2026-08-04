#include "vision/qwen3vl_pipeline.h"
#include "vision/qwen3vl_vision_load.h"

#include "core/logging.h"
#include "memory/engine_arena.h"
#include "vision/image_processor.h"
#include "vision/qwen3vl_vision_grid.h"
#include "vision/qwen3vl_vision_upload.h"

#include "stb_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>

namespace imp {

Qwen3VLPipeline::~Qwen3VLPipeline() { free_buffers(); }

void Qwen3VLPipeline::free_buffers() {
    encoder_.reset();
    // The tower's blocks live in the T2 arena and are not in `allocs_`; releasing
    // its slots here keeps a tower that outlives the arena from being read. What
    // `allocs_` still holds is this pipeline's own scratch, which is sized from
    // max_patches and so cannot be pre-charged to the arena at open time
    // (docs/audit/SETTLED.md F-12).
    if (tower_ && uploaded_tower_)
        qwen3vl_release_vision_tower(*tower_);
    uploaded_tower_ = false;
    taken_bytes_ = 0;
    d_patches_ = nullptr;
    d_out_ = nullptr;
    d_deepstack_.clear();
    max_patches_ = 0;
}

size_t Qwen3VLPipeline::taken_bytes() const {
    return taken_bytes_ + (encoder_ ? encoder_->taken_bytes() : 0);
}

size_t qwen3vl_vision_arena_bytes(VisionModel& tower, int configured_max_patches) {
    const int patches = Qwen3VLPipeline::patch_budget(tower, configured_max_patches);
    return qwen3vl_vision_tower_device_bytes(tower) + Qwen3VLPipeline::demand_bytes(tower, patches);
}

int Qwen3VLPipeline::patch_budget(const VisionModel& tower, int configured) {
    const int unit = tower.config.merge_size * tower.config.merge_size;
    int budget = configured > 0 ? configured : 4096;  // a 1024x1024 image at patch 16
    if (unit > 0)
        budget -= budget % unit;
    return budget;
}

size_t Qwen3VLPipeline::demand_bytes(const VisionModel& tower, int max_patches) {
    const VisionConfig& c = tower.config;
    const int unit = c.merge_size * c.merge_size;
    if (unit <= 0 || max_patches <= 0)
        return 0;
    const int64_t features = tower.patch_embd_w.shape[1];
    const int64_t merged = max_patches / unit;
    const size_t emb = static_cast<size_t>(merged) * c.out_hidden_size * sizeof(half);
    size_t total = static_cast<size_t>(max_patches) * features * sizeof(half);  // patches
    total += emb;                                                              // out
    total += emb * c.deepstack_indexes.size();                                 // deepstack taps
    return total + Qwen3VLEncoder::demand_bytes(c, max_patches);
}

int64_t Qwen3VLPipeline::max_pixels() const {
    if (!tower_)
        return 0;
    const int p = tower_->config.patch_size;
    return static_cast<int64_t>(max_patches_) * p * p;
}

bool Qwen3VLPipeline::init(VisionModel& tower, int max_patches) {
    free_buffers();
    const VisionConfig& c = tower.config;
    if (!c.is_qwen3vl) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: the tower is not a Qwen3-VL vision model");
        return false;
    }
    const int unit = c.merge_size * c.merge_size;
    if (max_patches <= 0 || max_patches % unit != 0) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: patch budget %d must be positive and a multiple of %d", max_patches,
                      unit);
        return false;
    }
    tower_ = &tower;
    max_patches_ = max_patches;

    // Idempotent: a tower already on the device (a second pipeline over the same
    // model) is left alone rather than uploaded twice.
    if (!tower.patch_embd_w.on_device) {
        size_t bytes = 0;
        std::string err;
        if (!qwen3vl_upload_vision_tower(tower, bytes, err)) {
            IMP_LOG_ERROR("Qwen3-VL pipeline: %s", err.c_str());
            free_buffers();
            return false;
        }
        uploaded_tower_ = true;
    }

    encoder_ = std::make_unique<Qwen3VLEncoder>();
    if (!encoder_->init(tower, max_patches)) {
        free_buffers();
        return false;
    }

    const int features = static_cast<int>(tower.patch_embd_w.shape[1]);
    const int merged = max_patches / unit;
    bool ok = true;
    auto take = [&](size_t bytes, const char* tag) -> half* {
        if (!ok)
            return nullptr;
        auto slab = engine_arena().take_bytes(bytes);
        if (slab.empty()) {
            IMP_LOG_ERROR("Qwen3-VL pipeline: engine arena exhausted for %s (%zu bytes) — the arena "
                          "was reserved without this pipeline",
                          tag, bytes);
            ok = false;
            return nullptr;
        }
        taken_bytes_ += bytes;
        return reinterpret_cast<half*>(slab.data());
    };
    d_patches_ = take(static_cast<size_t>(max_patches) * features * sizeof(half), "vision_patches");
    const size_t emb_bytes = static_cast<size_t>(merged) * c.out_hidden_size * sizeof(half);
    d_out_ = take(emb_bytes, "vision_embeddings");
    d_deepstack_.resize(c.deepstack_indexes.size());
    for (size_t i = 0; i < d_deepstack_.size(); ++i)
        d_deepstack_[i] = take(emb_bytes, "vision_deepstack");
    if (!ok) {
        free_buffers();
        return false;
    }

    IMP_LOG_INFO("Qwen3-VL pipeline ready: <= %d patches (%d image tokens, %lld pixels)", max_patches, merged,
                 static_cast<long long>(max_pixels()));
    return true;
}

QwenPatchifyConfig Qwen3VLPipeline::patchify_config() const {
    QwenPatchifyConfig pc;
    if (tower_) {
        const VisionConfig& c = tower_->config;
        pc.patch_size = c.patch_size;
        pc.merge_size = c.merge_size;
        pc.temporal_patch_size = c.temporal_patch_size;
        // The budget is a hard ceiling here, not a preference: every workspace
        // was sized from it, so an image is scaled down to fit rather than
        // refused.
        pc.max_pixels = std::min<int64_t>(pc.max_pixels, max_pixels());
    }
    return pc;
}

int Qwen3VLPipeline::merged_tokens_of(const QwenPatches& p) const {
    if (!tower_)
        return 0;
    const int unit = tower_->config.merge_size * tower_->config.merge_size;
    return unit > 0 ? p.tokens / unit : 0;
}

size_t Qwen3VLPipeline::embedding_bytes(int tokens) const {
    if (!tower_ || tokens <= 0)
        return 0;
    return static_cast<size_t>(tokens) * tower_->config.out_hidden_size * sizeof(half);
}

int Qwen3VLPipeline::embedding_dim() const { return tower_ ? tower_->config.out_hidden_size : 0; }

int Qwen3VLPipeline::deepstack_taps() const {
    return tower_ ? static_cast<int>(tower_->config.deepstack_indexes.size()) : 0;
}

bool Qwen3VLPipeline::preprocess(const uint8_t* data, size_t len, QwenPatches& out) const {
    if (!tower_)
        return false;
    int w = 0, h = 0, ch = 0;
    uint8_t* rgb = stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &ch, 3);
    if (!rgb) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not decode a %zu-byte image", len);
        return false;
    }
    const bool ok = qwen_patchify(rgb, w, h, patchify_config(), out);
    stbi_image_free(rgb);
    if (!ok)
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not patchify a %dx%d image", w, h);
    return ok;
}

bool Qwen3VLPipeline::encode_patches_to(const QwenPatches& patches, half* d_out,
                                        const std::vector<half*>& d_deepstack, Qwen3VLImage& shape_out,
                                        cudaStream_t stream) {
    if (!encode_patches(patches, shape_out, stream))
        return false;
    const size_t bytes = embedding_bytes(shape_out.tokens);
    if (d_out)
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(d_out, shape_out.d_embeddings, bytes, cudaMemcpyDeviceToDevice, stream));
    for (size_t i = 0; i < d_deepstack.size() && i < shape_out.d_deepstack.size(); ++i)
        if (d_deepstack[i])
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_deepstack[i], shape_out.d_deepstack[i], bytes,
                                               cudaMemcpyDeviceToDevice, stream));
    // Point the shape at the caller's memory so nothing keeps a handle on the
    // shared scratch, which the next request overwrites.
    shape_out.d_embeddings = d_out;
    shape_out.d_deepstack.assign(d_deepstack.begin(), d_deepstack.end());
    return true;
}

bool Qwen3VLPipeline::encode_rgb(const uint8_t* rgb, int width, int height, Qwen3VLImage& out,
                                 cudaStream_t stream) {
    if (!encoder_) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: encode before init");
        return false;
    }
    QwenPatches patches;
    if (!qwen_patchify(rgb, width, height, patchify_config(), patches)) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not patchify a %dx%d image", width, height);
        return false;
    }
    IMP_LOG_INFO("Qwen3-VL: %dx%d image -> %dx%d patches", width, height, patches.grid_h, patches.grid_w);
    return encode_patches(patches, out, stream);
}

bool Qwen3VLPipeline::encode_patches(const QwenPatches& patches, Qwen3VLImage& out, cudaStream_t stream) {
    if (!encoder_) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: encode before init");
        return false;
    }
    const VisionConfig& c = tower_->config;
    if (patches.tokens > max_patches_) {
        // smart_resize honours max_pixels, so this means the two disagree —
        // worth an error rather than a silent truncation.
        IMP_LOG_ERROR("Qwen3-VL pipeline: %d patches exceeds the %d-patch budget", patches.tokens,
                      max_patches_);
        return false;
    }

    QwenVisionGrid grid;
    std::string err;
    if (!qwen3vl_build_vision_grid(patches.grid_h, patches.grid_w, c.merge_size, c.pos_embed_grid, grid,
                                   err)) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: %s", err.c_str());
        return false;
    }

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_patches_, patches.data.data(), patches.data.size() * sizeof(half),
                                       cudaMemcpyHostToDevice, stream));

    std::vector<half*> deep(d_deepstack_.begin(), d_deepstack_.end());
    if (!encoder_->encode(d_patches_, grid, d_out_, deep, stream))
        return false;

    const int unit = c.merge_size * c.merge_size;
    out.grid_rows = patches.grid_h / c.merge_size;
    out.grid_cols = patches.grid_w / c.merge_size;
    out.tokens = patches.tokens / unit;
    out.d_embeddings = d_out_;
    out.d_deepstack.assign(d_deepstack_.begin(), d_deepstack_.end());
    return true;
}

bool Qwen3VLPipeline::encode_file(const std::string& path, Qwen3VLImage& out, cudaStream_t stream) {
    int w = 0, h = 0, ch = 0;
    uint8_t* rgb = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!rgb) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not read image '%s'", path.c_str());
        return false;
    }
    const bool ok = encode_rgb(rgb, w, h, out, stream);
    stbi_image_free(rgb);
    return ok;
}

}  // namespace imp
