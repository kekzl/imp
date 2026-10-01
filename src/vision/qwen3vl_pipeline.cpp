#include "vision/qwen3vl_pipeline.h"
#include "vision/qwen3vl_vision_load.h"

#include "core/logging.h"
#include "memory/engine_arena.h"
#include "vision/image_processor.h"
#include "vision/qwen3vl_vision_grid.h"
#include "vision/qwen3vl_vision_upload.h"

#include "vision/image_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>

namespace imp {

namespace {

InternVLPreprocessConfig internvl_pp_config(const VisionConfig& c) {
    InternVLPreprocessConfig pc;
    pc.image_size = c.image_size;
    pc.patch_size = c.patch_size;
    std::copy(c.image_mean, c.image_mean + 3, pc.mean);
    std::copy(c.image_std, c.image_std + 3, pc.std);
    return pc;
}

}  // namespace

Qwen3VLPipeline::~Qwen3VLPipeline() { free_buffers(); }

void Qwen3VLPipeline::free_buffers() {
    encoder_.reset();
    internvl_encoder_.reset();
    // The tower's blocks live in the T2 arena and are not in allocs_; releasing its slots here
    // keeps a tower that outlives the arena from being read. allocs_ still holds this
    // pipeline's own scratch, sized from max_patches so it cannot be pre-charged to the arena
    // at open time (docs/audit/SETTLED.md F-12).
    if (tower_ && uploaded_tower_)
        qwen3vl_release_vision_tower(*tower_);
    uploaded_tower_ = false;
    configured_ = false;
    taken_bytes_ = 0;
    d_patches_ = nullptr;
    d_out_ = nullptr;
    d_deepstack_.clear();
    max_patches_ = 0;
}

size_t Qwen3VLPipeline::taken_bytes() const {
    return taken_bytes_ + (encoder_ ? encoder_->taken_bytes() : 0) +
           (internvl_encoder_ ? internvl_encoder_->taken_bytes() : 0);
}

size_t qwen3vl_vision_arena_bytes(VisionModel& tower, int configured_max_patches) {
    const int patches = Qwen3VLPipeline::patch_budget(tower, configured_max_patches);
    return qwen3vl_vision_tower_device_bytes(tower) + Qwen3VLPipeline::demand_bytes(tower, patches);
}

int Qwen3VLPipeline::patch_budget(const VisionModel& tower, int configured) {
    if (tower.config.is_internvl)  // one fixed tile; runtime.vision_max_patches does not apply
        return tower.config.num_patches;
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
    return total +
           (c.is_internvl ? InternVLEncoder::demand_bytes(c) : Qwen3VLEncoder::demand_bytes(c, max_patches));
}

int64_t Qwen3VLPipeline::max_pixels() const {
    if (!tower_)
        return 0;
    const int p = tower_->config.patch_size;
    return static_cast<int64_t>(max_patches_) * p * p;
}

bool Qwen3VLPipeline::init(VisionModel& tower, int max_patches, bool lazy) {
    free_buffers();
    const VisionConfig& c = tower.config;
    if (!c.is_qwen3vl && !c.is_internvl) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: the tower is neither a Qwen3-VL nor an InternVL vision model");
        return false;
    }
    if (c.is_internvl && max_patches != c.num_patches) {
        IMP_LOG_ERROR("InternVL pipeline: patch budget %d, the tile has %d patches", max_patches,
                      c.num_patches);
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
    configured_ = true;
    lazy_ = lazy;
    if (lazy) {
        IMP_LOG_INFO(
            "Qwen3-VL pipeline configured: <= %d patches; tower upload and buffers deferred to the "
            "first image (%.1f MiB of the engine arena stay uncommitted until then)",
            max_patches,
            (qwen3vl_vision_tower_device_bytes(tower) + demand_bytes(tower, max_patches)) /
                (1024.0 * 1024.0));
        return true;
    }
    return build_();
}

bool Qwen3VLPipeline::ensure_ready_() {
    std::lock_guard<std::mutex> lock(ready_mu_);
    if (encoder_ || internvl_encoder_)
        return true;
    if (!configured_ || !tower_)
        return false;
    return build_();
}

// Upload the tower and take the encoder's buffers from the engine arena.
// Under lazy_commit this runs on the first image, on the worker thread, and
// the arena commits what the takes reach into.
bool Qwen3VLPipeline::build_() {
    VisionModel& tower = *tower_;
    const VisionConfig& c = tower.config;
    const int unit = c.merge_size * c.merge_size;
    const int max_patches = max_patches_;

    // Idempotent: a tower already on the device (a second pipeline over the same
    // model) is left alone rather than uploaded twice.
    if (!tower.patch_embd_w.on_device) {
        const auto uploaded = qwen3vl_upload_vision_tower(tower);
        if (!uploaded) {
            IMP_LOG_ERROR("Qwen3-VL pipeline: %s", uploaded.error().c_str());
            free_buffers();
            return false;
        }
        uploaded_tower_ = true;
    }

    const bool encoder_ok = c.is_internvl
                                ? (internvl_encoder_ = std::make_unique<InternVLEncoder>())->init(tower)
                                : (encoder_ = std::make_unique<Qwen3VLEncoder>())->init(tower, max_patches);
    if (!encoder_ok) {
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

bool Qwen3VLPipeline::preprocess(std::span<const uint8_t> data, QwenPatches& out) const {
    if (!tower_)
        return false;
    DecodedImage img;
    if (!decode_image(data, img)) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not decode a %zu-byte image", data.size());
        return false;
    }
    if (tower_->config.is_internvl) {
        const VisionConfig& c = tower_->config;
        if (!internvl_preprocess(img.rgb.data(), img.width, img.height, internvl_pp_config(c), out.data)) {
            IMP_LOG_ERROR("InternVL pipeline: could not preprocess a %dx%d image", img.width, img.height);
            return false;
        }
        out.grid_h = out.grid_w = c.pos_embed_grid;
        out.tokens = c.num_patches;
        out.features = 3 * c.patch_size * c.patch_size;
        return true;
    }
    const bool ok = qwen_patchify(img.rgb.data(), img.width, img.height, patchify_config(), out);
    if (!ok)
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not patchify a %dx%d image", img.width, img.height);
    return ok;
}

bool Qwen3VLPipeline::preprocess_video(std::span<const std::span<const uint8_t>> frames,
                                       std::vector<QwenPatches>& out) const {
    out.clear();
    if (!tower_ || frames.empty() || tower_->config.is_internvl)  // InternVL video: not implemented
        return false;
    std::vector<DecodedImage> rgb(frames.size());
    int w0 = 0, h0 = 0;
    for (size_t f = 0; f < frames.size(); ++f) {
        if (!decode_image(frames[f], rgb[f])) {
            IMP_LOG_ERROR("Qwen3-VL pipeline: could not decode video frame %zu (%zu bytes)", f,
                          frames[f].size());
            return false;
        }
        const int w = rgb[f].width, h = rgb[f].height;
        if (f == 0) {
            w0 = w;
            h0 = h;
        } else if (w != w0 || h != h0) {
            IMP_LOG_ERROR("Qwen3-VL pipeline: video frame %zu is %dx%d, frame 0 is %dx%d", f, w, h, w0, h0);
            return false;
        }
    }
    QwenPatchifyConfig cfg = qwen_video_patchify_config();
    const QwenPatchifyConfig img = patchify_config();
    cfg.patch_size = img.patch_size;
    cfg.merge_size = img.merge_size;
    cfg.temporal_patch_size = img.temporal_patch_size;
    cfg.max_pixels = std::min<int64_t>(cfg.max_pixels, static_cast<int64_t>(frames.size()) * max_pixels());
    std::vector<const uint8_t*> ptrs;
    ptrs.reserve(rgb.size());
    for (const DecodedImage& d : rgb)
        ptrs.push_back(d.rgb.data());
    const bool ok = qwen_patchify_video(ptrs, w0, h0, cfg, out);
    if (!ok)
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not patchify %zu video frames of %dx%d", frames.size(), w0,
                      h0);
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
    if (!ensure_ready_()) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: encode before init, or the deferred tower build failed");
        return false;
    }
    QwenPatches patches;
    if (tower_->config.is_internvl) {
        const VisionConfig& c = tower_->config;
        const InternVLPreprocessConfig pc = internvl_pp_config(c);
        if (!internvl_preprocess(rgb, width, height, pc, patches.data))
            return false;
        patches.grid_h = patches.grid_w = c.pos_embed_grid;
        patches.tokens = c.num_patches;
        patches.features = 3 * c.patch_size * c.patch_size;
        return encode_patches(patches, out, stream);
    }
    if (!qwen_patchify(rgb, width, height, patchify_config(), patches)) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not patchify a %dx%d image", width, height);
        return false;
    }
    IMP_LOG_INFO("Qwen3-VL: %dx%d image -> %dx%d patches", width, height, patches.grid_h, patches.grid_w);
    return encode_patches(patches, out, stream);
}

bool Qwen3VLPipeline::encode_patches(const QwenPatches& patches, Qwen3VLImage& out, cudaStream_t stream) {
    if (!ensure_ready_()) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: encode before init, or the deferred tower build failed");
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

    if (c.is_internvl) {
        if (patches.tokens != c.num_patches) {
            IMP_LOG_ERROR("InternVL pipeline: %d patches, the tile has %d", patches.tokens, c.num_patches);
            return false;
        }
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_patches_, patches.data.data(),
                                           patches.data.size() * sizeof(half), cudaMemcpyHostToDevice,
                                           stream));
        if (!internvl_encoder_->encode(d_patches_, d_out_, stream))
            return false;
        out.grid_rows = out.grid_cols = c.pos_embed_grid / 2;
        out.tokens = c.num_image_tokens;
        out.d_embeddings = d_out_;
        out.d_deepstack.clear();
        return true;
    }
    const auto built = qwen3vl_build_vision_grid(patches.grid_h, patches.grid_w, c.merge_size,
                                                 c.pos_embed_grid);
    if (!built) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: %s", built.error().c_str());
        return false;
    }
    const QwenVisionGrid& grid = *built;

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
    DecodedImage img;
    if (!decode_image_file(path, img)) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not read image '%s'", path.c_str());
        return false;
    }
    return encode_rgb(img.rgb.data(), img.width, img.height, out, stream);
}

}  // namespace imp
