#include "vision/qwen3vl_pipeline.h"

#include "core/logging.h"
#include "memory/vram_allocator.h"
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
    // Order matters: the tower's slots point into `allocs_`, so they are
    // invalidated before the memory goes away rather than after.
    if (tower_ && uploaded_tower_)
        qwen3vl_release_vision_tower(*tower_);
    uploaded_tower_ = false;
    if (alloc_)
        for (void* p : allocs_)
            alloc_->free(p);
    allocs_.clear();
    d_patches_ = nullptr;
    d_out_ = nullptr;
    d_deepstack_.clear();
    max_patches_ = 0;
}

int64_t Qwen3VLPipeline::max_pixels() const {
    if (!tower_)
        return 0;
    const int p = tower_->config.patch_size;
    return static_cast<int64_t>(max_patches_) * p * p;
}

bool Qwen3VLPipeline::init(VisionModel& tower, VRAMAllocator& alloc, int max_patches) {
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
    alloc_ = &alloc;
    max_patches_ = max_patches;

    // Idempotent: a tower already on the device (a second pipeline over the same
    // model) is left alone rather than uploaded twice.
    if (!tower.patch_embd_w.on_device) {
        size_t bytes = 0;
        std::string err;
        if (!qwen3vl_upload_vision_tower(tower, &alloc, allocs_, bytes, err)) {
            IMP_LOG_ERROR("Qwen3-VL pipeline: %s", err.c_str());
            free_buffers();
            return false;
        }
        uploaded_tower_ = true;
    }

    encoder_ = std::make_unique<Qwen3VLEncoder>();
    if (!encoder_->init(tower, &alloc, max_patches)) {
        free_buffers();
        return false;
    }

    const int features = static_cast<int>(tower.patch_embd_w.shape[1]);
    const int merged = max_patches / unit;
    bool ok = true;
    auto take = [&](size_t bytes, const char* tag) -> half* {
        if (!ok)
            return nullptr;
        void* p = alloc.allocate(bytes, tag);
        if (!p) {
            IMP_LOG_ERROR("Qwen3-VL pipeline: out of VRAM for %s", tag);
            ok = false;
            return nullptr;
        }
        allocs_.push_back(p);
        return static_cast<half*>(p);
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

bool Qwen3VLPipeline::encode_rgb(const uint8_t* rgb, int width, int height, Qwen3VLImage& out,
                                 cudaStream_t stream) {
    if (!encoder_) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: encode before init");
        return false;
    }
    const VisionConfig& c = tower_->config;

    QwenPatchifyConfig pc;
    pc.patch_size = c.patch_size;
    pc.merge_size = c.merge_size;
    pc.temporal_patch_size = c.temporal_patch_size;
    // The budget is a hard ceiling here, not a preference: every workspace was
    // sized from it, so an image is scaled down to fit rather than refused.
    pc.max_pixels = std::min<int64_t>(pc.max_pixels, max_pixels());

    QwenPatches patches;
    if (!qwen_patchify(rgb, width, height, pc, patches)) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not patchify a %dx%d image", width, height);
        return false;
    }
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
    IMP_LOG_INFO("Qwen3-VL: %dx%d image -> %dx%d patches -> %d image tokens", width, height, patches.grid_h,
                 patches.grid_w, out.tokens);
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

bool Qwen3VLPipeline::encode_memory(const uint8_t* data, size_t len, Qwen3VLImage& out, cudaStream_t stream) {
    int w = 0, h = 0, ch = 0;
    uint8_t* rgb = stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &ch, 3);
    if (!rgb) {
        IMP_LOG_ERROR("Qwen3-VL pipeline: could not decode a %zu-byte image", len);
        return false;
    }
    const bool ok = encode_rgb(rgb, w, h, out, stream);
    stbi_image_free(rgb);
    return ok;
}

}  // namespace imp
