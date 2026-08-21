// The engine side of Qwen3-VL: image entry points, per-request attachment,
// and M-RoPE position binding — turning a request's (t, h, w) layout into the
// per-step device array the rotary kernels read.
//
// Everything here is a no-op for a model without M-RoPE. For a model WITH it,
// the array is bound on every step — even a text-only prompt, where all three
// rows carry the same value and the rotation is bit-identical to the
// single-axis path. That is not waste: the rope dispatch branches on this
// pointer, and a branch that flips between CUDA-graph capture and replay would
// bake the wrong rotation into every replay.

#include "runtime/engine.h"

#include <fstream>

#include "core/buffer.h"
#include "core/logging.h"
#include "exec/inference_state.h"
#include "runtime/engine_internal.h"
#include "model/image_placeholders.h"
#include "model/mrope_positions.h"
#include "runtime/request.h"

#include <algorithm>

namespace imp {

void Engine::bind_mrope_(InferenceState& state, int n_tokens, int32_t*& buf, int& cap, bool fixed,
                         cudaStream_t stream) {
    const ModelConfig& mc = model_->config_;
    if (!mc.has_mrope() || n_tokens <= 0)
        return;

    // `cap` is the ALLOCATED capacity, so a null buffer always allocates —
    // setting a capacity before the memory exists would memcpy into nothing.
    if (!buf || n_tokens > cap) {
        if (buf && fixed) {
            // The decode buffer is baked into captured graphs; moving it would
            // leave them reading freed memory. Refuse this step's M-RoPE rather
            // than relocate underneath a replay.
            IMP_LOG_ERROR("M-RoPE: decode batch of %d exceeds the %d it was sized for", n_tokens, cap);
            state.mrope = MRopeParams{};
            return;
        }
        // A fixed buffer is sized once, to the batch ceiling, not to this step.
        const int want = fixed ? std::max(n_tokens, std::max(1, config_.max_batch_size)) : n_tokens;
        if (buf)
            vram_alloc_.free(buf);
        const size_t bytes = static_cast<size_t>(3) * want * sizeof(int32_t);
        buf = static_cast<int32_t*>(vram_alloc_.allocate(bytes, "mrope_positions"));
        if (!buf) {
            IMP_LOG_ERROR("M-RoPE: could not allocate %d positions — falling back to single-axis", want);
            cap = 0;
            state.mrope = MRopeParams{};
            return;
        }
        cap = want;
    }

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(buf, h_mrope_scratch_.data(),
                                       h_mrope_scratch_.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                                       stream));
    state.mrope.positions = buf;
    state.mrope.stride = n_tokens;
    state.mrope.sec_t = mc.mrope_section[0];
    state.mrope.sec_h = mc.mrope_section[1];
    state.mrope.sec_w = mc.mrope_section[2];
    state.mrope.interleaved = mc.mrope_interleaved;
}

void Engine::bind_mrope_prefill_(InferenceState& state, const Request& req, int offset, int chunk_len,
                                 cudaStream_t stream) {
    const ModelConfig& mc = model_->config_;
    if (!mc.has_mrope() || chunk_len <= 0)
        return;

    const size_t n_prompt = req.input_tokens.size();
    h_mrope_scratch_.assign(static_cast<size_t>(3) * chunk_len, 0);
    if (req.mrope_positions.size() == 3 * n_prompt && offset + chunk_len <= static_cast<int>(n_prompt)) {
        for (int a = 0; a < 3; ++a)
            for (int i = 0; i < chunk_len; ++i)
                h_mrope_scratch_[static_cast<size_t>(a) * chunk_len + i] =
                    req.mrope_positions[static_cast<size_t>(a) * n_prompt + offset + i];
    } else {
        // No image (or a prompt this engine did not lay out): plain ascending
        // positions on all three axes, which is what the single-axis path does.
        for (int a = 0; a < 3; ++a)
            for (int i = 0; i < chunk_len; ++i)
                h_mrope_scratch_[static_cast<size_t>(a) * chunk_len + i] = offset + i;
    }
    bind_mrope_(state, chunk_len, d_mrope_prefill_, mrope_prefill_cap_, /*fixed=*/false, stream);
}

void Engine::bind_mrope_single_(InferenceState& state, const Request& req, cudaStream_t stream) {
    (void)stream;
    const ModelConfig& mc = model_->config_;
    // No delta buffer means no image, and a zero offset — which is what the
    // unbound path already computes, bit for bit.
    if (!mc.has_mrope() || !req.mrope_delta_dev)
        return;
    state.mrope.pos_delta = static_cast<const int*>(req.mrope_delta_dev->ptr());
    state.mrope.sec_t = mc.mrope_section[0];
    state.mrope.sec_h = mc.mrope_section[1];
    state.mrope.sec_w = mc.mrope_section[2];
    state.mrope.interleaved = mc.mrope_interleaved;
}

void Engine::bind_mrope_decode_(InferenceState& state, const std::vector<std::shared_ptr<Request>>& reqs,
                                cudaStream_t stream) {
    const ModelConfig& mc = model_->config_;
    if (!mc.has_mrope() || reqs.empty())
        return;

    // One offset per token, because a batch mixes an image request (negative
    // offset) with text requests (zero). The kernel adds it to the device
    // `positions` it already reads, so this path and the single-sequence one
    // compute the same number from the same source.
    const int n = static_cast<int>(reqs.size());
    bool any = false;
    h_mrope_scratch_.assign(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        h_mrope_scratch_[static_cast<size_t>(i)] = reqs[i]->mrope_pos_delta;
        any = any || reqs[i]->mrope_pos_delta != 0;
    }
    if (!any)
        return;  // all text: identical to the unbound path, bit for bit

    if (!d_mrope_decode_ || n > mrope_decode_cap_) {
        if (d_mrope_decode_ && mrope_decode_cap_ > 0) {
            // Sized once; a captured graph holds this pointer.
            IMP_LOG_ERROR("M-RoPE: decode batch of %d exceeds the %d it was sized for", n, mrope_decode_cap_);
            return;
        }
        const int want = std::max(n, std::max(1, config_.max_batch_size));
        d_mrope_decode_ = static_cast<int32_t*>(
            vram_alloc_.allocate(static_cast<size_t>(want) * sizeof(int32_t), "mrope_delta"));
        if (!d_mrope_decode_)
            return;
        mrope_decode_cap_ = want;
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_mrope_decode_, h_mrope_scratch_.data(),
                                       static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyHostToDevice,
                                       stream));
    state.mrope.pos_delta = d_mrope_decode_;
    state.mrope.stride = n;
    state.mrope.sec_t = mc.mrope_section[0];
    state.mrope.sec_h = mc.mrope_section[1];
    state.mrope.sec_w = mc.mrope_section[2];
    state.mrope.interleaved = mc.mrope_interleaved;
}

void Engine::free_mrope_buffers_() {
    for (int32_t** b : {&d_mrope_prefill_, &d_mrope_decode_}) {
        if (*b) {
            vram_alloc_.free(*b);
            *b = nullptr;
        }
    }
    mrope_prefill_cap_ = 0;
    mrope_decode_cap_ = 0;
}

// ── Image entry points ────────────────────────────────────────────────
// Here rather than in engine.cpp because they are the Qwen3-VL half of the
// vision surface and belong next to the layout code above.

// Set-image is CPU-only for Qwen3-VL: it patchifies and stops there. That is
// what lets the caller learn the token count before tokenizing, and it keeps
// GPU work off whichever thread happened to call.
bool Engine::set_image(const std::string& path) {
    if (qwen_vision_.is_ready()) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            IMP_LOG_ERROR("Qwen3-VL: could not open image '%s'", path.c_str());
            return false;
        }
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                         std::istreambuf_iterator<char>());
        return set_image_from_memory(bytes);
    }
    return vision_.set_image(path, stream_);
}

bool Engine::set_image_from_memory(std::span<const uint8_t> data) {
    clear_image();
    return add_image_from_memory(data);
}

bool Engine::add_image(const std::string& path) {
    if (!qwen_vision_.is_ready())
        return false;  // the mmproj tower takes one image; there is nothing to add to
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        IMP_LOG_ERROR("Qwen3-VL: could not open image '%s'", path.c_str());
        return false;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return add_image_from_memory(bytes);
}

bool Engine::add_image_from_memory(std::span<const uint8_t> data) {
    if (qwen_vision_.is_ready()) {
        auto patches = std::make_shared<QwenPatches>();
        if (!qwen_vision_.preprocess(data, *patches))
            return false;
        qwen_pending_patches_.push_back(std::move(patches));
        pending_image_hash_ = combine_image_hash(pending_image_hash_, image_content_hash(data));
        return true;
    }
    // The mmproj path stores its image inside VisionPipeline, but the hash has
    // to reach the request either way — see the note at the prefill guard. It
    // holds exactly one image, so adding replaces.
    pending_image_hash_ = image_content_hash(data);
    return vision_.set_image_from_memory(data, stream_);
}

void Engine::clear_image() {
    qwen_pending_patches_.clear();
    pending_image_hash_ = 0;
    vision_.clear_image();
}

bool Engine::preprocess_image_qwen(std::span<const uint8_t> data, QwenPatches& out) const {
    return qwen_vision_.is_ready() && qwen_vision_.preprocess(data, out);
}

int Engine::image_tokens_of(const QwenPatches& patches) const {
    return qwen_vision_.merged_tokens_of(patches);
}

int Engine::pending_image_tokens() const {
    int n = 0;
    for (const auto& p : qwen_pending_patches_)
        if (p)
            n += qwen_vision_.merged_tokens_of(*p);
    return n;
}

std::vector<int> Engine::pending_image_token_counts() const {
    std::vector<int> counts;
    counts.reserve(qwen_pending_patches_.size());
    for (const auto& p : qwen_pending_patches_)
        counts.push_back(p ? qwen_vision_.merged_tokens_of(*p) : 0);
    return counts;
}

// Turns a request's CPU-preprocessed image into its own device buffers and its
// own (t, h, w) layout. Runs on the batch worker: it drives the GPU, so it
// cannot be on an HTTP thread.
bool Engine::encode_qwen_image_for_(Request& req, cudaStream_t stream) {
    if (req.qwen_patches.empty() || !qwen_vision_.is_ready() || qwen_image_pad_id_ < 0)
        return false;

    // One buffer for every image, laid out in prompt order. The kernels index
    // it by "the k-th image token in the prompt", which does not care where one
    // picture ends and the next begins, so concatenation IS the multi-image
    // representation — no per-image indirection is needed downstream.
    std::vector<int> tokens_per_image;
    tokens_per_image.reserve(req.qwen_patches.size());
    int total_tokens = 0;
    for (const auto& p : req.qwen_patches) {
        const int n = p ? qwen_vision_.merged_tokens_of(*p) : 0;
        if (n <= 0) {
            IMP_LOG_ERROR("Qwen3-VL: image %zu produced no tokens", tokens_per_image.size());
            req.qwen_patches.clear();
            return false;
        }
        tokens_per_image.push_back(n);
        total_tokens += n;
    }
    const size_t bytes = qwen_vision_.embedding_bytes(total_tokens);
    if (bytes == 0)
        return false;

    // Per-request buffers, so two concurrent image requests cannot share one
    // encoder output. shared_ptr<Buffer> frees on the last reference, which is
    // what keeps the cancel paths from needing a lifecycle of their own.
    auto emb = std::make_shared<Buffer>(Buffer::device(bytes));
    if (!*emb)
        return false;
    const int taps = qwen_vision_.deepstack_taps();
    std::vector<std::shared_ptr<Buffer>> deep;
    deep.reserve(static_cast<size_t>(taps));
    for (int i = 0; i < taps; ++i) {
        auto b = std::make_shared<Buffer>(Buffer::device(bytes));
        if (!*b)
            return false;
        deep.push_back(std::move(b));
    }

    // The encoder workspace is sized for ONE image (runtime.vision_max_patches),
    // so the images are encoded one at a time, each writing at its own offset.
    std::vector<Qwen3VLImage> shapes;
    shapes.reserve(req.qwen_patches.size());
    size_t token_offset = 0;
    for (size_t i = 0; i < req.qwen_patches.size(); ++i) {
        const size_t elem_offset = token_offset * static_cast<size_t>(qwen_vision_.embedding_dim());
        std::vector<half*> deep_ptrs;
        deep_ptrs.reserve(deep.size());
        for (const auto& b : deep)
            deep_ptrs.push_back(b->as<half>() + elem_offset);
        Qwen3VLImage shape;
        if (!qwen_vision_.encode_patches_to(*req.qwen_patches[i], emb->as<half>() + elem_offset, deep_ptrs,
                                            shape, stream))
            return false;
        shapes.push_back(shape);
        token_offset += static_cast<size_t>(tokens_per_image[i]);
    }
    req.qwen_patches.clear();  // host pixels no longer needed

    if (!build_qwen_layout_(req, shapes))
        return false;
    req.vision_emb = std::move(emb);
    req.deepstack_emb = std::move(deep);
    return true;
}

// The prompt-side half: which positions hold the image, and what that costs the
// M-RoPE sequence. Kept apart from the encode because it is pure bookkeeping
// over the token ids and fails for entirely different reasons.
bool Engine::build_qwen_layout_(Request& req, const std::vector<Qwen3VLImage>& shapes) {
    std::vector<uint8_t> is_image(req.input_tokens.size(), 0);
    int found = 0;
    for (size_t i = 0; i < req.input_tokens.size(); ++i)
        if (req.input_tokens[i] == qwen_image_pad_id_) {
            is_image[i] = 1;
            ++found;
        }
    int expected = 0;
    for (const auto& s : shapes)
        expected += s.tokens;
    if (found != expected) {
        // The prompt and the encoder describe different images. Every position
        // after the mismatch would be shifted, so this is fatal, not tolerated.
        IMP_LOG_ERROR("Qwen3-VL: prompt reserves %d image tokens but %zu image(s) produced %d", found,
                      shapes.size(), expected);
        return false;
    }

    // One grid per image, in prompt order. `qwen_build_mrope_positions` walks
    // the runs of image tokens and takes the next grid at each one, so a second
    // picture continues the (t, h, w) sequence rather than restarting it.
    std::vector<MRopeImageGrid> grids;
    grids.reserve(shapes.size());
    for (const auto& s : shapes)
        grids.push_back({s.grid_rows, s.grid_cols});
    const auto mrope = qwen_build_mrope_positions(is_image, grids, 0);
    if (!mrope) {
        IMP_LOG_ERROR("Qwen3-VL: %s", mrope.error().c_str());
        return false;
    }
    req.mrope_positions = std::move(mrope->pos);
    // Negative whenever the prompt held an image — it occupied more tokens than
    // it cost positions — and it is what keeps generation continuing where the
    // prompt left off instead of jumping past a gap.
    req.mrope_pos_delta = mrope->next_pos - static_cast<int>(req.input_tokens.size());
    // Its own device copy: decode replays dereference this pointer, and a
    // shared one would let a concurrent request's prefill change it mid-run.
    auto delta_buf = std::make_shared<Buffer>(Buffer::device(sizeof(int)));
    if (!*delta_buf)
        return false;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpy(delta_buf->ptr(), &req.mrope_pos_delta, sizeof(int), cudaMemcpyHostToDevice));
    req.mrope_delta_dev = std::move(delta_buf);
    req.vision_token_id = qwen_image_pad_id_;
    req.n_vision_tokens = expected;
    if (shapes.size() == 1)
        IMP_LOG_INFO("Qwen3-VL: %d image tokens (%dx%d), position delta %d", expected, shapes[0].grid_rows,
                     shapes[0].grid_cols, req.mrope_pos_delta);
    else
        IMP_LOG_INFO("Qwen3-VL: %zu images, %d image tokens total, position delta %d", shapes.size(),
                     expected, req.mrope_pos_delta);
    return true;
}

// CLI path: one pending image on the engine, moved onto the first request that
// reserves placeholders for it. The server does not use this — it sets
// `req.qwen_patches` directly, because it admits image requests concurrently.
bool Engine::attach_qwen_image_(Request& req) {
    if (qwen_pending_patches_.empty() || !req.qwen_patches.empty() || req.vision_emb)
        return false;
    bool has_pad = false;
    for (int32_t t : req.input_tokens)
        if (t == qwen_image_pad_id_) {
            has_pad = true;
            break;
        }
    if (!has_pad)
        return false;  // this prompt has no image; the pending one may be a later request's
    req.qwen_patches = std::move(qwen_pending_patches_);
    req.vision_content_hash = pending_image_hash_;
    qwen_pending_patches_.clear();
    pending_image_hash_ = 0;
    return true;
}

bool Engine::encode_image_for(Request& req) {
    if (!req.image || !vision_.is_available())
        return false;
    auto buf = std::make_shared<Buffer>(Buffer::device(vision_.embeddings_bytes()));
    if (!*buf)
        return false;
    if (!vision_.encode_to(*req.image, buf->as<half>(), stream_))
        return false;
    req.vision_emb = std::move(buf);
    req.vision_token_id = vision_.soft_token_id();
    req.n_vision_tokens = vision_.num_image_tokens();
    req.image.reset();  // host pixels no longer needed after encode
    return true;
}

}  // namespace imp
