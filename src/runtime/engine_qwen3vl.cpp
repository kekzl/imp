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

#include "core/logging.h"
#include "exec/inference_state.h"
#include "runtime/engine_internal.h"
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

// Decode usually runs as a CUDA-graph REPLAY, which never revisits the host
// binding — it only dereferences the pointer baked in at capture. So the delta
// has to be CURRENT on the device before the first replay, which is why it is
// published at prefill rather than only where the decode state is built.
void Engine::publish_mrope_delta_(int delta, cudaStream_t stream) {
    if (!d_mrope_delta_) {
        d_mrope_delta_ = static_cast<int*>(vram_alloc_.allocate(sizeof(int), "mrope_delta"));
        if (!d_mrope_delta_) {
            IMP_LOG_ERROR("M-RoPE: could not allocate the decode delta");
            return;
        }
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_mrope_delta_, &delta, sizeof(int), cudaMemcpyHostToDevice, stream));
}

void Engine::bind_mrope_prefill_(InferenceState& state, const Request& req, int offset, int chunk_len,
                                 cudaStream_t stream) {
    const ModelConfig& mc = model_->config_;
    if (!mc.has_mrope() || chunk_len <= 0)
        return;

    // Published at the START of the request's prefill, so it is in place before
    // the first decode replay — and reset here for a request without an image.
    if (offset == 0)
        publish_mrope_delta_(req.mrope_pos_delta, stream);

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
    const ModelConfig& mc = model_->config_;
    if (!mc.has_mrope())
        return;
    publish_mrope_delta_(req.mrope_pos_delta, stream);
    if (!d_mrope_delta_)
        return;
    state.mrope.pos_delta = d_mrope_delta_;
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

    // Generated text advances all three axes together, so the whole layout
    // collapses to one number per request. That is what makes this safe inside
    // a captured graph: `positions` is advanced device-side during replay and
    // the delta rides alongside it, rather than a host-filled array that a
    // replay would never see refreshed.
    int delta = reqs[0]->mrope_pos_delta;
    for (const auto& r : reqs) {
        if (r->mrope_pos_delta != delta) {
            // Mixed batch: one scalar cannot serve both. Refuse rather than
            // apply one request's image offset to another's text.
            IMP_LOG_WARN("M-RoPE: decode batch mixes position deltas — skipping the offset this step");
            state.mrope = MRopeParams{};
            return;
        }
    }
    publish_mrope_delta_(delta, stream);
    if (!d_mrope_delta_)
        return;
    state.mrope.pos_delta = d_mrope_delta_;
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
    if (d_mrope_delta_) {
        vram_alloc_.free(d_mrope_delta_);
        d_mrope_delta_ = nullptr;
    }
    mrope_prefill_cap_ = 0;
    mrope_decode_cap_ = 0;
}

// ── Image entry points ────────────────────────────────────────────────
// Here rather than in engine.cpp because they are the Qwen3-VL half of the
// vision surface and belong next to the layout code above.

bool Engine::set_image(const std::string& path) {
    // Qwen3-VL first: its tower came in with the checkpoint, so a model can
    // have one without any mmproj being configured.
    if (qwen_vision_.is_ready()) {
        if (!qwen_vision_.encode_file(path, qwen_image_, stream_))
            return false;
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream_));
        qwen_image_pending_ = true;
        return true;
    }
    return vision_.set_image(path, stream_);
}

bool Engine::set_image_from_memory(const uint8_t* data, size_t len) {
    if (qwen_vision_.is_ready()) {
        if (!qwen_vision_.encode_memory(data, len, qwen_image_, stream_))
            return false;
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream_));
        qwen_image_pending_ = true;
        return true;
    }
    return vision_.set_image_from_memory(data, len, stream_);
}

void Engine::clear_image() {
    qwen_image_pending_ = false;
    vision_.clear_image();
}

int Engine::pending_image_tokens() const { return qwen_image_pending_ ? qwen_image_.tokens : 0; }

bool Engine::attach_qwen_image_(Request& req) {
    if (!qwen_image_pending_ || req.vision_emb || qwen_image_pad_id_ < 0)
        return false;

    // Where the image sits in the prompt. The placeholders were expanded to the
    // encoder's token count before the request was submitted; if they were not,
    // the layout below would be built for a different prompt than the one the
    // embeddings land in, so the mismatch is fatal rather than tolerated.
    std::vector<uint8_t> is_image(req.input_tokens.size(), 0);
    int found = 0;
    for (size_t i = 0; i < req.input_tokens.size(); ++i)
        if (req.input_tokens[i] == qwen_image_pad_id_) {
            is_image[i] = 1;
            ++found;
        }
    if (found == 0) {
        // This prompt has no image in it. Leave the pending one alone rather
        // than consuming or complaining about it — a text turn between two
        // image turns is ordinary.
        return false;
    }
    if (found != qwen_image_.tokens) {
        IMP_LOG_ERROR("Qwen3-VL: prompt reserves %d image tokens but the encoder produced %d", found,
                      qwen_image_.tokens);
        return false;
    }

    std::string err;
    int next_pos = 0;
    const std::vector<MRopeImageGrid> grids = {{qwen_image_.grid_rows, qwen_image_.grid_cols}};
    if (!qwen_build_mrope_positions(is_image, grids, 0, req.mrope_positions, next_pos, err)) {
        IMP_LOG_ERROR("Qwen3-VL: %s", err.c_str());
        return false;
    }
    // Negative whenever the prompt held an image — it occupied more tokens than
    // it cost positions — and it is what keeps generation continuing where the
    // prompt left off instead of jumping past a gap.
    req.mrope_pos_delta = next_pos - static_cast<int>(req.input_tokens.size());

    req.vision_token_id = qwen_image_pad_id_;
    req.n_vision_tokens = qwen_image_.tokens;
    qwen_image_pending_ = false;
    IMP_LOG_INFO("Qwen3-VL: attached %d image tokens (%dx%d), position delta %d", qwen_image_.tokens,
                 qwen_image_.grid_rows, qwen_image_.grid_cols, req.mrope_pos_delta);
    return true;
}

}  // namespace imp
