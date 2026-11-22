// Inkling2-Loop in the engine: what the first serving path of the arch runs without.

#include "core/buffer.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "vision/image_decode.h"
#include "vision/image_processor.h"
#include "vision/vision_family.h"

#include <cmath>

#include <algorithm>

namespace imp {

// Eager, one kept sequence per slot: exec/inkling_forward.h keeps K/V and conv state outside the
// paged KV cache and reuses a prefix itself (inkling_resume_), so the engine's prefix cache is off.
// Context is capped at 32768 (8 KiB of global K/V per token); a slot costs 2288 MiB, so 2 by
// default and 4 at most (inkling_fit_slots_ lowers it to what the card took).
void Engine::init_inkling_limits_() {
    if (model_->config().arch != ModelArch::INKLING2)
        return;
    constexpr int kInklingMaxCtx = 32768, kInklingMaxSlots = 4;
    const int slots = config_.max_batch_size > 0 ? std::min(config_.max_batch_size, kInklingMaxSlots) : 2;
    config_.max_batch_size = slots;
    runtime_config_.runtime.max_batch_size = slots;
    if (scheduler_)
        scheduler_->clamp_max_batch_size(slots);
    config_.use_cuda_graphs = false;
    config_.use_prefix_caching = false;
    // The checkpoint's draft runs inside the forward (exec/inkling_draft.h), not as engine speculation;
    // an explicit speculative.mtp_k = 0 turns it off.
    inkling_lookahead_ = !(runtime_config_.was_set("speculative.mtp_k") &&
                           runtime_config_.speculative.mtp_k == 0);
    runtime_config_.speculative.mtp_k = 0;
    runtime_config_.speculative.ngram = false;
    runtime_config_.speculative.capture = false;
    runtime_config_.runtime.prefill_graph = false;
    runtime_config_.runtime.prefill_batch = false;
    // Prefill runs at about 440 rows/s here and a decode step waits for the whole chunk: 8 rows per
    // waiter (x4 fairness = 32 for one ingest beside one stream) instead of 1024.
    runtime_config_.runtime.prefill_chunk_decode_cap =
        std::min(std::max(runtime_config_.runtime.prefill_chunk_decode_cap, 1), 8);
    if (config_.max_seq_len <= 0 || config_.max_seq_len > kInklingMaxCtx)
        config_.max_seq_len = runtime_config_.runtime.max_seq_len > 0
                                  ? std::min(runtime_config_.runtime.max_seq_len, kInklingMaxCtx)
                                  : kInklingMaxCtx;
    runtime_config_.runtime.max_seq_len = config_.max_seq_len;
    IMP_LOG_INFO("inkling2_loop: up to %d sequences, prefix reuse in the forward, no speculation, context %d",
                 slots, config_.max_seq_len);
}

void Engine::inkling_fit_slots_() {
    const int got = executor_ ? static_cast<int>(executor_->inkling_slot_requests().size()) : 0;
    if (got == 0 || got >= config_.max_batch_size)
        return;
    IMP_LOG_WARN("inkling2_loop: %d of %d sequence slots fit, batch size lowered", got,
                 config_.max_batch_size);
    config_.max_batch_size = got;
    runtime_config_.runtime.max_batch_size = got;
    if (scheduler_)
        scheduler_->clamp_max_batch_size(got);
}

int Engine::inkling_resume_(Request& req, int offset, cudaStream_t stream) {
    if (offset != 0 || model_->config().arch != ModelArch::INKLING2 || !executor_)
        return offset;
    // A slot is busy while the request bound to it is in this step's prefill or decode batch.
    const std::vector<int> bound = executor_->inkling_slot_requests();
    std::vector<bool> busy(bound.size());
    const auto live = [&](int id) {
        const auto is = [id](const std::shared_ptr<Request>& r) { return r && r->id == id; };
        return std::any_of(sched_prefill_batch_.begin(), sched_prefill_batch_.end(), is) ||
               std::any_of(sched_decode_batch_.begin(), sched_decode_batch_.end(), is);
    };
    for (size_t i = 0; i < bound.size(); ++i)
        busy[i] = bound[i] >= 0 && bound[i] != req.id && live(bound[i]);
    // Always called at offset 0: an empty span makes the slot forget its sequence.
    const bool reuse = req.prefix_reuse_allowed() && !ppl_capture_.active;
    const int r = executor_->inkling_resume(req.id, reuse ? req.key_tokens() : std::span<const int32_t>(),
                                            busy, stream);
    if (r > 0) {
        IMP_LOG_INFO("inkling2_loop: seq %d resumes at %d of %zu prompt tokens", req.id, r,
                     req.input_tokens.size());
        req.prefill_offset = req.cached_tokens = r;
    }
    inkling_encode_images_(req, r, stream);
    return r;
}

void Engine::inkling_load_vision_() {
    if (model_->config().arch != ModelArch::INKLING2 || model_->inkling_dir.empty() || !model_->tokenizer())
        return;
    auto v = std::make_unique<InklingVision>();
    const int32_t pad = model_->tokenizer()->find_token(
        std::string(vision_prompt_layout(VisionFamily::Inkling2).image_pad));
    if (pad < 0 || !v->load(model_->inkling_dir, vram_alloc_)) {
        IMP_LOG_WARN("inkling2_loop: vision tower not loaded, continuing text-only");
        return;
    }
    qwen_image_pad_id_ = pad;
    inkling_vision_ = std::move(v);
}

bool Engine::inkling_preprocess_image_(std::span<const uint8_t> data, QwenPatches& out) const {
    DecodedImage img;
    if (!inkling_vision_ || !decode_image(data, img) || img.width < 1 || img.height < 1)
        return false;
    int rows = 0, cols = 0;
    inkling_vision_->grid(img.height, img.width, rows, cols);
    // The checkpoint's processor never resizes; a picture over the encoder's token budget is
    // box-averaged down to the largest size that fits.
    int w = img.width, h = img.height;
    while (rows * cols > InklingVision::kMaxTokens) {
        const double s = std::sqrt(static_cast<double>(InklingVision::kMaxTokens) / (rows * cols)) * 0.999;
        w = std::max(1, static_cast<int>(w * s));
        h = std::max(1, static_cast<int>(h * s));
        inkling_vision_->grid(h, w, rows, cols);
    }
    out = QwenPatches{};
    if (w == img.width && h == img.height) {
        out.rgb = std::move(img.rgb);
    } else {
        out.rgb.resize(static_cast<size_t>(w) * h * 3);
        for (int y = 0; y < h; ++y) {
            const int y0 = static_cast<int>(static_cast<int64_t>(y) * img.height / h);
            const int y1 = std::max(y0 + 1, static_cast<int>(static_cast<int64_t>(y + 1) * img.height / h));
            for (int x = 0; x < w; ++x) {
                const int x0 = static_cast<int>(static_cast<int64_t>(x) * img.width / w);
                const int x1 = std::max(x0 + 1,
                                        static_cast<int>(static_cast<int64_t>(x + 1) * img.width / w));
                uint32_t acc[3] = {0, 0, 0};
                for (int yy = y0; yy < y1; ++yy)
                    for (int xx = x0; xx < x1; ++xx)
                        for (int ch = 0; ch < 3; ++ch)
                            acc[ch] += img.rgb[(static_cast<size_t>(yy) * img.width + xx) * 3 + ch];
                const uint32_t n = static_cast<uint32_t>((y1 - y0) * (x1 - x0));
                for (int ch = 0; ch < 3; ++ch)
                    out.rgb[(static_cast<size_t>(y) * w + x) * 3 + ch] = static_cast<uint8_t>(
                        (acc[ch] + n / 2) / n);
            }
        }
    }
    out.px_h = h, out.px_w = w;
    out.grid_h = rows, out.grid_w = cols;
    out.tokens = rows * cols;
    out.features = 3;
    return true;
}

void Engine::inkling_encode_images_(Request& req, int resume, cudaStream_t stream) {
    if (!inkling_vision_ || req.qwen_patches.empty())
        return;
    const std::vector<std::shared_ptr<QwenPatches>> pics = std::move(req.qwen_patches);
    req.qwen_patches.clear();  // host pixels are not needed after this call
    // Prompt positions of the image tokens, in order: picture k owns the next tokens-of-k of them.
    std::vector<int> at;
    for (size_t i = 0; i < req.input_tokens.size(); ++i)
        if (req.input_tokens[i] == qwen_image_pad_id_)
            at.push_back(static_cast<int>(i));
    size_t total = 0;
    for (const auto& p : pics)
        total += p ? static_cast<size_t>(p->tokens) : 0;
    const size_t dim = static_cast<size_t>(inkling_vision_->config().out);
    auto emb = std::make_shared<Buffer>(Buffer::device(std::max<size_t>(total, 1) * dim * sizeof(half)));
    if (total == 0 || total != at.size() || !*emb) {
        IMP_LOG_ERROR(
            "inkling2_loop: request %d reserves %zu image tokens, its %zu picture(s) need %zu: "
            "pictures dropped",
            req.id, at.size(), pics.size(), total);
        return;
    }
    size_t off = 0, encoded = 0;
    for (const auto& p : pics) {
        const size_t n = static_cast<size_t>(p->tokens);
        // Rows before the resumed position are never read: the forward kept that prefix.
        if (at[off + n - 1] >= resume) {
            if (!inkling_vision_->encode(p->rgb.data(), p->px_h, p->px_w, emb->as<half>() + off * dim,
                                         stream)) {
                IMP_LOG_ERROR("inkling2_loop: request %d picture encode failed, pictures dropped", req.id);
                return;
            }
            ++encoded;
        }
        off += n;
    }
    req.vision_emb = std::move(emb);
    req.vision_token_id = qwen_image_pad_id_;
    req.n_vision_tokens = static_cast<int>(total);
    IMP_LOG_INFO("inkling2_loop: seq %d, %zu picture(s), %zu image tokens, %zu encoded", req.id, pics.size(),
                 total, encoded);
}

}  // namespace imp
