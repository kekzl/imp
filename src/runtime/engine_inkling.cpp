// Inkling2-Loop in the engine: what the first serving path of the arch runs without.

#include "memory/buffer.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "runtime/engine.h"
#include "runtime/prompt_tail.h"
#include "runtime/request.h"
#include "vision/image_decode.h"
#include "vision/image_processor.h"
#include "vision/vision_family.h"

#include <climits>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <algorithm>
#include <vector>

namespace imp {

// One kept sequence per slot (exec/inkling_forward.h): conv state and 40 sliding rings (1120 MiB) per slot,
// full-attention K/V in the paged pool at 36 KiB per position (8 x 4 KiB + the draft's), capped at slots x
// max_seq_len. Context 32768 unless set, up to model_max_length (1048576); 2 slots by default, 4 at most.
void Engine::init_inkling_limits_() {
    if (model_->config().arch != ModelArch::INKLING2)
        return;
    constexpr int kInklingDefaultCtx = 32768, kInklingMaxSlots = 4;
    const int model_ctx = model_->config().max_seq_len > 0 ? model_->config().max_seq_len
                                                           : kInklingDefaultCtx;
    const int slots = config_.max_batch_size > 0 ? std::min(config_.max_batch_size, kInklingMaxSlots) : 2;
    config_.max_batch_size = slots;
    runtime_config_.runtime.max_batch_size = slots;  // runs before Engine::init creates the scheduler
    demote_graphs_(GraphDemotionReason::InklingOwnForward);
    config_.streaming_kv_auto = false;  // dropping middle blocks would cut the forward's context
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
    const int asked = config_.max_seq_len > 0 ? config_.max_seq_len : runtime_config_.runtime.max_seq_len;
    config_.max_seq_len = std::min(asked > 0 ? asked : kInklingDefaultCtx, model_ctx);
    runtime_config_.runtime.max_seq_len = config_.max_seq_len;
    kv_ahead_tokens_ = kInklingKvAhead;
    // The forward writes its pool K/V as fp16: no other dtype (before the dtype resolver reads either).
    const std::string& kv_dt = runtime_config_.kv_cache.dtype;
    if (config_.kv_cache_dtype != QType::F16 || (kv_dt != "auto" && kv_dt != "fp16" && !kv_dt.empty()))
        IMP_LOG_WARN("inkling2_loop: KV cache fp16, kv_cache.dtype %s ignored", kv_dt.c_str());
    config_.kv_cache_dtype = QType::F16;
    runtime_config_.kv_cache.dtype = "fp16";
    IMP_LOG_INFO("inkling2_loop: up to %d sequences, K/V in the paged pool, no speculation, context %d",
                 slots, config_.max_seq_len);
}

void Engine::inkling_fit_slots_() {
    const InklingSlotTable* t = executor_ ? executor_->inkling_slots() : nullptr;
    const int got = t ? static_cast<int>(t->size()) : 0;
    if (got == 0 || got >= config_.max_batch_size)
        return;
    IMP_LOG_WARN("inkling2_loop: %d of %d sequence slots fit, batch size lowered", got,
                 config_.max_batch_size);
    config_.max_batch_size = got;
    runtime_config_.runtime.max_batch_size = got;
    if (scheduler_)
        scheduler_->clamp_max_batch_size(got);
}

void Engine::inkling_unbind_stale_() {
    InklingSlotTable* t = executor_ ? executor_->inkling_slots() : nullptr;
    for (size_t i = 0; t && i < t->size(); ++i)
        if (const int id = t->req_of(i); id >= 0 && !held_request_(id)) {
            IMP_LOG_ERROR("inkling2_loop: slot %zu still bound to ended request %d, unbound", i, id);
            t->release(id);
        }
}

// The slot's ring and conv state resume only its own sequence (InklingForward::match), the pool holds the
// full-attention K/V of the cached blocks: reuse the shorter, on the checkpoint grid. Binds the slot now so
// the first prefill chunk rewinds the same one (inkling_resume_).
int Engine::inkling_prefix_reuse_limit_(Request& req) {
    InklingRuntime* rt = executor_ ? executor_->inkling_runtime() : nullptr;
    if (req.bypass_prefix_cache || ppl_capture_.active || !rt || !kv_manager_)
        return 0;
    const std::span<const int32_t> key = req.key_tokens();
    int match = rt->bind_match(req.id, key);
    if (match < 0) {
        inkling_unbind_stale_();
        match = rt->bind_match(req.id, key);
    }
    if (match <= 0)
        return 0;
    const int bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    std::vector<size_t> hashes;
    const int cached = kv_manager_->longest_cached_prefix_blocks(key, hashes, req.prefix_salt);
    return inkling_prefix_cap_blocks(match, cached,
                                     prompt_reuse_cap_blocks(static_cast<int>(req.input_tokens.size()), bs),
                                     bs, InklingForward::kGrid);
}

int Engine::inkling_resume_(Request& req, int offset, cudaStream_t stream) {
    InklingRuntime* rt = executor_ ? executor_->inkling_runtime() : nullptr;
    if (!rt)
        return offset;
    // A slot is busy while a request is bound to it: the request_tables row "inkling_slot" unbinds it on
    // finish and cancel. An empty span makes the slot forget its sequence.
    const bool reuse = req.prefix_reuse_allowed() && !ppl_capture_.active;
    const std::span<const int32_t> key = reuse ? req.key_tokens() : std::span<const int32_t>();
    bool first = false;
    int r = rt->resume(req.id, key, offset, stream, first);
    if (r < 0) {
        inkling_unbind_stale_();
        r = rt->resume(req.id, key, offset, stream, first);
    }
    if (r < 0) {
        IMP_LOG_ERROR("inkling2_loop: no free sequence slot for request %d, cancelled", req.id);
        return -1;
    }
    if (!first)
        return offset;
    if (r < offset) {
        // The cached blocks below offset get rewritten with the same tokens' K/V (scheduler.cpp F-A10).
        IMP_LOG_WARN("inkling2_loop: seq %d slot resumes at %d, below the %d cached tokens", req.id, r,
                     offset);
        req.prefill_offset = req.cached_tokens = r;
    }
    if (r > 0)
        IMP_LOG_INFO("inkling2_loop: seq %d resumes at %d of %zu prompt tokens", req.id, r,
                     req.input_tokens.size());
    inkling_encode_images_(req, r, stream);
    return r;
}

int Engine::inkling_kv_layers_() const {
    const InklingRuntime* rt = executor_ ? executor_->inkling_runtime() : nullptr;
    if (!rt)
        return -1;
    // The draft's layer only when it can run (InklingForward lookahead, the checkpoint's mtp.safetensors).
    const bool draft = inkling_lookahead_ &&
                       std::filesystem::exists(model_->inkling_dir + "/mtp.safetensors");
    return inkling_kv_apps(rt->w.cfg, draft).n_layers();
}

int Engine::inkling_pool_cap_(int kv_bs) const {
    return executor_ && executor_->inkling_runtime()
               ? inkling_pool_blocks(config_.max_batch_size, config_.max_seq_len, kv_bs)
               : INT_MAX;
}

int Engine::inkling_initial_blocks_(int max_blocks, int ceiling, int kv_bs) const {
    // What the slots committed before the pool held their K/V: one growth step each; the pool grows.
    if (!executor_ || !executor_->inkling_runtime() || ceiling <= 0)
        return max_blocks;
    return std::min(max_blocks, std::max(16, config_.max_batch_size * kInklingKvGrowTokens / kv_bs));
}

void Engine::inkling_attach_pool_() {
    InklingRuntime* rt = executor_ ? executor_->inkling_runtime() : nullptr;
    if (!rt)
        return;
    if (!rt->bind_kv(*kv_cache_raw_))
        throw std::runtime_error("inkling2_loop: the KV pool does not hold the forward's layers");
    kv_cache_raw_->set_grow_step_blocks(kInklingKvGrowTokens / kv_cache_raw_->block_size());
    // A re-plan would move the pool under the slots' bound pointers.
    kv_replan_.armed = false;
    // Reuse stops where the slot can resume: the ring and conv state are per slot (B-111 S3/S4).
    if (kv_manager_->prefix_caching_enabled())
        scheduler_->set_prefix_reuse_limit([this](Request& r) { return inkling_prefix_reuse_limit_(r); });
}

void Engine::inkling_load_vision_() {
    if (model_->config().arch != ModelArch::INKLING2 || model_->inkling_dir.empty() || !model_->tokenizer())
        return;
    if (runtime_config_.runtime.no_vision) {
        IMP_LOG_INFO("runtime.no_vision: inkling vision tower not loaded (1133 MiB), text only");
        return;
    }
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
