// GraphExecutor's Inkling2-Loop branch: the standard path embeds, norms and projects; the 48 layer
// applications run in exec/inkling_forward.h, one kept sequence per slot.

#include "exec/executor.h"

#include "core/logging.h"
#include "exec/inkling_draft.h"
#include "exec/inkling_forward.h"
#include "model/inkling_weights.h"

#include <stdexcept>

namespace imp {

int InklingRuntime::slot_of_req(int req) const {
    for (size_t i = 0; i < slot_req.size(); ++i)
        if (slot_req[i] == req)
            return static_cast<int>(i);
    return -1;
}

int InklingRuntime::slot_of_block(int block) const {
    for (size_t i = 0; i < slot_block.size(); ++i)
        if (slot_block[i] == block)
            return static_cast<int>(i);
    return -1;
}

int InklingRuntime::bind(int req, std::span<const int32_t> tokens, const std::vector<bool>& busy) {
    int best = -1, best_match = -1;
    // Recycling order on a tie: (holds a checkpoint, bind tick), smallest first.
    const auto cost = [&](size_t i) { return std::pair(slots[i]->reusable(), slot_used[i]); };
    for (size_t i = 0; i < slots.size(); ++i) {
        if (i < busy.size() && busy[i])
            continue;
        const int m = slots[i]->match(tokens);
        if (m > best_match || (m == best_match && cost(i) < cost(static_cast<size_t>(best)))) {
            best = static_cast<int>(i);
            best_match = m;
        }
    }
    if (best >= 0) {
        slot_req[static_cast<size_t>(best)] = req;
        slot_block[static_cast<size_t>(best)] = -1;
        slot_used[static_cast<size_t>(best)] = ++tick;
    }
    return best;
}

std::unique_ptr<InklingRuntime> make_inkling_runtime(const Model& model, VRAMAllocator* va, int max_tokens,
                                                     int max_ctx, int slots) {
    const std::string& dir = model.inkling_dir;
    if (dir.empty())
        return nullptr;
    auto rt = std::make_unique<InklingRuntime>();
    bool ok = va && model.token_embedding().data && model.output_proj().data && model.output_norm().data &&
              load_inkling_weights(dir, rt->w, *va, /*with_vocab=*/false);
    const int want = slots > 0 ? slots : 2;
    for (int i = 0; ok && i < want; ++i) {
        auto f = std::make_unique<InklingForward>();
        if (!f->init(rt->w, *va, max_tokens, max_ctx)) {
            ok = i > 0;  // fewer slots than asked for: the engine lowers its batch size
            IMP_LOG_WARN("inkling2_loop: sequence slot %d of %d does not fit", i + 1, want);
            break;
        }
        rt->slots.push_back(std::move(f));
    }
    if (!ok)
        throw std::runtime_error("inkling2_loop: " + dir + " could not be loaded (see the error above)");
    rt->max_ctx = max_ctx;
    rt->cap = std::min(max_tokens, InklingForward::kGrid);
    rt->slot_req.assign(rt->slots.size(), -1);
    rt->slot_block.assign(rt->slots.size(), -1);
    rt->slot_used.assign(rt->slots.size(), 0);
    return rt;
}

int GraphExecutor::run_inkling_(const InferenceState& state, Tensor& h, int n, cudaStream_t stream) {
    if (!inkling_)
        return model_->config().n_layers;
    InklingRuntime& rt = *inkling_;
    if (!rt.draft_tried)
        inkling_attach_draft_(stream);
    const size_t N = static_cast<size_t>(n);
    // Host reads of the rows' positions, tokens and first KV blocks: this path is eager.
    const bool decode = !state.is_prefill;
    const int n_blk = state.block_tables ? (decode ? n : 1) : 0;
    rt.ids.resize(N);
    rt.pos.resize(N);
    rt.blk.assign(N, -1);
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    bool ok = h.qtype == QType::F16 &&
              cudaMemcpy(rt.pos.data(), state.positions, N * sizeof(int), cudaMemcpyDeviceToHost) ==
                  cudaSuccess &&
              cudaMemcpy(rt.ids.data(), state.token_ids, N * sizeof(int32_t), cudaMemcpyDeviceToHost) ==
                  cudaSuccess &&
              (n_blk == 0 ||
               cudaMemcpy2D(rt.blk.data(), sizeof(int), state.block_tables,
                            static_cast<size_t>(std::max(state.max_blocks_per_seq, 1)) * sizeof(int),
                            sizeof(int), static_cast<size_t>(n_blk), cudaMemcpyDeviceToHost) == cudaSuccess);
    half* hd = static_cast<half*>(h.data);
    if (ok && !decode) {
        // One sequence: the slot its request was bound to (a forward outside a request takes slot 0).
        int s = rt.slot_of_req(state.prefill_req_id);
        if (s < 0)
            s = rt.pos[0] == 0 ? 0 : rt.slot_of_block(rt.blk[0]);
        ok = s >= 0;
        if (ok) {
            rt.slot_block[static_cast<size_t>(s)] = rt.blk[0];
            ok = rt.slots[static_cast<size_t>(s)]->run(hd, n, rt.pos[0], stream, rt.ids.data(),
                                                       state.vision_embeddings ? state.vision_token_id : -1);
        }
    }
    // Decode: one row per sequence, each on the slot that holds its sequence.
    for (size_t i = 0; ok && decode && i < N; ++i) {
        const int s = rt.slots.size() == 1 ? 0 : rt.slot_of_block(rt.blk[i]);
        ok = s >= 0 && rt.slots[static_cast<size_t>(s)]->run(hd + i * static_cast<size_t>(h.shape[1]), 1,
                                                             rt.pos[i], stream, &rt.ids[i]);
    }
    if (!ok)
        throw std::runtime_error("inkling2_loop: forward failed at position " + std::to_string(rt.pos[0]));
    return 0;
}

// Prefix reuse: binds the request to a slot and returns the position its prefill resumes at.
int GraphExecutor::inkling_resume(int req, std::span<const int32_t> tokens, const std::vector<bool>& busy,
                                  cudaStream_t stream) {
    if (!inkling_)
        return 0;
    int s = inkling_->bind(req, tokens, busy);
    if (s < 0) {
        // The scheduler admits one request per slot, so this is a bookkeeping fault: say so and
        // recycle the oldest binding instead of failing the request.
        IMP_LOG_WARN("inkling2_loop: no free sequence slot for request %d, recycling the oldest", req);
        s = inkling_->bind(req, tokens, {});
    }
    return inkling_->slots[static_cast<size_t>(s)]->rewind(tokens, stream);
}

// The checkpoint's draft model behind every slot's decode lookahead; without it (no mtp.safetensors,
// no FP8 view of the LM head, speculative.mtp_k = 0, no VRAM) the slots decode one row per step.
void GraphExecutor::inkling_attach_draft_(cudaStream_t stream) {
    InklingRuntime& rt = *inkling_;
    rt.draft_tried = true;
    const Tensor& emb = model_->token_embedding();
    InklingHeadView head;
    head.vocab = model_->config().vocab_size;
    if (!inkling_lookahead_ || emb.qtype != QType::F16 || !emb.data ||
        !lm_head_fp8_view(head.w8, head.row_scales)) {
        IMP_LOG_INFO("inkling2_loop: decode lookahead off (%s)",
                     inkling_lookahead_ ? "needs an fp16 embedding table and the FP8 LM head"
                                        : "speculative.mtp_k = 0");
        return;
    }
    auto w = std::make_shared<InklingDraftWeights>();
    if (!w->load(model_->inkling_dir, *vram_alloc_, rt.w.embed_norm, head, stream))
        return;
    rt.draft_w = w;
    int on = 0;
    for (auto& slot : rt.slots) {
        auto d = std::make_shared<InklingDraft>();
        if (!d->init(*w, *vram_alloc_, rt.max_ctx, rt.cap) ||
            !slot->enable_lookahead(d->hooks(), w->cfg.taps, static_cast<const half*>(emb.data)))
            break;
        rt.drafts.push_back(std::move(d));
        ++on;
    }
    IMP_LOG_INFO("inkling2_loop: decode lookahead on %d of %zu slots, %d drafts per step", on,
                 rt.slots.size(), w->cfg.block - 1);
}

std::vector<int> GraphExecutor::inkling_slot_requests() const {
    return inkling_ ? inkling_->slot_req : std::vector<int>();
}

}  // namespace imp
