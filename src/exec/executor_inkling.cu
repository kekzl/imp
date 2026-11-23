// GraphExecutor's Inkling2-Loop branch: the standard path embeds, norms and projects; the 48 layer
// applications run in exec/inkling_forward.h, one kept sequence per slot.

#include "exec/executor.h"

#include "core/logging.h"
#include "exec/inkling_draft.h"
#include "exec/inkling_forward.h"
#include "model/inkling_weights.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace imp {

namespace {

// out = [pos[n]; ids[n]; first block of n_blk tables]: one device-to-host copy per call instead of three.
__global__ void inkling_pack_rows_k(const int* pos, const int32_t* ids, const int* blk, int stride, int n,
                                    int n_blk, int32_t* out) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = pos[i];
        out[n + i] = ids[i];
        if (i < n_blk)
            out[2 * n + i] = blk[(size_t)i * stride];
    }
}

}  // namespace

InklingRuntime::~InklingRuntime() {
    if (d_rows && va)
        va->free(d_rows);
}

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
    // Recycling order on a tie: (holds a checkpoint, decodes without lookahead, bind tick), smallest first.
    const auto cost = [&](size_t i) {
        return std::tuple(slots[i]->reusable(), !slots[i]->has_lookahead(), slot_used[i]);
    };
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
    rt->va = va;
    for (int i = 0; ok && i < want; ++i) {
        auto f = std::make_unique<InklingForward>();
        // The first slot's per-call buffers serve every slot: they run one after the other.
        if (!f->init(rt->w, *va, max_tokens, max_ctx, rt->slots.empty() ? nullptr : rt->slots[0].get())) {
            ok = i > 0;  // fewer slots than asked for: the engine lowers its batch size
            IMP_LOG_WARN("inkling2_loop: sequence slot %d of %d does not fit", i + 1, want);
            break;
        }
        rt->slots.push_back(std::move(f));
    }
    if (!ok)
        throw std::runtime_error("inkling2_loop: " + dir + " could not be loaded (see the error above)");
    // I8B32 copies of the dense projections for the decode steps, when the card keeps kDense8Spare free
    // behind them (draft contexts, library reserve, KV pool come later): 2 slots at 32768 take them, 4
    // slots or a 131072 context decode on fp16.
    constexpr size_t kDense8Spare = static_cast<size_t>(3) << 30;
    size_t need = 0, free_b = 0, total_b = 0;
    for (const InklingLayer& ly : rt->w.layers)
        need += (static_cast<size_t>(ly.proj_rows) * rt->w.cfg.hidden +
                 static_cast<size_t>(rt->w.cfg.hidden) * rt->w.cfg.n_heads * rt->w.cfg.head_dim) *
                17 / 16;
    auto d8 = std::make_unique<InklingDense8>();
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b >= need + kDense8Spare &&
        build_inkling_dense8(rt->w, *va, *d8)) {
        for (auto& f : rt->slots)
            f->set_dense8(d8.get());
        rt->dense8 = std::move(d8);
        IMP_LOG_INFO("inkling2_loop: dense projections of the decode steps as I8B32, %.0f MiB",
                     need / 1048576.0);
    } else {
        IMP_LOG_INFO(
            "inkling2_loop: decode steps read the fp16 dense projections (%.0f MiB free, I8B32 copies "
            "need %.0f MiB + %.0f MiB spare)",
            free_b / 1048576.0, need / 1048576.0, kDense8Spare / 1048576.0);
    }
    rt->max_ctx = max_ctx;
    rt->va = va;
    rt->rows_cap = max_tokens;
    rt->d_rows = static_cast<int32_t*>(
        va->allocate(3 * static_cast<size_t>(max_tokens) * sizeof(int32_t), "inkling_state"));
    rt->cap = std::min(max_tokens, InklingForward::kSeg);
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
    rt.rows.resize(3 * N);
    bool ok = h.qtype == QType::F16 && rt.d_rows && n <= rt.rows_cap;
    if (ok) {
        inkling_pack_rows_k<<<1, 256, 0, stream>>>(state.positions, state.token_ids, state.block_tables,
                                                   std::max(state.max_blocks_per_seq, 1), n, n_blk,
                                                   rt.d_rows);
        IMP_CUDA_CHECK_LAUNCH();
        // The copy waits for the stream's work.
        ok = cudaMemcpyAsync(rt.rows.data(), rt.d_rows, 3 * N * sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream) == cudaSuccess &&
             cudaStreamSynchronize(stream) == cudaSuccess;
        std::copy_n(rt.rows.begin(), N, rt.pos.begin());
        std::copy_n(rt.rows.begin() + static_cast<std::ptrdiff_t>(N), N, rt.ids.begin());
        std::copy_n(rt.rows.begin() + static_cast<std::ptrdiff_t>(2 * N), static_cast<size_t>(n_blk),
                    rt.blk.begin());
    }
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
    // A draft context holds 397 MiB at 32768 positions: two slots get one (4 slots with a context each
    // reached 31983 MiB on the 32 GB card).
    constexpr size_t kLookaheadSlots = 2;
    int on = 0;
    for (auto& slot : rt.slots) {
        if (rt.drafts.size() >= kLookaheadSlots)
            break;
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
