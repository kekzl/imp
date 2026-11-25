// GraphExecutor's Inkling2-Loop branch: the standard path embeds, norms and projects; the 48 layer
// applications run in exec/inkling_forward.h, one kept sequence per slot.

#include "exec/executor.h"
#include "memory/vram_query.h"

#include "core/logging.h"
#include "exec/inkling_draft.h"
#include "exec/inkling_forward.h"
#include "model/inkling_weights.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace imp {

namespace {

// out = [pos[n]; ids[n]]: one device-to-host copy per call instead of two.
__global__ void inkling_pack_rows_k(const int* pos, const int32_t* ids, int n, int32_t* out) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = pos[i];
        out[n + i] = ids[i];
    }
}

}  // namespace

InklingRuntime::~InklingRuntime() {
    // I5: before the slots, drafts and weights go (the order the hand frees had).
    d_rows.reset();
    for (auto& p : look_logits)
        p.reset();
}

int InklingRuntime::bind(int req, std::span<const int32_t> tokens) {
    // Smallest first: (longer kept prefix, holds a checkpoint, decodes without lookahead); the table
    // breaks a tie by bind tick.
    return table.bind(req, [&](size_t i) {
        return std::tuple(-slots[i]->match(tokens), slots[i]->reusable(), !slots[i]->has_lookahead());
    });
}

std::unique_ptr<InklingRuntime> make_inkling_runtime(const Model& model, VRAMAllocator* va, int max_tokens,
                                                     int max_ctx, int slots, Backend* growable) {
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
        if (!f->init(rt->w, *va, max_tokens, max_ctx, rt->slots.empty() ? nullptr : rt->slots[0].get(),
                     growable)) {
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
    size_t need = 0, free_b = 0;
    for (const InklingLayer& ly : rt->w.layers)
        need += (static_cast<size_t>(ly.proj_rows) * rt->w.cfg.hidden +
                 static_cast<size_t>(rt->w.cfg.hidden) * rt->w.cfg.n_heads * rt->w.cfg.head_dim) *
                17 / 16;
    auto d8 = std::make_unique<InklingDense8>();
    // The budget view: context rows reserved and not committed yet are not free.
    if (vram_fits(need, kDense8Spare, 0, &free_b) && build_inkling_dense8(rt->w, *va, *d8)) {
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
    rt->growable = growable;
    rt->va = va;
    rt->rows_cap = max_tokens;
    rt->d_rows = VramOwned<int32_t>(*va, 2 * static_cast<size_t>(max_tokens), "inkling_state");
    rt->cap = std::min(max_tokens, InklingForward::kSeg);
    rt->table = InklingSlotTable(rt->slots.size());
    return rt;
}

int GraphExecutor::run_inkling_(const InferenceState& state, Tensor& h, int n, cudaStream_t stream) {
    if (!inkling_)
        return model_->config().n_layers;
    InklingRuntime& rt = *inkling_;
    if (!rt.draft_tried)
        inkling_attach_draft_(stream);
    const size_t N = static_cast<size_t>(n);
    rt.logit_src = nullptr;
    // Host reads of the rows' positions and tokens: this path is eager. Slots are keyed by request
    // id only (inkling_slot_table.h); unbound ids throw.
    const bool decode = !state.is_prefill;
    if (decode && state.n_sequences != n)
        throw std::runtime_error("inkling2_loop: a decode call carries one row per sequence");
    if (decode)
        rt.table.rows_to_slots(state.h_seq_ids, n, rt.row_slot);
    rt.ids.resize(N);
    rt.pos.resize(N);
    rt.rows.resize(2 * N);
    bool ok = h.qtype == QType::F16 && rt.d_rows && n <= rt.rows_cap;
    if (ok) {
        inkling_pack_rows_k<<<1, 256, 0, stream>>>(state.positions, state.token_ids, n, rt.d_rows.get());
        IMP_CUDA_CHECK_LAUNCH();
        // The copy waits for the stream's work.
        ok = cudaMemcpyAsync(rt.rows.data(), rt.d_rows.get(), 2 * N * sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream) == cudaSuccess &&
             cudaStreamSynchronize(stream) == cudaSuccess;
        std::copy_n(rt.rows.begin(), N, rt.pos.begin());
        std::copy_n(rt.rows.begin() + static_cast<std::ptrdiff_t>(N), N, rt.ids.begin());
    }
    half* hd = static_cast<half*>(h.data);
    if (ok && !decode) {
        // One sequence on its request's slot; a forward outside a request may start an unbound slot 0.
        const int s = state.prefill_req_id >= 0 ? rt.table.slot_of(state.prefill_req_id) : 0;
        if (state.prefill_req_id < 0 && (rt.pos[0] != 0 || rt.table.bound(0)))
            throw std::runtime_error("inkling2_loop: a prefill without a request id at position " +
                                     std::to_string(rt.pos[0]) + " (only position 0 of an unbound slot 0)");
        ok = rt.slots[static_cast<size_t>(s)]->run(hd, n, rt.pos[0], stream, rt.ids.data(),
                                                   state.vision_embeddings ? state.vision_token_id : -1);
    }
    // Decode: one row per sequence, each on the slot its request is bound to.
    for (size_t i = 0; ok && decode && i < N; ++i) {
        const int s = rt.row_slot[i];
        ok = rt.slots[static_cast<size_t>(s)]->run(hd + i * static_cast<size_t>(h.shape[1]), 1, rt.pos[i],
                                                   stream, &rt.ids[i]);
        if (ok && N == 1)
            inkling_look_logits_(s, stream);
    }
    if (!ok)
        throw std::runtime_error("inkling2_loop: forward failed at position " + std::to_string(rt.pos[0]));
    return 0;
}

// A lookahead call's rows go through the LM head in one pass (0.30 ms of head weights per pass, whatever
// the rows): the step and the steps served from the call then copy their row instead.
void GraphExecutor::inkling_look_logits_(int slot, cudaStream_t stream) {
    InklingRuntime& rt = *inkling_;
    const InklingForward& f = *rt.slots[static_cast<size_t>(slot)];
    float* stash = static_cast<size_t>(slot) < rt.look_logits.size()
                       ? rt.look_logits[static_cast<size_t>(slot)].get()
                       : nullptr;
    const int row = f.look_row();
    if (!stash || row < 0)
        return;
    const int64_t V = model_->config().vocab_size;
    if (row == 0) {
        int rows = 0;
        const half* hs = f.look_states(rows);
        const int64_t hshape[2] = {rows, model_->config().d_model}, lshape[2] = {rows, V};
        Tensor ht(const_cast<half*>(hs), QType::F16, 2, hshape, true);
        Tensor lt(stash, QType::F32, 2, lshape, true);
        rt.look_serial[static_cast<size_t>(slot)] = -1;
        if (!lm_head_fp8_(ht, lt, stream))
            return;
        rt.look_serial[static_cast<size_t>(slot)] = f.spec_stats().steps;
    }
    // A call that ran while two sequences decoded has no logits here: its rows take the head themselves.
    if (rt.look_serial[static_cast<size_t>(slot)] != f.spec_stats().steps)
        return;
    rt.logit_src = stash + static_cast<size_t>(row) * static_cast<size_t>(V);
}

bool GraphExecutor::inkling_take_logits_(Tensor& lg, cudaStream_t stream) {
    if (!inkling_ || !inkling_->logit_src || lg.shape[0] != 1)
        return false;
    const float* src = inkling_->logit_src;
    inkling_->logit_src = nullptr;
    return cudaMemcpyAsync(lg.data, src, static_cast<size_t>(lg.shape[1]) * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
}

// Prefix reuse: binds the request to a slot and returns the position its prefill resumes at, -1 when
// every slot is bound (the caller fails the request; no bound slot is taken from its request).
int GraphExecutor::inkling_resume(int req, std::span<const int32_t> tokens, cudaStream_t stream) {
    if (!inkling_)
        return 0;
    const int s = inkling_->bind(req, tokens);
    return s < 0 ? -1 : inkling_->slots[static_cast<size_t>(s)]->rewind(tokens, stream);
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
        if (!d->init(*w, *vram_alloc_, rt.max_ctx, rt.cap, InklingForward::kSeg + InklingForward::kSlack,
                     rt.growable) ||
            !slot->enable_lookahead(d->hooks(), w->cfg.taps, static_cast<const half*>(emb.data)))
            break;
        rt.drafts.push_back(std::move(d));
        rt.look_logits.emplace_back(*vram_alloc_,
                                    InklingForward::kGraphRows * static_cast<size_t>(head.vocab),
                                    "inkling_state");
        rt.look_serial.push_back(-1);
        ++on;
    }
    IMP_LOG_INFO("inkling2_loop: decode lookahead on %d of %zu slots, %d drafts per step", on,
                 rt.slots.size(), w->cfg.block - 1);
}

}  // namespace imp
