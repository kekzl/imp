#include "compute/token_recycle_device.h"
#include "core/logging.h"

namespace imp {

namespace {

// Serial MRU promote — the device mirror of TokenRecycleTable::promote_.
// Returns true when `next` was already recorded (re-observation).
__device__ bool tr_promote(int32_t* succ, uint8_t* streak, int slots, int32_t prev,
                           int32_t next) {
    int32_t* r = succ + static_cast<int64_t>(prev) * slots;
    int pos = slots - 1;
    bool existed = false;
    for (int i = 0; i < slots; ++i) {
        if (r[i] == next) {
            pos = i;
            existed = true;
            break;
        }
        if (r[i] == -1) {
            pos = i;
            break;
        }
    }
    for (int i = pos; i > 0; --i)
        r[i] = r[i - 1];
    r[0] = next;
    return existed;
}

__device__ void tr_observe_pair_dev(int32_t* succ, uint8_t* streak, int vocab, int slots,
                                    int32_t prev, int32_t next) {
    if (prev < 0 || prev >= vocab || next < 0 || next >= vocab)
        return;
    const bool existed = tr_promote(succ, streak, slots, prev, next);
    streak[prev] = existed ? static_cast<uint8_t>(min(255, streak[prev] + 1)) : uint8_t{0};
}

__global__ void tr_init_kernel(int32_t* succ, uint8_t* streak, int64_t n_succ, int vocab) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_succ)
        succ[i] = -1;
    if (i < vocab)
        streak[i] = 0;
}

__global__ void tr_observe_pairs_kernel(int32_t* succ, uint8_t* streak, int vocab, int slots,
                                        const int32_t* __restrict__ toks, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;  // serial by design — MRU promotion is order-dependent
    for (int i = 1; i < n; ++i)
        tr_observe_pair_dev(succ, streak, vocab, slots, toks[i - 1], toks[i]);
}

__global__ void tr_observe_topk_kernel(int32_t* succ, uint8_t* streak, int vocab, int slots,
                                       const int32_t* __restrict__ row_tokens,
                                       const int32_t* __restrict__ topm, int rows, int m) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;  // serial by design
    for (int r = 0; r < rows; ++r) {
        const int32_t tok = row_tokens[r];
        if (tok < 0 || tok >= vocab)
            continue;
        const int32_t* ids = topm + static_cast<int64_t>(r) * m;
        // Streak follows the rank-0 candidate vs the PRE-update row content
        // (host observe_topk semantics).
        bool front_existed = false;
        const int32_t front = ids[0];
        if (front >= 0 && front < vocab) {
            const int32_t* row = succ + static_cast<int64_t>(tok) * slots;
            for (int i = 0; i < slots; ++i)
                if (row[i] == front) {
                    front_existed = true;
                    break;
                }
        }
        for (int i = m - 1; i >= 0; --i)
            if (ids[i] >= 0 && ids[i] < vocab)
                tr_promote(succ, streak, slots, tok, ids[i]);
        streak[tok] = front_existed ? static_cast<uint8_t>(min(255, streak[tok] + 1))
                                    : uint8_t{0};
    }
}

__global__ void tr_draft_kernel(const int32_t* __restrict__ succ,
                                const uint8_t* __restrict__ streak, int vocab, int slots,
                                const int32_t* __restrict__ last_token, int depth,
                                int min_streak, int32_t* __restrict__ out_draft,
                                int32_t* __restrict__ out_len) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    int32_t cur = *last_token;
    int len = 0;
    for (int i = 0; i < depth; ++i) {
        if (cur < 0 || cur >= vocab)
            break;
        if (streak[cur] < min_streak)
            break;
        const int32_t s = succ[static_cast<int64_t>(cur) * slots];
        if (s < 0)
            break;
        out_draft[len++] = s;
        cur = s;
    }
    *out_len = len;
}

// Accept the just-verified chunk, feed the adjacency, draft + stage the
// next chunk, emit accepted tokens to the ring, and decide whether the
// loop continues. Single-threaded by design (order-dependent bookkeeping,
// a handful of slot shuffles — the post_decode_step_kernel pattern).
template <bool kHasHandle>
__global__ void tr_verify_step_kernel(TrLoopView v, TrLoopParams p,
                                      cudaGraphConditionalHandle handle) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    int32_t* succ = v.tab.succ;
    uint8_t* streak = v.tab.streak;
    const int vocab = v.tab.vocab;
    const int slots = v.tab.slots;

    const int L = *v.chunk_len;
    const int p0 = *v.past_len;
    int count = *v.emit_count;
    int emitted = 0;
    int exit_reason = 0;
    int32_t last = v.tokens[0];

    for (int j = 0; j < L; ++j) {
        if (count + emitted >= p.token_limit) {
            exit_reason = 3;  // budget
            break;
        }
        const int32_t tok = v.argmax[j];
        v.ring[count + emitted] = tok;
        ++emitted;
        tr_observe_pair_dev(succ, streak, vocab, slots, last, tok);
        last = tok;
        bool stop = tok == p.eos_id;
        for (int s = 0; !stop && s < p.n_stop_ids; ++s)
            stop = tok == p.stop_ids[s];
        if (stop) {
            exit_reason = 2;  // stop token
            break;
        }
        if (j == L - 1)
            break;  // bonus token reached
        if (tok != v.tokens[j + 1])
            break;  // divergence — tok becomes the next t0
    }

    // Top-M harvest over the real rows (the model's own candidates — valid
    // regardless of acceptance; this breadth is what makes drafts fire).
    if (p.topm > 0 && v.topm != nullptr) {
        for (int j = 0; j < L; ++j) {
            const int32_t tok = v.tokens[j];
            if (tok < 0 || tok >= vocab)
                continue;
            const int32_t* ids = v.topm + static_cast<int64_t>(j) * p.topm;
            bool front_existed = false;
            if (ids[0] >= 0 && ids[0] < vocab) {
                const int32_t* row = succ + static_cast<int64_t>(tok) * slots;
                for (int i = 0; i < slots; ++i)
                    if (row[i] == ids[0]) {
                        front_existed = true;
                        break;
                    }
            }
            for (int i = p.topm - 1; i >= 0; --i)
                if (ids[i] >= 0 && ids[i] < vocab)
                    tr_promote(succ, streak, slots, tok, ids[i]);
            streak[tok] = front_existed ? static_cast<uint8_t>(min(255, streak[tok] + 1))
                                        : uint8_t{0};
        }
    }

    count += emitted;
    *v.emit_count = count;
    __threadfence_system();
    *v.ring_count_mapped = count;

    if (exit_reason == 0 && count >= p.token_limit)
        exit_reason = 3;

    const int new_p0 = p0 + emitted;
    if (exit_reason == 0) {
        // Draft the next chunk from the last emitted (not yet forwarded) token.
        int32_t draft[8];
        int len = 0;
        int32_t cur = last;
        const int depth = p.depth < 8 ? p.depth : 8;
        for (int i = 0; i < depth; ++i) {
            if (cur < 0 || cur >= vocab)
                break;
            if (streak[cur] < p.min_streak)
                break;
            const int32_t s = succ[static_cast<int64_t>(cur) * slots];
            if (s < 0)
                break;
            draft[len++] = s;
            cur = s;
        }
        if (len == 0) {
            exit_reason = 1;  // draft miss — hand back to the host
        } else if (new_p0 + p.chunk_pad > p.ctx_ceiling) {
            exit_reason = 4;  // baked context ceiling reached
        } else {
            const int newL = len + 1;
            for (int i = 0; i < p.chunk_pad; ++i) {
                v.tokens[i] = (i == 0) ? last : (i <= len ? draft[i - 1] : last);
                v.positions[i] = new_p0 + i;
                v.row_ctx_lens[i] = (i < newL) ? (new_p0 + i + 1) : 1;
            }
            *v.chunk_len = newL;
            *v.past_len = new_p0;
            *v.ctx_len = new_p0 + p.chunk_pad;
        }
    }
    *v.exit_reason = exit_reason;
    if (kHasHandle && exit_reason != 0)
        cudaGraphSetConditional(handle, 0);
}

}  // namespace

void tr_verify_step(const TrLoopView& v, const TrLoopParams& p, bool /*no_handle*/,
                    cudaStream_t stream) {
    tr_verify_step_kernel<false><<<1, 32, 0, stream>>>(v, p, cudaGraphConditionalHandle{});
    IMP_CUDA_CHECK_LAUNCH();
}

void tr_verify_step_conditional(const TrLoopView& v, const TrLoopParams& p,
                                cudaGraphConditionalHandle handle, cudaStream_t stream) {
    tr_verify_step_kernel<true><<<1, 32, 0, stream>>>(v, p, handle);
    IMP_CUDA_CHECK_LAUNCH();
}

bool tr_device_init(TrDeviceTable& t, int vocab, int slots, cudaStream_t stream) {
    t.vocab = vocab;
    t.slots = slots;
    const int64_t n_succ = static_cast<int64_t>(vocab) * slots;
    if (cudaMalloc(&t.succ, n_succ * sizeof(int32_t)) != cudaSuccess)
        return false;
    if (cudaMalloc(&t.streak, vocab) != cudaSuccess) {
        cudaFree(t.succ);
        t.succ = nullptr;
        return false;
    }
    const int64_t n = n_succ > vocab ? n_succ : vocab;
    tr_init_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(t.succ, t.streak,
                                                                               n_succ, vocab);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

void tr_device_free(TrDeviceTable& t) {
    if (t.succ)
        IMP_CUDA_CHECK_LOG(cudaFree(t.succ));
    if (t.streak)
        IMP_CUDA_CHECK_LOG(cudaFree(t.streak));
    t = TrDeviceTable{};
}

void tr_observe_pairs(TrDeviceTable& t, const int32_t* d_toks, int n, cudaStream_t stream) {
    if (n < 2)
        return;
    tr_observe_pairs_kernel<<<1, 32, 0, stream>>>(t.succ, t.streak, t.vocab, t.slots, d_toks, n);
    IMP_CUDA_CHECK_LAUNCH();
}

void tr_observe_topk(TrDeviceTable& t, const int32_t* d_row_tokens, const int32_t* d_topm,
                     int rows, int m, cudaStream_t stream) {
    if (rows <= 0 || m <= 0)
        return;
    tr_observe_topk_kernel<<<1, 32, 0, stream>>>(t.succ, t.streak, t.vocab, t.slots,
                                                 d_row_tokens, d_topm, rows, m);
    IMP_CUDA_CHECK_LAUNCH();
}

void tr_draft(TrDeviceTable& t, const int32_t* d_last_token, int depth, int min_streak,
              int32_t* d_out_draft, int32_t* d_out_len, cudaStream_t stream) {
    tr_draft_kernel<<<1, 32, 0, stream>>>(t.succ, t.streak, t.vocab, t.slots, d_last_token,
                                          depth, min_streak, d_out_draft, d_out_len);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
