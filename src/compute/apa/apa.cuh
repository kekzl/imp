// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
// MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Copyright (c) 2026 Raphael Friedmann (github.com/kekzl). APA: https://github.com/kekzl/apa
// Vendored from kekzl/apa include/apa/apa.cuh (8346520, v0.2.0); change there first, then copy.
// APA (adaptive-precision attention) prefill for sm_120a, host API (header-only). Pass 1: all-FP4 flash
// attention; a KV tile whose share of the running row sum exceeds eps goes to pass 2 (exact FP16), merged by
// log-sum-exp. Attention: hd 128, causal (q_offset = position of Q row 0), GQA (H % Hkv == 0). Q/O
// [B][Sq][H][D] FP16; K/V flat [B][Skv][Hkv][D] or a paged FP16 pool + flat tail. false = declined.
#pragma once
#include <cuda_runtime.h>

#include "apa_pass2.cuh"
#include "apa_prep.cuh"

// Version of this header set (kekzl/apa); see CHANGELOG.md.
#define APA_VERSION_MAJOR 0
#define APA_VERSION_MINOR 2
#define APA_VERSION_PATCH 0

namespace apa {

struct Problem {
    int B, Sq, Skv, H, Hkv, D;
    int q_offset;  // causal: Q row i sits at position q_offset + i (chunked prefill: Skv = q_offset + Sq)
    bool causal;
    float scale;  // softmax scale (usually 1/sqrt(D))
};

enum class KvKind { Flat, PagedF16, PagedNvfp4 };
struct KvSource {
    KvKind kind = KvKind::Flat;
    const void* k = nullptr;  // Flat: [B][Skv][Hkv][D] T; Paged: pool base of this layer
    const void* v = nullptr;
    const uint8_t* k_scales = nullptr;  // PagedNvfp4: [block][slot][Hkv][D/16] UE4M3
    const uint8_t* v_scales = nullptr;
    const int* block_table = nullptr;  // Paged: [B][max_blocks], -1 = hole
    int block_size = 0, max_blocks = 0;
};

inline Dims make_dims(const Problem& p) {
    const int G = p.H / p.Hkv;
    const int ntkv = (p.Skv + BKV - 1) / BKV;
    return Dims{p.B, p.Sq, p.Skv, p.H, p.Hkv, G, p.Sq * G, ntkv, p.q_offset, ntkv};
}

// Workspace carve-up (256 B aligned): amax, ksum, hs, Qr, Qq, Qs, KV blobs, masks, part, ml, kpart.
struct Workspace {
    unsigned* amax;
    float* ksum;
    HeadScale* hs;
    float* Qr;
    uint8_t *Qq, *Qs, *KV;
    uint32_t* warp_hot;  // pass-1 hot-tile masks
    uint32_t* ready;     // [bhk][nqb] pass-1 q block done; after warp_hot (one memset)
    uint32_t* ticket;    // [2] fused launch: CTA tickets, pass-2 group items; after ready
    float* part;         // [rows][D] cold-tile partials
    float2* ml;          // [rows] (m, l) of the partials
    int W, nqb;          // mask words per warp / CTA, q blocks per (b, kv head)
    int kvcap;           // KV / hs / ksum from a KvState of this many tiles (0: own, ntkv)
    float* kpart;        // [bhk][nchunk][D] K column sums per stats chunk (deterministic mean)
    int nchunk;
    size_t bytes;
};
inline size_t tile_bytes(int D) {
    return D == 64 ? Cfg<64>::TILE : D == 128 ? Cfg<128>::TILE : Cfg<256>::TILE;
}
// own_kv = false: no ksum / hs / KV (a KvState provides them).
inline Workspace carve(const Problem& p, void* base, bool own_kv = true) {
    const Dims dm = make_dims(p);
    const size_t bhk = (size_t)p.B * p.Hkv, rows = bhk * dm.R;
    auto al = [](size_t x) { return (x + 255) & ~size_t(255); };
    size_t off = 0;
    auto take = [&](size_t n) {
        const size_t o = off;
        off += al(n);
        return reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(base) +
                                          o);  // base may be null (size query)
    };
    Workspace w{};
    w.amax = reinterpret_cast<unsigned*>(take(bhk * 3 * 4));
    if (own_kv) {
        w.ksum = reinterpret_cast<float*>(take(bhk * p.D * 4));
        w.hs = reinterpret_cast<HeadScale*>(take(bhk * sizeof(HeadScale)));
    }
    w.Qr = reinterpret_cast<float*>(take(rows * 4));
    w.Qq = take(rows * p.D / 2);
    w.Qs = take(rows * p.D / 16);
    if (own_kv)
        w.KV = take(bhk * dm.ntkv * tile_bytes(p.D));
    constexpr int NW = 12;  // APA v1: hd 128 launch (Launch<128>)
    w.W = (dm.ntkv + 31) / 32;
    w.nqb = (dm.R + NW * 16 - 1) / (NW * 16);
    w.warp_hot = reinterpret_cast<uint32_t*>(take((bhk * w.nqb * (NW * w.W + 1) + 2) * 4));
    w.ready = w.warp_hot + bhk * w.nqb * NW * w.W;
    w.ticket = w.ready + bhk * w.nqb;
    w.part = reinterpret_cast<float*>(take(rows * p.D * 4));
    w.ml = reinterpret_cast<float2*>(take(rows * 8));
    w.nchunk = (dm.ntkv + STATS_TILES - 1) / STATS_TILES;
    w.kpart = reinterpret_cast<float*>(take(bhk * w.nchunk * p.D * 4));
    w.bytes = off;
    return w;
}
inline size_t workspace_bytes(const Problem& p, bool own_kv = true) {
    return carve(p, nullptr, own_kv).bytes;
}

[[nodiscard]] inline bool supported(const Problem& p, const KvSource& kv) {
    if (p.D != 64 && p.D != 128 && p.D != 256)
        return false;
    if (p.Hkv <= 0 || p.H % p.Hkv != 0 || p.Sq <= 0 || p.Skv <= 0 || p.B <= 0)
        return false;
    if (p.causal && p.q_offset + p.Sq > p.Skv)
        return false;  // every query needs its own key
    if (kv.kind != KvKind::Flat && (kv.block_table == nullptr || kv.block_size <= 0))
        return false;
    if (kv.kind == KvKind::PagedNvfp4 && (kv.k_scales == nullptr || kv.v_scales == nullptr))
        return false;
    return true;
}

// Launch config per head dim: warps per CTA, pipeline stages (1 CTA per SM).
template <int D>
struct Launch;
template <>
struct Launch<64> {
#ifndef APA_NW64
#define APA_NW64 16
#define APA_ST64 6
#endif
    static constexpr int NW = APA_NW64, ST = APA_ST64;
};
template <>
struct Launch<128> {
    static constexpr int NW = 12, ST = 6;
};
template <>
struct Launch<256> {
    static constexpr int NW = 8, ST = 4;
};

// Prep (stats + KV blobs + Q packing) only; attn() consumes the workspace.
template <int D, typename T, typename Reader>
inline cudaError_t prep_d(const T* Q, const Reader& rd, const Problem& p, const Workspace& w,
                          cudaStream_t st) {
    const Dims dm = make_dims(p);
    const size_t bhk = (size_t)p.B * p.Hkv;
    cudaError_t e = cudaMemsetAsync(w.amax, 0, bhk * 3 * 4, st);
    if (e != cudaSuccess)
        return e;
    const dim3 gkv(dm.ntkv, p.Hkv, p.B);
    stats_kernel<D, Reader><<<dim3(w.nchunk, p.Hkv, p.B), 256, 0, st>>>(rd, dm, w.amax, w.kpart);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    finalize_stats_kernel<D>
        <<<(unsigned)bhk, D, 0, st>>>(w.amax, w.kpart, w.nchunk, w.ksum, w.hs, p.Skv, 1.f);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    quant_kv_kernel<D, Reader><<<gkv, 256, 0, st>>>(rd, dm, w.ksum, w.hs, w.KV, 0);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    quant_q_kernel<D, T>
        <<<dim3((dm.R + BKV - 1) / BKV, p.Hkv, p.B), 256, 0, st>>>(Q, dm, p.scale * LOG2E, w.Qq, w.Qs, w.Qr);
    return cudaGetLastError();
}

template <typename T, typename Reader>
inline cudaError_t prep_r(const T* Q, const Reader& rd, const Problem& p, const Workspace& w,
                          cudaStream_t st) {
    switch (p.D) {
        case 64:
            return prep_d<64>(Q, rd, p, w, st);
        case 128:
            return prep_d<128>(Q, rd, p, w, st);
        default:
            return prep_d<256>(Q, rd, p, w, st);
    }
}

template <typename T>
inline cudaError_t prep(const T* Q, const KvSource& kv, const Problem& p, const Workspace& w,
                        cudaStream_t st) {
    switch (kv.kind) {
        case KvKind::Flat:
            return prep_r(Q, FlatKV<T>{static_cast<const T*>(kv.k), static_cast<const T*>(kv.v)}, p, w, st);
        case KvKind::PagedF16:
            return prep_r(Q,
                          PagedF16KV{static_cast<const __half*>(kv.k), static_cast<const __half*>(kv.v),
                                     kv.block_table, kv.block_size, kv.max_blocks},
                          p, w, st);
        default:
            return prep_r(Q,
                          PagedNvfp4KV{static_cast<const uint8_t*>(kv.k), static_cast<const uint8_t*>(kv.v),
                                       kv.k_scales, kv.v_scales, kv.block_table, kv.block_size,
                                       kv.max_blocks},
                          p, w, st);
    }
}

// APA attention on a prepared workspace (hd 128, flat K/V): pass 1 (all-FP4, hot tiles diverted), then
// pass 2 (exact FP16 on the hot tiles, merge). eps: tile share of the running row sum above which a tile is
// hot; eps <= 0 sends every tile to pass 2 (exact baseline from the same code). One fused launch
// (apa_kernel). with_pass1 = false: pass 2 alone on the masks/partials of a previous attn() (reads pass-1
// output only).
template <bool CAUSAL, typename T, typename RS>
inline cudaError_t launch_d128(const T* Q, const RS& rs, T* O, const Problem& p, const Workspace& w,
                               float eps, bool with_pass1, cudaStream_t st) {
    constexpr int NW = Launch<128>::NW;
    constexpr int smem = Launch<128>::ST * Cfg<128>::TILE > 3 * P2_SMEM ? Launch<128>::ST * Cfg<128>::TILE
                                                                        : 3 * P2_SMEM;
    Dims dm = make_dims(p);
    if (w.kvcap > 0)
        dm.kvcap = w.kvcap;
    const size_t bhk = (size_t)p.B * p.Hkv, masks = bhk * w.nqb * (NW * w.W + 1) * 4;  // warp_hot + ready
    cudaError_t e = cudaSuccess;
    if (with_pass1 && eps >= 0.f)
        e = cudaMemsetAsync(w.warp_hot, 0, masks, st);
    if (with_pass1 && eps < 0.f) {  // exact baseline: every tile to pass 2, no cold partials (l = 0)
        if ((e = cudaMemsetAsync(w.warp_hot, 0xFF, masks, st)) == cudaSuccess)
            e = cudaMemsetAsync(w.ml, 0, bhk * dm.R * 8, st);
    }
    if (e == cudaSuccess)
        e = cudaMemsetAsync(w.ticket, 0, 8, st);
    if (e != cudaSuccess)
        return e;
    const Pass1Out out{eps, w.W, w.warp_hot, w.ready, w.part, w.ml, w.ksum};
    // per instantiation: SM count (pass-2 workers, one CTA per SM) and the device the smem opt-in was set on
    static int nsm = 0, attr_dev = -1;
    int dev = 0;
    if ((e = cudaGetDevice(&dev)) != cudaSuccess)
        return e;
    if (nsm == 0 && (e = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)) != cudaSuccess)
        return e;
    const int n1 = (with_pass1 && eps >= 0.f) ? (int)bhk * w.nqb : 0;
    auto k = apa_kernel<CAUSAL, T, RS>;
    if (attr_dev != dev) {
        if ((e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) != cudaSuccess)
            return e;
        attr_dev = dev;
    }
    k<<<n1 + nsm, NW * 32, smem, st>>>(w.Qq, w.Qs, w.Qr, w.KV, w.hs, Q, rs, O, dm, p.scale * LOG2E, out, n1,
                                       w.nqb, w.ticket);
    return cudaPeekAtLastError();
}

template <typename T>
inline cudaError_t pass2(const T* Q, const T* K, const T* V, T* O, const Problem& p, const Workspace& w,
                         float eps, cudaStream_t st) {
    return p.causal ? launch_d128<true>(Q, FlatKV<T>{K, V}, O, p, w, eps, false, st)
                    : launch_d128<false>(Q, FlatKV<T>{K, V}, O, p, w, eps, false, st);
}

template <typename T>
inline cudaError_t attn(const T* Q, const T* K, const T* V, T* O, const Problem& p, const Workspace& w,
                        float eps, cudaStream_t st) {
    return p.causal ? launch_d128<true>(Q, FlatKV<T>{K, V}, O, p, w, eps, true, st)
                    : launch_d128<false>(Q, FlatKV<T>{K, V}, O, p, w, eps, true, st);
}

// Full prefill (flat K/V, hd 128). ws must hold workspace_bytes(p). false = declined or launch error.
template <typename T>
[[nodiscard]] inline bool prefill(const T* Q, const T* K, const T* V, T* O, const Problem& p, float eps,
                                  void* ws, size_t ws_bytes, cudaStream_t st) {
    KvSource kv{};
    kv.k = K;
    kv.v = V;
    if (p.D != 128 || !supported(p, kv) || ws_bytes < workspace_bytes(p))
        return false;
    const Workspace w = carve(p, ws);
    if (prep(Q, kv, p, w, st) != cudaSuccess)
        return false;
    return attn(Q, K, V, O, p, w, eps, st) == cudaSuccess;
}

// Reader over a paged FP16 pool (keys [0, tail) through block_table) plus a flat tail [Skv - tail][Hkv][D] (B
// = 1).
inline PagedF16KV paged_reader(const __half* k_pool, const __half* v_pool, const int* block_table,
                               int block_size, const __half* k_tail, const __half* v_tail, int tail) {
    PagedF16KV rd{k_pool, v_pool, block_table, block_size, 0};
    rd.kt = k_tail;
    rd.vt = v_tail;
    rd.tail = tail;
    if ((block_size & (block_size - 1)) == 0)
        rd.bs_shift = __builtin_ctz((unsigned)block_size);
    return rd;
}

// Prefill straight from a paged FP16 cache (B = 1, hd 128): keys [0, tail) through block_table (block_size
// slots of [Hkv][D]), keys [tail, Skv) from flat k_tail / v_tail [Skv - tail][Hkv][D] (imp's current chunk).
// No gathered K/V copy. false = declined or launch error.
template <typename T>
[[nodiscard]] inline bool prefill_paged(const T* Q, const __half* k_pool, const __half* v_pool,
                                        const int* block_table, int block_size, const __half* k_tail,
                                        const __half* v_tail, int tail, T* O, const Problem& p, float eps,
                                        void* ws, size_t ws_bytes, cudaStream_t st) {
    KvSource kv{};
    kv.kind = KvKind::PagedF16;
    kv.k = k_pool;
    kv.v = v_pool;
    kv.block_table = block_table;
    kv.block_size = block_size;
    if (p.D != 128 || p.B != 1 || tail < 0 || tail > p.Skv || !supported(p, kv) ||
        ws_bytes < workspace_bytes(p))
        return false;
    const PagedF16KV rd = paged_reader(k_pool, v_pool, block_table, block_size, k_tail, v_tail, tail);
    const Workspace w = carve(p, ws);
    if (prep_r(Q, rd, p, w, st) != cudaSuccess)
        return false;
    const cudaError_t e = p.causal ? launch_d128<true>(Q, rd, O, p, w, eps, true, st)
                                   : launch_d128<false>(Q, rd, O, p, w, eps, true, st);
    return e == cudaSuccess;
}

// Persistent K/V tiles of one sequence and layer for chunked prefill: a chunk quantizes only its new tiles.
// K mean and head scales are frozen (rebuilt when the context doubles), widened by KV_HEADROOM (later keys up
// to 4x the first chunk's maxima keep UE4M3 block scales in range). len = keys quantized so far; a call whose
// q_offset != len (new sequence, gap) starts over.
#ifndef APA_KV_HEADROOM
#define APA_KV_HEADROOM 4.f
#endif
constexpr float KV_HEADROOM = APA_KV_HEADROOM;
struct KvState {
    uint8_t* KV;    // [B * Hkv][cap][TILE]
    HeadScale* hs;  // [B * Hkv]
    float* kmean;   // [B * Hkv][D]
    float* fp;      // [48] fingerprint of the cached keys (kv_fingerprint_kernel)
    int* redo;      // device flag: cached tiles do not match the keys (set by the check)
    int cap;        // tiles per (b, kv head)
    int len;        // keys quantized so far (host view)
    int stats_len;  // keys the frozen stats were taken over (host view)
};
inline size_t kv_state_bytes(int B, int Hkv, int D, int cap_tokens) {
    auto al = [](size_t x) { return (x + 255) & ~size_t(255); };
    const size_t bhk = (size_t)B * Hkv, cap = (cap_tokens + BKV - 1) / BKV;
    return al(bhk * cap * tile_bytes(D)) + al(bhk * sizeof(HeadScale)) + al(bhk * D * 4) + al(48 * 4) + al(4);
}
inline KvState kv_state_carve(void* base, int B, int Hkv, int D, int cap_tokens) {
    auto al = [](size_t x) { return (x + 255) & ~size_t(255); };
    const size_t bhk = (size_t)B * Hkv, cap = (cap_tokens + BKV - 1) / BKV;
    uint8_t* b = static_cast<uint8_t*>(base);
    KvState st{};
    st.KV = b;
    b += al(bhk * cap * tile_bytes(D));
    st.hs = reinterpret_cast<HeadScale*>(b);
    b += al(bhk * sizeof(HeadScale));
    st.kmean = reinterpret_cast<float*>(b);
    b += al(bhk * D * 4);
    st.fp = reinterpret_cast<float*>(b);
    st.redo = reinterpret_cast<int*>(b + al(48 * 4));
    st.cap = (int)cap;
    return st;
}

// Prep of one chunk on a KvState. Host restart (new sequence: q_offset != len; context doubled since the
// stats) = stats over all keys, all tiles. Otherwise the GPU checks the fingerprint and redoes everything
// on a mismatch (another sequence in the cache), else quantizes the tiles from len / BKV on.
template <int D, typename T, typename Reader>
inline cudaError_t prep_incremental(const T* Q, const Reader& rd, const Problem& p, const Workspace& w,
                                    KvState& st, cudaStream_t s) {
    const Dims dm = make_dims(p);
    Dims dq = dm;
    dq.kvcap = st.cap;
    const unsigned bhk = (unsigned)(p.B * p.Hkv);
    const dim3 gkv(dm.ntkv, p.Hkv, p.B);
    const bool restart = st.len == 0 || p.q_offset != st.len || p.Skv >= 2 * st.stats_len;
    const int* gate = restart ? nullptr : st.redo;
    cudaError_t e = cudaSuccess;
    if (restart) {
        e = cudaMemsetAsync(w.amax, 0, bhk * 3 * 4, s);
    } else {
        kv_fingerprint_kernel<D, Reader><<<1, 32, 0, s>>>(rd, dm, st.fp, st.len, st.redo, true);
        if ((e = cudaPeekAtLastError()) == cudaSuccess)
            reset_stats_kernel<D><<<bhk, 32, 0, s>>>(w.amax, st.redo);
        if (e == cudaSuccess)
            e = cudaPeekAtLastError();
    }
    if (e != cudaSuccess)
        return e;
    stats_kernel<D, Reader><<<dim3(w.nchunk, p.Hkv, p.B), 256, 0, s>>>(rd, dm, w.amax, w.kpart, gate);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    finalize_stats_kernel<D>
        <<<bhk, D, 0, s>>>(w.amax, w.kpart, w.nchunk, st.kmean, st.hs, p.Skv, KV_HEADROOM, gate);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    quant_kv_kernel<D, Reader>
        <<<gkv, 256, 0, s>>>(rd, dq, st.kmean, st.hs, st.KV, 0, restart ? 0 : st.len / BKV, gate);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    kv_fingerprint_kernel<D, Reader><<<1, 32, 0, s>>>(rd, dm, st.fp, p.Skv, st.redo, false);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    quant_q_kernel<D, T>
        <<<dim3((dm.R + BKV - 1) / BKV, p.Hkv, p.B), 256, 0, s>>>(Q, dm, p.scale * LOG2E, w.Qq, w.Qs, w.Qr);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    st.len = p.Skv;
    if (restart)
        st.stats_len = p.Skv;
    return cudaSuccess;
}

// Prefill of the next chunk (hd 128) with tiles cached in st across calls; ws holds workspace_bytes(p,
// false).
template <typename T, typename Reader>
[[nodiscard]] inline bool prefill_incremental(const T* Q, const Reader& rd, T* O, const Problem& p, float eps,
                                              KvState& st, void* ws, size_t ws_bytes, cudaStream_t s) {
    if (p.D != 128 || make_dims(p).ntkv > st.cap || ws_bytes < workspace_bytes(p, false))
        return false;
    Workspace w = carve(p, ws, false);
    w.KV = st.KV;
    w.hs = st.hs;
    w.ksum = st.kmean;
    w.kvcap = st.cap;
    if (prep_incremental<128>(Q, rd, p, w, st, s) != cudaSuccess) {
        st.len = 0;  // unknown state: the next call restarts
        return false;
    }
    const cudaError_t e = p.causal ? launch_d128<true>(Q, rd, O, p, w, eps, true, s)
                                   : launch_d128<false>(Q, rd, O, p, w, eps, true, s);
    return e == cudaSuccess;
}

}  // namespace apa
