// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
// MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Copyright (c) 2026 Raphael Friedmann (github.com/kekzl). APA: https://github.com/kekzl/apa
// Vendored from kekzl/apa include/apa/apa.cuh (v0.8.2, abe6994); change there first, then copy.
// APA (adaptive-precision attention) prefill for sm_120a, host API (header-only). Pass 1: all-FP4 flash
// attention; a KV tile whose share of the running row sum exceeds eps goes to pass 2 (exact FP16), merged by
// log-sum-exp. Attention: hd 128 or 256, causal (q_offset = position of Q row 0), GQA (H % Hkv == 0). Q/O
// [B][Sq][H][D] FP16; K/V flat [B][Skv][Hkv][D] or a paged FP16 pool + flat tail. false = declined.
#pragma once
#include <cuda_runtime.h>

#include "apa_pass2.cuh"
#include "apa_prep.cuh"

// Version of this header set (kekzl/apa); see CHANGELOG.md.
#define APA_VERSION_MAJOR 0
#define APA_VERSION_MINOR 8
#define APA_VERSION_PATCH 2

namespace apa {

struct Problem {
    int B, Sq, Skv, H, Hkv, D;
    int q_offset;  // causal: Q row i sits at position q_offset + i (chunked prefill: Skv = q_offset + Sq)
    bool causal;
    float scale;  // softmax scale (usually 1/sqrt(D))
    int window =
        0;  // causal sliding window: query at pos sees keys (pos - window, pos]; 0 = every earlier key
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
    // a window covering every query's keys (window >= q_offset + Sq) runs the plain causal kernel
    const int swin = p.causal && p.window > 0 && p.window < p.q_offset + p.Sq ? p.window : 0;
    return Dims{p.B, p.Sq, p.Skv, p.H, p.Hkv, G, p.Sq * G, ntkv, p.q_offset, ntkv, swin};
}

// KV window: tiles per pass-1 segment; the FP4 KV copy holds the sink tile + WIN_SLOTS segments per (b, kv
// head) instead of every tile (Hkv 8, 96 x 2: 13.6 MiB at any context, every tile at 122880 keys: 135 MiB). 0
// (or a context of at most WIN_SLOTS segments): every tile, quantized in prep.
#ifndef APA_WIN
#define APA_WIN 96
#endif
constexpr int WIN = APA_WIN;

// Workspace carve-up (256 B aligned): amax, ksum, hs, Qr, Qq, Qs, KV blobs, masks, part, ml, carry, kpart.
struct Workspace {
    unsigned* amax;
    int* ovf;  // after amax: prep overflow flag (sampled stats too narrow)
    float* ksum;
    HeadScale* hs;
    float* Qr;
    uint8_t *Qq, *Qs, *KV;
    uint32_t* warp_hot;       // pass-1 hot-tile masks
    uint32_t* ready;          // [bhk][nqb] pass-1 segments done; after warp_hot (one memset)
    uint32_t* ticket;         // [2] fused launch: CTA tickets, pass-2 group items; after ready
    uint32_t *qdone, *pdone;  // [nseg][bhk] window counters (nseg > 1); after ticket
    uint32_t* p2ready;        // [bhk][nqb] pass-2 segments done (nseg > 1); after pdone
    float* part;              // [rows][D] cold-tile partials (rows: bhk * rp)
    float2* ml;               // [rows] (m, l) of the partials
    float4* carry;            // [rows] row state between pass-1 segments (nseg > 1)
    uint4* p2o;               // pass-2 O between segments (nseg > 1): rows * D * 2 B (APA_P2_F32O: 4 B)
    float4* p2ml;             // pass-2 (m, l) per lane between segments (nseg > 1): rows * 32 B
    int W, nqb, rp;           // mask words per warp / CTA, q blocks per (b, kv head), rows per (b, kv head)
    int S, nseg;              // KV window: tiles per segment (0: none), pass-1 segments
    int kvcap;                // KV / hs / ksum from a KvState of this many tiles (0: own, ntkv)
    float* kpart;             // [bhk][nchunk][D] K column sums per stats chunk (deterministic mean)
    int nchunk;
#ifdef APA_DBG
    float4* dbg = nullptr;  // [rows] pass-1 row state (bench APA_FULL); nullptr: not written
#endif
    size_t bytes;
};
inline size_t tile_bytes(int D) {
    return D == 64 ? Cfg<64>::TILE : D == 128 ? Cfg<128>::TILE : Cfg<256>::TILE;
}
// own_kv = false: no ksum / hs / KV (a KvState provides them). win: tiles per segment of the KV window.
inline Workspace carve(const Problem& p, void* base, bool own_kv = true, int win = WIN) {
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
    const int NW = p.D == 256 ? Launch<256>::NW : Launch<128>::NW;
    w.W = (dm.ntkv + 31) / 32;
    w.nqb = (dm.R + NW * 16 - 1) / (NW * 16);
    w.rp = w.nqb * NW * 16;
    // sliding window: segments below tile tlo (first key of query 0) hold no visible key: not quantized, not
    // walked
    const int tlo = dm.swin > 0 ? std::max(1, (p.q_offset - dm.swin + 1) / BKV) : 1, need = dm.ntkv - tlo;
    const bool wnd = own_kv && APA_ORDER == 1 && win > 0 &&
                     (dm.ntkv - 1 > WIN_SLOTS * win || (tlo > 1 && need > 0));
    w.S = wnd ? win : 0;
    w.nseg = wnd ? std::max(2, (need + win - 1) / win) : 1;
    const size_t rows_p = bhk * w.rp;
    w.amax = reinterpret_cast<unsigned*>(take((bhk * 3 + 1) * 4));
    w.ovf = reinterpret_cast<int*>(w.amax + bhk * 3);
    if (own_kv) {
        w.ksum = reinterpret_cast<float*>(take(bhk * p.D * 4));
        w.hs = reinterpret_cast<HeadScale*>(take(bhk * sizeof(HeadScale)));
    }
    w.Qr = reinterpret_cast<float*>(take(rows * 4));
    w.Qq = take(rows * p.D / 2 * 2);  // two E2M1 terms of Q (term 2 = residual of term 1)
    w.Qs = take(rows * p.D / 16 * 2);
    if (own_kv)
        w.KV = take(bhk * (wnd ? 1 + (size_t)WIN_SLOTS * w.S : (size_t)dm.ntkv) * tile_bytes(p.D));
    const size_t cnt = w.nseg > 1 ? 2 * (size_t)w.nseg * bhk : 0;
    const size_t p2r = cnt ? bhk * w.nqb : 0;  // p2ready
    w.warp_hot = reinterpret_cast<uint32_t*>(take((bhk * w.nqb * (NW * w.W + 1) + 2 + cnt + p2r) * 4));
    w.ready = w.warp_hot + bhk * w.nqb * NW * w.W;
    w.ticket = w.ready + bhk * w.nqb;
    w.qdone = w.ticket + 2;
    w.pdone = w.qdone + cnt / 2;
    w.p2ready = w.pdone + cnt / 2;
    w.part = reinterpret_cast<float*>(take(rows_p * p.D * 4));
    w.ml = reinterpret_cast<float2*>(take(rows_p * 8));
    if (w.nseg > 1) {
        w.carry = reinterpret_cast<float4*>(take(rows_p * 16));
        if (APA_P2_SEG)
            w.p2o = reinterpret_cast<uint4*>(take(rows_p * p.D * (APA_P2_F32O ? 4 : 2)));
        if (APA_P2_SEG)
            w.p2ml = reinterpret_cast<float4*>(take(rows_p * 32));
    }
    w.nchunk = (dm.ntkv + STATS_TILES - 1) / STATS_TILES;
    w.kpart = reinterpret_cast<float*>(take(bhk * w.nchunk * p.D * 4));
    w.bytes = off;
    return w;
}
inline size_t workspace_bytes(const Problem& p, bool own_kv = true, int win = WIN) {
    return carve(p, nullptr, own_kv, win).bytes;
}

[[nodiscard]] inline bool supported(const Problem& p, const KvSource& kv) {
    if (p.D != 64 && p.D != 128 && p.D != 256)
        return false;
    if (p.Hkv <= 0 || p.H % p.Hkv != 0 || p.Sq <= 0 || p.Skv <= 0 || p.B <= 0)
        return false;
    if (p.causal && p.q_offset + p.Sq > p.Skv)
        return false;  // every query needs its own key
    if (p.window < 0 || (p.window > 0 && (!p.causal || APA_ORDER != 1)))
        return false;
    if (kv.kind != KvKind::Flat && (kv.block_table == nullptr || kv.block_size <= 0))
        return false;
    if (kv.kind == KvKind::PagedNvfp4 && (kv.k_scales == nullptr || kv.v_scales == nullptr))
        return false;
    return true;
}

// Prep (stats + KV blobs + Q packing) only; attn() consumes the workspace. Stats read every stride-th
// 1024-key chunk
// (>= APA_PREP_SAMPLE chunks; 0: all chunks, exact head scales). Head scales x4, rounded up to powers of two
// (as the KvState: equal tiles at restats). A 16-block past the UE4M3 range sets ovf: exact stats + every
// tile again, gated. APA_PREP_SAMPLE default: apa_common.cuh.
#ifndef APA_KV_HEADROOM  // head scales x this: later keys (KvState) or unsampled chunks up to 4x the seen
                         // maxima
#define APA_KV_HEADROOM 4.f
#endif
constexpr float KV_HEADROOM = APA_KV_HEADROOM;
// Stats sampling: every stride-th 1024-key chunk, >= APA_PREP_SAMPLE chunks (0: all); keys the sample covers.
inline int stats_stride(int nchunk) {
    return APA_PREP_SAMPLE > 0 ? std::max(1, nchunk / (APA_PREP_SAMPLE > 0 ? APA_PREP_SAMPLE : 1)) : 1;
}
inline int stats_keys(int skv, int nchunk, int stride) {
    int n = 0;
    for (int c = 0; c * stride < nchunk; ++c)
        n += std::min(STATS_TILES * BKV, skv - c * stride * STATS_TILES * BKV);
    return n;
}
#ifndef APA_PREP_HEADROOM  // sampled head scales x this (power of two); tiny values force the overflow redo
                           // (test)
#define APA_PREP_HEADROOM 4.f
#endif
inline int prep_stride(const Workspace& w) {
#if defined(APA_DBG_KMEAN_LEN) || defined(APA_DBG_KMEAN_FROM)
    return 1;
#else
    return stats_stride(w.nchunk);
#endif
}
// Sampled stats may miss a key: the KV quantization checks for overflow (exact stats x >= 1 cannot overflow).
[[nodiscard]] inline bool prep_check(const Workspace& w) {
    return APA_PREP_SAMPLE > 0 && (prep_stride(w) > 1 || APA_PREP_HEADROOM < 1.f);
}
// Overflow redo, gated on w.ovf: exact stats over all keys.
template <int D, typename Reader>
inline cudaError_t redo_stats(const Reader& rd, const Problem& p, const Workspace& w, cudaStream_t st) {
    const Dims dm = make_dims(p);
    const unsigned bhk = (unsigned)(p.B * p.Hkv);
    reset_stats_kernel<D><<<bhk, 32, 0, st>>>(w.amax, w.ovf);
    stats_kernel<D, Reader><<<dim3(w.nchunk, p.Hkv, p.B), 256, 0, st>>>(rd, dm, w.amax, w.kpart, w.ovf);
    finalize_stats_kernel<D><<<bhk, D, 0, st>>>(w.amax, w.kpart, w.nchunk, w.ksum, w.hs, p.Skv,
                                                std::max(KV_HEADROOM, 1.f), w.ovf, APA_PREP_SAMPLE > 0);
    return cudaPeekAtLastError();
}

// KV window (w.nseg > 1): no KV quantization here (the attention launch quantizes per segment and redoes).
template <int D, typename T, typename Reader>
inline cudaError_t prep_d(const T* Q, const Reader& rd, const Problem& p, const Workspace& w,
                          cudaStream_t st) {
    const Dims dm = make_dims(p);
    const size_t bhk = (size_t)p.B * p.Hkv;
    cudaError_t e = cudaMemsetAsync(w.amax, 0, (bhk * 3 + 1) * 4, st);  // amax and ovf
    if (e != cudaSuccess)
        return e;
    const dim3 gkv(dm.ntkv, p.Hkv, p.B);
    const int stride = prep_stride(w);
    const int ns = (w.nchunk + stride - 1) / stride, nkeys = stats_keys(p.Skv, w.nchunk, stride);
    stats_kernel<D, Reader><<<dim3(ns, p.Hkv, p.B), 256, 0, st>>>(rd, dm, w.amax, w.kpart, nullptr, stride);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
#ifdef APA_DBG_KMEAN_LEN  // diagnostic: K mean over the first N keys only (multiple of 1024), maxima over all
    const int kml = std::min(p.Skv, APA_DBG_KMEAN_LEN), kmc = kml / (STATS_TILES * BKV);
    if ((e = cudaMemset2DAsync(w.kpart + (size_t)kmc * D, (size_t)w.nchunk * D * 4, 0,
                               (size_t)(w.nchunk - kmc) * D * 4, bhk, st)) != cudaSuccess)
        return e;
#elif defined(APA_DBG_KMEAN_FROM)  // diagnostic: K mean over keys [N, Skv) only (N multiple of 1024)
    const int kmf = std::min(p.Skv - 1024, APA_DBG_KMEAN_FROM) / (STATS_TILES * BKV);
    const int kml = p.Skv - kmf * STATS_TILES * BKV;
    if ((e = cudaMemset2DAsync(w.kpart, (size_t)w.nchunk * D * 4, 0, (size_t)kmf * D * 4, bhk, st)) !=
        cudaSuccess)
        return e;
#else
    const int kml = nkeys;
#endif
    constexpr bool P2 = APA_PREP_SAMPLE > 0;  // 0: exact stats, exact head scales (0.3.0 prep)
    const bool chk = prep_check(w);
    finalize_stats_kernel<D><<<(unsigned)bhk, D, 0, st>>>(w.amax, w.kpart, ns, w.ksum, w.hs, kml,
                                                          P2 ? APA_PREP_HEADROOM : 1.f, nullptr, P2);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    if (w.nseg == 1) {
        quant_kv_kernel<D, Reader>
            <<<gkv, 256, 0, st>>>(rd, dm, w.ksum, w.hs, w.KV, 0, 0, nullptr, chk ? w.ovf : nullptr);
        if ((e = cudaPeekAtLastError()) != cudaSuccess)
            return e;
    }
    if (chk && w.nseg == 1) {  // redo on overflow (kernels exit early when ovf == 0)
        if ((e = redo_stats<D>(rd, p, w, st)) != cudaSuccess)
            return e;
        quant_kv_kernel<D, Reader><<<gkv, 256, 0, st>>>(rd, dm, w.ksum, w.hs, w.KV, 0, dm.ntkv, w.ovf);
        if ((e = cudaPeekAtLastError()) != cudaSuccess)
            return e;
    }
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

// Zero n words if *gate (gated window redo: masks, ready, tickets, counters).
template <typename W>
__global__ void zero_if_kernel(W* p, size_t n, const int* gate) {
    if (*gate == 0)
        return;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = W(0);
}

// APA attention on a prepared workspace (hd 128 or 256): pass 1 (all-FP4, hot tiles diverted), then
// pass 2 (exact FP16 on the hot tiles, merge). eps: tile share of the running row sum above which a tile is
// hot; eps <= 0 sends every tile to pass 2 (exact baseline from the same code). One fused launch
// (apa_kernel). with_pass1 = false: pass 2 alone on the masks/partials of a previous attn() (reads pass-1
// output only).
template <int D, bool CAUSAL, typename T, typename RS>
inline cudaError_t launch_d(const T* Q, const RS& rs, T* O, const Problem& p, const Workspace& w, float eps,
                            bool with_pass1, cudaStream_t st, const int* gate = nullptr) {
    constexpr int NW = Launch<D>::NW;
    constexpr int smem = std::max(std::max(Launch<D>::ST * Cfg<D>::TILE, P2S_STG * P2Cfg<D>::SMEM),
                                  QuantCfg<D>::SMEM);
    Dims dm = make_dims(p);
    if (w.kvcap > 0)
        dm.kvcap = w.kvcap;
    const size_t bhk = (size_t)p.B * p.Hkv, masks = bhk * w.nqb * (NW * w.W + 1) * 4;  // warp_hot + ready
    const size_t cnt = 2 + (w.nseg > 1 ? (2 * (size_t)w.nseg + w.nqb) * bhk : 0);      // tickets + counters
    cudaError_t e = cudaSuccess;
    if (gate !=
        nullptr) {  // window redo (eps >= 0): zero only if it runs (else the masks of the first launch stay)
        zero_if_kernel<<<64, 256, 0, st>>>(w.warp_hot, masks / 4 + cnt, gate);
        e = cudaPeekAtLastError();
    } else {
        if (with_pass1 && eps >= 0.f)
            e = cudaMemsetAsync(w.warp_hot, 0, masks, st);
        if (with_pass1 && eps < 0.f) {  // exact baseline: every tile to pass 2, no cold partials (l = 0)
            if ((e = cudaMemsetAsync(w.warp_hot, 0xFF, masks, st)) == cudaSuccess)
                e = cudaMemsetAsync(w.ml, 0, bhk * w.rp * 8, st);
        }
        if (e == cudaSuccess)
            e = cudaMemsetAsync(w.ticket, 0, cnt * 4, st);
    }
    if (e != cudaSuccess)
        return e;
#ifdef APA_DBG
    const Pass1Out out{eps,     w.W,  w.warp_hot, w.ready, w.part,    w.ml,  w.ksum,
                       w.carry, w.rp, w.nseg,     w.dbg,   w.p2ready, w.p2o, w.p2ml};
#else
    const Pass1Out out{eps,     w.W,  w.warp_hot, w.ready,   w.part, w.ml,  w.ksum,
                       w.carry, w.rp, w.nseg,     w.p2ready, w.p2o,  w.p2ml};
#endif
    Win wn{};
    wn.nseg = w.nseg;
    const bool chk = with_pass1 && eps >= 0.f && w.nseg > 1 && prep_check(w);
    if (w.nseg > 1) {
        wn.kv = w.KV;
        wn.qdone = w.qdone;
        wn.pdone = w.pdone;
        wn.ovf = chk && gate == nullptr ? w.ovf : nullptr;
        wn.S = w.S;
        wn.qpb = (w.S + WIN_QT - 1) / WIN_QT;
    }
    // per instantiation: SM count (pass-2 workers, one CTA per SM) and the device the smem opt-in was set on
    static int nsm = 0, attr_dev[4] = {-1, -1, -1, -1};
    int dev = 0;
    if ((e = cudaGetDevice(&dev)) != cudaSuccess)
        return e;
    if (nsm == 0 && (e = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)) != cudaSuccess)
        return e;
    const int n1 = (with_pass1 && eps >= 0.f) ? (int)bhk * w.nqb : 0;
    const int sw = CAUSAL && dm.swin > 0;                // sliding window kernel (causal only)
    const int seg = APA_P2_SEG && w.nseg > 1 && n1 > 0;  // pass 2 per KV-window segment
    auto k = sw ? (seg ? apa_kernel<D, CAUSAL, CAUSAL, true, T, RS>
                       : apa_kernel<D, CAUSAL, CAUSAL, false, T, RS>)
                : (seg ? apa_kernel<D, CAUSAL, false, true, T, RS>
                       : apa_kernel<D, CAUSAL, false, false, T, RS>);
    if (attr_dev[sw * 2 + seg] != dev) {
        if ((e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) != cudaSuccess)
            return e;
        attr_dev[sw * 2 + seg] = dev;
    }
    k<<<nsm, NW * 32, smem, st>>>(w.Qq, w.Qs, w.Qr, w.KV, w.hs, Q, rs, O, dm, p.scale * LOG2E, out, n1, w.nqb,
                                  w.ticket, wn, gate);
    if ((e = cudaPeekAtLastError()) != cudaSuccess || !chk || gate != nullptr)
        return e;
    // window: a 16-block past the sampled scales (ovf) redoes exact stats and this launch, gated on the
    // device flag
    if ((e = redo_stats<D>(rs, p, w, st)) != cudaSuccess)
        return e;
    return launch_d<D, CAUSAL>(Q, rs, O, p, w, eps, true, st, w.ovf);
}

// Head dim dispatch (hd 128 or 256), causal or not.
template <typename T, typename RS>
inline cudaError_t launch(const T* Q, const RS& rs, T* O, const Problem& p, const Workspace& w, float eps,
                          bool with_pass1, cudaStream_t st) {
    if (p.D == 256)
        return p.causal ? launch_d<256, true>(Q, rs, O, p, w, eps, with_pass1, st)
                        : launch_d<256, false>(Q, rs, O, p, w, eps, with_pass1, st);
    if (p.D == 128)
        return p.causal ? launch_d<128, true>(Q, rs, O, p, w, eps, with_pass1, st)
                        : launch_d<128, false>(Q, rs, O, p, w, eps, with_pass1, st);
    return cudaErrorInvalidValue;
}

template <typename T>
inline cudaError_t pass2(const T* Q, const T* K, const T* V, T* O, const Problem& p, const Workspace& w,
                         float eps, cudaStream_t st) {
    return launch(Q, FlatKV<T>{K, V}, O, p, w, eps, false, st);
}

template <typename T>
inline cudaError_t attn(const T* Q, const T* K, const T* V, T* O, const Problem& p, const Workspace& w,
                        float eps, cudaStream_t st) {
    return launch(Q, FlatKV<T>{K, V}, O, p, w, eps, true, st);
}

// Full prefill (flat K/V, hd 128 or 256). ws must hold workspace_bytes(p, true, win). false = declined or
// launch error.
template <typename T>
[[nodiscard]] inline bool prefill(const T* Q, const T* K, const T* V, T* O, const Problem& p, float eps,
                                  void* ws, size_t ws_bytes, cudaStream_t st, int win = WIN) {
    KvSource kv{};
    kv.k = K;
    kv.v = V;
    if ((p.D != 128 && p.D != 256) || !supported(p, kv) || ws_bytes < workspace_bytes(p, true, win))
        return false;
    const Workspace w = carve(p, ws, true, win);
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
    if (block_size > 0 && (block_size & (block_size - 1)) == 0)
        rd.bs_shift = __builtin_ctz((unsigned)block_size);
    return rd;
}

// Prefill straight from a paged FP16 cache (B = 1, hd 128 or 256): keys [0, tail) through block_table
// (block_size slots of [Hkv][D]), keys [tail, Skv) from flat k_tail / v_tail [Skv - tail][Hkv][D] (imp's
// current chunk). No gathered K/V copy. false = declined or launch error.
template <typename T>
[[nodiscard]] inline bool prefill_paged(const T* Q, const __half* k_pool, const __half* v_pool,
                                        const int* block_table, int block_size, const __half* k_tail,
                                        const __half* v_tail, int tail, T* O, const Problem& p, float eps,
                                        void* ws, size_t ws_bytes, cudaStream_t st, int win = WIN) {
    KvSource kv{};
    kv.kind = KvKind::PagedF16;
    kv.k = k_pool;
    kv.v = v_pool;
    kv.block_table = block_table;
    kv.block_size = block_size;
    if (tail == 0)
        kv.kind = KvKind::Flat;  // empty pool (first chunk): block_table is never read
    if ((p.D != 128 && p.D != 256) || p.B != 1 || tail < 0 || tail > p.Skv || !supported(p, kv) ||
        ws_bytes < workspace_bytes(p, true, win))
        return false;
    const PagedF16KV rd = paged_reader(k_pool, v_pool, block_table, block_size, k_tail, v_tail, tail);
    const Workspace w = carve(p, ws, true, win);
    if (prep_r(Q, rd, p, w, st) != cudaSuccess)
        return false;
    return launch(Q, rd, O, p, w, eps, true, st) == cudaSuccess;
}

// Persistent K/V tiles of one sequence and layer for chunked prefill: a chunk quantizes only its new tiles.
// K mean and head scales are frozen (rebuilt when the context grows by KV_RESTAT), widened by KV_HEADROOM
// (later keys up to 4x the first chunk's maxima keep UE4M3 block scales in range). len = keys quantized so
// far; a call whose q_offset != len (new sequence, gap) starts over.
#ifndef APA_KV_RESTAT
#define APA_KV_RESTAT 1.125f
#endif
// Restat when Skv >= KV_RESTAT * stats_len. Stale K mean, lc_122880_2 eps 0.005: 1.84x -> 431 pairs <
// 0.9, 1.11x -> 56.
constexpr float KV_RESTAT = APA_KV_RESTAT;
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

// Prep of one chunk on a KvState. Host restart (new sequence: q_offset != len; context x KV_RESTAT since the
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
    const bool restart = st.len == 0 || p.q_offset != st.len || p.Skv >= KV_RESTAT * st.stats_len;
    const int* gate = restart ? nullptr : st.redo;
    constexpr bool P2 = APA_PREP_SAMPLE > 0;  // same sampling and scales as prep_d (0: exact, unrounded)
    const int stride = stats_stride(w.nchunk), ns = (w.nchunk + stride - 1) / stride;
    cudaError_t e = cudaMemsetAsync(w.amax, 0, (bhk * 3 + 1) * 4, s);  // amax and ovf
    if (e == cudaSuccess && !restart) {
        kv_fingerprint_kernel<D, Reader><<<1, 32, 0, s>>>(rd, dm, st.fp, st.len, st.redo, true);
        e = cudaPeekAtLastError();
    }
    if (e != cudaSuccess)
        return e;
    stats_kernel<D, Reader><<<dim3(ns, p.Hkv, p.B), 256, 0, s>>>(rd, dm, w.amax, w.kpart, gate, stride);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    finalize_stats_kernel<D><<<bhk, D, 0, s>>>(w.amax, w.kpart, ns, st.kmean, st.hs,
                                               stats_keys(p.Skv, w.nchunk, stride), KV_HEADROOM, gate, P2);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    quant_kv_kernel<D, Reader><<<gkv, 256, 0, s>>>(rd, dq, st.kmean, st.hs, st.KV, 0,
                                                   restart ? 0 : st.len / BKV, gate, P2 ? w.ovf : nullptr);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    if (P2) {  // a key beyond the frozen or sampled scales: exact stats over all keys, every tile again
               // (gated)
        reset_stats_kernel<D><<<bhk, 32, 0, s>>>(w.amax, w.ovf);
        stats_kernel<D, Reader><<<dim3(w.nchunk, p.Hkv, p.B), 256, 0, s>>>(rd, dm, w.amax, w.kpart, w.ovf);
        finalize_stats_kernel<D><<<bhk, D, 0, s>>>(w.amax, w.kpart, w.nchunk, st.kmean, st.hs, p.Skv,
                                                   std::max(KV_HEADROOM, 1.f), w.ovf, true);
        quant_kv_kernel<D, Reader><<<gkv, 256, 0, s>>>(rd, dq, st.kmean, st.hs, st.KV, 0, dm.ntkv, w.ovf);
        if ((e = cudaPeekAtLastError()) != cudaSuccess)
            return e;
    }
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

// Prefill of the next chunk (hd 128 or 256) with tiles cached in st across calls; ws holds workspace_bytes(p,
// false).
template <typename T, typename Reader>
[[nodiscard]] inline bool prefill_incremental(const T* Q, const Reader& rd, T* O, const Problem& p, float eps,
                                              KvState& st, void* ws, size_t ws_bytes, cudaStream_t s) {
    if ((p.D != 128 && p.D != 256) || make_dims(p).ntkv > st.cap || ws_bytes < workspace_bytes(p, false))
        return false;
    Workspace w = carve(p, ws, false);
    w.KV = st.KV;
    w.hs = st.hs;
    w.ksum = st.kmean;
    w.kvcap = st.cap;
    const cudaError_t e = p.D == 256 ? prep_incremental<256>(Q, rd, p, w, st, s)
                                     : prep_incremental<128>(Q, rd, p, w, st, s);
    if (e != cudaSuccess) {
        st.len = 0;  // unknown state: the next call restarts
        return false;
    }
    return launch(Q, rd, O, p, w, eps, true, s) == cudaSuccess;
}

}  // namespace apa
