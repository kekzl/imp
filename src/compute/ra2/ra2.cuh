// RA2: INT8 QK^T + two-term NVFP4 PV flash attention prefill for sm_120a. Host API (header-only).
// Vendored from the ra2 standalone repo (kekzl/ra2); change there first, then copy.
// Q/O [B][Sq][H][D] (FP16 or BF16), K/V flat [B][Skv][Hkv][D] or paged (imp layout, FP16 or NVFP4).
// Head dim 64/128/256, GQA (H % Hkv == 0), causal with q_offset (position of Q row 0), any Sq/Skv.
// Returns false (declines) for unsupported problems, like imp's attention tiers.
#pragma once
#include <cuda_runtime.h>

#include "ra2_attn.cuh"
#include "ra2_prep.cuh"

namespace ra2 {

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
    return Dims{p.B, p.Sq, p.Skv, p.H, p.Hkv, G, p.Sq * G, (p.Skv + BKV - 1) / BKV, p.q_offset};
}

// Workspace carve-up (256 B aligned): amax, ksum, hs, Qr, Qq (int8), KV blobs.
struct Workspace {
    unsigned* amax;
    float* ksum;
    HeadScale* hs;
    float* Qr;
    uint8_t *Qq, *KV;
    size_t bytes;
};
inline size_t tile_bytes(int D) {
    return D == 64 ? Cfg<64>::TILE : D == 128 ? Cfg<128>::TILE : Cfg<256>::TILE;
}
inline Workspace carve(const Problem& p, void* base) {
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
    w.ksum = reinterpret_cast<float*>(take(bhk * p.D * 4));
    w.hs = reinterpret_cast<HeadScale*>(take(bhk * sizeof(HeadScale)));
    w.Qr = reinterpret_cast<float*>(take(rows * 4));
    w.Qq = take(rows * p.D);
    w.KV = take(bhk * dm.ntkv * tile_bytes(p.D));
    w.bytes = off;
    return w;
}
inline size_t workspace_bytes(const Problem& p) { return carve(p, nullptr).bytes; }

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
#ifndef RA2_NW64
#define RA2_NW64 16
#define RA2_ST64 6
#endif
    static constexpr int NW = RA2_NW64, ST = RA2_ST64;
};
template <>
struct Launch<128> {
    static constexpr int NW = 12, ST = 5;
};
template <>
struct Launch<256> {
    static constexpr int NW = 8, ST = 2;
};

template <int D, bool CAUSAL, typename T>
inline cudaError_t launch_attn(const Workspace& w, T* O, const Dims& dm, cudaStream_t st) {
    constexpr int NW = Launch<D>::NW, ST = Launch<D>::ST, smem = ST * Cfg<D>::TILE;
    auto kern = attn_kernel<D, NW, ST, CAUSAL, T>;
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess)
        return e;
    const dim3 grid((dm.R + NW * 16 - 1) / (NW * 16), dm.Hkv, dm.B);
    kern<<<grid, NW * 32, smem, st>>>(w.Qq, w.Qr, w.KV, w.hs, O, dm);
    return cudaPeekAtLastError();
}

// Prep (stats + KV blobs + Q packing) only; attn() consumes the workspace.
template <int D, typename T, typename Reader>
inline cudaError_t prep_d(const T* Q, const Reader& rd, const Problem& p, const Workspace& w,
                          cudaStream_t st) {
    const Dims dm = make_dims(p);
    const size_t bhk = (size_t)p.B * p.Hkv;
    cudaError_t e = cudaMemsetAsync(w.amax, 0, bhk * 3 * 4, st);
    if (e == cudaSuccess)
        e = cudaMemsetAsync(w.ksum, 0, bhk * D * 4, st);
    if (e != cudaSuccess)
        return e;
    const dim3 gkv(dm.ntkv, p.Hkv, p.B);
    stats_kernel<D, Reader><<<gkv, 256, 0, st>>>(rd, dm, w.amax, w.ksum);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    quant_kv_kernel<D, Reader><<<gkv, 256, 0, st>>>(rd, dm, w.amax, w.ksum, w.KV, w.hs);
    if ((e = cudaPeekAtLastError()) != cudaSuccess)
        return e;
    quant_q_kernel<D, T>
        <<<dim3((dm.R + BKV - 1) / BKV, p.Hkv, p.B), 256, 0, st>>>(Q, dm, p.scale * LOG2E, w.Qq, w.Qr);
    return cudaPeekAtLastError();
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

template <typename T>
inline cudaError_t attn(T* O, const Problem& p, const Workspace& w, cudaStream_t st) {
    const Dims dm = make_dims(p);
    if (p.causal) {
        switch (p.D) {
            case 64:
                return launch_attn<64, true>(w, O, dm, st);
            case 128:
                return launch_attn<128, true>(w, O, dm, st);
            default:
                return launch_attn<256, true>(w, O, dm, st);
        }
    }
    switch (p.D) {
        case 64:
            return launch_attn<64, false>(w, O, dm, st);
        case 128:
            return launch_attn<128, false>(w, O, dm, st);
        default:
            return launch_attn<256, false>(w, O, dm, st);
    }
}

// Full prefill. ws must hold workspace_bytes(p). false = declined (nothing launched) or launch error.
template <typename T>
[[nodiscard]] inline bool prefill(const T* Q, const KvSource& kv, T* O, const Problem& p, void* ws,
                                  size_t ws_bytes, cudaStream_t st) {
    if (!supported(p, kv) || ws_bytes < workspace_bytes(p))
        return false;
    const Workspace w = carve(p, ws);
    if (prep(Q, kv, p, w, st) != cudaSuccess)
        return false;
    return attn(O, p, w, st) == cudaSuccess;
}

}  // namespace ra2
