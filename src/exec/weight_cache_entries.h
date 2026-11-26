#pragma once

// Value types of the WeightCaches maps (MEMORY.md I5). Each entry is move-only: its base is the
// plain struct the kernels read (raw pointers, no indirection added on a lookup), its DeviceOwned
// members are the buffers the entry allocated. A pointer of the base with an empty owner beside
// it is a view: a slice of a WeightCaches slab, of another entry or of the model's weights.
// The ownership flags of the base structs (borrowed, borrowed_data) always
// read "not mine" inside an entry, so a sliced copy of the base is a view
// and the src/quant free functions release nothing when handed one. NvFP4CacheEntry and
// CutlassNvFP4CacheEntry are the producer owners (quant/nvfp4_owned.h, compute/cutlass_nvfp4_owned.h)
// that can also hold a view.

#include <cstddef>
#include <utility>

#include "compute/cutlass_nvfp4_owned.h"  // CutlassNvFP4Owned
#include "compute/gemm_cutlass_sm120.h"   // CutlassNvFP4Weight
#include "core/tensor.h"
#include "memory/device_owned.h"
#include "quant/cutlass_mxfp4_weight.h"  // CutlassMxFP4Weight
#include "quant/mxfp8_sfatom.h"          // CutlassMxFP8Weight
#include "quant/nvfp4_owned.h"           // NvFP4Owned, NvFP4MoEOwned
#include "quant/nvfp4_quant.h"           // NvFP4QuantResult, NvFP4MoEQuantResult

namespace imp {

namespace detail {
// Moves a view struct out and leaves the default behind: a moved-from entry holds no address.
template <typename V>
V take_view(V& v) noexcept {
    return std::exchange(v, V{});
}
}  // namespace detail

// FP16 copy of a weight (fp16, fused_kv, fused_gate_up). `owned` empty: data is a slice of
// WeightCaches::fp16_bulk_data.
struct Fp16CacheEntry : Tensor {
    DeviceOwned<void> owned;

    Fp16CacheEntry() = default;
    explicit Fp16CacheEntry(const Tensor& view) : Tensor(view) {}
    Fp16CacheEntry(const Tensor& t, DeviceOwned<void>&& buf) : Tensor(t), owned(std::move(buf)) {}

    Fp16CacheEntry(Fp16CacheEntry&& o) noexcept
        : Tensor(detail::take_view<Tensor>(o)), owned(std::move(o.owned)) {}
    Fp16CacheEntry& operator=(Fp16CacheEntry&& o) noexcept {
        if (this != &o) {
            Tensor::operator=(detail::take_view<Tensor>(o));
            owned = std::move(o.owned);
        }
        return *this;
    }
};

// NVFP4 decode copy. Empty owners: packed_data / micro_scales are the model's weights or slices
// of an nvfp4_moe entry.
struct NvFP4CacheEntry : NvFP4Owned {
    NvFP4CacheEntry() = default;
    // A view: owns nothing.
    explicit NvFP4CacheEntry(const NvFP4QuantResult& view) noexcept { NvFP4QuantResult::operator=(view); }
    // Takes over a quantize_* owner (plain family). Empties `r`.
    explicit NvFP4CacheEntry(NvFP4Owned&& r) noexcept : NvFP4Owned(std::move(r)) {}
    NvFP4CacheEntry(const NvFP4Owned&) = delete;  // an owner passed without std::move would become a view
};

// Contiguous per-expert NVFP4. Empty owners: the pointers are the model's contiguous upload.
struct NvFP4MoECacheEntry : NvFP4MoEQuantResult {
    DeviceOwned<void> packed_owned;
    DeviceOwned<void> scales_owned;
    DeviceOwned<float> tscales_owned;

    NvFP4MoECacheEntry() { borrowed = true; }
    // Takes over a converter result (plain family) unless it is marked borrowed. Empties `r`.
    explicit NvFP4MoECacheEntry(NvFP4MoEQuantResult&& r) noexcept : NvFP4MoEQuantResult(r) {
        if (!r.borrowed) {
            const size_t ne = static_cast<size_t>(r.n_experts);
            packed_owned.adopt(r.packed_data, ne * r.expert_stride_packed, nullptr);
            scales_owned.adopt(r.micro_scales, ne * r.expert_stride_ms, nullptr);
            tscales_owned.adopt(r.tensor_scales, ne * sizeof(float), nullptr);
        }
        borrowed = true;
        r = NvFP4MoEQuantResult{};
        r.borrowed = true;
    }
    // `view` describes the layout; each non-empty owner is the buffer its pointer names
    // (VRAMAllocator family of the native and zero-copy builders).
    NvFP4MoECacheEntry(const NvFP4MoEQuantResult& view, DeviceOwned<void>&& packed, DeviceOwned<void>&& ms,
                       DeviceOwned<float>&& ts) noexcept
        : NvFP4MoEQuantResult(view),
          packed_owned(std::move(packed)),
          scales_owned(std::move(ms)),
          tscales_owned(std::move(ts)) {
        borrowed = true;
    }
    // Takes over a quantize_packed_experts_to_nvfp4 owner. Empties `r`.
    explicit NvFP4MoECacheEntry(NvFP4MoEOwned&& r) noexcept
        : NvFP4MoECacheEntry(r, std::move(r.packed_owned), std::move(r.scales_owned),
                             std::move(r.tscales_owned)) {
        static_cast<NvFP4MoEQuantResult&>(r) = NvFP4MoEQuantResult{};
        r.borrowed = true;
    }

    NvFP4MoECacheEntry(NvFP4MoECacheEntry&& o) noexcept
        : NvFP4MoEQuantResult(take_(o)),
          packed_owned(std::move(o.packed_owned)),
          scales_owned(std::move(o.scales_owned)),
          tscales_owned(std::move(o.tscales_owned)) {}
    NvFP4MoECacheEntry& operator=(NvFP4MoECacheEntry&& o) noexcept {
        if (this != &o) {
            NvFP4MoEQuantResult::operator=(take_(o));
            packed_owned = std::move(o.packed_owned);
            scales_owned = std::move(o.scales_owned);
            tscales_owned = std::move(o.tscales_owned);
        }
        return *this;
    }

private:
    static NvFP4MoEQuantResult take_(NvFP4MoEQuantResult& v) noexcept {
        NvFP4MoEQuantResult out = detail::take_view(v);
        v.borrowed = true;
        return out;
    }
};

// SfAtom scale factors of an NVFP4 weight. `data` is always a view (an nvfp4 entry's packed
// data or the model's). `sf_owned` empty: scale_factors is a slab slice or null (G3 SfAtom-less).
struct CutlassNvFP4CacheEntry : CutlassNvFP4Owned {
    CutlassNvFP4CacheEntry() = default;
    // A view: a slab slice, another entry's SfAtom or none (G3 SfAtom-less).
    explicit CutlassNvFP4CacheEntry(const CutlassNvFP4Weight& view) noexcept {
        CutlassNvFP4Weight::operator=(view);
    }
    // Takes over a convert_nvfp4_to_cutlass owner (plain family). Empties `w`.
    explicit CutlassNvFP4CacheEntry(CutlassNvFP4Owned&& w) noexcept : CutlassNvFP4Owned(std::move(w)) {}
    CutlassNvFP4CacheEntry(const CutlassNvFP4Owned&) =
        delete;  // an owner passed without std::move would become a view
};

// MXFP8 prefill copy. `data_owned` empty: data is rows of an FP8 row-scale sidecar
// (borrowed_data stays the dispatch's route flag). `sf_owned` also holds a row view's col_scale.
struct CutlassMxFP8CacheEntry : CutlassMxFP8Weight {
    DeviceOwned<void> data_owned;
    DeviceOwned<void> sf_owned;

    CutlassMxFP8CacheEntry() = default;
    CutlassMxFP8CacheEntry(const CutlassMxFP8Weight& view, DeviceOwned<void>&& data_buf,
                           DeviceOwned<void>&& sf_buf) noexcept
        : CutlassMxFP8Weight(view), data_owned(std::move(data_buf)), sf_owned(std::move(sf_buf)) {}

    CutlassMxFP8CacheEntry(CutlassMxFP8CacheEntry&& o) noexcept
        : CutlassMxFP8Weight(detail::take_view<CutlassMxFP8Weight>(o)),
          data_owned(std::move(o.data_owned)),
          sf_owned(std::move(o.sf_owned)) {}
    CutlassMxFP8CacheEntry& operator=(CutlassMxFP8CacheEntry&& o) noexcept {
        if (this != &o) {
            CutlassMxFP8Weight::operator=(detail::take_view<CutlassMxFP8Weight>(o));
            data_owned = std::move(o.data_owned);
            sf_owned = std::move(o.sf_owned);
        }
        return *this;
    }
};

// MXFP4 weight. `data` is always a view: unpack_mxfp4_gguf compacts the packed nibbles in place
// in the model's own buffer (native GGUF). Both scale layouts are the entry's.
struct CutlassMxFP4CacheEntry : CutlassMxFP4Weight {
    DeviceOwned<void> sf_owned;
    DeviceOwned<void> linear_owned;

    CutlassMxFP4CacheEntry() = default;
    // Takes over the two scale buffers of unpack_mxfp4_gguf (plain family). Empties `w`.
    explicit CutlassMxFP4CacheEntry(CutlassMxFP4Weight&& w) noexcept : CutlassMxFP4Weight(w) {
        sf_owned.adopt(w.scale_factors, w.sf_bytes, nullptr);
        linear_owned.adopt(w.linear_scales, static_cast<size_t>(w.N) * (w.K / 32), nullptr);
        w = CutlassMxFP4Weight{};
    }

    CutlassMxFP4CacheEntry(CutlassMxFP4CacheEntry&& o) noexcept
        : CutlassMxFP4Weight(detail::take_view<CutlassMxFP4Weight>(o)),
          sf_owned(std::move(o.sf_owned)),
          linear_owned(std::move(o.linear_owned)) {}
    CutlassMxFP4CacheEntry& operator=(CutlassMxFP4CacheEntry&& o) noexcept {
        if (this != &o) {
            CutlassMxFP4Weight::operator=(detail::take_view<CutlassMxFP4Weight>(o));
            sf_owned = std::move(o.sf_owned);
            linear_owned = std::move(o.linear_owned);
        }
        return *this;
    }
};

}  // namespace imp
