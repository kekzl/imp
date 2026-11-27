#pragma once

// Value types of the WeightCaches maps (MEMORY.md I5). Each entry is move-only: its base is the
// plain struct the kernels read (raw pointers, no indirection added on a lookup), its DeviceOwned
// members are the buffers the entry allocated. A pointer of the base with an empty owner beside
// it is a view: a slice of a WeightCaches slab, of another entry or of the model's weights.
// The ownership flags of the base structs (borrowed, borrowed_data) always
// read "not mine" inside an entry, so a sliced copy of the base is a view
// and the src/quant free functions release nothing when handed one. NvFP4CacheEntry and
// NvFP4SfAtomCacheEntry are the producer owners (quant/nvfp4_owned.h, compute/nvfp4_sfatom_owned.h)
// that can also hold a view.

#include <cstddef>
#include <utility>

#include "compute/nvfp4_sfatom_owned.h"  // NvFP4SfAtomOwned
#include "compute/nvfp4_sfatom_quant.h"  // NvFP4SfAtomWeight
#include "core/tensor.h"
#include "memory/device_owned.h"
#include "quant/mxfp4_sfatom_weight.h"  // MxFP4SfAtomWeight
#include "quant/mxfp8_sfatom.h"         // MxFP8SfAtomWeight
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
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved): POD view copied, then reset
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
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved): owners moved, view reset
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
struct NvFP4SfAtomCacheEntry : NvFP4SfAtomOwned {
    NvFP4SfAtomCacheEntry() = default;
    // A view: a slab slice, another entry's SfAtom or none (G3 SfAtom-less).
    explicit NvFP4SfAtomCacheEntry(const NvFP4SfAtomWeight& view) noexcept {
        NvFP4SfAtomWeight::operator=(view);
    }
    // Takes over a convert_nvfp4_to_sfatom owner (plain family). Empties `w`.
    explicit NvFP4SfAtomCacheEntry(NvFP4SfAtomOwned&& w) noexcept : NvFP4SfAtomOwned(std::move(w)) {}
    NvFP4SfAtomCacheEntry(const NvFP4SfAtomOwned&) =
        delete;  // an owner passed without std::move would become a view
};

// MXFP8 prefill copy. `data_owned` empty: data is rows of an FP8 row-scale sidecar
// (borrowed_data stays the dispatch's route flag). `sf_owned` also holds a row view's col_scale.
struct MxFP8SfAtomCacheEntry : MxFP8SfAtomWeight {
    DeviceOwned<void> data_owned;
    DeviceOwned<void> sf_owned;

    MxFP8SfAtomCacheEntry() = default;
    MxFP8SfAtomCacheEntry(const MxFP8SfAtomWeight& view, DeviceOwned<void>&& data_buf,
                          DeviceOwned<void>&& sf_buf) noexcept
        : MxFP8SfAtomWeight(view), data_owned(std::move(data_buf)), sf_owned(std::move(sf_buf)) {}

    MxFP8SfAtomCacheEntry(MxFP8SfAtomCacheEntry&& o) noexcept
        : MxFP8SfAtomWeight(detail::take_view<MxFP8SfAtomWeight>(o)),
          data_owned(std::move(o.data_owned)),
          sf_owned(std::move(o.sf_owned)) {}
    MxFP8SfAtomCacheEntry& operator=(MxFP8SfAtomCacheEntry&& o) noexcept {
        if (this != &o) {
            MxFP8SfAtomWeight::operator=(detail::take_view<MxFP8SfAtomWeight>(o));
            data_owned = std::move(o.data_owned);
            sf_owned = std::move(o.sf_owned);
        }
        return *this;
    }
};

// MXFP4 weight. `data` is always a view: unpack_mxfp4_gguf compacts the packed nibbles in place
// in the model's own buffer (native GGUF). Both scale layouts are the entry's.
struct MxFP4SfAtomCacheEntry : MxFP4SfAtomWeight {
    DeviceOwned<void> sf_owned;
    DeviceOwned<void> linear_owned;

    MxFP4SfAtomCacheEntry() = default;
    // Takes over the two scale buffers of unpack_mxfp4_gguf (plain family). Empties `w`.
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved): POD view copied, then reset
    explicit MxFP4SfAtomCacheEntry(MxFP4SfAtomWeight&& w) noexcept : MxFP4SfAtomWeight(w) {
        sf_owned.adopt(w.scale_factors, w.sf_bytes, nullptr);
        linear_owned.adopt(w.linear_scales, static_cast<size_t>(w.N) * (w.K / 32), nullptr);
        w = MxFP4SfAtomWeight{};
    }

    MxFP4SfAtomCacheEntry(MxFP4SfAtomCacheEntry&& o) noexcept
        : MxFP4SfAtomWeight(detail::take_view<MxFP4SfAtomWeight>(o)),
          sf_owned(std::move(o.sf_owned)),
          linear_owned(std::move(o.linear_owned)) {}
    MxFP4SfAtomCacheEntry& operator=(MxFP4SfAtomCacheEntry&& o) noexcept {
        if (this != &o) {
            MxFP4SfAtomWeight::operator=(detail::take_view<MxFP4SfAtomWeight>(o));
            sf_owned = std::move(o.sf_owned);
            linear_owned = std::move(o.linear_owned);
        }
        return *this;
    }
};

}  // namespace imp
