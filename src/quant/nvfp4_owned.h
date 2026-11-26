#pragma once

// Owners of the NVFP4 quantize results (MEMORY.md I5): the plain view struct the kernels read as
// the base, the buffers the producer allocated as DeviceOwned members (plain family). Move-only;
// a moved-from owner holds no address, a sliced copy of the base is a view.

#include <utility>

#include "memory/device_owned.h"
#include "quant/nvfp4_quant.h"

namespace imp {

// quantize_fp16_to_nvfp4, _with_scale, _async: packed [N, K/2] and micro-scales [N, K/16].
struct NvFP4Owned : NvFP4QuantResult {
    DeviceOwned<void> packed_owned;
    DeviceOwned<void> scales_owned;

    NvFP4Owned() = default;
    NvFP4Owned(NvFP4Owned&& o) noexcept
        : NvFP4QuantResult(std::exchange(static_cast<NvFP4QuantResult&>(o), NvFP4QuantResult{})),
          packed_owned(std::move(o.packed_owned)),
          scales_owned(std::move(o.scales_owned)) {}
    NvFP4Owned& operator=(NvFP4Owned&& o) noexcept {
        if (this != &o) {
            NvFP4QuantResult::operator=(std::exchange(static_cast<NvFP4QuantResult&>(o), NvFP4QuantResult{}));
            packed_owned = std::move(o.packed_owned);
            scales_owned = std::move(o.scales_owned);
        }
        return *this;
    }
    ~NvFP4Owned() = default;
};

// quantize_packed_experts_to_nvfp4: packed, micro-scales and one tensor scale per expert.
// `borrowed` reads true: a sliced copy of the base is a view (NvFP4MoECacheEntry's rule).
struct NvFP4MoEOwned : NvFP4MoEQuantResult {
    DeviceOwned<void> packed_owned;
    DeviceOwned<void> scales_owned;
    DeviceOwned<float> tscales_owned;

    NvFP4MoEOwned() { borrowed = true; }
    NvFP4MoEOwned(NvFP4MoEOwned&& o) noexcept
        : NvFP4MoEQuantResult(take_(o)),
          packed_owned(std::move(o.packed_owned)),
          scales_owned(std::move(o.scales_owned)),
          tscales_owned(std::move(o.tscales_owned)) {}
    NvFP4MoEOwned& operator=(NvFP4MoEOwned&& o) noexcept {
        if (this != &o) {
            NvFP4MoEQuantResult::operator=(take_(o));
            packed_owned = std::move(o.packed_owned);
            scales_owned = std::move(o.scales_owned);
            tscales_owned = std::move(o.tscales_owned);
        }
        return *this;
    }
    ~NvFP4MoEOwned() = default;

private:
    static NvFP4MoEQuantResult take_(NvFP4MoEQuantResult& v) noexcept {
        NvFP4MoEQuantResult out = std::exchange(v, NvFP4MoEQuantResult{});
        v.borrowed = true;
        return out;
    }
};

}  // namespace imp
