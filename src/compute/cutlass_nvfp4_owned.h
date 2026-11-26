#pragma once

// Owner of a convert_nvfp4_to_cutlass result (MEMORY.md I5): the SfAtom buffer the converter
// allocated (plain family); `data` stays a view of the packed weight. Move-only; a moved-from
// owner holds no address, a sliced copy of the base is a view.

#include <utility>

#include "compute/gemm_cutlass_sm120.h"
#include "memory/device_owned.h"

namespace imp {

struct CutlassNvFP4Owned : CutlassNvFP4Weight {
    DeviceOwned<void> sf_owned;

    CutlassNvFP4Owned() = default;
    CutlassNvFP4Owned(CutlassNvFP4Owned&& o) noexcept
        : CutlassNvFP4Weight(std::exchange(static_cast<CutlassNvFP4Weight&>(o), CutlassNvFP4Weight{})),
          sf_owned(std::move(o.sf_owned)) {}
    CutlassNvFP4Owned& operator=(CutlassNvFP4Owned&& o) noexcept {
        if (this != &o) {
            CutlassNvFP4Weight::operator=(
                std::exchange(static_cast<CutlassNvFP4Weight&>(o), CutlassNvFP4Weight{}));
            sf_owned = std::move(o.sf_owned);
        }
        return *this;
    }
    ~CutlassNvFP4Owned() = default;
};

}  // namespace imp
