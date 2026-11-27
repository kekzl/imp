#pragma once

// Owner of a convert_nvfp4_to_sfatom result (MEMORY.md I5): the SfAtom buffer the converter
// allocated (plain family); `data` stays a view of the packed weight. Move-only; a moved-from
// owner holds no address, a sliced copy of the base is a view.

#include <utility>

#include "compute/nvfp4_sfatom_quant.h"
#include "memory/device_owned.h"

namespace imp {

struct NvFP4SfAtomOwned : NvFP4SfAtomWeight {
    DeviceOwned<void> sf_owned;

    NvFP4SfAtomOwned() = default;
    NvFP4SfAtomOwned(NvFP4SfAtomOwned&& o) noexcept
        : NvFP4SfAtomWeight(std::exchange(static_cast<NvFP4SfAtomWeight&>(o), NvFP4SfAtomWeight{})),
          sf_owned(std::move(o.sf_owned)) {}
    NvFP4SfAtomOwned& operator=(NvFP4SfAtomOwned&& o) noexcept {
        if (this != &o) {
            NvFP4SfAtomWeight::operator=(
                std::exchange(static_cast<NvFP4SfAtomWeight&>(o), NvFP4SfAtomWeight{}));
            sf_owned = std::move(o.sf_owned);
        }
        return *this;
    }
    ~NvFP4SfAtomOwned() = default;
};

}  // namespace imp
