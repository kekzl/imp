#include "exec/quant_scratch.h"

namespace imp {

// Whole-struct reset: the DeviceOwned members release through their own family; the FP8
// reduction trio, the dp4a staging family, split-K and sparse scratch are T2 arena tenants
// (closed by ~Engine) and are only nulled. A member added later cannot be missed here.
void QuantScratch::free() { *this = QuantScratch{}; }

}  // namespace imp
