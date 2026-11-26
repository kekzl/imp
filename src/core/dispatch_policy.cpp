#include "core/dispatch_policy.h"
#include "core/dispatch_policy_fwd.h"

namespace imp {

const DispatchPolicy& dispatch_policy_or_default(const DispatchPolicy* p) noexcept {
    // Cold default: set_dispatch_policy() wires the engine's snapshot before any forward.
    static const DispatchPolicy kDefault;
    return p ? *p : kDefault;
}

}  // namespace imp
