#include "core/dispatch_policy.h"
#include "core/dispatch_policy_fwd.h"

namespace imp {

const DispatchPolicy& dispatch_policy_or_default(const DispatchPolicy* p) noexcept {
    // Cold default: set_dispatch_policy() wires the engine's snapshot before any forward.
    static const DispatchPolicy kDefault;
    return p ? *p : kDefault;
}

const cfg::Attention& attention_section(const DispatchPolicy& p) noexcept { return p.attention; }
const cfg::MoE& moe_section(const DispatchPolicy& p) noexcept { return p.moe; }
const cfg::GDN& gdn_section(const DispatchPolicy& p) noexcept { return p.gdn; }
const cfg::Speculative& speculative_section(const DispatchPolicy& p) noexcept { return p.speculative; }
const cfg::Diagnostics& diagnostics_section(const DispatchPolicy& p) noexcept { return p.diagnostics; }

}  // namespace imp
