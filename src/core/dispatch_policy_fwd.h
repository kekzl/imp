#pragma once

// DispatchPolicy for headers that only hold or pass it (audit X-4): a key added to core/config/*
// recompiles the TUs that read the policy, not every includer of exec/executor.h. A TU that reads
// one or two sections includes core/config/<section>.h and calls <section>_section() (B-123), so a
// key in another section does not recompile it; a TU that reads most of them includes
// core/dispatch_policy.h.

namespace imp {

struct DispatchPolicy;

namespace cfg {
struct Attention;
struct MoE;
struct GDN;
struct Speculative;
struct Diagnostics;
}  // namespace cfg

// `p`, or a process-wide default-constructed policy when null (a bare executor in a test).
[[nodiscard]] const DispatchPolicy& dispatch_policy_or_default(const DispatchPolicy* p) noexcept;

// One section of `p`, out of line (core/dispatch_policy.cpp).
[[nodiscard]] const cfg::Attention& attention_section(const DispatchPolicy& p) noexcept;
[[nodiscard]] const cfg::MoE& moe_section(const DispatchPolicy& p) noexcept;
[[nodiscard]] const cfg::GDN& gdn_section(const DispatchPolicy& p) noexcept;
[[nodiscard]] const cfg::Speculative& speculative_section(const DispatchPolicy& p) noexcept;
[[nodiscard]] const cfg::Diagnostics& diagnostics_section(const DispatchPolicy& p) noexcept;

}  // namespace imp
