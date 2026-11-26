#pragma once

// DispatchPolicy for headers that only hold or pass it (audit X-4): a key added to core/config/*
// recompiles the TUs that read the policy, not every includer of exec/executor.h. A TU that reads
// a field includes core/dispatch_policy.h itself.

namespace imp {

struct DispatchPolicy;

// `p`, or a process-wide default-constructed policy when null (a bare executor in a test).
[[nodiscard]] const DispatchPolicy& dispatch_policy_or_default(const DispatchPolicy* p) noexcept;

}  // namespace imp
