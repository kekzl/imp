#pragma once

// The RuntimeConfig -> process_diag snapshot. Kept apart from
// core/process_diag.h so the 40+ leaf TUs that read the snapshot do not
// depend on runtime/ (AUDIT_arch_2026 G-1 / P0). Called once at startup by
// the tool mains and by Engine::init.

namespace imp {

struct RuntimeConfig;

void process_diag_install(const RuntimeConfig& cfg);

}  // namespace imp
