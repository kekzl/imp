#include "core/pdl.h"
#include "core/device_info.h"
#include "core/process_diag.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <unordered_set>

namespace imp::pdl {

// Registry of kernel functions with PDL enabled.
static std::unordered_set<const void*>& enabled_kernels() {
    static std::unordered_set<const void*> s;
    return s;
}

static bool s_pdl_available = false;
static bool s_pdl_checked = false;

void enable(const void* kernel_func) {
    enabled_kernels().insert(kernel_func);
    IMP_LOG_DEBUG("PDL: enabled for kernel %p (registry size: %zu)", kernel_func, enabled_kernels().size());
}

void disable(const void* kernel_func) {
    enabled_kernels().erase(kernel_func);
    IMP_LOG_DEBUG("PDL: disabled for kernel %p", kernel_func);
}

bool is_enabled(const void* kernel_func) {
    // Disable PDL globally via [runtime] no_pdl = true in imp.conf.
    if (process_diag_no_pdl())
        return false;
    return enabled_kernels().contains(kernel_func);
}

bool is_available() {
    if (s_pdl_checked)
        return s_pdl_available;
    s_pdl_checked = true;

    // PDL needs cc >= 9.0; device_cc() is 0 on a failed query.
    const int major = device_cc() / 10;
    s_pdl_available = (major >= 9);

    if (s_pdl_available) {
        IMP_LOG_INFO("PDL: available (sm_%d0+)", major);
    }
    return s_pdl_available;
}

}  // namespace imp::pdl
