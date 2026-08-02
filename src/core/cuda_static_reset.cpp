#include "core/cuda_static_reset.h"

#include <cuda_runtime.h>

#include <vector>

namespace imp {

namespace {

// Function-local static so a registrar in another TU cannot run before this
// container is constructed (static-init order fiasco). Registration happens
// during static init from arbitrary TUs, so the container has to be created
// on first use, not at namespace scope.
std::vector<void (*)()>& hooks() {
    static std::vector<void (*)()> v;
    return v;
}

}  // namespace

namespace detail {

CudaStaticResetRegistrar::CudaStaticResetRegistrar(void (*fn)()) {
    if (fn)
        hooks().push_back(fn);
}

}  // namespace detail

int cuda_static_reset_hook_count() { return static_cast<int>(hooks().size()); }

void reset_static_cuda_state() {
    // Registration order is link order, which is arbitrary — every hook is
    // independent and idempotent by contract, so that is fine. What is NOT
    // fine is a hook missing entirely, which is what the old hand-maintained
    // call list allowed; see the header.
    for (auto* fn : hooks())
        fn();

    // Best-effort teardown: clear any sticky error left by the frees above.
    (void)cudaGetLastError();
}

}  // namespace imp
