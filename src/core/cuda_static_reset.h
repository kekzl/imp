#pragma once

// Reset of lazily-created module-static CUDA resources. Called ONLY from
// imp_gpu_release() immediately BEFORE cudaDeviceReset(), while the context
// is still valid: several TUs hold static handles/buffers behind lazy
// `if (!ptr)` guards that would otherwise dangle post-reset while the guard
// stays armed. Not part of normal engine teardown. All hooks idempotent,
// safe when the module was never used.
// Registration is automatic (#1207): each owning TU registers itself at
// static-init time instead of being hand-listed here (a forgotten entry
// used to dangle behind an armed guard with nothing to catch it).
// namespace {
// void my_module_reset() { ... }
// IMP_REGISTER_CUDA_STATIC_RESET(my_module_reset);
// }

namespace imp {

// Runs every registered hook, then clears any sticky CUDA error.
void reset_static_cuda_state();

// Number of registered hooks. Lets a test assert the registry is populated
// rather than silently empty — a link-order or --gc-sections accident would
// otherwise turn the whole mechanism into a no-op that still "passes".
int cuda_static_reset_hook_count();

namespace detail {

// Appends `fn` to the hook list when constructed. File-scope instances run
// before main(), so every TU linked into the binary is registered by the time
// imp_gpu_release() can be called.
struct CudaStaticResetRegistrar {
    explicit CudaStaticResetRegistrar(void (*fn)());
};

}  // namespace detail

}  // namespace imp

// Registers `fn` (a `void()` in the current TU) as a pre-cudaDeviceReset hook.
// Place at file scope, inside an anonymous namespace.
#define IMP_REGISTER_CUDA_STATIC_RESET(fn) \
    const ::imp::detail::CudaStaticResetRegistrar imp_cuda_reset_registrar_##fn { &fn }
