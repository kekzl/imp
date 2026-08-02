#pragma once

// Reset of lazily-created module-static CUDA resources.
//
// Called ONLY from imp_gpu_release() immediately BEFORE cudaDeviceReset(),
// while the CUDA context is still valid. Several translation units hold
// file/function-scope statics (cuBLAS/cuBLASLt handles, device scratch
// buffers) behind lazy `if (!ptr)` / capacity guards. After a
// cudaDeviceReset() those pointers dangle but the guards stay armed, so the
// next use after an in-process reload would touch freed device memory.
// reset_static_cuda_state() frees + nulls every such resource so its guard
// re-arms on the next use.
//
// This is NOT part of normal engine teardown (~Engine's gemm_cleanup() etc.
// stay untouched). All hooks are idempotent and safe to call when the
// corresponding module was never used (its statics are still null).
//
// REGISTRATION IS AUTOMATIC (#1207). This header used to declare each owning
// module's hook by name, and cuda_static_reset.cpp called all eleven from a
// hand-maintained list. That cost two things: a twelfth lazy static added
// without an entry dangled behind an armed guard — exactly the bug this file
// exists to prevent, with nothing to catch it — and the aggregator had to
// include a compute/ header to reach one of them, inverting the layer graph
// (core -> compute).
//
// Each owning TU now registers itself at static-init time:
//
//     namespace {
//     void my_module_reset() { ... }
//     IMP_REGISTER_CUDA_STATIC_RESET(my_module_reset);
//     }
//
// No declaration here, no call site to forget, and core/ no longer includes
// anything above it.

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
