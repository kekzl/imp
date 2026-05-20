#include "exec/gemm_kernel_registry.h"
#include "compute/gemm.h"
#include "core/logging.h"

namespace imp {

// ---------------------------------------------------------------------------
// Singleton storage
// ---------------------------------------------------------------------------
GemmKernelRegistry& GemmKernelRegistry::instance() {
    static GemmKernelRegistry registry;
    return registry;
}

void GemmKernelRegistry::register_kernel(GemmStrategy strategy, GemmKernelFn fn) {
    IMP_CHECK(fn != nullptr, "GemmKernelRegistry::register_kernel: fn is null");
    // Check for existing entry (slice 1 expects at most one per strategy).
    for (std::size_t i = 0; i < count_; ++i) {
        if (entries_[i].strategy == strategy) {
            IMP_LOG_WARN(
                "GemmKernelRegistry: overriding existing entry (tier=%d qtype=%d m_is_one=%d)",
                static_cast<int>(strategy.tier), static_cast<int>(strategy.weight_qtype),
                strategy.m_is_one ? 1 : 0);
            entries_[i].fn = fn;
            return;
        }
    }
    IMP_CHECK(count_ < (sizeof(entries_) / sizeof(entries_[0])),
              "GemmKernelRegistry: capacity exceeded (max=%zu)",
              sizeof(entries_) / sizeof(entries_[0]));
    entries_[count_].strategy = strategy;
    entries_[count_].fn = fn;
    ++count_;
}

GemmDispatchResult GemmKernelRegistry::dispatch(const GemmStrategy& strategy,
                                                const GemmKernelArgs& args) const {
    for (std::size_t i = 0; i < count_; ++i) {
        if (entries_[i].strategy == strategy)
            return entries_[i].fn(args);
    }
    return GemmDispatchResult::NoMatch;
}

std::size_t GemmKernelRegistry::size() const noexcept { return count_; }

// ---------------------------------------------------------------------------
// FP16 tier — proof-of-concept registration. Input + weight + output all
// FP16; delegate to the existing cuBLAS-backed `gemm()` host wrapper.
//
// Strategy: tier=FP16, weight_qtype=F16, m_is_one == either (cuBLAS
// handles both M=1 GEMV and M>1 GEMM through the same API; no need to
// register a separate GEMV strategy for FP16).
// ---------------------------------------------------------------------------
static GemmDispatchResult fp16_gemm_kernel(const GemmKernelArgs& args) {
    IMP_CHECK(args.input != nullptr, "fp16_gemm_kernel: input is null");
    IMP_CHECK(args.output != nullptr, "fp16_gemm_kernel: output is null");
    IMP_CHECK(args.weight_payload != nullptr, "fp16_gemm_kernel: weight_payload is null");

    const Tensor& weight = *static_cast<const Tensor*>(args.weight_payload);
    IMP_CHECK(weight.qtype == QType::F16, "fp16_gemm_kernel: weight qtype=%d, expected F16",
              static_cast<int>(weight.qtype));
    gemm(*args.input, weight, *args.output, /*alpha=*/1.0f, args.beta, args.stream);
    return GemmDispatchResult::Ok;
}

// Static registration. The .cu translation unit gets linked into libimp
// when the registry header is included downstream; this static initializer
// runs at library load. Slice 1 registers only FP16 (m_is_one=false; the
// dispatcher uses this entry for both M=1 and M>1 — see GemmKernelArgs.m_is_one
// comment in the header).
namespace {
struct FP16Registration {
    FP16Registration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::FP16, QType::F16, /*m_is_one=*/false}, &fp16_gemm_kernel);
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::FP16, QType::F16, /*m_is_one=*/true}, &fp16_gemm_kernel);
    }
};
static FP16Registration s_fp16_registration;
}  // namespace

}  // namespace imp
