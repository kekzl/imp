#pragma once

#include "imp/imp.h"
#include "core/logging.h"
#include "memory/weight_snapshot.h"
#include "model/model.h"
#include "runtime/request.h"

#include <exception>
#include <memory>
#include <new>

namespace imp {

// Nothing throws across the C ABI. Every `ImpError imp_*()` body runs inside
// this or an equivalent inline try/catch; tools/check_api_guard.py gates the
// convention (AUDIT_arch_2026 G-10: 4 of 23 entry points had no catch, and
// the invariant was 19 hand-copied blocks).
template <class F>
ImpError api_guard(const char* fn, F&& body) noexcept {
    try {
        return body();
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("%s: %s", fn, e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

// Forward-declared on purpose. ImpContext_T only stores an Engine, and pulling
// runtime/engine.h in here put it in front of every TU that includes this header
// — most of which never touch Engine at all. The out-of-line destructor below is
// what makes the forward declaration legal: std::unique_ptr needs the complete
// type where the deleter is instantiated, and that is now imp_api.cpp alone.
class Engine;
}  // namespace imp

// Internal handle types backing the opaque C API handles.
// Shared between imp_api.cpp and tool binaries that need
// direct access to the engine (imp-cli, imp-server).

struct ImpModel_T {
    std::shared_ptr<imp::Model> model;
};

struct ImpWeightSnapshot_T {
    std::unique_ptr<imp::WeightSnapshot> snap;
};

struct ImpContext_T {
    ImpContext_T();
    ~ImpContext_T();

    ImpModel model_handle = nullptr;
    std::unique_ptr<imp::Engine> engine;

    // State for token-level prefill/decode API
    std::shared_ptr<imp::Request> active_request;

    // Multi-token step consumption (self-speculative decode produces N tokens per step)
    size_t consumed_output = 0;
};
