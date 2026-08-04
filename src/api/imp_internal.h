#pragma once

#include "imp/imp.h"
#include "memory/weight_snapshot.h"
#include "model/model.h"
#include "runtime/request.h"

#include <memory>

namespace imp {
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
