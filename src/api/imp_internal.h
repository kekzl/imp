#pragma once

#include "imp/imp.h"
#include "model/model.h"
#include "runtime/engine.h"
#include "runtime/request.h"

#include <memory>

// Internal handle types backing the opaque C API handles.
// Shared between imp_api.cpp and tool binaries that need
// direct access to the engine (imp-cli, imp-server).

struct ImpModel_T {
    std::shared_ptr<imp::Model> model;
};

struct ImpContext_T {
    ImpModel model_handle = nullptr;
    std::unique_ptr<imp::Engine> engine;

    // State for token-level prefill/decode API
    std::shared_ptr<imp::Request> active_request;
};
