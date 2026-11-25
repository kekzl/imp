// Model lifetime on the server side (AUDIT_ARCH_2026_11_24 C-2): leases for handlers, the epoch
// check at submit, and the one wait every path that frees state.model / state.ctx takes first.
// The protocol itself is ModelLifetimeGate (model_lifetime.h, CPU-tested).

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <string>

void send_model_busy(httplib::Response& res, const std::string& message) {
    res.set_header("Retry-After", std::to_string(kModelBusyRetryAfterSeconds));
    send_json_error(res, 503, "server_error", message);
}

std::shared_ptr<ModelLease> lease_model_locked(ServerState& state, httplib::Response& res) {
    auto lease = acquire_model_lease(state.lifetime);
    if (!lease)
        send_model_busy(res, "Server busy (model swap or suspend in progress), retry shortly");
    return lease;
}

std::shared_ptr<ModelLease> lease_model_snapshot(ServerState& state, httplib::Response& res, ImpModel& model,
                                                 int* max_seq_len) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);
    if (!resume_if_idle_locked(state, res))
        return nullptr;
    auto lease = lease_model_locked(state, res);
    if (lease) {
        model = state.model;
        if (max_seq_len)
            *max_seq_len = state.max_seq_len;
    }
    return lease;
}

bool submit_for_lease(ServerState& state, const std::shared_ptr<ModelLease>& lease,
                      const std::shared_ptr<ServerRequest>& req) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);
    // Epoch compare under the lock that guards state.batching: tokens of load N never reach N+1.
    if (!lease || !lease->current() || !state.batching || !state.batching->is_running())
        return false;
    state.batching->submit(req);
    return true;
}

ModelExclusiveScope::ModelExclusiveScope(ServerState& state)
    : state_(state), budget_ms_(std::max(0, state.runtime_config.server.model_swap_drain_ms)) {
    const auto t0 = std::chrono::steady_clock::now();
    // Pre-routing and /ready read `swapping` without the mutex: shed before the wait starts.
    const bool was_pending = state.lifetime.exclusive_pending();
    if (!was_pending)
        state.swapping.store(true);
    result_ = state.lifetime.begin_exclusive(std::chrono::milliseconds(budget_ms_), &state.mtx);
    const auto waited = std::chrono::steady_clock::now() - t0;
    waited_ms_ = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(waited).count());
    readers_left_ = state.lifetime.readers();
    if (result_ == ModelLifetimeGate::Exclusive::TimedOut)
        state.swapping.store(false);
}

ModelExclusiveScope::~ModelExclusiveScope() {
    if (!acquired())
        return;
    state_.swapping.store(false);
    state_.lifetime.end_exclusive();
}

std::string ModelExclusiveScope::refusal() const {
    if (result_ == ModelLifetimeGate::Exclusive::Busy)
        return "another model swap or suspend is in progress";
    return std::to_string(readers_left_) + " in-flight request(s) did not finish within " +
           std::to_string(budget_ms_) + "ms (server.model_swap_drain_ms)";
}

int ModelExclusiveScope::pause_budget_ms() const { return std::max(1000, budget_ms_ - waited_ms_); }
