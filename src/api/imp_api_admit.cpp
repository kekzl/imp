// Prefill admission loop of imp_prefill_with_params, split out of imp_api.cpp (800-line TU gate).

#include "api/imp_internal.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "runtime/engine.h"

void imp_prefill_until_admitted(ImpContext ctx, imp::Request& req, int n_tokens) {
    // A request the scheduler holds back stays PENDING; after 8 rounds it is a capacity refusal,
    // not IMP_SUCCESS followed by "no token" in imp_decode_step.
    for (int held = 0; held < 8; ++held) {
        do {
            (void)ctx->engine->step();
        } while (req.status == imp::RequestStatus::PREFILLING);
        if (req.status != imp::RequestStatus::PENDING)
            return;
    }
    IMP_LOG_ERROR(
        "imp_prefill: request of %d tokens still not admitted after 8 scheduler rounds "
        "(KV pool %d blocks) - refusing",
        n_tokens, ctx->engine->kv_manager()->kv_cache()->total_blocks());
    req.status = imp::RequestStatus::CANCELLED;
    req.cancel_reason = imp::CancelReason::KvCapacity;
}
