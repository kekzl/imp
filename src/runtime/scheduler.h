#pragma once

#include "runtime/request.h"
#include <vector>
#include <queue>
#include <memory>

namespace imp {

class KVCacheManager;  // forward declare

class Scheduler {
public:
    explicit Scheduler(int max_batch_size = 32);
    ~Scheduler() = default;

    void add_request(std::shared_ptr<Request> req);

    // Schedule next batch: returns requests for prefill and decode
    void schedule(std::vector<std::shared_ptr<Request>>& prefill_batch,
                  std::vector<std::shared_ptr<Request>>& decode_batch);

    bool has_pending() const;
    int active_count() const;

    // Memory-aware scheduling: set KV cache manager to check budget
    void set_kv_manager(KVCacheManager* mgr) { kv_manager_ = mgr; }

private:
    int max_batch_size_;
    int next_seq_id_ = 0;
    std::queue<std::shared_ptr<Request>> pending_;
    std::vector<std::shared_ptr<Request>> active_;
    KVCacheManager* kv_manager_ = nullptr;  // optional, for memory-aware scheduling
};

} // namespace imp
