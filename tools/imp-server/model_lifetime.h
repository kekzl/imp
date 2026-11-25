#pragma once

// ModelLifetimeGate (AUDIT_ARCH_2026_11_24 C-2): who may hold a pointer into the loaded model, and
// when it may be freed. Header-only, no engine types, so the CPU lane drives it with threads.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

// Readers are handlers that snapshotted model pointers (tokenizer, context); the exclusive side is
// a swap, suspend or unload. Exclusive is granted at 0 readers and refuses new ones while pending.
class ModelLifetimeGate {
public:
    enum class Exclusive { Acquired, Busy, TimedOut };

    // False and not counted while an exclusive holder is pending or active.
    bool try_enter() {
        readers_.fetch_add(1, std::memory_order_seq_cst);
        if (!pending_.load(std::memory_order_seq_cst))
            return true;
        leave();
        return false;
    }
    void leave() {
        readers_.fetch_sub(1, std::memory_order_seq_cst);
        if (pending_.load(std::memory_order_seq_cst)) {
            std::lock_guard<std::mutex> lock(wait_mtx_);
            wait_cv_.notify_all();
        }
    }

    // Marks "exclusive pending" (no reader is admitted from here on), then waits up to `timeout`
    // for the readers inside to leave. `held` (may be null) is a lock the caller owns and a reader
    // needs to finish: unlocked for the wait, owned again on return. Busy: another caller is pending.
    template <class Lock>
    Exclusive begin_exclusive(std::chrono::milliseconds timeout, Lock* held) {
        bool expected = false;
        if (!pending_.compare_exchange_strong(expected, true, std::memory_order_seq_cst))
            return Exclusive::Busy;
        if (readers_.load(std::memory_order_seq_cst) == 0)
            return Exclusive::Acquired;
        if (held)
            held->unlock();
        bool drained = false;
        {
            std::unique_lock<std::mutex> lock(wait_mtx_);
            drained = wait_cv_.wait_for(lock, timeout,
                                        [this] { return readers_.load(std::memory_order_seq_cst) == 0; });
        }
        if (held)
            held->lock();
        if (!drained) {
            pending_.store(false, std::memory_order_seq_cst);
            return Exclusive::TimedOut;
        }
        return Exclusive::Acquired;
    }
    Exclusive begin_exclusive(std::chrono::milliseconds timeout) {
        return begin_exclusive<std::mutex>(timeout, nullptr);
    }
    // Only after begin_exclusive returned Acquired.
    void end_exclusive() { pending_.store(false, std::memory_order_seq_cst); }

    // Load epoch: bumped on every load and unload. A snapshot of another epoch is stale.
    void bump_epoch() { epoch_.fetch_add(1, std::memory_order_seq_cst); }
    uint64_t epoch() const { return epoch_.load(std::memory_order_seq_cst); }
    bool is_current(uint64_t snapshot_epoch) const { return snapshot_epoch == epoch(); }

    int readers() const { return readers_.load(std::memory_order_seq_cst); }
    bool exclusive_pending() const { return pending_.load(std::memory_order_seq_cst); }

private:
    std::atomic<int> readers_{0};
    std::atomic<bool> pending_{false};
    std::atomic<uint64_t> epoch_{0};
    std::mutex wait_mtx_;
    std::condition_variable wait_cv_;
};

// One reader's stay plus the load epoch it snapshotted. Shared: a streaming response copies its
// request context into the content provider, and the model must outlive the last copy.
class ModelLease {
public:
    ModelLease(ModelLifetimeGate& gate, uint64_t epoch) : gate_(&gate), epoch_(epoch) {}
    ~ModelLease() { gate_->leave(); }
    ModelLease(const ModelLease&) = delete;
    ModelLease& operator=(const ModelLease&) = delete;
    uint64_t epoch() const { return epoch_; }
    bool current() const { return gate_->is_current(epoch_); }

private:
    ModelLifetimeGate* gate_;
    uint64_t epoch_;
};

// Null while an exclusive holder is pending; nothing is counted then.
inline std::shared_ptr<ModelLease> acquire_model_lease(ModelLifetimeGate& gate) {
    if (!gate.try_enter())
        return nullptr;
    return std::make_shared<ModelLease>(gate, gate.epoch());
}
