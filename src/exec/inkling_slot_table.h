#pragma once
// Inkling sequence slots keyed by request id (audit D-4): every row finds its slot through its
// request id (InferenceState::h_seq_ids, prefill_req_id), never through a KV block id. An id no slot
// is bound to throws. Header-only and CUDA-free for the CPU lane (tests/test_inkling_slot_table.cpp).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace imp {

class InklingSlotTable {
public:
    InklingSlotTable() = default;
    explicit InklingSlotTable(size_t slots) : req_(slots, -1), tick_(slots, 0) {}

    [[nodiscard]] size_t size() const { return req_.size(); }
    // Request bound to slot s, -1 = free. A bound slot is busy.
    [[nodiscard]] int req_of(size_t s) const { return req_[s]; }
    [[nodiscard]] bool bound(size_t s) const { return req_[s] >= 0; }

    // Slot bound to req, -1 = none.
    [[nodiscard]] int find(int req) const {
        for (size_t i = 0; req >= 0 && i < req_.size(); ++i)
            if (req_[i] == req)
                return static_cast<int>(i);
        return -1;
    }

    // Slot bound to req; throws for an id no slot is bound to.
    [[nodiscard]] int slot_of(int req) const {
        const int s = find(req);
        if (s < 0)
            throw std::runtime_error("inkling2_loop: request " + std::to_string(req) +
                                     " holds no sequence slot");
        return s;
    }

    // Binds req (dropping a binding it holds) to the free slot with the smallest (rank(slot), bind
    // tick): rank orders kept prefix and recycling cost, the tick takes the oldest binding on a tie.
    // Returns the slot, -1 = every slot is bound.
    template <class Rank>
    int bind(int req, Rank rank) {
        if (req < 0)
            throw std::invalid_argument("inkling2_loop: a slot binds a request id >= 0, got " +
                                        std::to_string(req));
        release(req);
        using Key = decltype(std::pair(rank(size_t{0}), uint64_t{0}));
        std::optional<Key> best_key;
        int best = -1;
        for (size_t i = 0; i < req_.size(); ++i) {
            if (bound(i))
                continue;
            Key key(rank(i), tick_[i]);
            if (!best_key || key < *best_key) {
                best_key = std::move(key);
                best = static_cast<int>(i);
            }
        }
        if (best >= 0) {
            req_[static_cast<size_t>(best)] = req;
            tick_[static_cast<size_t>(best)] = ++clock_;
        }
        return best;
    }

    // Unbinds req (unknown id = no-op). The slot keeps its sequence for the next bind's prefix match.
    void release(int req) {
        if (const int s = find(req); s >= 0)
            req_[static_cast<size_t>(s)] = -1;
    }

    // Slot of each of n rows' request ids, one row per sequence; throws on an unbound or repeated id.
    void rows_to_slots(const int* ids, int n, std::vector<int>& out) const {
        if (n > 0 && ids == nullptr)
            throw std::runtime_error("inkling2_loop: decode rows without request ids");
        out.resize(static_cast<size_t>(n > 0 ? n : 0));
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = slot_of(ids[i]);
            for (size_t j = 0; j < i; ++j)
                if (out[j] == out[i])
                    throw std::runtime_error("inkling2_loop: request " + std::to_string(ids[i]) +
                                             " has two rows in one decode call");
        }
    }

private:
    std::vector<int> req_;        // request bound to the slot, -1 = free
    std::vector<uint64_t> tick_;  // when the slot was last bound, kept across release
    uint64_t clock_ = 0;
};

}  // namespace imp
