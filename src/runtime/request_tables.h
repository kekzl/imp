#pragma once

// The engine tables keyed by request id, as one ordered list: finish and cancel both release
// through it, so a table cannot be in one teardown and missing from the other.
// Header-only and CUDA-free so the CPU lane pins the list (tests/test_request_tables.cpp).

#include "runtime/recurrent_slot.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace imp {

struct Request;

// What Engine::cancel_request does with device lanes that still reference the request.
// Wait: collect all 3 (pipelined step, graph burst, constrained forward) before KV and slot go.
// Abandon: fault path, no device wait, the caller runs invalidate_graphs() next. None: no lane holds it.
enum class CancelLanes { Wait, Abandon, None };

// One table. `finished` is the request on a graceful finish, nullptr on a cancel.
struct RequestTableRow {
    const char* name = "";
    std::function<bool(int id)> holds;  // empty = the owner cannot answer (release stays idempotent)
    std::function<void(int id, const Request* finished)> release;
};

class RequestTableList {
public:
    void add(RequestTableRow row) { rows_.push_back(std::move(row)); }
    [[nodiscard]] bool empty() const { return rows_.empty(); }
    [[nodiscard]] size_t size() const { return rows_.size(); }
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(rows_.size());
        for (const auto& r : rows_)
            out.emplace_back(r.name);
        return out;
    }
    // Releases id from every table, in list order. An id no table holds changes nothing.
    void release_all(int id, const Request* finished = nullptr) const {
        for (const auto& r : rows_)
            r.release(id, finished);
    }
    // First table that still holds id, nullptr = none: the post-teardown invariant.
    [[nodiscard]] const char* first_holding(int id) const {
        for (const auto& r : rows_)
            if (r.holds && r.holds(id))
                return r.name;
        return nullptr;
    }

private:
    std::vector<RequestTableRow> rows_;
};

// id -> pool slot: the entry goes, its slot returns to the free list. Unknown id = no-op.
[[nodiscard]] inline RequestTableRow pooled_slot_table(const char* name,
                                                       std::unordered_map<int, int>& slot_of,
                                                       std::vector<int>& free_slots) {
    return {.name = name,
            .holds = [&slot_of](int id) { return slot_of.contains(id); },
            .release =
                [&slot_of, &free_slots](int id, const Request*) {
                    const auto it = slot_of.find(id);
                    if (it == slot_of.end())
                        return;
                    free_slots.push_back(it->second);
                    slot_of.erase(it);
                }};
}

// id -> value with no pool behind it.
template <class V>
[[nodiscard]] RequestTableRow keyed_table(const char* name, std::unordered_map<int, V>& table) {
    return {.name = name,
            .holds = [&table](int id) { return table.contains(id); },
            .release = [&table](int id, const Request*) { table.erase(id); }};
}

// The engine's members. Host containers by reference; the 4 tables behind a device or a
// foreign owner (KV manager, phrase slabs, snapshot store, MTP pool) as the engine's rows.
template <class SuffixIndex>
struct EngineRequestTables {
    RequestTableRow kv_sequence;
    std::unordered_map<int, int>& verify_spare_of;
    std::vector<int>& verify_spare_free;
    RequestTableRow phrase_ban_state;
    RequestTableRow snapshot_protection;
    RecurrentSlotMap& recurrent_slot_of;
    std::vector<int>& recurrent_slot_free;
    RequestTableRow mtp_bind;
    std::unordered_map<int, SuffixIndex>& spec_suffix_index;
};

// THE teardown list (7 tables). Order = the finish path's since #1632: KV first, the recurrent
// slot after its spare, phrase slabs and snapshot protection, MTP after the slot.
template <class SuffixIndex>
[[nodiscard]] RequestTableList make_request_table_list(EngineRequestTables<SuffixIndex> t) {
    RequestTableList list;
    list.add(std::move(t.kv_sequence));
    list.add(pooled_slot_table("verify_spare", t.verify_spare_of, t.verify_spare_free));
    list.add(std::move(t.phrase_ban_state));
    list.add(std::move(t.snapshot_protection));
    list.add(pooled_slot_table("recurrent_slot", t.recurrent_slot_of, t.recurrent_slot_free));
    list.add(std::move(t.mtp_bind));
    list.add(keyed_table("spec_suffix_index", t.spec_suffix_index));
    return list;
}

}  // namespace imp
