// Regression (audit C-4): request teardown was 6 hand-written lists and 4 of them skipped
// spec_suffix_idx_, so every disconnected request left its suffix index behind. Finish and cancel
// now release through one list (runtime/request_tables.h); these tests build that list over the
// engine's real host container types and pin that a teardown leaves no entry for the id anywhere.

#include "runtime/request.h"
#include "runtime/request_tables.h"
#include "runtime/scheduler.h"
#include "runtime/suffix_draft.h"

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using imp::EngineRequestTables;
using imp::make_request_table_list;
using imp::RecurrentSlotMap;
using imp::Request;
using imp::RequestTableList;
using imp::RequestTableRow;
using imp::SuffixDraftIndex;

namespace {

// The engine's 7 tables: the 3 host containers as the engine declares them, the 4 tables behind
// a device or a foreign owner (KV manager, phrase slabs, snapshot store, MTP pool) as id sets.
struct Tables {
    std::set<int> kv, phrase, snapshot, mtp;
    std::unordered_map<int, int> spare_of;
    std::vector<int> spare_free;
    RecurrentSlotMap slot_of;
    std::vector<int> slot_free;
    std::unordered_map<int, SuffixDraftIndex> suffix;
    std::vector<std::string> calls;         // row names in release order
    const Request* mtp_finished = nullptr;  // what the MTP row was handed

    RequestTableRow fake(const char* name, std::set<int>& ids, bool answers = true) {
        RequestTableRow row{.name = name,
                            .holds = {},
                            .release = [this, name, &ids](int id, const Request* fin) {
                                calls.emplace_back(name);
                                if (std::string(name) == "mtp_bind")
                                    mtp_finished = fin;
                                ids.erase(id);
                            }};
        if (answers)
            row.holds = [&ids](int id) { return ids.contains(id); };
        return row;
    }

    RequestTableList list() {
        return make_request_table_list(EngineRequestTables<SuffixDraftIndex>{
            .kv_sequence = fake("kv_sequence", kv),
            .verify_spare_of = spare_of,
            .verify_spare_free = spare_free,
            .phrase_ban_state = fake("phrase_ban_state", phrase),
            .snapshot_protection = fake("snapshot_protection", snapshot, /*answers=*/false),
            .recurrent_slot_of = slot_of,
            .recurrent_slot_free = slot_free,
            .mtp_bind = fake("mtp_bind", mtp),
            .spec_suffix_index = suffix,
        });
    }

    // A request that holds an entry in every table.
    void admit(int id, int slot, int spare) {
        for (auto* s : {&kv, &phrase, &snapshot, &mtp})
            s->insert(id);
        spare_of[id] = spare;
        slot_of[id] = slot;
        suffix.emplace(id, SuffixDraftIndex(3, 8));
    }

    // Every container, asked directly (not through the list): does id have an entry anywhere?
    [[nodiscard]] std::vector<std::string> holding(int id) const {
        std::vector<std::string> out;
        if (kv.contains(id))
            out.emplace_back("kv_sequence");
        if (spare_of.contains(id))
            out.emplace_back("verify_spare");
        if (phrase.contains(id))
            out.emplace_back("phrase_ban_state");
        if (snapshot.contains(id))
            out.emplace_back("snapshot_protection");
        if (slot_of.contains(id))
            out.emplace_back("recurrent_slot");
        if (mtp.contains(id))
            out.emplace_back("mtp_bind");
        if (suffix.contains(id))
            out.emplace_back("spec_suffix_index");
        return out;
    }
};

const std::vector<std::string> kAllTables = {"kv_sequence",         "verify_spare",   "phrase_ban_state",
                                             "snapshot_protection", "recurrent_slot", "mtp_bind",
                                             "spec_suffix_index"};

}  // namespace

TEST(RequestTables, TheListIsTheSevenTablesInFinishOrder) {
    Tables t;
    EXPECT_EQ(t.list().names(), kAllTables);
}

TEST(RequestTables, CancelLeavesNoEntryForTheIdInAnyTable) {
    Tables t;
    t.admit(7, /*slot=*/1, /*spare=*/5);
    t.admit(8, /*slot=*/0, /*spare=*/4);
    ASSERT_EQ(t.holding(7), kAllTables);
    const RequestTableList list = t.list();
    EXPECT_STREQ(list.first_holding(7), "kv_sequence");

    list.release_all(7);  // cancel: no finished request

    EXPECT_EQ(t.holding(7), std::vector<std::string>{});  // the audited leak: spec_suffix_index stayed
    EXPECT_EQ(list.first_holding(7), nullptr);
    EXPECT_EQ(t.slot_free, std::vector<int>{1});  // its slots are back in their pools
    EXPECT_EQ(t.spare_free, std::vector<int>{5});
    EXPECT_EQ(t.holding(8), kAllTables);  // the other sequence keeps everything
    EXPECT_EQ(t.slot_of.at(8), 0);
    EXPECT_EQ(t.spare_of.at(8), 4);
}

TEST(RequestTables, FinishAndCancelReleaseTheSameTables) {
    Tables t;
    const RequestTableList list = t.list();
    Request req;
    req.id = 7;

    t.admit(7, 1, 5);
    list.release_all(7, &req);  // finish
    const std::vector<std::string> finish_calls = t.calls;
    EXPECT_EQ(t.mtp_finished, &req);  // finish only: the MTP row flushes its carry first
    EXPECT_EQ(t.holding(7), std::vector<std::string>{});

    t.calls.clear();
    t.slot_free.clear();
    t.spare_free.clear();
    t.admit(7, 1, 5);
    list.release_all(7);  // cancel
    EXPECT_EQ(t.mtp_finished, nullptr);
    EXPECT_EQ(t.calls, finish_calls);
    EXPECT_EQ(t.holding(7), std::vector<std::string>{});
}

TEST(RequestTables, UnknownIdReleasesNothingOfAnotherSequence) {
    // The C-1 class: a cancel for an id without entries (queued, finished) touches nobody else.
    Tables t;
    t.admit(5, /*slot=*/1, /*spare=*/3);
    const RequestTableList list = t.list();
    EXPECT_EQ(list.first_holding(9), nullptr);

    list.release_all(9);  // 9 % 4 == slot 1, which request 5 holds
    list.release_all(0);  // the id of a Request that was never admitted

    EXPECT_EQ(t.holding(5), kAllTables);
    EXPECT_TRUE(t.slot_free.empty());
    EXPECT_TRUE(t.spare_free.empty());
}

TEST(RequestTables, ReleaseIsIdempotent) {
    Tables t;
    t.admit(7, 2, 6);
    const RequestTableList list = t.list();
    list.release_all(7);
    list.release_all(7);                          // cancel after finish, or twice
    EXPECT_EQ(t.slot_free, std::vector<int>{2});  // one slot, returned once
    EXPECT_EQ(t.spare_free, std::vector<int>{6});
}

TEST(RequestTables, ATableThatCannotAnswerIsSkippedByTheInvariant) {
    Tables t;
    t.snapshot.insert(7);
    EXPECT_EQ(t.list().first_holding(7), nullptr);  // snapshot_protection has no holds()
}

// Engine::cancel_request looks its request up here; an id the scheduler does not hold is nullptr = no-op.
TEST(RequestTables, SchedulerFindsOnlyTheRequestsItHolds) {
    imp::Scheduler sched(4);
    EXPECT_EQ(sched.find(0), nullptr);  // empty scheduler, default Request id
    auto a = std::make_shared<Request>();
    a->id = 11;
    auto b = std::make_shared<Request>();
    b->id = 12;
    sched.add_request(a);
    sched.add_request(b);
    EXPECT_EQ(sched.find(11), a);
    EXPECT_EQ(sched.find(12), b);
    EXPECT_EQ(sched.find(13), nullptr);
    EXPECT_EQ(sched.find(0), nullptr);
}
