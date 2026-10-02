// Golden equivalence for request-field parsing (roadmap row 83, #2461): every case body goes through
// each dialect's real parse path; the parsed params (diff vs the dialect's base body) or the 4xx
// (status + message) must match tests/fixtures/sampling_golden.json. IMP_UPDATE_SAMPLING_GOLDEN=1 rewrites it.
#include <gtest/gtest.h>

#include "anthropic.h"
#include "handlers_internal.h"
#include "request_field_types.h"
#include "responses.h"
#include "utils.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

const char* const kGoldenPath = IMP_TEST_FIXTURES_DIR "/sampling_golden.json";

ServerState& test_state() {
    static ServerState* s = [] {
        auto* st = new ServerState();
        st->armed_mtp_k.store(3);
        st->max_n = 4;
        return st;
    }();
    return *s;
}

json error_of(const httplib::Response& res) {
    std::string msg = res.body;
    const json b = json::parse(res.body, nullptr, false);
    if (b.is_object() && b.contains("error") && b["error"].is_object())
        msg = b["error"].value("message", res.body);
    return {{"status", res.status}, {"error", msg}};
}

json chat_params_json(const ChatRequestParams& p) {
    return {{"temperature", p.temperature},
            {"top_p", p.top_p},
            {"top_k", p.top_k},
            {"min_p", p.min_p},
            {"typical_p", p.typical_p},
            {"repetition_penalty", p.repetition_penalty},
            {"frequency_penalty", p.frequency_penalty},
            {"presence_penalty", p.presence_penalty},
            {"repeat_last_n", p.repeat_last_n},
            {"dry_multiplier", p.dry_multiplier},
            {"dry_base", p.dry_base},
            {"dry_allowed_length", p.dry_allowed_length},
            {"dry_penalty_last_n", p.dry_penalty_last_n},
            {"mirostat", p.mirostat},
            {"mirostat_tau", p.mirostat_tau},
            {"mirostat_eta", p.mirostat_eta},
            {"think_budget", p.think_budget},
            {"max_tokens", p.max_tokens},
            {"seed", p.seed},
            {"priority", p.priority},
            {"stream", p.stream},
            {"n", p.n_completions},
            {"logprobs", p.req_logprobs},
            {"top_logprobs", p.top_logprobs},
            {"include_usage", p.include_usage},
            {"ignore_eos", p.ignore_eos},
            {"top_p_explicit", p.top_p_explicit},
            {"top_k_explicit", p.top_k_explicit},
            {"rep_pen_explicit", p.rep_pen_explicit},
            {"cache_prompt", p.cache_prompt},
            {"cache_prefix_messages", p.cache_prefix_messages},
            {"spec_override", p.spec_override},
            {"spec_mtp_k", p.spec_mtp_k},
            {"stop", p.stop_sequences},
            {"max_stop_len", p.max_stop_len},
            {"prediction_text", p.prediction_text},
            {"enable_thinking_set", p.enable_thinking_set},
            {"enable_thinking_requested", p.enable_thinking_requested},
            {"reasoning_effort", p.reasoning_effort},
            {"lora", p.lora_name},
            {"json_mode", p.json_mode},
            {"json_schema", p.json_schema_str},
            {"regex", p.regex_pattern},
            {"grammar", p.grammar},
            {"logit_bias", p.logit_bias},
            {"model", p.requested_model}};
}

// The server's global exception handler (main.cpp) and the /v1/messages wrapper map a
// json::exception to 400 + wrong_field_type_message, else e.what().
json json_exception_error(const std::string& raw, const nlohmann::json::exception& e) {
    const std::string field = wrong_field_type_message(raw);
    return {{"status", 400}, {"error", field.empty() ? std::string(e.what()) : field}};
}

json run_chat_parse(const json& oai_body, const std::string& raw) {
    httplib::Request req;
    req.body = oai_body.dump();
    httplib::Response res;
    ChatRequestContext ctx;
    try {
        if (!parse_chat_request_params(req, res, test_state(), ctx))
            return error_of(res);
    } catch (const nlohmann::json::exception& e) {
        return json_exception_error(raw, e);
    }
    return {{"params", chat_params_json(ctx.params)}};
}

json run_chat(const json& body) { return run_chat_parse(body, body.dump()); }

json run_messages(const json& body) {
    json anth = body;
    drop_null_fields(anth);
    json oai;
    try {
        oai = imp_server::anthropic::anthropic_to_openai_body(anth);
    } catch (const nlohmann::json::exception& e) {
        return json_exception_error(body.dump(), e);
    } catch (const std::exception& e) {
        const std::string field = wrong_field_type_message(body.dump());
        return {{"status", 400},
                {"error", field.empty() ? std::string("Failed to transform Anthropic body: ") + e.what() : field}};
    }
    return run_chat_parse(oai, body.dump());
}

json run_responses(const json& body) {
    json rsp = body;
    drop_null_fields(rsp);
    json oai;
    try {
        oai = imp_server::responses::responses_to_openai_body(rsp);
    } catch (const std::exception& e) {
        const std::string field = wrong_field_type_message(body.dump());
        return {{"status", 400}, {"error", field.empty() ? std::string(e.what()) : field}};
    }
    return run_chat_parse(oai, body.dump());
}

struct Dialect {
    const char* name;
    json base;
    json (*run)(const json&);
};

std::vector<Dialect> dialects() {
    return {
        {"chat", {{"model", "m"}, {"messages", {{{"role", "user"}, {"content", "hi"}}}}}, run_chat},
        {"messages",
         {{"model", "m"}, {"max_tokens", 64}, {"messages", {{{"role", "user"}, {"content", "hi"}}}}},
         run_messages},
        {"responses", {{"model", "m"}, {"input", "hi"}}, run_responses},
    };
}

// Field values: set, boundary, out of range, wrong type. Integers stay inside int64 so no
// float-to-int conversion is undefined.
std::vector<std::pair<std::string, json>> cases() {
    const std::vector<json> kFloatVals = {0.5, 0, -0.5, 1, 1.0, 2, 2.5, 100, "x", true, json::array(), json::object()};
    const std::vector<json> kIntVals = {0, 1, -1, 3, 7, 2.7, 65, 2147483647, -2147483648LL, 4294967297LL, "x", false};
    const std::vector<json> kBoolVals = {true, false, 1, "x"};
    const char* kFloatKeys[] = {"temperature",      "top_p",         "min_p",          "typical_p",
                                "repetition_penalty", "frequency_penalty", "presence_penalty", "dry_multiplier",
                                "dry_base",         "mirostat_tau",  "mirostat_eta",   "think_budget"};
    const char* kIntKeys[] = {"top_k",          "seed",       "priority",           "repeat_last_n",
                              "dry_allowed_length", "dry_penalty_last_n", "mirostat", "n",
                              "top_logprobs",   "max_tokens", "max_completion_tokens", "max_output_tokens",
                              "best_of"};
    const char* kBoolKeys[] = {"stream", "ignore_eos", "cache_prompt", "echo", "logprobs"};
    std::vector<std::pair<std::string, json>> out;
    out.emplace_back("<base>", json::object());
    for (const char* k : kFloatKeys)
        for (const auto& v : kFloatVals)
            out.emplace_back(std::string(k) + "=" + v.dump(), json{{k, v}});
    for (const char* k : kIntKeys)
        for (const auto& v : kIntVals)
            out.emplace_back(std::string(k) + "=" + v.dump(), json{{k, v}});
    for (const char* k : kBoolKeys)
        for (const auto& v : kBoolVals)
            out.emplace_back(std::string(k) + "=" + v.dump(), json{{k, v}});
    const std::vector<json> kMisc = {
        {{"stop", "END"}},
        {{"stop", {"a", "bcd"}}},
        {{"stop", 5}},
        {{"stop_sequences", {"a", "bcd"}}},
        {{"speculative", true}},
        {{"speculative", false}},
        {{"speculative", {{"mtp_k", 1}}}},
        {{"speculative", {{"mtp_k", 99}}}},
        {{"speculative", "x"}},
        {{"stream_options", {{"include_usage", true}}}},
        {{"stream", true}, {"stream_options", {{"include_usage", true}}}},
        {{"stream", true}, {"n", 2}},
        {{"logprobs", true}, {"top_logprobs", 25}},
        {{"logprobs", true}, {"top_logprobs", -3}},
        {{"logit_bias", {{"5", 10}}}},
        {{"temperature", nullptr}},
        {{"top_p", nullptr}, {"top_k", nullptr}},
        {{"thinking", {{"type", "enabled"}, {"budget_tokens", 32}}}},
        {{"thinking", {{"type", "disabled"}}}},
        {{"reasoning", {{"effort", "low"}}}},
        {{"reasoning", {{"effort", "high"}}}},
        {{"enable_thinking", true}},
        {{"chat_template_kwargs", {{"enable_thinking", false}}}},
        {{"reasoning_effort", "low"}},
        {{"lora", "a"}},
        {{"guided_regex", "[a-z]+"}},
        {{"grammar", "root ::= \"a\""}},
        {{"response_format", {{"type", "json_object"}}}},
        {{"text", {{"format", {{"type", "json_object"}}}}}},
        {{"prediction", {{"type", "content"}, {"content", "abc"}}}},
        {{"metadata", {{"user_id", "u"}}}},
        {{"temperature", 0.2}, {"top_p", 0.5}, {"top_k", 9}, {"seed", 4}, {"priority", -2}, {"min_p", 0.1},
         {"typical_p", 0.9}, {"repetition_penalty", 1.2}, {"frequency_penalty", 0.3}, {"presence_penalty", -0.3},
         {"repeat_last_n", 64}, {"dry_multiplier", 0.8}, {"dry_base", 2.0}, {"dry_allowed_length", 3},
         {"dry_penalty_last_n", 128}, {"mirostat", 2}, {"mirostat_tau", 4.0}, {"mirostat_eta", 0.2},
         {"think_budget", 0.25}, {"ignore_eos", true}, {"cache_prompt", true}},
    };
    for (const auto& m : kMisc)
        out.emplace_back(m.dump(), m);
    return out;
}

// Result minus the dialect's base params: a case records only what it changed.
json diff_from_base(const json& r, const json& base) {
    if (!r.contains("params") || !base.contains("params"))
        return r;
    json d = json::object();
    for (const auto& [k, v] : r["params"].items())
        if (!base["params"].contains(k) || base["params"][k] != v)
            d[k] = v;
    return {{"params", d}};
}

json compute_golden() {
    json g = json::object();
    const auto cs = cases();
    for (const auto& d : dialects()) {
        const json base_res = d.run(d.base);
        json section = json::object();
        for (const auto& [name, delta] : cs) {
            if (name == "<base>") {
                section[name] = base_res;
                continue;
            }
            json body = d.base;
            for (const auto& [k, v] : delta.items())
                body[k] = v;
            section[name] = diff_from_base(d.run(body), base_res);
        }
        g[d.name] = std::move(section);
    }
    return g;
}

TEST(SamplingParamsGolden, EveryDialectMatchesTheRecordedParse) {
    const json got = compute_golden();
    if (const char* upd = std::getenv("IMP_UPDATE_SAMPLING_GOLDEN"); upd && std::string(upd) == "1") {
        std::ofstream(kGoldenPath) << got.dump(1) << "\n";
        GTEST_SKIP() << "rewrote " << kGoldenPath;
    }
    std::ifstream in(kGoldenPath);
    ASSERT_TRUE(in.good()) << kGoldenPath;
    const json want = json::parse(in);
    size_t n = 0;
    for (const auto& [dialect, section] : want.items()) {
        ASSERT_TRUE(got.contains(dialect)) << dialect;
        for (const auto& [name, expected] : section.items()) {
            ++n;
            ASSERT_TRUE(got[dialect].contains(name)) << dialect << " " << name;
            EXPECT_EQ(got[dialect][name], expected) << dialect << " " << name;
        }
    }
    EXPECT_EQ(n, got.size() * cases().size());
    std::printf("[sampling golden] %zu dialects x %zu bodies = %zu cases\n", got.size(), cases().size(), n);
}

}  // namespace
