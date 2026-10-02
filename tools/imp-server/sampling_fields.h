#pragma once

// The one parser for the sampling fields every endpoint shares (#2461). A new field: one member
// below, one kSamplingKeys row (its `via` bits carry it through /v1/messages and /v1/responses),
// one apply_to line. Member initializers are the server defaults.

#include "runtime/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

struct SamplingFields {
    float temperature = 0.7f, top_p = 0.95f, min_p = 0.0f, typical_p = 1.0f;
    // 1.05: breaks 40-turn repetition spirals, keeps structural repetition; pass 1.0 to disable.
    float repetition_penalty = 1.05f, frequency_penalty = 0.0f, presence_penalty = 0.0f;
    float dry_multiplier = 0.0f, dry_base = 1.75f, mirostat_tau = 5.0f, mirostat_eta = 0.1f;
    float think_budget = 0.0f;  // parsed default: --think-budget
    int top_k = 40, seed = -1, repeat_last_n = 0;
    int dry_allowed_length = 2, dry_penalty_last_n = 0, mirostat = 0;
    int priority = 0;  // vLLM-compatible admission priority, lower = earlier
    bool stream = false;
    bool ignore_eos = false;    // vLLM-style: run to max_tokens, never stop on EOS (benchmarks)
    bool cache_prompt = false;  // pin the prompt's KV blocks (llama.cpp field, Anthropic cache_control)
    std::string session_id;     // agent session (#2407): pins the prompt blocks across turns
    bool top_p_explicit = false, top_k_explicit = false, rep_pen_explicit = false;

    // Everything but seed and stream, which callers set per completion.
    void apply_to(imp::Request& r) const {
        r.temperature = temperature;
        r.top_p = top_p;
        r.top_k = top_k;
        r.priority = priority;
        r.min_p = min_p;
        r.typical_p = typical_p;
        r.repetition_penalty = repetition_penalty;
        r.frequency_penalty = frequency_penalty;
        r.presence_penalty = presence_penalty;
        r.repeat_last_n = repeat_last_n;
        r.dry_multiplier = dry_multiplier;
        r.dry_base = dry_base;
        r.dry_allowed_length = dry_allowed_length;
        r.dry_penalty_last_n = dry_penalty_last_n;
        r.mirostat = mirostat;
        r.mirostat_tau = mirostat_tau;
        r.mirostat_eta = mirostat_eta;
        r.ignore_eos = ignore_eos;
        r.think_budget = think_budget;
        r.pin_kv_prefix = cache_prompt;
        r.session_id = session_id;
    }
};

// Dialects whose body transform copies the key verbatim into the OpenAI body.
inline constexpr uint8_t kViaMessages = 1, kViaResponses = 2;

struct SamplingKey {
    const char* key;
    // monostate: passed through only, parsed elsewhere (speculative: parse_spec_field_).
    std::variant<std::monostate, float SamplingFields::*, int SamplingFields::*, bool SamplingFields::*,
                 std::string SamplingFields::*>
        member;
    uint8_t via;
};

// Parse order = row order: the first wrong-typed field is the one a 400 reports.
inline constexpr SamplingKey kSamplingKeys[] = {
    {"temperature", &SamplingFields::temperature, kViaMessages | kViaResponses},
    {"top_p", &SamplingFields::top_p, kViaMessages | kViaResponses},
    {"top_k", &SamplingFields::top_k, kViaMessages},
    {"seed", &SamplingFields::seed, 0},
    {"priority", &SamplingFields::priority, kViaMessages | kViaResponses},
    {"stream", &SamplingFields::stream, kViaMessages | kViaResponses},
    {"min_p", &SamplingFields::min_p, 0},
    {"typical_p", &SamplingFields::typical_p, 0},
    {"repetition_penalty", &SamplingFields::repetition_penalty, 0},
    {"frequency_penalty", &SamplingFields::frequency_penalty, 0},
    {"presence_penalty", &SamplingFields::presence_penalty, 0},
    {"repeat_last_n", &SamplingFields::repeat_last_n, 0},
    {"dry_multiplier", &SamplingFields::dry_multiplier, 0},
    {"dry_base", &SamplingFields::dry_base, 0},
    {"dry_allowed_length", &SamplingFields::dry_allowed_length, 0},
    {"dry_penalty_last_n", &SamplingFields::dry_penalty_last_n, 0},
    {"mirostat", &SamplingFields::mirostat, 0},
    {"mirostat_tau", &SamplingFields::mirostat_tau, 0},
    {"mirostat_eta", &SamplingFields::mirostat_eta, 0},
    {"think_budget", &SamplingFields::think_budget, 0},
    {"ignore_eos", &SamplingFields::ignore_eos, 0},
    {"cache_prompt", &SamplingFields::cache_prompt, 0},
    {"session_id", &SamplingFields::session_id, kViaMessages | kViaResponses},
    {"speculative", std::monostate{}, kViaMessages | kViaResponses},
};

// Same contract as body.value(key, default): absent keeps the default, a wrong JSON type throws
// json::type_error (the route's handler maps it to a 400).
inline void parse_sampling_fields(const nlohmann::json& body, float default_think_budget,
                                  SamplingFields& out) {
    out = SamplingFields{};
    out.think_budget = default_think_budget;
    for (const auto& k : kSamplingKeys) {
        const auto it = body.find(k.key);
        if (it == body.end())
            continue;
        std::visit(
            [&](auto m) {
                if constexpr (!std::is_same_v<decltype(m), std::monostate>)
                    out.*m = it->template get<std::remove_reference_t<decltype(out.*m)>>();
            },
            k.member);
    }
    out.top_p_explicit = body.contains("top_p");
    out.top_k_explicit = body.contains("top_k");
    out.rep_pen_explicit = body.contains("repetition_penalty");
}

// Copies the keys a dialect carries (`via` bit) from its body into the OpenAI body.
inline void pass_sampling_keys(const nlohmann::json& from, uint8_t via, nlohmann::json& to) {
    for (const auto& k : kSamplingKeys)
        if ((k.via & via) != 0 && from.contains(k.key))
            to[k.key] = from[k.key];
}

// session_id (#2407): 1-128 chars of [A-Za-z0-9._:-], so it is also a /v1/sessions/{id}/close
// path segment.
inline constexpr const char* kSessionIdRule = "\"session_id\" must be 1-128 characters of [A-Za-z0-9._:-]";
inline bool session_id_valid(const std::string& s) {
    bool ok = !s.empty() && s.size() <= 128;
    for (const char c : s)
        ok = ok && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' ||
                    c == '_' || c == ':' || c == '-');
    return ok;
}
// After parse_sampling_fields (type already checked). Empty when valid or absent.
inline std::string session_id_error(const SamplingFields& f, const nlohmann::json& body) {
    return !body.contains("session_id") || session_id_valid(f.session_id) ? std::string() : kSessionIdRule;
}
