#pragma once

#include <nlohmann/json.hpp>

#include <cstdio>
#include <limits>
#include <string>

// A request scalar of the wrong JSON type threw json::type_error out of body.value(), and the client
// got "[json.exception.type_error.302] type must be number, but is string" without the field name
// (15 of 26 fuzzed fields across chat, completions, responses, embeddings, tokenize; /v1/messages
// answered 500). Returns "" or the 400 message naming the first known field with the wrong type.
inline std::string wrong_field_type_message(const std::string& raw_body) {
    const nlohmann::json body = nlohmann::json::parse(raw_body, nullptr, /*allow_exceptions=*/false);
    if (!body.is_object())
        return "";
    struct Field {
        const char* key;
        char kind;  // n = number, b = boolean, s = string, l = boolean or number (logprobs)
    };
    static constexpr Field kFields[] = {
        {"temperature", 'n'},
        {"top_p", 'n'},
        {"top_k", 'n'},
        {"min_p", 'n'},
        {"typical_p", 'n'},
        {"max_tokens", 'n'},
        {"max_completion_tokens", 'n'},
        {"max_output_tokens", 'n'},
        {"n", 'n'},
        {"seed", 'n'},
        {"presence_penalty", 'n'},
        {"frequency_penalty", 'n'},
        {"repetition_penalty", 'n'},
        {"think_budget", 'n'},
        {"priority", 'n'},
        {"top_logprobs", 'n'},
        {"stream", 'b'},
        {"echo", 'b'},
        {"model", 's'},
        {"encoding_format", 's'},
        {"content", 's'},
        {"logprobs", 'l'},
        {"top_n", 'n'},
        {"return_documents", 'b'},
    };
    for (const auto& f : kFields) {
        const auto it = body.find(f.key);
        if (it == body.end() || it->is_null())
            continue;
        const bool ok = f.kind == 'n'   ? it->is_number()
                        : f.kind == 'b' ? it->is_boolean()
                        : f.kind == 'l' ? it->is_boolean() || it->is_number()
                                        : it->is_string();
        if (!ok)
            return std::string("\"") + f.key + "\" must be a " +
                   (f.kind == 'n'   ? "number"
                    : f.kind == 'b' ? "boolean"
                    : f.kind == 'l' ? "boolean or a number"
                                    : "string") +
                   ", got " + it->type_name();
    }
    if (const auto so = body.find("stream_options"); so != body.end() && so->is_object()) {
        const auto iu = so->find("include_usage");
        if (iu != so->end() && !iu->is_null() && !iu->is_boolean())
            return "\"stream_options.include_usage\" must be a boolean, got " + std::string(iu->type_name());
    }
    if (const auto tools = body.find("tools"); tools != body.end() && tools->is_array()) {
        for (size_t i = 0; i < tools->size(); ++i)
            if (!(*tools)[i].is_object())
                return "\"tools[" + std::to_string(i) + "]\" must be an object, got " +
                       std::string((*tools)[i].type_name());
    }
    if (const auto msgs = body.find("messages"); msgs != body.end() && msgs->is_array()) {
        for (size_t i = 0; i < msgs->size(); ++i) {
            const auto& m = (*msgs)[i];
            if (!m.is_object())
                return "\"messages[" + std::to_string(i) + "]\" must be an object, got " +
                       std::string(m.type_name());
            const auto c = m.find("content");
            if (c != m.end() && !c->is_null() && !c->is_string() && !c->is_array())
                return "\"messages[" + std::to_string(i) +
                       "].content\" must be a string or an array of content "
                       "parts, got " +
                       std::string(c->type_name());
        }
    }
    return "";
}

// Sampler ranges the parser passed through unchecked: the penalty kernel divides positive logits by
// repetition_penalty (sampling_penalties.cu), so 0 gives inf and < 0 flips the penalty into a boost.
// Penalty bounds follow OpenAI (-2..2). Returns "" or the 400 message. Non-numbers are left to the
// type check above.
inline std::string out_of_range_message(const nlohmann::json& body) {
    struct Range {
        const char* key;
        double lo, hi;
        bool lo_open;  // true: the lower bound itself is invalid
    };
    constexpr double kInf = std::numeric_limits<double>::infinity();
    static constexpr Range kRanges[] = {
        {"repetition_penalty", 0.0, kInf, true}, {"min_p", 0.0, 1.0, false},
        {"typical_p", 0.0, 1.0, true},           {"presence_penalty", -2.0, 2.0, false},
        {"frequency_penalty", -2.0, 2.0, false},
    };
    for (const auto& r : kRanges) {
        const auto it = body.find(r.key);
        if (it == body.end() || !it->is_number())
            continue;
        const double v = it->get<double>();
        if (v >= r.lo && v <= r.hi && !(r.lo_open && v == r.lo))
            continue;
        char buf[112];
        if (r.hi == kInf)
            snprintf(buf, sizeof(buf), "\"%s\" must be greater than %g", r.key, r.lo);
        else if (r.lo_open)
            snprintf(buf, sizeof(buf), "\"%s\" must be greater than %g and at most %g", r.key, r.lo, r.hi);
        else
            snprintf(buf, sizeof(buf), "\"%s\" must be between %g and %g", r.key, r.lo, r.hi);
        return buf;
    }
    return "";
}
