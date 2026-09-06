// Admission-time validation of constrained-decoding requests.
//
// Split out of handlers.cpp for the same reason bearer_token_matches() lives in
// utils.cpp: the CPU test lane compiles this file, and handlers.cpp (with its
// engine, CUDA and httplib-server dependencies) it does not. A validation rule
// that only runs inside the real handler is a rule CI never checks.

#include "handlers_internal.h"
#include "utils.h"

#include "compute/regex_constrain.h"
#include "compute/gbnf_grammar.h"
#include "compute/json_schema.h"

#include <string>
#include <vector>

// A constraint imp cannot compile used to be dropped and the request answered
// anyway: HTTP 200, free-form text, nothing in the reply to distinguish it from
// a satisfied constraint (#1256). Constrained decoding is a guarantee, so a
// pattern that cannot back it is a bad request, not a silent downgrade.
//
// Validated here, at admission, rather than where the constrainer is built:
// ensure_constraints_() runs from prefill and decode, where the only way to
// report this would be to abort a request that was already accepted.
//
// The same parsers the engine uses are called, so the two cannot drift into
// disagreeing about what is enforceable.
bool validate_constraints(const json& body, httplib::Response& res) {
    auto reject = [&res](const std::string& message) {
        res.status = 400;
        json err = {{"error", {{"message", message}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    };

    // Every spelling the parameter parser accepts, so validation cannot be
    // sidestepped by using the vLLM/llama.cpp aliases.
    std::string pattern, grammar, schema;
    if (body.contains("response_format") && body["response_format"].is_object()) {
        const auto& rf = body["response_format"];
        const std::string fmt = rf.value("type", "text");
        if (fmt == "json_schema" && rf.contains("json_schema") && rf["json_schema"].is_object()) {
            const auto& js = rf["json_schema"];
            if (js.contains("schema") && js["schema"].is_object())
                schema = dump_safe(js["schema"]);
        }
        if (fmt == "regex") {
            if (rf.contains("regex") && rf["regex"].is_string())
                pattern = rf["regex"].get<std::string>();
            else if (rf.contains("pattern") && rf["pattern"].is_string())
                pattern = rf["pattern"].get<std::string>();
        } else if (fmt == "grammar") {
            if (rf.contains("grammar") && rf["grammar"].is_string())
                grammar = rf["grammar"].get<std::string>();
            else if (rf.contains("gbnf") && rf["gbnf"].is_string())
                grammar = rf["gbnf"].get<std::string>();
        }
    }
    if (pattern.empty() && body.contains("guided_regex") && body["guided_regex"].is_string())
        pattern = body["guided_regex"].get<std::string>();
    if (grammar.empty() && body.contains("grammar") && body["grammar"].is_string())
        grammar = body["grammar"].get<std::string>();
    if (grammar.empty() && body.contains("guided_grammar") && body["guided_grammar"].is_string())
        grammar = body["guided_grammar"].get<std::string>();

    if (!pattern.empty()) {
        // Pattern-only init: no tokenizer needed, so this runs before a model
        // is touched and costs nothing on the happy path.
        imp::RegexConstrainer probe;
        if (!probe.init_pattern_only(pattern))
            return reject("\"" + pattern +
                          "\" cannot be enforced as a constraint: unsupported or malformed regex. "
                          "Lookaround, word boundaries, backreferences and interior anchors are not "
                          "supported; a pattern matching nothing is refused too.");
    }

    if (!grammar.empty()) {
        std::vector<imp::GbnfRule> rules;
        int32_t root = -1;
        std::string err;
        if (!imp::parse_gbnf(grammar, rules, root, &err))
            return reject("the GBNF grammar cannot be enforced as a constraint: " + err);
    }

    // A schema the parser cannot build rejects for a narrow set of reasons — an
    // unresolvable or unsupported `$ref`, or a document that is not JSON. The
    // engine logged "Failed to parse JSON schema" and carried on, which for a
    // `json_schema` request means falling back to any-JSON: the reply is still
    // JSON, so it looks right, while the structure the caller asked for was
    // never enforced. That is harder to notice than the regex case, not easier.
    //
    // Deliberately NOT rejected here: a schema the parser accepts but cannot
    // extract structure from (`{"type":"object"}` with no properties) is
    // documented to mean json_object, and an unknown `type` falls back to
    // string rather than failing. Those are tolerances, not failures, so
    // turning them into 400s would break working clients.
    if (!schema.empty() && !imp::parse_json_schema(schema))
        return reject(
            "the JSON schema cannot be enforced as a constraint: it could not be "
            "parsed. Usual causes: a keyword this build cannot enforce "
            "(minimum/maximum/multipleOf/allOf/not/uniqueItems and the rest listed "
            "in docs/LIMITATIONS.md), a non-string \"enum\" or \"const\" member, "
            "or an unresolvable \"$ref\" (only local \"#/$defs/...\" and "
            "\"#/definitions/...\" are supported). The server log names the one "
            "that fired.");

    return true;
}

// A content part this server cannot read used to fall through the parsing chain
// in silence: `video_url` (imp has no video path at all) produced a 200
// answering a prompt the model never saw, and an `image_url` part with the
// object missing did the same. Answering as if the input had been understood is
// worse than refusing it — the caller cannot tell that reply apart from one that
// actually used its picture.
//
// Checked at admission alongside the constraints, and in this TU for the same
// reason: the CPU lane compiles it, so the rule runs in CI.
bool validate_content_parts(const json& body, httplib::Response& res) {
    if (!body.contains("messages") || !body["messages"].is_array())
        return true;
    for (const auto& msg : body["messages"]) {
        if (!msg.is_object() || !msg.contains("content") || !msg["content"].is_array())
            continue;
        for (const auto& part : msg["content"]) {
            if (!part.is_object())
                continue;
            const std::string type = part.value("type", "");
            if (type == "text")
                continue;
            if (type == "image_url" && part.contains("image_url"))
                continue;
            res.status = 400;
            json err = {{"error",
                         {{"message", "unsupported content part \"" +
                                          (type.empty() ? std::string("(missing type)") : type) +
                                          "\": this endpoint reads \"text\" and \"image_url\" parts"},
                          {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return false;
        }
    }
    return true;
}

// The same rule for the Anthropic dialect, in the Anthropic spelling.
//
// `/v1/messages` converts to an OpenAI body first (`anthropic_to_openai_body`)
// and only then reaches `validate_content_parts`. The converter's block loop
// has no `else`, so a block it does not know is deleted and the transformed
// body arrives clean: the check meant to catch the problem stands behind a gate
// that already removed the evidence (the #1384 shape). Measured on the
// model-less server: `input_audio` and `video_url` were 400 on
// /v1/chat/completions and /v1/responses, and fell through /v1/messages.
//
// The allowlist is the set `anthropic.cpp` can actually convert, read off its
// own loops: `text` and `image` (convert_message_content), `tool_use` and
// `thinking` (push_assistant_turn), `tool_result` (the user-turn split).
// `redacted_thinking` carries no input and rides along. Anything else - a
// `document`, a `search_result`, an audio block - is content the caller
// believes was read.
//
// Returns true when a block is unreadable, with `why` describing it. The caller
// owns the response because the Anthropic error envelope differs from OpenAI's.
namespace {

// An `image` block converts only from a `base64` or a `url` source; every other
// source pushes nothing at all. Shared by both levels, because the converter
// uses the same routine for both.
bool anthropic_image_is_readable(const json& block, std::string& why) {
    const std::string src = block.contains("source") && block["source"].is_object()
                                ? block["source"].value("type", "")
                                : "";
    if (src == "base64" || src == "url")
        return true;
    why = "image block with source type \"" + (src.empty() ? std::string("(missing)") : src) +
          "\": this endpoint reads \"base64\" and \"url\" image sources";
    return false;
}

// `tool_result.content` may itself be an array of blocks, and the converter's
// inner loop (anthropic.cpp, push_user_turn) reads only `text` and `image`
// there. Accepting `tool_result` wholesale at the outer level left that array
// unguarded, which costs more than a drop: an unreadable block leaves the tool
// body EMPTY, so the model is told the tool returned nothing and answers 200 on
// it.
bool anthropic_tool_result_unreadable(const json& block, std::string& why) {
    if (!block.contains("content"))
        return false;
    const auto& c = block["content"];
    if (!c.is_array())  // a plain string is the common shape and converts whole
        return false;
    for (const auto& p : c) {
        if (!p.is_object())
            continue;
        const std::string ptype = p.value("type", "");
        if (ptype == "text")
            continue;
        if (ptype == "image") {
            if (anthropic_image_is_readable(p, why))
                continue;
            why = "inside a tool_result: " + why;
            return true;
        }
        why = "unsupported block \"" + (ptype.empty() ? std::string("(missing type)") : ptype) +
              "\" inside a tool_result: this endpoint reads \"text\" and \"image\" there";
        return true;
    }
    return false;
}

}  // namespace

// The `system` field, which the block walk below never reached: it keys on
// "messages" only. `flatten_system` (anthropic.cpp) reads a string, or an array
// from which it keeps `text` blocks; every other shape returns "" and the whole
// system prompt is gone. Measured on the model-less binary before this: a bare
// object, a number, and an array carrying an image block all reached the model
// lookup, so with weights the model would have answered without its
// instructions and said nothing about it.
//
// The allowlist is `flatten_system`'s own capability, which is also what the
// upstream API allows in this field. `cache_control` rides on a `text` block and
// is unaffected.
bool anthropic_system_unreadable(const json& system_field, std::string& why) {
    if (system_field.is_null() || system_field.is_string())
        return false;
    if (!system_field.is_array()) {
        why = "\"system\" must be a string or an array of text blocks";
        return true;
    }
    for (const auto& block : system_field) {
        if (!block.is_object()) {
            why = "\"system\" array holds a non-object entry; it takes text blocks";
            return true;
        }
        const std::string type = block.value("type", "");
        if (type == "text")
            continue;
        why = "unsupported \"system\" block \"" + (type.empty() ? std::string("(missing type)") : type) +
              "\": this field takes \"text\" blocks only";
        return true;
    }
    return false;
}

bool anthropic_unreadable_block(const json& body, std::string& why) {
    if (body.contains("system") && anthropic_system_unreadable(body["system"], why))
        return true;
    if (!body.contains("messages") || !body["messages"].is_array())
        return false;
    // The Messages API has no `system` role, but clients ported from the OpenAI
    // dialect send one and imp folds the LEADING run of them into the system
    // prompt through flatten_system (anthropic.cpp), consuming each one whether
    // or not anything survived the fold. Those carry the system field's narrower
    // allowlist. A system message AFTER the first turn is not folded; it reaches
    // push_user_turn and keeps its images, so it is checked as an ordinary
    // message. Getting this boundary wrong in either direction is a false
    // refusal or a silent drop, so it is read off the converter's own loop.
    bool still_leading = true;
    for (const auto& msg : body["messages"]) {
        const bool is_leading_system = still_leading && msg.is_object() &&
                                       msg.value("role", "user") == "system";
        if (!is_leading_system)
            still_leading = false;
        if (!msg.is_object() || !msg.contains("content"))
            continue;
        if (is_leading_system) {
            if (anthropic_system_unreadable(msg["content"], why))
                return true;
            continue;
        }
        if (!msg["content"].is_array())
            continue;
        for (const auto& block : msg["content"]) {
            if (!block.is_object())
                continue;
            const std::string type = block.value("type", "");
            if (type == "text" || type == "tool_use" || type == "thinking" || type == "redacted_thinking")
                continue;
            if (type == "tool_result") {
                if (anthropic_tool_result_unreadable(block, why))
                    return true;
                continue;
            }
            if (type == "image") {
                if (anthropic_image_is_readable(block, why))
                    continue;
                return true;
            }
            why = "unsupported content block \"" + (type.empty() ? std::string("(missing type)") : type) +
                  "\": this endpoint reads \"text\", \"image\", \"tool_use\", \"tool_result\" and "
                  "\"thinking\" blocks";
            return true;
        }
    }
    return false;
}

// `tool_choice` naming a tool that is not in `tools`, or demanding a tool call
// with no tools at all, is a CONTRADICTORY request rather than a loose one —
// unlike a tool whose schema simply cannot be enforced, which legitimately
// degrades to prompt-hint choice. Answering it anyway produced a model
// inventing a call to a function the caller never described (measured: a
// request naming "nonexistent" came back with a call to "g"). OpenAI answers
// 400 for both.
bool validate_tool_choice(const json& body, httplib::Response& res) {
    if (!body.contains("tool_choice"))
        return true;

    auto reject = [&res](const std::string& message) {
        res.status = 400;
        json err = {{"error", {{"message", message}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    };

    const json& tc = body["tool_choice"];
    const bool has_tools = body.contains("tools") && body["tools"].is_array() && !body["tools"].empty();

    // "none" is satisfiable without tools; "auto" is a no-op there.
    if (tc.is_string()) {
        const std::string s = tc.get<std::string>();
        if (s == "required" && !has_tools)
            return reject("\"tool_choice\": \"required\" needs a non-empty \"tools\" array");
        return true;
    }

    if (!tc.is_object() || !tc.contains("function") || !tc["function"].is_object())
        return true;  // shapes this server does not interpret are left alone
    const std::string want = tc["function"].value("name", "");
    if (want.empty())
        return true;
    if (!has_tools)
        return reject("\"tool_choice\" names the function \"" + want +
                      "\" but the request carries no \"tools\"");
    for (const auto& t : body["tools"]) {
        if (!t.is_object() || !t.contains("function") || !t["function"].is_object())
            continue;
        if (t["function"].value("name", "") == want)
            return true;
    }
    return reject("\"tool_choice\" names the function \"" + want +
                  "\", which is not among the tools in this request");
}
