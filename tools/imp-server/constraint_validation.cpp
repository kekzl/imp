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
