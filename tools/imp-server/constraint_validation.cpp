// Admission-time validation of constrained-decoding requests. Split out of handlers.cpp (same
// reason as bearer_token_matches() in utils.cpp): the CPU test lane compiles this file but not
// handlers.cpp's engine/CUDA/httplib dependencies, so a rule only inside the real handler is a
// rule CI never checks.

#include "handlers_internal.h"
#include "utils.h"

#include "compute/regex_constrain.h"
#include "compute/gbnf_grammar.h"
#include "compute/json_schema.h"

#include <string>
#include <vector>

// A constraint imp can't compile used to be silently dropped and the request answered anyway
// (HTTP 200, free-form text, no signal it wasn't enforced, #1256). Constrained decoding is a
// guarantee, so an uncompilable pattern is a bad request, not a silent downgrade. Validated at
// admission rather than where the constrainer is built, since ensure_constraints_() runs from
// prefill/decode where the only way to report this is aborting an already-accepted request. The
// same parsers the engine uses are called, so the two can't disagree about what's enforceable.
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

    // A schema the parser can't build rejects for a narrow set of reasons (unresolvable/
    // unsupported $ref, non-JSON document). Previously logged and fell back to any-JSON, which
    // still looks like valid JSON while the requested structure was never enforced - harder to
    // notice than the regex case.
    // Deliberately NOT rejected: a schema the parser accepts but can't extract structure from
    // ({"type":"object"} with no properties, documented as meaning json_object) and an unknown
    // `type` (falls back to string) - those are tolerances, not failures.
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

// A content part this server cannot read (video_url, malformed image_url) must be rejected,
// not silently dropped: a 200 answering unread input looks identical to a real answer.
// Checked at admission; compiled in the CPU-only lane so the rule runs in CI.
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

// Anthropic dialect equivalent of the OpenAI content-part check (id 441).
// Allowlist = what anthropic_to_openai_body converts: text, image, tool_use, thinking,
// tool_result; redacted_thinking passes through. Runs before conversion silently drops unknowns.
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

// tool_result.content array: only text/image blocks are read downstream (push_user_turn).
// An unguarded unreadable block leaves the tool body empty, so the model answers 200 on nothing.
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

// system field allowlist matches flatten_system: string, or array of text blocks; other shapes
// silently become "" and the whole system prompt is lost.
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
    // Leading run of system-role messages folds into the system prompt via flatten_system and uses
    // its allowlist; a system message after the first turn is an ordinary message (keeps images).
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

// tool_choice naming a tool absent from `tools`, or requiring a call with no tools at all, is a
// contradictory request: 400, not a degrade-to-prompt-hint (OpenAI parity).
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
