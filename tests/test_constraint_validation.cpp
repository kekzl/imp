// =============================================================================
// Admission-time validation of constrained-decoding requests (#1256).
//
// WHY: a constraint imp cannot compile used to be dropped and the request
// answered anyway — HTTP 200, free-form text, nothing in the reply telling the
// caller its guarantee was not applied. These assert the 400 contract on the
// CPU, in CI, where the real handler never runs.
//
// The alias coverage is the part most likely to rot: `regex`/`pattern`,
// `grammar`/`gbnf`, `guided_regex`, `guided_grammar` and llama.cpp's bare
// `grammar` all reach the same engine, so validation that misses one is a
// bypass, not a gap. A new spelling added to the parameter parser without a
// matching line here is exactly the regression these catch.
// =============================================================================

#include <gtest/gtest.h>

#include "handlers_internal.h"
#include "responses.h"

#include <nlohmann/json.hpp>
#include <httplib.h>

using json = nlohmann::json;

namespace {

// Returns true when the body is accepted (no 400 written).
bool accepts(const json& body) {
    httplib::Response res;
    return validate_constraints(body, res);
}

// Returns the error message a rejected body produces (empty when accepted).
std::string rejection_message(const json& body) {
    httplib::Response res;
    if (validate_constraints(body, res))
        return {};
    EXPECT_EQ(res.status, 400);
    auto parsed = json::parse(res.body, nullptr, false);
    if (parsed.is_discarded() || !parsed.contains("error"))
        return "<unparseable>";
    EXPECT_EQ(parsed["error"].value("type", ""), "invalid_request_error");
    return parsed["error"].value("message", "");
}

json regex_body(const std::string& pattern) {
    return json{{"response_format", {{"type", "regex"}, {"regex", pattern}}}};
}

json grammar_body(const std::string& gbnf) {
    return json{{"response_format", {{"type", "grammar"}, {"grammar", gbnf}}}};
}

}  // namespace

TEST(ConstraintValidation, UnconstrainedRequestIsUntouched) {
    EXPECT_TRUE(accepts(json::object()));
    EXPECT_TRUE(accepts(json{{"max_tokens", 16}}));
    EXPECT_TRUE(accepts(json{{"response_format", {{"type", "text"}}}}));
    // json_object carries no pattern to compile.
    EXPECT_TRUE(accepts(json{{"response_format", {{"type", "json_object"}}}}));
}

TEST(ConstraintValidation, EnforceablePatternsAreAccepted) {
    EXPECT_TRUE(accepts(regex_body("[0-9]{3}")));
    // Edge anchors are redundant, not unsupported (#1255) — they must NOT 400.
    EXPECT_TRUE(accepts(regex_body("^[0-9]{3}$")));
    EXPECT_TRUE(accepts(regex_body("^(yes|no)$")));
    // Non-capturing groups compile since #1257.
    EXPECT_TRUE(accepts(regex_body("(?:ab)+")));
    EXPECT_TRUE(accepts(grammar_body("root ::= \"yes\" | \"no\"")));
}

TEST(ConstraintValidation, MalformedRegexIsRejectedWithItsPattern) {
    const std::string msg = rejection_message(regex_body("^[0-9{3}$"));
    EXPECT_NE(msg.find("^[0-9{3}$"), std::string::npos)
        << "the message must quote the pattern the caller sent";
    EXPECT_NE(msg.find("cannot be enforced"), std::string::npos);
}

TEST(ConstraintValidation, UnsupportedConstructsAreRejected) {
    for (const char* bad : {"(?=lookahead)x", "a\\b", "(a)\\1", "a^b"}) {
        EXPECT_FALSE(accepts(regex_body(bad))) << "should have been rejected: " << bad;
    }
}

// A pattern that matches nothing would decode to an empty string with no
// explanation, which is the failure this whole change exists to remove.
TEST(ConstraintValidation, EmptyLanguageIsRejected) {
    EXPECT_FALSE(accepts(regex_body("[z-a]")));
}

TEST(ConstraintValidation, MalformedGrammarIsRejectedWithTheParserError) {
    const std::string msg = rejection_message(grammar_body("root ::= <<<"));
    EXPECT_NE(msg.find("GBNF"), std::string::npos);
    // The parser's own diagnostic is forwarded rather than replaced by a generic one.
    EXPECT_NE(msg.find("unexpected character"), std::string::npos) << "got: " << msg;
}

// Every spelling the parameter parser accepts must be validated, or the check
// is bypassable by using an alias.
TEST(ConstraintValidation, AllRegexSpellingsAreValidated) {
    EXPECT_FALSE(accepts(json{{"response_format", {{"type", "regex"}, {"regex", "(?=x)y"}}}}));
    EXPECT_FALSE(accepts(json{{"response_format", {{"type", "regex"}, {"pattern", "(?=x)y"}}}}));
    EXPECT_FALSE(accepts(json{{"guided_regex", "(?=x)y"}})) << "vLLM spelling";
}

TEST(ConstraintValidation, AllGrammarSpellingsAreValidated) {
    EXPECT_FALSE(accepts(json{{"response_format", {{"type", "grammar"}, {"grammar", "root ::= <<<"}}}}));
    EXPECT_FALSE(accepts(json{{"response_format", {{"type", "grammar"}, {"gbnf", "root ::= <<<"}}}}));
    EXPECT_FALSE(accepts(json{{"grammar", "root ::= <<<"}})) << "llama.cpp spelling";
    EXPECT_FALSE(accepts(json{{"guided_grammar", "root ::= <<<"}})) << "vLLM spelling";
}

// Wrong JSON types must not be read as a constraint — and must not crash.
TEST(ConstraintValidation, NonStringConstraintFieldsAreIgnored) {
    EXPECT_TRUE(accepts(json{{"guided_regex", 42}}));
    EXPECT_TRUE(accepts(json{{"grammar", json::array({1, 2})}}));
    EXPECT_TRUE(accepts(json{{"response_format", {{"type", "regex"}, {"regex", nullptr}}}}));
    EXPECT_TRUE(accepts(json{{"response_format", "not-an-object"}}));
}

// ---------------------------------------------------------------------------
// JSON Schema. The failure here is quieter than the regex one: the engine fell
// back to any-JSON, so the reply was still JSON and still looked right, while
// the structure the caller asked for was never enforced.
// ---------------------------------------------------------------------------

namespace {
json schema_body(const json& schema) {
    return json{{"response_format", {{"type", "json_schema"}, {"json_schema", {{"schema", schema}}}}}};
}
}  // namespace

TEST(ConstraintValidation, EnforceableSchemasAreAccepted) {
    EXPECT_TRUE(accepts(schema_body(json::parse(R"({"type":"object","properties":{"a":{"type":"string"}}})"))));
    EXPECT_TRUE(accepts(schema_body(json::parse(R"({"type":"string","enum":["a","b"]})"))));
    // A local $ref resolves and must keep working.
    EXPECT_TRUE(accepts(schema_body(json::parse(
        R"({"$defs":{"S":{"type":"string"}},"type":"object","properties":{"a":{"$ref":"#/$defs/S"}}})"))));
}

TEST(ConstraintValidation, UnresolvableRefIsRejected) {
    const std::string msg = rejection_message(schema_body(json::parse(R"({"$ref":"#/definitions/missing"})")));
    EXPECT_NE(msg.find("$ref"), std::string::npos) << "got: " << msg;
}

TEST(ConstraintValidation, NonLocalRefIsRejected) {
    EXPECT_FALSE(accepts(schema_body(json::parse(R"({"$ref":"https://example.com/x.json"})"))));
}

// Tolerances, not failures — turning these into 400s would break working
// clients, so they are pinned as accepted on purpose.
TEST(ConstraintValidation, SchemaTolerancesStayAccepted) {
    // Free-form object: documented to mean json_object.
    EXPECT_TRUE(accepts(schema_body(json::parse(R"({"type":"object"})"))));
    // Unknown type falls back to string rather than failing the parse.
    EXPECT_TRUE(accepts(schema_body(json::parse(R"({"type":"nonsense"})"))));
    // json_schema with no schema member at all carries nothing to validate.
    EXPECT_TRUE(accepts(json{{"response_format", {{"type", "json_schema"}}}}));
    EXPECT_TRUE(accepts(json{{"response_format", {{"type", "json_schema"}, {"json_schema", {{"name", "x"}}}}}}));
}

// ---------------------------------------------------------------------------
// Dialect coverage. validate_constraints() is called once, from
// validate_sampling_params, which every dialect reaches through
// parse_chat_request_params. /v1/responses gets there via a shim that BUILDS
// the response_format, so the validation sees the converted body — asserted
// here rather than assumed, because a shim that stopped converting (or started
// converting after validation) would reopen the hole for that dialect only.
// ---------------------------------------------------------------------------

TEST(ConstraintValidation, ResponsesDialectSchemaReachesValidation) {
    // What a /v1/responses caller sends: text.format, not response_format.
    json rsp = {{"model", "m"},
                {"input", "hi"},
                {"text", {{"format", {{"type", "json_schema"},
                                      {"name", "r"},
                                      {"schema", json::parse(R"({"$ref":"#/definitions/missing"})")}}}}}};
    const json converted = imp_server::responses::responses_to_openai_body(rsp);
    ASSERT_TRUE(converted.contains("response_format")) << "the shim must produce a response_format";
    EXPECT_FALSE(accepts(converted)) << "an unenforceable schema must not survive the shim";
}

TEST(ConstraintValidation, ResponsesDialectGoodSchemaStillPasses) {
    json rsp = {{"model", "m"},
                {"input", "hi"},
                {"text", {{"format", {{"type", "json_schema"},
                                      {"name", "r"},
                                      {"schema", json::parse(
                                          R"({"type":"object","properties":{"a":{"type":"string"}}})")}}}}}};
    EXPECT_TRUE(accepts(imp_server::responses::responses_to_openai_body(rsp)));
}

// ---------------------------------------------------------------------------
// Content parts. A part this server cannot read used to fall through the
// parsing chain in silence: `video_url` (imp has no video path) produced a 200
// answering a prompt the model never saw. The caller cannot tell that reply
// apart from one that actually used its input.
// ---------------------------------------------------------------------------

namespace {
json parts_body(const json& parts) {
    return json{{"messages", json::array({{{"role", "user"}, {"content", parts}}})}};
}
bool parts_ok(const json& parts) {
    httplib::Response res;
    return validate_content_parts(parts_body(parts), res);
}
}  // namespace

TEST(ContentParts, TextAndImageAreAccepted) {
    EXPECT_TRUE(parts_ok(json::array({{{"type", "text"}, {"text", "hi"}}})));
    EXPECT_TRUE(parts_ok(json::array({{{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AA"}}}}})));
    EXPECT_TRUE(parts_ok(json::array({{{"type", "text"}, {"text", "what is this"}},
                                      {{"type", "image_url"}, {"image_url", {{"url", "x"}}}}})));
}

TEST(ContentParts, UnknownTypeIsRejectedAndNamed) {
    httplib::Response res;
    ASSERT_FALSE(validate_content_parts(
        parts_body(json::array({{{"type", "video_url"}, {"video_url", {{"url", "x"}}}}})), res));
    EXPECT_EQ(res.status, 400);
    EXPECT_NE(res.body.find("video_url"), std::string::npos)
        << "the message must name the part it could not read";
}

// `{"type":"image_url"}` with the object missing took the same silent path.
TEST(ContentParts, ImagePartWithoutTheObjectIsRejected) {
    EXPECT_FALSE(parts_ok(json::array({{{"type", "image_url"}}})));
}

TEST(ContentParts, MissingTypeIsRejected) {
    httplib::Response res;
    ASSERT_FALSE(validate_content_parts(parts_body(json::array({{{"text", "hi"}}})), res));
    EXPECT_NE(res.body.find("missing type"), std::string::npos);
}

// A plain string content (the overwhelmingly common shape) must not be touched.
TEST(ContentParts, StringContentIsUntouched) {
    httplib::Response res;
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "plain text"}}})}};
    EXPECT_TRUE(validate_content_parts(body, res));
    // ...and so must a request with no messages at all.
    EXPECT_TRUE(validate_content_parts(json::object(), res));
}

// Later messages are checked too, not just the first.
TEST(ContentParts, EveryMessageIsChecked) {
    json body = {{"messages", json::array({
        {{"role", "user"}, {"content", "fine"}},
        {{"role", "assistant"}, {"content", "also fine"}},
        {{"role", "user"}, {"content", json::array({{{"type", "audio"}, {"audio", "x"}}})}}})}};
    httplib::Response res;
    EXPECT_FALSE(validate_content_parts(body, res));
}

// ---------------------------------------------------------------------------
// The same rule in the Anthropic spelling.
//
// `/v1/messages` runs anthropic_to_openai_body FIRST, and that converter's
// block loop has no `else`: an unknown block is deleted, so the OpenAI body
// reaching validate_content_parts above is clean and the check finds nothing.
// Measured on the model-less binary before the fix: `input_audio` was 400 on
// /v1/chat/completions and /v1/responses, and fell through /v1/messages.
//
// The allowlist has to be exactly what the converter handles, or a legitimate
// replay starts getting refused. These tests pin both edges.
// ---------------------------------------------------------------------------

namespace {
json anth_body(const json& blocks) {
    return json{{"messages", json::array({{{"role", "user"}, {"content", blocks}}})}};
}
bool anth_refused(const json& blocks, std::string& why) {
    return anthropic_unreadable_block(anth_body(blocks), why);
}
}  // namespace

TEST(AnthropicContentBlocks, ConvertibleBlocksAreAccepted) {
    std::string why;
    EXPECT_FALSE(anth_refused(json::array({{{"type", "text"}, {"text", "hi"}}}), why));
    EXPECT_FALSE(anth_refused(
        json::array({{{"type", "image"},
                      {"source", {{"type", "base64"}, {"media_type", "image/png"}, {"data", "AA"}}}}}),
        why));
    EXPECT_FALSE(anth_refused(
        json::array({{{"type", "image"}, {"source", {{"type", "url"}, {"url", "https://x/y.png"}}}}}), why));
    EXPECT_FALSE(anth_refused(
        json::array({{{"type", "tool_use"}, {"id", "t1"}, {"name", "f"}, {"input", json::object()}}}), why));
    EXPECT_FALSE(
        anth_refused(json::array({{{"type", "tool_result"}, {"tool_use_id", "t1"}, {"content", "42"}}}),
                     why));
    EXPECT_FALSE(anth_refused(json::array({{{"type", "thinking"}, {"thinking", "hm"}}}), why));
    // Carries no input of its own; the converter ignoring it costs nothing.
    EXPECT_FALSE(anth_refused(json::array({{{"type", "redacted_thinking"}, {"data", "zz"}}}), why));
}

TEST(AnthropicContentBlocks, UnknownBlockIsRefusedAndNamed) {
    std::string why;
    ASSERT_TRUE(
        anth_refused(json::array({{{"type", "input_audio"}, {"input_audio", {{"data", "AA"}}}}}), why));
    EXPECT_NE(why.find("input_audio"), std::string::npos)
        << "the message must name the block it could not read";

    why.clear();
    EXPECT_TRUE(
        anth_refused(json::array({{{"type", "document"}, {"source", {{"type", "base64"}, {"data", "x"}}}}}),
                     why));
}

// The image branch drops one level further down: only `base64` and `url`
// sources convert, and any other source pushed nothing at all.
TEST(AnthropicContentBlocks, ImageWithAnUnreadableSourceIsRefused) {
    std::string why;
    ASSERT_TRUE(
        anth_refused(json::array({{{"type", "image"}, {"source", {{"type", "file"}, {"file_id", "f1"}}}}}),
                     why));
    EXPECT_NE(why.find("file"), std::string::npos);

    why.clear();
    EXPECT_TRUE(anth_refused(json::array({{{"type", "image"}}}), why));
}

// `tool_result.content` may itself be an array, and the converter reads only
// `text` and `image` there. Accepting `tool_result` wholesale left that array
// unguarded, and the cost is higher than a drop: an unreadable block leaves the
// tool body empty, so the model is told the tool returned nothing.
//
// Worse, the image half used to count the block rather than the conversion, so
// a `file` source injected "[1 image(s) ... follow]" into the prompt with no
// image following - the prompt asserting an input the model never received.
TEST(AnthropicContentBlocks, ToolResultInnerBlocksAreChecked) {
    std::string why;
    auto tr = [](const json& inner) {
        return json::array({{{"type", "tool_result"}, {"tool_use_id", "t1"}, {"content", inner}}});
    };

    // The two shapes the converter reads.
    EXPECT_FALSE(anth_refused(tr("42"), why)) << "a plain string result converts whole";
    EXPECT_FALSE(anth_refused(tr(json::array({{{"type", "text"}, {"text", "42"}}})), why));
    EXPECT_FALSE(anth_refused(
        tr(json::array({{{"type", "image"},
                         {"source", {{"type", "base64"}, {"media_type", "image/png"}, {"data", "AA"}}}}})),
        why));

    // A block the inner loop drops.
    why.clear();
    ASSERT_TRUE(
        anth_refused(tr(json::array({{{"type", "document"}, {"source", {{"type", "base64"}}}}})), why));
    EXPECT_NE(why.find("tool_result"), std::string::npos)
        << "the message must say where the unreadable block was";

    // The image-source hole, one level down.
    why.clear();
    ASSERT_TRUE(anth_refused(
        tr(json::array({{{"type", "image"}, {"source", {{"type", "file"}, {"file_id", "f1"}}}}})), why));
    EXPECT_NE(why.find("tool_result"), std::string::npos);
}

// The `system` field was outside the walk entirely: it keys on "messages".
// `flatten_system` reads a string, or an array from which it keeps `text`
// blocks, and returns "" for everything else - so the whole system prompt
// vanished and the model answered without its instructions. Measured on the
// model-less binary before this: a bare object, a number and an array carrying
// an image all reached the model lookup.
TEST(AnthropicContentBlocks, SystemFieldIsChecked) {
    std::string why;
    auto sys = [](const json& v) { return json{{"messages", json::array()}, {"system", v}}; };

    // What flatten_system can actually fold.
    EXPECT_FALSE(anthropic_unreadable_block(sys("you are terse"), why));
    EXPECT_FALSE(anthropic_unreadable_block(sys(json::array({{{"type", "text"}, {"text", "s"}}})), why));
    EXPECT_FALSE(anthropic_unreadable_block(sys(json(nullptr)), why));
    // cache_control rides on a text block and must stay accepted (#1046).
    EXPECT_FALSE(anthropic_unreadable_block(
        sys(json::array({{{"type", "text"}, {"text", "s"}, {"cache_control", {{"type", "ephemeral"}}}}})),
        why));

    // The shapes that folded to "".
    why.clear();
    ASSERT_TRUE(anthropic_unreadable_block(sys(json{{"type", "text"}, {"text", "s"}}), why))
        << "a bare object is not an array, so flatten_system returns \"\"";
    EXPECT_NE(why.find("system"), std::string::npos);
    EXPECT_TRUE(anthropic_unreadable_block(sys(json(42)), why));
    why.clear();
    ASSERT_TRUE(anthropic_unreadable_block(
        sys(json::array({{{"type", "image"}, {"source", {{"type", "base64"}, {"data", "AA"}}}}})), why));
    EXPECT_NE(why.find("image"), std::string::npos);
}

// A LEADING `role: "system"` message is folded through flatten_system too, and
// `leading_system++` consumes it whether or not anything survived. It therefore
// carries the system field's narrower allowlist. A system message after the
// first turn is NOT folded: it reaches push_user_turn and keeps its images, so
// refusing it would be a false refusal. The boundary is read off the converter.
TEST(AnthropicContentBlocks, LeadingSystemMessageUsesTheSystemAllowlist) {
    const json txt = {{"type", "text"}, {"text", "s"}};
    const json img = {{"type", "image"}, {"source", {{"type", "base64"}, {"data", "AA"}}}};
    const json user = {{"role", "user"}, {"content", "hi"}};
    std::string why;

    // Leading, and unfoldable: refused.
    ASSERT_TRUE(anthropic_unreadable_block(
        json{{"messages", json::array({{{"role", "system"}, {"content", json::array({txt, img})}}, user})}},
        why));

    // Leading and foldable, in both content shapes.
    why.clear();
    EXPECT_FALSE(anthropic_unreadable_block(
        json{{"messages", json::array({{{"role", "system"}, {"content", json::array({txt})}}, user})}}, why));
    EXPECT_FALSE(anthropic_unreadable_block(
        json{{"messages", json::array({{{"role", "system"}, {"content", "be terse"}}, user})}}, why));

    // Not leading: the image survives the transform, so it must not be refused.
    EXPECT_FALSE(anthropic_unreadable_block(
        json{{"messages", json::array({user, {{"role", "system"}, {"content", json::array({txt, img})}}})}},
        why))
        << "a later system message is not folded; refusing it would be a false refusal";
}

TEST(AnthropicContentBlocks, MissingTypeIsRefused) {
    std::string why;
    ASSERT_TRUE(anth_refused(json::array({{{"text", "hi"}}}), why));
    EXPECT_NE(why.find("missing type"), std::string::npos);
}

// The common shapes must be untouched: a plain string body, and no messages.
TEST(AnthropicContentBlocks, StringContentAndEmptyBodyAreUntouched) {
    std::string why;
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "plain text"}}})}};
    EXPECT_FALSE(anthropic_unreadable_block(body, why));
    EXPECT_FALSE(anthropic_unreadable_block(json::object(), why));
}

TEST(AnthropicContentBlocks, EveryMessageIsChecked) {
    json body = {
        {"messages",
         json::array({{{"role", "user"}, {"content", "fine"}},
                      {{"role", "assistant"}, {"content", json::array({{{"type", "text"}, {"text", "ok"}}})}},
                      {{"role", "user"}, {"content", json::array({{{"type", "video"}, {"video", "x"}}})}}})}};
    std::string why;
    EXPECT_TRUE(anthropic_unreadable_block(body, why));
}

// ---------------------------------------------------------------------------
// tool_choice contradictions. Distinct from a tool whose SCHEMA cannot be
// enforced — that legitimately degrades to prompt-hint choice, because `tools`
// offers capabilities rather than promising a shape. Naming a tool that is not
// there is not loose, it is self-contradictory, and answering it anyway had the
// model invent a call to a function the caller never described.
// ---------------------------------------------------------------------------

namespace {
json tool(const std::string& name) {
    return json{{"type", "function"},
                {"function", {{"name", name}, {"parameters", {{"type", "object"}}}}}};
}
bool tc_ok(const json& body) {
    httplib::Response res;
    return validate_tool_choice(body, res);
}
}  // namespace

TEST(ToolChoice, ConsistentRequestsAreAccepted) {
    EXPECT_TRUE(tc_ok(json::object())) << "no tool_choice at all";
    EXPECT_TRUE(tc_ok(json{{"tool_choice", "auto"}}));
    EXPECT_TRUE(tc_ok(json{{"tool_choice", "none"}})) << "satisfiable without tools";
    EXPECT_TRUE(tc_ok(json{{"tool_choice", "required"}, {"tools", json::array({tool("f")})}}));
    EXPECT_TRUE(tc_ok(json{{"tool_choice", {{"type", "function"}, {"function", {{"name", "f"}}}}},
                           {"tools", json::array({tool("g"), tool("f")})}}))
        << "named tool present, not necessarily first";
}

TEST(ToolChoice, RequiredWithoutToolsIsRejected) {
    httplib::Response res;
    ASSERT_FALSE(validate_tool_choice(json{{"tool_choice", "required"}}, res));
    EXPECT_EQ(res.status, 400);
    EXPECT_NE(res.body.find("required"), std::string::npos);

    // An empty array is the same contradiction as none at all.
    EXPECT_FALSE(tc_ok(json{{"tool_choice", "required"}, {"tools", json::array()}}));
}

TEST(ToolChoice, NamedToolMustExistAndIsNamedInTheError) {
    httplib::Response res;
    json body = {{"tool_choice", {{"type", "function"}, {"function", {{"name", "nonexistent"}}}}},
                 {"tools", json::array({tool("f")})}};
    ASSERT_FALSE(validate_tool_choice(body, res));
    EXPECT_NE(res.body.find("nonexistent"), std::string::npos)
        << "the error must name the function that was asked for";
}

TEST(ToolChoice, NamedToolWithNoToolsAtAllIsRejected) {
    EXPECT_FALSE(tc_ok(json{{"tool_choice", {{"type", "function"}, {"function", {{"name", "f"}}}}}}));
}

// Shapes this server does not interpret must pass through rather than 400 —
// rejecting an unfamiliar dialect would be worse than ignoring it.
TEST(ToolChoice, UnrecognisedShapesArePassedThrough) {
    EXPECT_TRUE(tc_ok(json{{"tool_choice", 42}}));
    EXPECT_TRUE(tc_ok(json{{"tool_choice", {{"type", "function"}}}})) << "no function object";
    EXPECT_TRUE(tc_ok(json{{"tool_choice", {{"type", "function"}, {"function", {{"name", ""}}}}}}))
        << "empty name carries no requirement";
}
