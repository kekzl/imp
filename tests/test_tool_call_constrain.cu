#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/json_schema.h"
#include "compute/schema_constrain.h"
#include "model/tokenizer.h"

#include <string>
#include <vector>

#include "test_cuda_skip.h"

// Constrained tool calling (#1002): the TOOL_CALL schema FSM, envelope literals
// (forced vs strict-optional), $defs hoisting, and parallel_tool_calls re-arm.
// The core schema-FSM grammar (patterns, $ref, enums, arrays, forced_text) lives
// in test_schema_constrain.cu; the any-JSON constrainer in test_json_constrain.cu.

namespace imp {
namespace {

// Run apply_mask over a vocab of `n` tokens and return which token ids survive.
static std::vector<bool> schema_allowed(SchemaConstrainer& sc, int n) {
    std::vector<float> h(n, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, n, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
    std::vector<bool> out(n);
    for (int i = 0; i < n; i++)
        out[i] = h[i] > -1e30f;
    return out;
}

// ---------------------------------------------------------------------------
// TOOL_CALL enforcement (#1002): envelope literals forced, "name" before
// "arguments", name enum restricted to the tool set, arguments bound to the
// CHOSEN tool's parameter schema, EOS forced after the close literal.
// ---------------------------------------------------------------------------
TEST(SchemaConstrainTest, ToolCallEnvelopeAndNameBinding) {
    SKIP_IF_NO_CUDA();
    // Single-char vocab over the emission corpus + a negative probe 'x'.
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
    std::string chars = "<>tol_ca\n{\"nme:usrgd,1}/x";
    for (char c : chars)
        toks.push_back(std::string(1, c));
    auto id = [&](char c) {
        for (size_t i = 3; i < toks.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    // Two tools: add / sub, both {"a" hm... use params with property "d" ...
    std::vector<std::pair<std::string, std::string>> tools = {
        {"add", R"({"type":"object","properties":{"d":{"type":"number"}},"required":["d"]})"},
        {"sub", R"({"type":"object","properties":{"u":{"type":"number"}},"required":["u"]})"},
    };
    auto schema = build_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.reset();

    // 1. Envelope first: '<' legal, '{' not.
    auto at_start = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_start[id('<')]) << "envelope open must be legal";
    EXPECT_FALSE(at_start[id('{')]) << "the body may not start before the envelope";

    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };
    feed("<tool_call>\n{\"");
    // 2. Key order: only "name" may open.
    auto at_key = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_key[id('n')]) << "'name' must be available first";
    EXPECT_FALSE(at_key[id('a')]) << "'arguments' may not precede 'name'";

    feed("name\":\"");
    // 3. Name enum: 'a'(add)/'s'(sub) legal, 'x' not.
    auto at_enum = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_enum[id('a')]);
    EXPECT_TRUE(at_enum[id('s')]);
    EXPECT_FALSE(at_enum[id('x')]) << "non-tool names must be masked";

    feed("add\",\"arguments\":{\"");
    // 4. Binding: only add's parameter 'd' is a legal key — not sub's 'u'.
    auto at_args = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_args[id('d')]) << "chosen tool's parameter must be legal";
    EXPECT_FALSE(at_args[id('u')]) << "the OTHER tool's parameter must be masked";

    feed("d\":1}}");
    // 5. Close literal forced.
    auto at_close = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_close[id('\n')]) << "close literal must be legal after the body";
    EXPECT_FALSE(at_close[id('{')]);

    feed("\n</tool_call>");
    // 6. Stack drained: EOS forced.
    auto at_done = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_done[2]) << "EOS must be allowed after the envelope closes";
    EXPECT_FALSE(at_done[id('<')]) << "no trailing text after the close literal";
}

TEST(SchemaConstrainTest, ToolCallBuilderRejectsUnenforceable) {
    // Free-form parameters (no properties) → decline.
    EXPECT_TRUE(build_tool_call_schema({{"t", R"({"type":"object"})"}}) == nullptr);
    // Unresolvable $ref → parse fails → decline (would enforce a wrong grammar).
    EXPECT_TRUE(build_tool_call_schema(
                    {{"t", R"({"type":"object","properties":{"i":{"$ref":"#/$defs/Missing"}}})"}}) ==
                nullptr);
    // Empty tool list → decline.
    EXPECT_TRUE(build_tool_call_schema({}) == nullptr);
    // Well-formed → builds.
    EXPECT_TRUE(build_tool_call_schema(
                    {{"t", R"({"type":"object","properties":{"i":{"type":"integer"}}})"}}) != nullptr);
    // $defs inside a parameter schema → now HOISTED (stage 2), builds. The
    // ToolCallHoistedDefsEnforced test proves the nested keys are enforced.
    EXPECT_TRUE(build_tool_call_schema(
                    {{"t", R"({"type":"object","properties":{"i":{"$ref":"#/$defs/I"}},)"
                           R"("required":["i"],"$defs":{"I":{"type":"object",)"
                           R"("properties":{"x":{"type":"integer"}},"required":["x"]}}})"}}) != nullptr);
}

// A tool whose parameter schema uses $defs+$ref (the pydantic/zod norm for any
// nested model) is now enforceable (#1002 stage 2): the nested defs are hoisted
// into the TOOL_CALL root under a "<tool>/<def>" namespace and the refs rewired,
// so the chosen tool's arguments constrain down through the nested model.
TEST(SchemaConstrainTest, ToolCallHoistedDefsEnforced) {
    SKIP_IF_NO_CUDA();
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
    std::string chars = "<>tol_ca\n{\"nme:d,rguspx1}/z";
    for (char c : chars)
        toks.push_back(std::string(1, c));
    auto id = [&](char c) {
        for (size_t i = 3; i < toks.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    // One tool "add" whose only argument "pt" is a nested $def-referenced model
    // Point = {"x": integer}. Two ref layers: root object → $ref Point → object.
    std::vector<std::pair<std::string, std::string>> tools = {
        {"add", R"({"type":"object","properties":{"pt":{"$ref":"#/$defs/Point"}},)"
                R"("required":["pt"],"$defs":{"Point":{"type":"object",)"
                R"("properties":{"x":{"type":"integer"}},"required":["x"]}}})"},
    };
    auto schema = build_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr) << "a tool with $defs must now build (hoisting)";
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.reset();

    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };
    feed("<tool_call>\n{\"name\":\"add\",\"arguments\":{\"");
    // Only the tool's own parameter "pt" may open the arguments object.
    auto at_arg_key = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_arg_key[id('p')]) << "tool parameter 'pt' must be legal";
    EXPECT_FALSE(at_arg_key[id('x')]) << "the nested key must not leak to the arguments level";

    feed("pt\":{\"");
    // Inside pt's value we are now in the hoisted Point model: only 'x' is a key.
    auto at_nested = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_nested[id('x')]) << "hoisted nested model's key must be enforced";
    EXPECT_FALSE(at_nested[id('z')]) << "a non-member nested key must be masked";
    EXPECT_FALSE(at_nested[id('p')]) << "the parent key must not be reachable inside the nested model";

    feed("x\":1}}}");  // close Point, arguments, and the tool-call body
    // Body complete → close literal forced.
    auto at_close = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_close[id('\n')]) << "close literal must be legal after the body";
    EXPECT_FALSE(at_close[id('{')]);

    feed("\n</tool_call>");
    auto at_done = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_done[2]) << "EOS must be allowed after the envelope closes";
}

// Strict OPTIONAL tool call (#1002, OpenAI strict:true + tool_choice auto): the
// envelope is NOT forced — the model may emit free text (mask off), but once it
// opens the tool tag the preamble gate hands off to the body FSM, which enforces
// the arguments, then forces the close literal + EOS.
TEST(SchemaConstrainTest, ToolCallStrictOptionalEnforced) {
    SKIP_IF_NO_CUDA();
    // Single-char body/close vocab, then the opener + a free-text token last so
    // the single-char id() lookup never collides with the multi-char tokens.
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
    std::string chars = "{\"name:d,rgus1}\n</tol_c>x";
    for (char c : chars)
        toks.push_back(std::string(1, c));
    const int opener_id = static_cast<int>(toks.size());
    toks.push_back("<tool_call>");
    const int free_id = static_cast<int>(toks.size());
    toks.push_back("FREE");
    auto id = [&](char c) {
        for (size_t i = 3; i < 3 + chars.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    std::vector<std::pair<std::string, std::string>> tools = {
        {"add", R"({"type":"object","properties":{"d":{"type":"number"}},"required":["d"]})"},
    };
    auto schema = build_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.set_strict_optional_envelope(true);
    sc.set_preamble_with_tools(/*close_token=*/-1, /*max_tokens=*/512, /*open_tokens=*/{opener_id},
                               /*close_tokens=*/{}, /*open_prefix=*/"<tool_call>",
                               /*close_suffix=*/"</tool_call>", /*thinking_open=*/false,
                               /*strict_tool=*/true);
    sc.reset();

    // 1. Free generation: the gate is ACTIVE, mask bypassed — a plain-text token
    // and even a non-structural char pass. The model is NOT forced to call.
    auto at_free = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_free[free_id]) << "free text must pass — no tool call is forced";
    EXPECT_TRUE(at_free[id('x')]) << "arbitrary chars pass during free generation";

    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };

    // 2. Model opens the tool tag → gate hands off to the body FSM.
    sc.update(opener_id);
    sc.update(id('\n'));  // the \n after <tool_call> (VALUE_START tolerates ws)
    auto at_body = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_body[id('{')]) << "the tool-call body must open with '{'";
    EXPECT_FALSE(at_body[free_id]) << "free text is masked once the body FSM engaged";
    EXPECT_FALSE(at_body[id('x')]) << "non-structural chars masked inside the body";

    // 3. Key order + name binding, exactly as the forced path.
    feed("{\"");
    auto at_key = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_key[id('n')]) << "'name' first";
    EXPECT_FALSE(at_key[id('a')]) << "'arguments' may not precede 'name'";

    feed("name\":\"add\",\"arguments\":{\"");
    auto at_args = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_args[id('d')]) << "chosen tool's parameter must be legal";

    feed("d\":1}}");
    // 4. Body complete → the close literal is FORCED (model emitted the open tag
    // freely, but the close is enforced so the tool call parses).
    auto at_close = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_close[id('\n')]) << "close literal '\\n</tool_call>' forced";
    EXPECT_FALSE(at_close[id('{')]);

    feed("\n</tool_call>");
    // 5. Envelope closed → EOS forced.
    auto at_done = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_done[2]) << "EOS forced after the tool call closes";
    EXPECT_FALSE(at_done[free_id]) << "no trailing free text after the close literal";
}

// parallel_tool_calls (#1002): in strict optional mode with allow_parallel, the
// gate RE-ARMS after each tool-call body instead of forcing EOS — the model may
// emit a second `<tool_call>` (fresh body FSM) or stop.
TEST(SchemaConstrainTest, ToolCallStrictParallelReArms) {
    SKIP_IF_NO_CUDA();
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
    std::string chars = "{\"name:d,rgus1}\n</tol_c>x";
    for (char c : chars)
        toks.push_back(std::string(1, c));
    const int opener_id = static_cast<int>(toks.size());
    toks.push_back("<tool_call>");
    const int free_id = static_cast<int>(toks.size());
    toks.push_back("FREE");
    auto id = [&](char c) {
        for (size_t i = 3; i < 3 + chars.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    std::vector<std::pair<std::string, std::string>> tools = {
        {"add", R"({"type":"object","properties":{"d":{"type":"number"}},"required":["d"]})"},
    };
    auto schema = build_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.set_strict_optional_envelope(true);
    sc.set_allow_parallel(true);
    sc.set_preamble_with_tools(-1, 512, {opener_id}, {}, "<tool_call>", "</tool_call>",
                               /*thinking_open=*/false, /*strict_tool=*/true);
    sc.reset();

    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };

    // First call: open → body → close literal.
    sc.update(opener_id);
    feed("\n{\"name\":\"add\",\"arguments\":{\"d\":1}}\n</tool_call>");

    // Re-armed: the gate is ACTIVE again (mask off) — NOT a forced EOS. Free
    // text, EOS, and another opener are all legal now.
    auto at_between = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_between[free_id]) << "after a call the model may emit more (parallel)";
    EXPECT_TRUE(at_between[2]) << "EOS is also legal — the model may stop";

    // Second call opens → the body FSM engages a fresh frame.
    sc.update(opener_id);
    sc.update(id('\n'));
    auto at_body2 = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_body2[id('{')]) << "second tool-call body must open with '{'";
    EXPECT_FALSE(at_body2[id('x')]) << "second body is FSM-constrained too";
    EXPECT_FALSE(at_body2[free_id]) << "free text masked inside the second body";

    feed("{\"name\":\"add\",\"arguments\":{\"d\":1}}\n</tool_call>");
    // Re-armed once more → EOS still legal to finish.
    auto at_end = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_end[2]) << "EOS legal after the second call";
}

// Llama3 forced tool call (#1002): `<function=NAME>{args}</function>` — the body
// is the bare arguments object, so the constraint root is the parameter schema
// directly (per-tool envelope), not a TOOL_CALL wrapper.
TEST(SchemaConstrainTest, Llama3BareArgsForcedEnvelope) {
    SKIP_IF_NO_CUDA();
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
    std::string chars = "<functio=ad>/{\":1}x";
    for (char c : chars)
        toks.push_back(std::string(1, c));
    auto id = [&](char c) {
        for (size_t i = 3; i < toks.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    // Bare parameter schema (NOT a TOOL_CALL wrapper).
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"d":{"type":"number"}},"required":["d"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<function=add>", "</function>");  // forced (no strict flag)
    sc.reset();

    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };

    // 1. Envelope open forced: '<' legal, the body '{' is not yet.
    auto at_start = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_start[id('<')]) << "the <function=...> envelope must open first";
    EXPECT_FALSE(at_start[id('{')]) << "the args body may not start before the envelope";

    feed("<function=add>{\"");
    // 2. Inside the bare args object: the parameter 'd' is a legal key, 'x' not.
    auto at_key = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_key[id('d')]) << "the tool parameter must be legal";
    EXPECT_FALSE(at_key[id('x')]) << "a non-parameter key must be masked";

    feed("d\":1}");
    // 3. Body complete → the close literal '</function>' is forced.
    auto at_close = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_close[id('<')]) << "close literal must be legal after the body";
    EXPECT_FALSE(at_close[id('{')]);

    feed("</function>");
    auto at_done = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_done[2]) << "EOS forced after the envelope closes";
}

// ---------------------------------------------------------------------------
// Qwen-Coder XML tool calls (#1002 follow-up): the body is
//   <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>
// inside the ChatML <tool_call> envelope. Parameter VALUES are raw text
// (multi-line, unescaped) delimited by "\n</parameter>" — the reason the JSON
// body FSM must never engage on this dialect (it masked raw newlines and
// produced single-line code, see the PR).
// ---------------------------------------------------------------------------

static std::vector<std::string> xml_test_vocab(std::string& chars) {
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
    chars = "<>=/\ntol_cafunirdepmshvx{1";
    for (char c : chars)
        toks.push_back(std::string(1, c));
    return toks;
}

TEST(SchemaConstrainTest, XmlToolCallForcedGrammar) {
    SKIP_IF_NO_CUDA();
    std::string chars;
    auto toks = xml_test_vocab(chars);
    auto id = [&](char c) {
        for (size_t i = 3; i < toks.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    // "read" has a required param (path) and an optional one (limit); "sum"
    // exists to prove name-enum narrowing and cross-tool key isolation.
    std::vector<std::pair<std::string, std::string>> tools = {
        {"read", R"({"type":"object","properties":{"path":{"type":"string"},)"
                 R"("limit":{"type":"integer"}},"required":["path"]})"},
        {"sum", R"({"type":"object","properties":{"vals":{"type":"string"}},"required":["vals"]})"},
    };
    auto schema = build_xml_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.reset();

    auto allowed = [&] { return schema_allowed(sc, static_cast<int>(toks.size())); };
    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };

    // 1. Envelope forced; no JSON body anywhere.
    auto at_start = allowed();
    EXPECT_TRUE(at_start[id('<')]) << "envelope open must be legal";
    EXPECT_FALSE(at_start[id('{')]) << "JSON body must not open on the XML dialect";

    feed("<tool_call>\n");
    auto at_body = allowed();
    EXPECT_TRUE(at_body[id('<')]) << "the <function= tag must open the body";
    EXPECT_FALSE(at_body[id('{')]) << "no JSON body inside the XML envelope";

    // 2. Name enum: 'r'(read) / 's'(sum) legal, 'x' not; '>' only on a full name.
    feed("<function=");
    auto at_name = allowed();
    EXPECT_TRUE(at_name[id('r')]);
    EXPECT_TRUE(at_name[id('s')]);
    EXPECT_FALSE(at_name[id('x')]) << "non-tool names must be masked";
    EXPECT_FALSE(at_name[id('>')]) << "the name tag may not close on an empty name";

    feed("rea");
    auto at_partial = allowed();
    EXPECT_TRUE(at_partial[id('d')]);
    EXPECT_FALSE(at_partial[id('>')]) << "'rea' is not a complete tool name";

    feed("d");
    auto at_full = allowed();
    EXPECT_TRUE(at_full[id('>')]) << "'read' is complete — the tag may close";
    EXPECT_FALSE(at_full[id('x')]) << "'readx' is not a tool-name prefix";

    // 3. Params: "\n<parameter=" is mandatory while required params are missing.
    feed(">");
    auto at_params = allowed();
    EXPECT_TRUE(at_params[id('\n')]);
    EXPECT_FALSE(at_params[id('<')]) << "param/close tags start on a fresh line";

    feed("\n<");
    auto at_tag = allowed();
    EXPECT_TRUE(at_tag[id('p')]) << "<parameter= must be available";
    EXPECT_FALSE(at_tag[id('/')]) << "</function> is illegal while required 'path' is missing";

    // 4. Key enum: the chosen tool's keys only.
    feed("parameter=");
    auto at_key = allowed();
    EXPECT_TRUE(at_key[id('p')]) << "'path' must be legal";
    EXPECT_TRUE(at_key[id('l')]) << "'limit' must be legal";
    EXPECT_FALSE(at_key[id('v')]) << "the OTHER tool's parameter must be masked";

    feed("path");
    auto at_keyend = allowed();
    EXPECT_TRUE(at_keyend[id('>')]);

    // 5. Raw value: forced leading newline, then ANY text — including raw
    // newlines (the whole point of the XML dialect) — until \n</parameter>.
    feed(">");
    auto at_vopen = allowed();
    EXPECT_TRUE(at_vopen[id('\n')]) << "the value region opens with a forced newline";
    EXPECT_FALSE(at_vopen[id('x')]);

    feed("\n");
    auto at_value = allowed();
    EXPECT_TRUE(at_value[id('x')]) << "raw value content must be legal";
    EXPECT_TRUE(at_value[id('\n')]) << "raw newlines must be legal inside the value";
    EXPECT_TRUE(at_value[id('{')]) << "braces are plain text inside the value";
    EXPECT_FALSE(at_value[2]) << "EOS must be masked mid-value (the call must complete)";

    feed("x/h\n1");
    EXPECT_TRUE(allowed()[id('x')]) << "still inside the raw value";

    // 6. Delimiter closes the param; the next tag needs its fresh line.
    feed("\n</parameter>");
    auto at_after = allowed();
    EXPECT_TRUE(at_after[id('\n')]);
    EXPECT_FALSE(at_after[id('x')]) << "value text may not continue past the delimiter";

    // 7. Required satisfied → both another param and the function close are legal.
    feed("\n<");
    auto at_choice = allowed();
    EXPECT_TRUE(at_choice[id('p')]) << "'limit' is still available";
    EXPECT_TRUE(at_choice[id('/')]) << "required params emitted — the body may close";

    // 8. Close: </function> then the forced envelope close, then EOS.
    feed("/function>");
    auto at_close = allowed();
    EXPECT_TRUE(at_close[id('\n')]) << "envelope close literal must follow";
    EXPECT_FALSE(at_close[id('<')]);

    feed("\n</tool_call>");
    auto at_done = allowed();
    EXPECT_TRUE(at_done[2]) << "EOS forced after the envelope closes";
    EXPECT_FALSE(at_done[id('<')]) << "no trailing text after the close literal";
}

// The delimiter matcher must survive value text that CONTAINS partial
// delimiters ("\n</param" + more text) and still close on the real one later.
TEST(SchemaConstrainTest, XmlToolCallRawValueDelimiterOverlap) {
    SKIP_IF_NO_CUDA();
    std::string chars;
    auto toks = xml_test_vocab(chars);
    auto id = [&](char c) {
        for (size_t i = 3; i < toks.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    std::vector<std::pair<std::string, std::string>> tools = {
        {"sum", R"({"type":"object","properties":{"vals":{"type":"string"}},"required":["vals"]})"},
    };
    auto schema = build_xml_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.reset();

    auto allowed = [&] { return schema_allowed(sc, static_cast<int>(toks.size())); };
    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };

    feed("<tool_call>\n<function=sum>\n<parameter=vals>\n");
    // A partial delimiter that dies ("\n</param" + 'x') keeps the value open.
    feed("a\n</param");
    auto mid = allowed();
    EXPECT_TRUE(mid[id('x')]) << "a dead partial delimiter is plain value text";
    feed("x");
    EXPECT_TRUE(allowed()[id('x')]) << "still inside the raw value after the dead match";

    // An extra newline directly before the delimiter must re-anchor the match.
    feed("\n\n</parameter>");
    auto after = allowed();
    EXPECT_TRUE(after[id('\n')]) << "the delimiter must close the param";
    EXPECT_FALSE(after[id('x')]) << "no value text after the delimiter";

    feed("\n</function>\n</tool_call>");
    EXPECT_TRUE(allowed()[2]) << "EOS forced after the envelope closes";
}

// An empty value renders as <parameter=k>\n\n</parameter> (forced newline, no
// content, delimiter) — the FSM must accept the immediate close.
TEST(SchemaConstrainTest, XmlToolCallEmptyValue) {
    SKIP_IF_NO_CUDA();
    std::string chars;
    auto toks = xml_test_vocab(chars);
    auto id = [&](char c) {
        for (size_t i = 3; i < toks.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    std::vector<std::pair<std::string, std::string>> tools = {
        {"sum", R"({"type":"object","properties":{"vals":{"type":"string"}},"required":["vals"]})"},
    };
    auto schema = build_xml_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.reset();

    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };
    feed("<tool_call>\n<function=sum>\n<parameter=vals>\n\n</parameter>\n</function>\n</tool_call>");
    auto at_done = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_done[2]) << "an empty value must be a legal, closable call";
}

// Strict OPTIONAL on the XML dialect: free text passes until the model opens
// <tool_call>; then the XML body FSM engages (never the JSON body).
TEST(SchemaConstrainTest, XmlToolCallStrictOptionalEnforced) {
    SKIP_IF_NO_CUDA();
    std::string chars;
    auto toks = xml_test_vocab(chars);
    const int opener_id = static_cast<int>(toks.size());
    toks.push_back("<tool_call>");
    const int free_id = static_cast<int>(toks.size());
    toks.push_back("FREE");
    auto id = [&](char c) {
        for (size_t i = 3; i < 3 + chars.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    std::vector<std::pair<std::string, std::string>> tools = {
        {"sum", R"({"type":"object","properties":{"vals":{"type":"string"}},"required":["vals"]})"},
    };
    auto schema = build_xml_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.set_strict_optional_envelope(true);
    sc.set_preamble_with_tools(-1, 512, {opener_id}, {}, "<tool_call>", "</tool_call>",
                               /*thinking_open=*/false, /*strict_tool=*/true);
    sc.reset();

    auto allowed = [&] { return schema_allowed(sc, static_cast<int>(toks.size())); };
    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };

    // Free generation: nothing forced.
    auto at_free = allowed();
    EXPECT_TRUE(at_free[free_id]) << "free text must pass — no tool call is forced";

    // Opener → the XML body FSM engages.
    sc.update(opener_id);
    sc.update(id('\n'));
    auto at_body = allowed();
    EXPECT_TRUE(at_body[id('<')]) << "the <function= tag must open the body";
    EXPECT_FALSE(at_body[id('{')]) << "the JSON body must NOT engage on the XML dialect";
    EXPECT_FALSE(at_body[free_id]) << "free text is masked once the body FSM engaged";

    feed("<function=sum>\n<parameter=vals>\nx\n</parameter>\n</function>");
    auto at_close = allowed();
    EXPECT_TRUE(at_close[id('\n')]) << "close literal '\\n</tool_call>' forced";
    EXPECT_FALSE(at_close[free_id]);

    feed("\n</tool_call>");
    auto at_done = allowed();
    EXPECT_TRUE(at_done[2]) << "EOS forced after the tool call closes";
    EXPECT_FALSE(at_done[free_id]) << "no trailing free text (parallel off)";
}

// parallel_tool_calls on the XML dialect: the gate re-arms after each call.
TEST(SchemaConstrainTest, XmlToolCallStrictParallelReArms) {
    SKIP_IF_NO_CUDA();
    std::string chars;
    auto toks = xml_test_vocab(chars);
    const int opener_id = static_cast<int>(toks.size());
    toks.push_back("<tool_call>");
    const int free_id = static_cast<int>(toks.size());
    toks.push_back("FREE");
    auto id = [&](char c) {
        for (size_t i = 3; i < 3 + chars.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    std::vector<std::pair<std::string, std::string>> tools = {
        {"sum", R"({"type":"object","properties":{"vals":{"type":"string"}},"required":["vals"]})"},
    };
    auto schema = build_xml_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.set_strict_optional_envelope(true);
    sc.set_allow_parallel(true);
    sc.set_preamble_with_tools(-1, 512, {opener_id}, {}, "<tool_call>", "</tool_call>",
                               /*thinking_open=*/false, /*strict_tool=*/true);
    sc.reset();

    auto allowed = [&] { return schema_allowed(sc, static_cast<int>(toks.size())); };
    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };

    sc.update(opener_id);
    feed("\n<function=sum>\n<parameter=vals>\nx\n</parameter>\n</function>\n</tool_call>");

    auto at_between = allowed();
    EXPECT_TRUE(at_between[free_id]) << "after a call the model may emit more (parallel)";
    EXPECT_TRUE(at_between[2]) << "EOS is also legal — the model may stop";

    sc.update(opener_id);
    sc.update(id('\n'));
    auto at_body2 = allowed();
    EXPECT_TRUE(at_body2[id('<')]) << "second XML body must open with <function=";
    EXPECT_FALSE(at_body2[free_id]) << "free text masked inside the second body";

    feed("<function=sum>\n<parameter=vals>\nx\n</parameter>\n</function>\n</tool_call>");
    EXPECT_TRUE(allowed()[2]) << "EOS legal after the second call";
}

TEST(SchemaConstrainTest, XmlBuilderGates) {
    // Same enforceability gates as the JSON builder: free-form params,
    // unresolvable $refs and an empty tool list decline to the prompt hint.
    EXPECT_TRUE(build_xml_tool_call_schema({}) == nullptr);
    EXPECT_TRUE(build_xml_tool_call_schema({{"t", R"({"type":"object"})"}}) == nullptr);
    EXPECT_TRUE(build_xml_tool_call_schema(
                    {{"t", R"({"type":"object","properties":{"i":{"$ref":"#/$defs/Missing"}}})"}}) ==
                nullptr);
    EXPECT_TRUE(build_xml_tool_call_schema(
                    {{"t", R"({"type":"object","properties":{"i":{"type":"integer"}}})"}}) != nullptr);
    // Per-tool $defs parse and hoist exactly like the JSON path.
    EXPECT_TRUE(build_xml_tool_call_schema(
                    {{"t", R"({"type":"object","properties":{"i":{"$ref":"#/$defs/I"}},)"
                           R"("required":["i"],"$defs":{"I":{"type":"object",)"
                           R"("properties":{"x":{"type":"integer"}},"required":["x"]}}})"}}) != nullptr);
}

}  // namespace
}  // namespace imp
