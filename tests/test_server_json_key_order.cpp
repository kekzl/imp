// Request JSON keeps the client's key order (nlohmann::ordered_json), as vLLM's Python dicts do:
// tool schemas and past tool-call arguments reach the chat template in the order the client sent.

#include "anthropic.h"
#include "responses.h"

#include <gtest/gtest.h>

#include <string>

using json = imp_server::anthropic::json;

namespace {

// path, offset, limit: not alphabetical, so a sorted parse shows up as limit, offset, path.
constexpr const char* kSchema =
    R"({"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer"},)"
    R"("limit":{"type":"integer"}},"required":["path"]})";
constexpr const char* kOrdered = R"("properties":{"path":{"type":"string"},"offset":{"type":"integer"},)"
                                 R"("limit":{"type":"integer"}})";

}  // namespace

TEST(ServerJsonKeyOrder, ParseDumpKeepsClientOrder) {
    const json schema = json::parse(kSchema);
    EXPECT_EQ(schema.dump(), kSchema);
}

TEST(ServerJsonKeyOrder, AnthropicToolSchemaAndToolUseInputKeepOrder) {
    const json anth = json::parse(
        std::string(R"({"model":"m","max_tokens":16,"tools":[{"name":"read_file",)") +
        R"("description":"Read.","input_schema":)" + kSchema +
        R"(}],"messages":[{"role":"user","content":"go"},{"role":"assistant","content":[)"
        R"({"type":"tool_use","id":"toolu_1","name":"read_file",)"
        R"("input":{"path":"a.py","offset":200,"limit":120}}]},)"
        R"({"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1",)"
        R"("content":"ok"}]}]})");
    const json oai = imp_server::anthropic::anthropic_to_openai_body(anth);
    EXPECT_NE(oai["tools"][0]["function"]["parameters"].dump().find(kOrdered), std::string::npos)
        << oai["tools"][0]["function"]["parameters"].dump();
    const std::string args = oai["messages"][1]["tool_calls"][0]["function"]["arguments"].get<std::string>();
    EXPECT_EQ(args, R"({"path":"a.py","offset":200,"limit":120})");
}

TEST(ServerJsonKeyOrder, ResponsesToolSchemaKeepsOrder) {
    const json rsp = json::parse(std::string(R"({"model":"m","input":"go","tools":[{"type":"function",)") +
                                 R"("name":"read_file","description":"Read.","parameters":)" + kSchema +
                                 "}]}");
    const json oai = imp_server::responses::responses_to_openai_body(rsp);
    EXPECT_NE(oai["tools"][0]["function"]["parameters"].dump().find(kOrdered), std::string::npos)
        << oai["tools"][0]["function"]["parameters"].dump();
}
