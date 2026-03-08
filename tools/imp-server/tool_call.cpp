#include "tool_call.h"

std::string build_tool_prompt(imp::ChatTemplateFamily family,
                              const json& tools,
                              const json& tool_choice) {
    if (tools.empty()) return "";

    // tool_choice "none" means no tool injection
    if (tool_choice.is_string() && tool_choice.get<std::string>() == "none")
        return "";

    std::string prompt;

    if (family == imp::ChatTemplateFamily::LLAMA3) {
        // Llama3 function calling format
        prompt = "\n\nYou have access to the following functions:\n\n";
        for (const auto& tool : tools) {
            if (!tool.contains("function")) continue;
            const auto& fn = tool["function"];
            json fn_desc = {
                {"name", fn.value("name", "")},
                {"description", fn.value("description", "")},
                {"parameters", fn.value("parameters", json::object())}
            };
            prompt += fn_desc.dump() + "\n\n";
        }
        prompt += "For each function call, return a JSON object within <function=function_name> tags:\n"
                  "<function=function_name>{\"param\": \"value\"}</function>\n\n"
                  "If no function call is needed, respond normally without any function tags.";
    } else {
        // ChatML (Qwen3, Hermes) and all other families — use <tool_call> format
        prompt = "\n\n# Tools\n\n"
                 "You may call one or more functions to assist with the user query.\n\n"
                 "<tools>\n" + tools.dump() + "\n</tools>\n\n"
                 "For each function call, return a JSON object within <tool_call></tool_call> XML tags:\n"
                 "<tool_call>\n"
                 "{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}\n"
                 "</tool_call>\n\n"
                 "If no function call is needed, respond normally without any tool_call tags.";
    }

    // Add constraints based on tool_choice
    if (tool_choice.is_string()) {
        std::string choice = tool_choice.get<std::string>();
        if (choice == "required") {
            prompt += "\n\nYou MUST call at least one tool.";
        }
    } else if (tool_choice.is_object() && tool_choice.contains("function")) {
        std::string fn_name = tool_choice["function"].value("name", "");
        if (!fn_name.empty()) {
            prompt += "\n\nYou MUST call the " + fn_name + " tool.";
        }
    }

    return prompt;
}

std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls_chatml(const std::string& text, std::atomic<int>& next_tool_call_id) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    size_t pos = 0;
    size_t first_tag = text.find("<tool_call>");
    if (first_tag == std::string::npos) {
        return {text, {}};
    }

    // Content is everything before the first <tool_call>
    content = text.substr(0, first_tag);
    // Trim trailing whitespace
    auto last = content.find_last_not_of("\n\r\t ");
    if (last != std::string::npos) content = content.substr(0, last + 1);
    else content.clear();

    pos = first_tag;
    while (pos < text.size()) {
        size_t start = text.find("<tool_call>", pos);
        if (start == std::string::npos) break;
        start += 11; // skip "<tool_call>"

        size_t end = text.find("</tool_call>", start);
        if (end == std::string::npos) break; // incomplete tag

        std::string body = text.substr(start, end - start);
        // Trim whitespace
        auto bs = body.find_first_not_of("\n\r\t ");
        auto be = body.find_last_not_of("\n\r\t ");
        if (bs != std::string::npos && be != std::string::npos)
            body = body.substr(bs, be - bs + 1);

        // Parse JSON
        try {
            json j = json::parse(body);
            ParsedToolCall tc;
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            tc.name = j.value("name", "");
            if (j.contains("arguments")) {
                tc.arguments = j["arguments"].dump();
            } else {
                // Some models put params at top level minus "name"
                json args = j;
                args.erase("name");
                tc.arguments = args.dump();
            }
            if (!tc.name.empty()) {
                calls.push_back(std::move(tc));
            }
        } catch (...) {
            // Malformed JSON — skip
        }

        pos = end + 12; // skip "</tool_call>"
    }

    return {content, calls};
}

std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls_llama3(const std::string& text, std::atomic<int>& next_tool_call_id) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    size_t first_tag = text.find("<function=");
    if (first_tag == std::string::npos) {
        return {text, {}};
    }

    content = text.substr(0, first_tag);
    auto last = content.find_last_not_of("\n\r\t ");
    if (last != std::string::npos) content = content.substr(0, last + 1);
    else content.clear();

    size_t pos = first_tag;
    while (pos < text.size()) {
        size_t start = text.find("<function=", pos);
        if (start == std::string::npos) break;
        start += 10; // skip "<function="

        size_t name_end = text.find('>', start);
        if (name_end == std::string::npos) break;

        std::string name = text.substr(start, name_end - start);

        size_t body_start = name_end + 1;
        size_t end = text.find("</function>", body_start);
        if (end == std::string::npos) break;

        std::string body = text.substr(body_start, end - body_start);
        auto bs = body.find_first_not_of("\n\r\t ");
        auto be = body.find_last_not_of("\n\r\t ");
        if (bs != std::string::npos && be != std::string::npos)
            body = body.substr(bs, be - bs + 1);

        try {
            // Validate it's valid JSON
            json j = json::parse(body);
            ParsedToolCall tc;
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            tc.name = name;
            tc.arguments = j.dump();
            calls.push_back(std::move(tc));
        } catch (...) {
            // Malformed JSON — skip
        }

        pos = end + 11; // skip "</function>"
    }

    return {content, calls};
}

std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls(imp::ChatTemplateFamily family, const std::string& text,
                 std::atomic<int>& next_tool_call_id) {
    if (family == imp::ChatTemplateFamily::LLAMA3)
        return parse_tool_calls_llama3(text, next_tool_call_id);
    return parse_tool_calls_chatml(text, next_tool_call_id);
}

std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family,
                                         const json& tool_calls,
                                         const std::string& content) {
    std::string result;
    if (!content.empty() && content != "null") {
        result = content;
    }

    for (const auto& tc : tool_calls) {
        if (!tc.contains("function")) continue;
        std::string name = tc["function"].value("name", "");
        std::string args = tc["function"].value("arguments", "{}");

        if (family == imp::ChatTemplateFamily::LLAMA3) {
            result += "\n<function=" + name + ">" + args + "</function>";
        } else {
            // ChatML format
            json call_obj = {{"name", name}, {"arguments", json::parse(args, nullptr, false)}};
            if (call_obj["arguments"].is_discarded()) call_obj["arguments"] = args;
            result += "\n<tool_call>\n" + call_obj.dump() + "\n</tool_call>";
        }
    }

    return result;
}

std::string format_tool_response(imp::ChatTemplateFamily family,
                                 const json& msg) {
    std::string content = msg.value("content", "");
    std::string tool_call_id = msg.value("tool_call_id", "");

    if (family == imp::ChatTemplateFamily::LLAMA3) {
        return content;
    }
    // ChatML: wrap in <tool_response> tags
    return "<tool_response>\n" + content + "\n</tool_response>";
}
