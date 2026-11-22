#pragma once

// OpenAI wire: pictures inside a role "tool" message. A tool turn renders as text, so its image_url
// parts used to be serialized into the prompt (a 1280 x 800 screenshot became 35313 tokens of
// base64). They move to the user message behind the run of tool messages, as the Anthropic
// conversion does for tool_result blocks (anthropic.cpp push_user_turn).

#include <nlohmann/json.hpp>

#include <string>

namespace imp::server {

// messages with every tool message's image_url parts moved: the tool message keeps its text parts
// and a note, the pictures go in front of the next user message (a new one when another role
// follows, or nothing does). moved (optional): pictures moved.
inline nlohmann::ordered_json rehome_tool_images(const nlohmann::ordered_json& messages,
                                                 int* moved = nullptr) {
    using json = nlohmann::ordered_json;
    json out = json::array(), pending = json::array();
    int n_moved = 0;
    const auto flush = [&](json* user) {
        if (pending.empty())
            return;
        json parts = std::move(pending);
        pending = json::array();
        if (user != nullptr) {
            const json& c = (*user)["content"];
            if (c.is_string())
                parts.push_back({{"type", "text"}, {"text", c.get<std::string>()}});
            else if (c.is_array())
                for (const auto& p : c)
                    parts.push_back(p);
            (*user)["content"] = std::move(parts);
        } else {
            out.push_back({{"role", "user"}, {"content", std::move(parts)}});
        }
    };
    if (!messages.is_array())
        return messages;
    for (const auto& msg : messages) {
        const std::string role = msg.is_object() ? msg.value("role", "") : "";
        if (role != "tool") {
            json copy = msg;
            flush(role == "user" ? &copy : nullptr);
            out.push_back(std::move(copy));
            continue;
        }
        json copy = msg;
        const auto it = msg.find("content");
        int n = 0;
        if (it != msg.end() && it->is_array())
            for (const auto& p : *it)
                n += (p.is_object() && p.value("type", "") == "image_url") ? 1 : 0;
        if (n > 0) {
            std::string text;
            for (const auto& p : *it) {
                if (!p.is_object())
                    continue;
                if (p.value("type", "") == "image_url") {
                    pending.push_back(p);
                } else if (p.value("type", "") == "text" && p.contains("text") && p["text"].is_string()) {
                    text += (text.empty() ? "" : "\n") + p["text"].get<std::string>();
                }
            }
            text += (text.empty() ? "" : "\n") + std::string("[") + std::to_string(n) +
                    " image(s) from this tool result follow in the next user message]";
            copy["content"] = std::move(text);
            n_moved += n;
        }
        out.push_back(std::move(copy));
    }
    flush(nullptr);
    if (moved != nullptr)
        *moved = n_moved;
    return out;
}

}  // namespace imp::server
