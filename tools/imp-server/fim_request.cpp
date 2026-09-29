#include "fim_request.h"

#include "model/tokenizer.h"

#include <utility>

using nlohmann::json;

namespace {

// Absent or null = "", any other non-string is an error.
bool optional_string(const json& body, const char* key, std::string& out) {
    if (!body.contains(key) || body[key].is_null())
        return true;
    if (!body[key].is_string())
        return false;
    out = body[key].get<std::string>();
    return true;
}

std::string parse_extra(const json& body, std::vector<imp::FimExtraChunk>& out) {
    if (!body.contains("input_extra") || body["input_extra"].is_null())
        return "";
    const json& extra = body["input_extra"];
    if (!extra.is_array())
        return "\"input_extra\" must be an array of {\"filename\", \"text\"} objects";
    for (const auto& chunk : extra) {
        imp::FimExtraChunk c;
        if (!chunk.is_object() || !chunk.contains("text") || !chunk["text"].is_string() ||
            !optional_string(chunk, "filename", c.filename))
            return "each \"input_extra\" entry must be {\"filename\": string, \"text\": string}";
        c.text = chunk["text"].get<std::string>();
        out.push_back(std::move(c));
    }
    return "";
}

}  // namespace

size_t FimRequest::bytes() const {
    size_t n = input.prefix.size() + input.suffix.size() + input.prompt.size();
    for (const auto& c : input.extra)
        n += c.filename.size() + c.text.size();
    return n;
}

std::string parse_infill_fields(json& body, FimRequest& out) {
    const std::pair<const char*, std::string*> fields[] = {{"input_prefix", &out.input.prefix},
                                                           {"input_suffix", &out.input.suffix},
                                                           {"prompt", &out.input.prompt}};
    for (const auto& [key, dst] : fields) {
        if (!optional_string(body, key, *dst))
            return std::string("\"") + key + "\" must be a string";
    }
    if (std::string err = parse_extra(body, out.input.extra); !err.empty())
        return err;
    // llama.cpp clients (llama.vim, llama.vscode) send n_predict; max_tokens wins when both are set.
    if (body.contains("n_predict") && body["n_predict"].is_number_integer() && !body.contains("max_tokens") &&
        body["n_predict"].get<int64_t>() > 0)
        body["max_tokens"] = body["n_predict"];
    out.active = true;
    return "";
}

std::string parse_completion_suffix(const json& body, const std::string& prompt, bool token_prompt,
                                    FimRequest& out) {
    std::string suffix;
    if (!optional_string(body, "suffix", suffix))
        return "\"suffix\" must be a string";
    if (suffix.empty())
        return "";
    if (token_prompt)
        return "\"suffix\" needs a text prompt; token-id prompts cannot be split into prefix and suffix";
    out.input.prefix = prompt;
    out.input.suffix = std::move(suffix);
    out.active = true;
    return "";
}

std::string build_fim_request_tokens(const FimRequest& fim, const imp::Tokenizer& tok,
                                     std::vector<int32_t>& tokens, std::vector<std::string>& stops) {
    const imp::FimTokens ft = imp::find_fim_tokens(tok);
    if (!ft.supported())
        return "fill-in-the-middle is not supported by this model: its tokenizer has no FIM "
               "prefix/suffix/middle "
               "tokens";
    tokens = imp::build_fim_prompt(tok, ft, fim.input);
    for (auto& s : imp::fim_stop_texts(tok, ft))
        stops.push_back(std::move(s));
    return "";
}
