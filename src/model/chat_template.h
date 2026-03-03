#pragma once

#include "model/tokenizer.h"
#include "model/model_arch.h"
#include <string>
#include <vector>
#include <cstdint>

namespace imp {

enum class ChatTemplateFamily {
    RAW,        // No template — pass raw text
    CHATML,     // <|im_start|>...<|im_end|> (Qwen3, etc.)
    LLAMA2,     // [INST]...[/INST] (Llama 2, Mistral)
    LLAMA3,     // <|start_header_id|>...<|end_header_id|>...<|eot_id|>
    NEMOTRON,   // <extra_id_0>System\n...<extra_id_1>\n<extra_id_0>User\n...
    GEMMA,      // <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
    DEEPSEEK_R1,// <｜User｜>...<｜Assistant｜>...<｜end▁of▁sentence｜>
};

const char* chat_template_family_name(ChatTemplateFamily family);

struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

class ChatTemplate {
public:
    ChatTemplate() = default;

    // Detect family from GGUF Jinja2 template string via substring matching
    static ChatTemplateFamily detect_family(const std::string& jinja2_str);

    // Parse family name from CLI string (e.g. "chatml", "llama3", "none")
    static ChatTemplateFamily parse_family(const std::string& name);

    // Default template family for a given model architecture (fallback)
    static ChatTemplateFamily default_family_for_arch(ModelArch arch);

    // Initialize: resolve special token IDs via tokenizer
    bool init(ChatTemplateFamily family, const Tokenizer& tokenizer);

    // Build token ID vector: special tokens as raw IDs, text segments encoded
    std::vector<int32_t> apply(const Tokenizer& tok,
                               const std::vector<ChatMessage>& messages) const;

    // Build token ID vector with image tokens inserted before the first user message.
    // Produces: <boi> <img_soft_token>*n_image_tokens <eoi> \n {text}
    std::vector<int32_t> apply_with_image(const Tokenizer& tok,
                                           const std::vector<ChatMessage>& messages,
                                           int n_image_tokens) const;

    const std::vector<int32_t>& stop_token_ids() const { return stop_token_ids_; }
    ChatTemplateFamily family() const { return family_; }
    bool is_raw() const { return family_ == ChatTemplateFamily::RAW; }

private:
    ChatTemplateFamily family_ = ChatTemplateFamily::RAW;
    std::vector<int32_t> stop_token_ids_;

    // Resolved special token IDs (set during init)
    int32_t bos_id_ = -1;

    // ChatML tokens
    int32_t im_start_id_ = -1;
    int32_t im_end_id_ = -1;

    // Llama3 tokens
    int32_t start_header_id_ = -1;
    int32_t end_header_id_ = -1;
    int32_t eot_id_ = -1;

    // Llama2 tokens
    int32_t inst_start_id_ = -1;  // [INST]
    int32_t inst_end_id_ = -1;    // [/INST]

    // Nemotron tokens
    int32_t extra_id_0_ = -1;
    int32_t extra_id_1_ = -1;

    // Gemma tokens
    int32_t start_of_turn_id_ = -1;
    int32_t end_of_turn_id_ = -1;

    // DeepSeek R1 tokens
    int32_t ds_user_id_ = -1;        // <｜User｜>
    int32_t ds_assistant_id_ = -1;   // <｜Assistant｜>
    int32_t ds_eos_id_ = -1;         // <｜end▁of▁sentence｜>

    // Vision tokens (Gemma-3)
    int32_t boi_id_ = -1;           // <start_of_image>
    int32_t eoi_id_ = -1;           // <end_of_image>
    int32_t img_soft_token_id_ = -1; // <image_soft_token>

    // Template-specific apply methods
    std::vector<int32_t> apply_chatml(const Tokenizer& tok,
                                       const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_llama3(const Tokenizer& tok,
                                       const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_llama2(const Tokenizer& tok,
                                       const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_nemotron(const Tokenizer& tok,
                                         const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_gemma(const Tokenizer& tok,
                                      const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_deepseek_r1(const Tokenizer& tok,
                                            const std::vector<ChatMessage>& msgs) const;
};

} // namespace imp
