#pragma once

#include "model/tokenizer.h"
#include "model/model_arch.h"
#include "model/jinja.h"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace imp {

enum class ChatTemplateFamily {
    RAW,          // No template — pass raw text
    CHATML,       // <|im_start|>...<|im_end|> (Qwen3, etc.)
    LLAMA2,       // [INST]...[/INST] (Llama 2, older Mistral V1/V2)
    MISTRAL_V3,   // [INST]...[/INST] + [TOOL_CALLS]/[AVAILABLE_TOOLS] (Mistral V3-Tekken: 3.x family)
    LLAMA3,       // <|start_header_id|>...<|end_header_id|>...<|eot_id|>
    NEMOTRON,     // <extra_id_0>System\n...<extra_id_1>\n<extra_id_0>User\n...
    GEMMA,        // <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
    DEEPSEEK_R1,  // <｜User｜>...<｜Assistant｜>...<｜end▁of▁sentence｜>
    PHI,          // <|user|>...<|end|>...<|assistant|>
    HARMONY,      // gpt-oss: <|start|>role<|message|>...<|end|>, channels, <|return|> stop
};

const char* chat_template_family_name(ChatTemplateFamily family);

struct ChatMessage {
    std::string role;  // "system", "user", "assistant"
    std::string content;
};

struct ToolFunction {
    std::string name;
    std::string description;
    std::string parameters_json;  // JSON string of parameters schema
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

    // Initialize: resolve special token IDs via tokenizer.
    // jinja_str: raw Jinja2 template from GGUF (optional). When provided and
    // parseable, the engine renders via Jinja2 instead of hardcoded families.
    bool init(ChatTemplateFamily family, const Tokenizer& tokenizer, const std::string& jinja_str = "");

    // Build token ID vector: special tokens as raw IDs, text segments encoded.
    // suppress_thinking: inject /no_think + stamp enable_thinking=false so the
    //   template renders its answer-directly branch.
    // force_thinking: stamp enable_thinking=true so a template that defaults the
    //   variable to a *closed* block (e.g. Qwen3.5-4B: `<think>\n\n</think>\n\n`)
    //   instead opens the block for an explicit caller request. suppress wins if
    //   both are set. Default (neither) leaves the variable undefined so each
    //   template author's own default applies (Qwen3 open vs Gemma-4 closed).
    std::vector<int32_t> apply(const Tokenizer& tok, const std::vector<ChatMessage>& messages,
                               bool suppress_thinking = false, bool force_thinking = false) const;

    // Build token ID vector with tool definitions passed to Jinja2 context.
    // Falls back to standard apply() if Jinja2 doesn't handle tools.
    std::vector<int32_t> apply_with_tools(const Tokenizer& tok, const std::vector<ChatMessage>& messages,
                                          const std::vector<ToolFunction>& tools,
                                          const std::string& tool_choice = "auto",
                                          bool suppress_thinking = false,
                                          bool force_thinking = false) const;

    // Build token ID vector with image tokens inserted before the first user message.
    // Produces: <boi> <img_soft_token>*n_image_tokens <eoi> \n {text}
    std::vector<int32_t> apply_with_image(const Tokenizer& tok, const std::vector<ChatMessage>& messages,
                                          int n_image_tokens, bool suppress_thinking = false,
                                          bool force_thinking = false) const;

    const std::vector<int32_t>& stop_token_ids() const { return stop_token_ids_; }
    ChatTemplateFamily family() const { return family_; }
    // True when the raw Jinja template references reasoning ("<think>" or
    // "enable_thinking"). Used to decide the server-side thinking DEFAULT:
    // vocab-level <think> tokens alone are NOT evidence of a think-trained
    // model (Qwen3-*-Instruct-2507 ships the Qwen3 vocab incl. <think>
    // specials but is not think-trained and its template never opens a
    // think block — defaulting it to thinking traps the whole answer in
    // reasoning_content).
    bool mentions_thinking() const { return mentions_thinking_; }
    // True when a raw Jinja template is driving rendering (mentions_thinking
    // is only meaningful evidence in that case).
    bool has_jinja() const { return use_jinja_; }
    bool is_raw() const { return family_ == ChatTemplateFamily::RAW; }
    bool supports_tools() const;
    const std::string& default_system_message() const { return default_system_message_; }

    // Special token accessors (for banned token list)
    int32_t im_start_id() const { return im_start_id_; }
    int32_t start_header_id() const { return start_header_id_; }
    int32_t end_header_id() const { return end_header_id_; }

private:
    ChatTemplateFamily family_ = ChatTemplateFamily::RAW;
    std::vector<int32_t> stop_token_ids_;
    std::string default_system_message_;

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
    int32_t ds_user_id_ = -1;       // <｜User｜>
    int32_t ds_assistant_id_ = -1;  // <｜Assistant｜>
    int32_t ds_eos_id_ = -1;        // <｜end▁of▁sentence｜>

    // Phi tokens
    int32_t phi_user_id_ = -1;       // <|user|>
    int32_t phi_assistant_id_ = -1;  // <|assistant|>
    int32_t phi_end_id_ = -1;        // <|end|>

    // Harmony tokens (gpt-oss)
    int32_t hm_start_id_ = -1;    // <|start|>
    int32_t hm_end_id_ = -1;      // <|end|>     (message separator — NOT a stop token)
    int32_t hm_message_id_ = -1;  // <|message|>
    int32_t hm_channel_id_ = -1;  // <|channel|>
    int32_t hm_return_id_ = -1;   // <|return|>  (stop)
    int32_t hm_call_id_ = -1;     // <|call|>    (stop, tool call)

    // Vision tokens (Gemma-3)
    int32_t boi_id_ = -1;             // <start_of_image>
    int32_t eoi_id_ = -1;             // <end_of_image>
    int32_t img_soft_token_id_ = -1;  // <image_soft_token>

    // Jinja2 engine (set during init if template string provided)
    std::shared_ptr<jinja::Template> jinja_tpl_;
    bool use_jinja_ = false;
    bool mentions_thinking_ = false;

    // Jinja2-based apply: render template, split on control tokens, encode
    bool probe_render_mentions_think(const Tokenizer& tok) const;
    std::vector<int32_t> apply_jinja(const Tokenizer& tok, const std::vector<ChatMessage>& msgs,
                                     bool add_generation_prompt = true, bool suppress_thinking = false,
                                     bool force_thinking = false) const;

    // Jinja2-based apply with tool definitions in context
    std::vector<int32_t> apply_jinja_with_tools(const Tokenizer& tok, const std::vector<ChatMessage>& msgs,
                                                const std::vector<ToolFunction>& tools,
                                                const std::string& tool_choice,
                                                bool add_generation_prompt = true,
                                                bool suppress_thinking = false,
                                                bool force_thinking = false) const;

    // Shared helper: split rendered string on control tokens and encode
    std::vector<int32_t> tokenize_rendered(const Tokenizer& tok, const std::string& rendered) const;

    // Auto-detect stop tokens from a rendered Jinja2 context
    void auto_detect_stop_tokens(const jinja::Context& ctx) const;

    // Build control token lookup table for splitting rendered output
    void build_control_token_map(const Tokenizer& tok);
    std::vector<std::pair<std::string, int32_t>> control_tokens_;  // sorted longest-first

    // Template-specific apply methods
    std::vector<int32_t> apply_chatml(const Tokenizer& tok, const std::vector<ChatMessage>& msgs,
                                      bool suppress_thinking = false) const;
    std::vector<int32_t> apply_llama3(const Tokenizer& tok, const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_llama2(const Tokenizer& tok, const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_nemotron(const Tokenizer& tok, const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_gemma(const Tokenizer& tok, const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_deepseek_r1(const Tokenizer& tok, const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_phi(const Tokenizer& tok, const std::vector<ChatMessage>& msgs) const;
    std::vector<int32_t> apply_harmony(const Tokenizer& tok, const std::vector<ChatMessage>& msgs) const;
};

}  // namespace imp
