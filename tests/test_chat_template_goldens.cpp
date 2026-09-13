// Byte-for-byte chat-template goldens per family through the PRODUCTION path (#1572): most
// families had only structural smoke (token count, substring), which cannot catch a wrong
// prompt - jinja.cpp skips unknown tags with no log, so a regression reaches the model unseen.
// Oracle: transformers apply_chat_template, captured by tests/refs/gen_chat_goldens.py, never
// imp-vs-imp. Goes through ChatTemplate::init()+render_jinja(), unlike the pre-existing
// harmony golden which bypasses the context builder. No model/GPU needed: only bos/eos strings.

#include <gtest/gtest.h>

#include "model/chat_template.h"
#include "model/tokenizer.h"
#include "refs/chat_template_goldens.h"

#include <string>
#include <vector>

namespace imp {
namespace {

// Extracts every `<...>`-shaped literal (special tokens across all ten families:
// <|im_start|>, <start_of_turn>, DeepSeek's fullwidth-bar tokens, <extra_id_0>).
// Derived from the template, not tabulated: init() refuses when a family's tokens are absent
// from the vocab, so a hand-kept table would need editing per checkpoint and rot.
std::vector<std::string> special_tokens_in(const std::string& tpl) {
    std::vector<std::string> out;
    for (size_t i = 0; i < tpl.size(); ++i) {
        // Both bracket shapes: `<|im_start|>` and Mistral's `[INST]`. Scanning
        // only `<...>` left the Llama2 family with inst_start=-1, and `init()`
        // fell back to raw - a failure that reads like a render divergence.
        const char close = tpl[i] == '<' ? '>' : (tpl[i] == '[' ? ']' : '\0');
        if (close == '\0')
            continue;
        const size_t end = tpl.find(close, i + 1);
        if (end == std::string::npos || end - i > 64)
            continue;
        const std::string cand = tpl.substr(i, end - i + 1);
        // A marker has no whitespace and no quote: `<think>` qualifies,
        // `<a href="x">` and a stray `<` in prose do not.
        if (cand.find_first_of(" \t\n\r\"\'") != std::string::npos)
            continue;
        out.push_back(cand);
        i = end;
    }
    return out;
}

// A vocabulary whose job is to report the family's bos/eos TEXT and to carry
// its special tokens, which is all `init()` and `render_jinja` read from a
// tokenizer. Byte fallback keeps `load_vocab` happy for anything else.
Tokenizer make_tokenizer(const std::string& bos, const std::string& eos, const std::string& tpl_src) {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    tokens.push_back("<unk>");
    scores.push_back(0.0f);
    for (int b = 0; b < 256; b++) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.push_back(buf);
        scores.push_back(-10.0f);
    }
    for (const auto& t : special_tokens_in(tpl_src)) {
        tokens.push_back(t);
        scores.push_back(0.0f);
    }
    // An empty bos is not a token: Qwen3 has none, and inventing one would put
    // a string in the render that the reference does not have.
    int bos_id = -1;
    if (!bos.empty()) {
        bos_id = static_cast<int>(tokens.size());
        tokens.push_back(bos);
        scores.push_back(0.0f);
    }
    const int eos_id = static_cast<int>(tokens.size());
    tokens.push_back(eos.empty() ? "</s>" : eos);
    scores.push_back(0.0f);

    Tokenizer tok;
    EXPECT_TRUE(tok.load_vocab(tokens, scores, bos_id, eos_id));
    return tok;
}

struct GoldenCase {
    const char* name;
    const std::string* expected;
    std::vector<ChatMessage> msgs;
};

std::vector<ChatMessage> to_msgs(const std::vector<std::pair<std::string, std::string>>& rows) {
    std::vector<ChatMessage> out;
    for (const auto& [role, content] : rows)
        out.push_back({role, content});
    return out;
}

std::vector<GoldenCase> cases(const std::string& user_only, const std::string& system_user,
                              const std::string& multi_turn) {
    return {
        {"user_only", &user_only, to_msgs(chat_goldens::k_conv_user_only)},
        {"system_user", &system_user, to_msgs(chat_goldens::k_conv_system_user)},
        {"multi_turn", &multi_turn, to_msgs(chat_goldens::k_conv_multi_turn)},
    };
}

// Family comes from imp's own detect_family(), not a hand-picked label, pinned against a
// value: Nemotron-3-Nano and Phi-4-reasoning both ship ChatML-shaped templates while the enum
// comments describe older checkpoints, so hand-picking failed twice and init() fell back to
// raw for an unrelated reason.
void check_family(ChatTemplateFamily expect_family, const std::string& tpl_src, const std::string& bos,
                  const std::string& eos, const std::vector<GoldenCase>& cs) {
    const ChatTemplateFamily family = ChatTemplate::detect_family(tpl_src);
    ASSERT_EQ(family, expect_family)
        << "detect_family moved on this template; the golden below was rendered for the pinned "
           "family, so re-check both rather than just re-pinning";
    Tokenizer tok = make_tokenizer(bos, eos, tpl_src);
    ChatTemplate tpl;
    ASSERT_TRUE(tpl.init(family, tok, tpl_src)) << "init failed";
    ASSERT_TRUE(tpl.has_jinja()) << "the Jinja engine must be the one driving, or this golden "
                                    "measures the hardcoded family instead of the template";
    for (const auto& c : cs) {
        const std::string got = tpl.render_jinja(tok, c.msgs, /*add_generation_prompt=*/true);
        EXPECT_EQ(got, *c.expected) << "family render diverged from the HF reference: " << c.name;
    }
}

}  // namespace

TEST(ChatTemplateGolden, ChatML) {
    check_family(ChatTemplateFamily::CHATML, chat_goldens::k_chatml_template, chat_goldens::k_chatml_bos,
                 chat_goldens::k_chatml_eos,
                 cases(chat_goldens::k_chatml_user_only, chat_goldens::k_chatml_system_user,
                       chat_goldens::k_chatml_multi_turn));
}

TEST(ChatTemplateGolden, ChatMLNemotronNano) {
    check_family(ChatTemplateFamily::CHATML, chat_goldens::k_chatml_nemotron_nano_template,
                 chat_goldens::k_chatml_nemotron_nano_bos, chat_goldens::k_chatml_nemotron_nano_eos,
                 cases(chat_goldens::k_chatml_nemotron_nano_user_only,
                       chat_goldens::k_chatml_nemotron_nano_system_user,
                       chat_goldens::k_chatml_nemotron_nano_multi_turn));
}

TEST(ChatTemplateGolden, Gemma) {
    check_family(ChatTemplateFamily::GEMMA, chat_goldens::k_gemma_template, chat_goldens::k_gemma_bos,
                 chat_goldens::k_gemma_eos,
                 cases(chat_goldens::k_gemma_user_only, chat_goldens::k_gemma_system_user,
                       chat_goldens::k_gemma_multi_turn));
}

TEST(ChatTemplateGolden, Llama3) {
    check_family(ChatTemplateFamily::LLAMA3, chat_goldens::k_llama3_template, chat_goldens::k_llama3_bos,
                 chat_goldens::k_llama3_eos,
                 cases(chat_goldens::k_llama3_user_only, chat_goldens::k_llama3_system_user,
                       chat_goldens::k_llama3_multi_turn));
}

TEST(ChatTemplateGolden, Llama2) {
    check_family(ChatTemplateFamily::LLAMA2, chat_goldens::k_llama2_template, chat_goldens::k_llama2_bos,
                 chat_goldens::k_llama2_eos,
                 cases(chat_goldens::k_llama2_user_only, chat_goldens::k_llama2_system_user,
                       chat_goldens::k_llama2_multi_turn));
}

TEST(ChatTemplateGolden, MistralV3) {
    check_family(ChatTemplateFamily::MISTRAL_V3, chat_goldens::k_mistral_v3_template,
                 chat_goldens::k_mistral_v3_bos, chat_goldens::k_mistral_v3_eos,
                 cases(chat_goldens::k_mistral_v3_user_only, chat_goldens::k_mistral_v3_system_user,
                       chat_goldens::k_mistral_v3_multi_turn));
}

TEST(ChatTemplateGolden, Nemotron) {
    check_family(ChatTemplateFamily::NEMOTRON, chat_goldens::k_nemotron_template,
                 chat_goldens::k_nemotron_bos, chat_goldens::k_nemotron_eos,
                 cases(chat_goldens::k_nemotron_user_only, chat_goldens::k_nemotron_system_user,
                       chat_goldens::k_nemotron_multi_turn));
}

TEST(ChatTemplateGolden, DeepSeekR1) {
    check_family(ChatTemplateFamily::DEEPSEEK_R1, chat_goldens::k_deepseek_r1_template,
                 chat_goldens::k_deepseek_r1_bos, chat_goldens::k_deepseek_r1_eos,
                 cases(chat_goldens::k_deepseek_r1_user_only, chat_goldens::k_deepseek_r1_system_user,
                       chat_goldens::k_deepseek_r1_multi_turn));
}

TEST(ChatTemplateGolden, Phi) {
    check_family(ChatTemplateFamily::PHI, chat_goldens::k_phi_template, chat_goldens::k_phi_bos,
                 chat_goldens::k_phi_eos,
                 cases(chat_goldens::k_phi_user_only, chat_goldens::k_phi_system_user,
                       chat_goldens::k_phi_multi_turn));
}

// The generation prompt is the half a structural test cannot see: without it the model is
// handed a finished transcript and answers by writing the next turn's role marker as text
// (the Phi-4 {% generation %} derailment, src/model/chat_template.cpp).
TEST(ChatTemplateGolden, GenerationPromptIsWhatTheGoldenPins) {
    Tokenizer tok = make_tokenizer(chat_goldens::k_chatml_bos, chat_goldens::k_chatml_eos,
                                   chat_goldens::k_chatml_template);
    ChatTemplate tpl;
    ASSERT_TRUE(tpl.init(ChatTemplateFamily::CHATML, tok, chat_goldens::k_chatml_template));
    const std::vector<ChatMessage> msgs = {{"user", "What is the capital of France?"}};

    const std::string with = tpl.render_jinja(tok, msgs, /*add_generation_prompt=*/true);
    const std::string without = tpl.render_jinja(tok, msgs, /*add_generation_prompt=*/false);

    EXPECT_EQ(with, chat_goldens::k_chatml_user_only);
    EXPECT_NE(without, with) << "dropping the generation prompt must change the render, or the "
                                "golden cannot detect it going missing";
    EXPECT_TRUE(with.size() > without.size());
}

}  // namespace imp
