#include "model/chat_template.h"
#include "model/model_arch.h"
#include <gtest/gtest.h>

namespace imp {
namespace {

// ---- Helper: build a tokenizer with byte fallback + all special tokens ----

static Tokenizer make_chat_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<float> scores;

    // 0: <unk>, 1: <s> (BOS), 2: </s> (EOS)
    tokens.push_back("<unk>"); scores.push_back(0.0f);
    tokens.push_back("<s>");   scores.push_back(0.0f);
    tokens.push_back("</s>"); scores.push_back(0.0f);

    // 3-258: SPM byte fallback <0x00>..<0xFF>
    for (int b = 0; b < 256; b++) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.push_back(buf);
        scores.push_back(-10.0f);
    }

    // 259+: Special tokens for all chat template families
    tokens.push_back("<|im_start|>"); scores.push_back(0.0f);           // 259
    tokens.push_back("<|im_end|>");   scores.push_back(0.0f);           // 260
    tokens.push_back("<|start_header_id|>"); scores.push_back(0.0f);    // 261
    tokens.push_back("<|end_header_id|>");   scores.push_back(0.0f);    // 262
    tokens.push_back("<|eot_id|>");          scores.push_back(0.0f);    // 263
    tokens.push_back("[INST]");  scores.push_back(0.0f);                // 264
    tokens.push_back("[/INST]"); scores.push_back(0.0f);                // 265
    tokens.push_back("<extra_id_0>"); scores.push_back(0.0f);           // 266
    tokens.push_back("<extra_id_1>"); scores.push_back(0.0f);           // 267
    tokens.push_back("<start_of_turn>"); scores.push_back(0.0f);        // 268
    tokens.push_back("<end_of_turn>");   scores.push_back(0.0f);        // 269
    // DeepSeek R1 (fullwidth vertical bars U+FF5C)
    tokens.push_back("<\xef\xbd\x9c" "User\xef\xbd\x9c>");             // 270
    scores.push_back(0.0f);
    tokens.push_back("<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>");        // 271
    scores.push_back(0.0f);
    tokens.push_back("<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>"); // 272
    scores.push_back(0.0f);
    tokens.push_back("<|user|>");      scores.push_back(0.0f);          // 273
    tokens.push_back("<|assistant|>"); scores.push_back(0.0f);          // 274
    tokens.push_back("<|end|>");       scores.push_back(0.0f);          // 275
    tokens.push_back("<start_of_image>"); scores.push_back(0.0f);       // 276
    tokens.push_back("<end_of_image>");   scores.push_back(0.0f);       // 277
    tokens.push_back("<image_soft_token>"); scores.push_back(0.0f);     // 278

    Tokenizer tok;
    tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2);
    tok.set_type("spm");
    tok.set_add_bos(true);
    tok.set_add_space_prefix(false);
    return tok;
}

// Token IDs for readability
static constexpr int BOS = 1;
static constexpr int EOS = 2;
static constexpr int IM_START = 259;
static constexpr int IM_END = 260;
static constexpr int START_HEADER = 261;
static constexpr int END_HEADER = 262;
static constexpr int EOT = 263;
static constexpr int INST_START = 264;
static constexpr int INST_END = 265;
static constexpr int EXTRA_0 = 266;
static constexpr int EXTRA_1 = 267;
static constexpr int START_TURN = 268;
static constexpr int END_TURN = 269;
static constexpr int DS_USER = 270;
static constexpr int DS_ASSISTANT = 271;
static constexpr int DS_EOS = 272;
static constexpr int PHI_USER = 273;
static constexpr int PHI_ASSISTANT = 274;
static constexpr int PHI_END = 275;
static constexpr int BOI = 276;
static constexpr int EOI = 277;
static constexpr int IMG_SOFT = 278;

// Helper: check if a token ID appears in a vector
static bool contains(const std::vector<int32_t>& v, int32_t id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

// Helper: find first occurrence of id in vector, returns -1 if not found
static int find_pos(const std::vector<int32_t>& v, int32_t id) {
    auto it = std::find(v.begin(), v.end(), id);
    if (it == v.end()) return -1;
    return static_cast<int>(it - v.begin());
}

// ---- detect_family ----

TEST(ChatTemplateDetectTest, EmptyReturnsRaw) {
    EXPECT_EQ(ChatTemplate::detect_family(""), ChatTemplateFamily::RAW);
}

TEST(ChatTemplateDetectTest, ChatML) {
    EXPECT_EQ(ChatTemplate::detect_family("{% for m in messages %}<|im_start|>{{ m.role }}"),
              ChatTemplateFamily::CHATML);
}

TEST(ChatTemplateDetectTest, Llama3) {
    EXPECT_EQ(ChatTemplate::detect_family("<|start_header_id|>{{ role }}<|end_header_id|>"),
              ChatTemplateFamily::LLAMA3);
}

TEST(ChatTemplateDetectTest, Gemma) {
    EXPECT_EQ(ChatTemplate::detect_family("<start_of_turn>user\n"),
              ChatTemplateFamily::GEMMA);
}

TEST(ChatTemplateDetectTest, Llama2) {
    EXPECT_EQ(ChatTemplate::detect_family("[INST] {{ message.content }} [/INST]"),
              ChatTemplateFamily::LLAMA2);
}

TEST(ChatTemplateDetectTest, Nemotron) {
    EXPECT_EQ(ChatTemplate::detect_family("<extra_id_0>System\n"),
              ChatTemplateFamily::NEMOTRON);
}

TEST(ChatTemplateDetectTest, DeepSeekR1) {
    // Fullwidth vertical bars U+FF5C = \xef\xbd\x9c
    EXPECT_EQ(ChatTemplate::detect_family("<\xef\xbd\x9c" "User\xef\xbd\x9c>"),
              ChatTemplateFamily::DEEPSEEK_R1);
}

TEST(ChatTemplateDetectTest, Phi) {
    EXPECT_EQ(ChatTemplate::detect_family("<|user|>\n{{ content }}<|end|>\n"),
              ChatTemplateFamily::PHI);
}

TEST(ChatTemplateDetectTest, UnknownReturnsRaw) {
    EXPECT_EQ(ChatTemplate::detect_family("some random template {{ message }}"),
              ChatTemplateFamily::RAW);
}

TEST(ChatTemplateDetectTest, ChatMLTakesPriorityOverPhi) {
    // Both <|im_start|> and <|end|> present -> ChatML wins (checked first)
    EXPECT_EQ(ChatTemplate::detect_family("<|im_start|>role<|im_end|><|end|>"),
              ChatTemplateFamily::CHATML);
}

// ---- parse_family ----

TEST(ChatTemplateParseFamilyTest, AllNames) {
    EXPECT_EQ(ChatTemplate::parse_family("chatml"), ChatTemplateFamily::CHATML);
    EXPECT_EQ(ChatTemplate::parse_family("llama2"), ChatTemplateFamily::LLAMA2);
    EXPECT_EQ(ChatTemplate::parse_family("llama3"), ChatTemplateFamily::LLAMA3);
    EXPECT_EQ(ChatTemplate::parse_family("nemotron"), ChatTemplateFamily::NEMOTRON);
    EXPECT_EQ(ChatTemplate::parse_family("gemma"), ChatTemplateFamily::GEMMA);
    EXPECT_EQ(ChatTemplate::parse_family("deepseek_r1"), ChatTemplateFamily::DEEPSEEK_R1);
    EXPECT_EQ(ChatTemplate::parse_family("deepseek-r1"), ChatTemplateFamily::DEEPSEEK_R1);
    EXPECT_EQ(ChatTemplate::parse_family("phi"), ChatTemplateFamily::PHI);
    EXPECT_EQ(ChatTemplate::parse_family("none"), ChatTemplateFamily::RAW);
    EXPECT_EQ(ChatTemplate::parse_family("auto"), ChatTemplateFamily::RAW);
    EXPECT_EQ(ChatTemplate::parse_family("unknown_garbage"), ChatTemplateFamily::RAW);
}

// ---- default_family_for_arch ----

TEST(ChatTemplateArchDefaultTest, KnownArchitectures) {
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::LLAMA), ChatTemplateFamily::LLAMA3);
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::MISTRAL), ChatTemplateFamily::LLAMA2);
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::MIXTRAL), ChatTemplateFamily::LLAMA2);
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::DEEPSEEK), ChatTemplateFamily::DEEPSEEK_R1);
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::NEMOTRON_H_MOE), ChatTemplateFamily::NEMOTRON);
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::QWEN3), ChatTemplateFamily::CHATML);
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::QWEN3_MOE), ChatTemplateFamily::CHATML);
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::GEMMA3), ChatTemplateFamily::GEMMA);
    EXPECT_EQ(ChatTemplate::default_family_for_arch(ModelArch::GENERIC), ChatTemplateFamily::RAW);
}

// ---- chat_template_family_name ----

TEST(ChatTemplateFamilyNameTest, AllFamilies) {
    EXPECT_STREQ(chat_template_family_name(ChatTemplateFamily::RAW), "raw");
    EXPECT_STREQ(chat_template_family_name(ChatTemplateFamily::CHATML), "chatml");
    EXPECT_STREQ(chat_template_family_name(ChatTemplateFamily::LLAMA2), "llama2");
    EXPECT_STREQ(chat_template_family_name(ChatTemplateFamily::LLAMA3), "llama3");
    EXPECT_STREQ(chat_template_family_name(ChatTemplateFamily::NEMOTRON), "nemotron");
    EXPECT_STREQ(chat_template_family_name(ChatTemplateFamily::GEMMA), "gemma");
    EXPECT_STREQ(chat_template_family_name(ChatTemplateFamily::DEEPSEEK_R1), "deepseek_r1");
    EXPECT_STREQ(chat_template_family_name(ChatTemplateFamily::PHI), "phi");
}

// ---- init ----

TEST(ChatTemplateInitTest, RawAlwaysSucceeds) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::RAW, tok));
    EXPECT_TRUE(tpl.is_raw());
    EXPECT_TRUE(tpl.stop_token_ids().empty());
}

TEST(ChatTemplateInitTest, ChatMLSuccess) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::CHATML, tok));
    EXPECT_EQ(tpl.family(), ChatTemplateFamily::CHATML);
    ASSERT_EQ(tpl.stop_token_ids().size(), 1u);
    EXPECT_EQ(tpl.stop_token_ids()[0], IM_END);
}

TEST(ChatTemplateInitTest, ChatMLMissingTokensFallsBack) {
    // Tokenizer without <|im_start|> / <|im_end|>
    Tokenizer tok;
    std::vector<std::string> v = {"<unk>", "<s>", "</s>"};
    std::vector<float> s = {0, 0, 0};
    tok.load_vocab(v, s, 1, 2);
    tok.set_type("spm");

    ChatTemplate tpl;
    EXPECT_FALSE(tpl.init(ChatTemplateFamily::CHATML, tok));
    EXPECT_EQ(tpl.family(), ChatTemplateFamily::RAW);
}

TEST(ChatTemplateInitTest, Llama3Success) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::LLAMA3, tok));
    EXPECT_EQ(tpl.family(), ChatTemplateFamily::LLAMA3);
    ASSERT_EQ(tpl.stop_token_ids().size(), 1u);
    EXPECT_EQ(tpl.stop_token_ids()[0], EOT);
}

TEST(ChatTemplateInitTest, Llama2Success) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::LLAMA2, tok));
    EXPECT_EQ(tpl.family(), ChatTemplateFamily::LLAMA2);
    ASSERT_EQ(tpl.stop_token_ids().size(), 1u);
    EXPECT_EQ(tpl.stop_token_ids()[0], EOS);
}

TEST(ChatTemplateInitTest, NemotronSuccess) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::NEMOTRON, tok));
    EXPECT_EQ(tpl.family(), ChatTemplateFamily::NEMOTRON);
    ASSERT_EQ(tpl.stop_token_ids().size(), 1u);
    EXPECT_EQ(tpl.stop_token_ids()[0], EXTRA_1);
}

TEST(ChatTemplateInitTest, GemmaSuccess) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::GEMMA, tok));
    EXPECT_EQ(tpl.family(), ChatTemplateFamily::GEMMA);
    // Gemma has 2 stop tokens: end_of_turn + EOS
    ASSERT_EQ(tpl.stop_token_ids().size(), 2u);
    EXPECT_EQ(tpl.stop_token_ids()[0], END_TURN);
    EXPECT_EQ(tpl.stop_token_ids()[1], EOS);
}

TEST(ChatTemplateInitTest, DeepSeekR1Success) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::DEEPSEEK_R1, tok));
    EXPECT_EQ(tpl.family(), ChatTemplateFamily::DEEPSEEK_R1);
    ASSERT_EQ(tpl.stop_token_ids().size(), 1u);
    EXPECT_EQ(tpl.stop_token_ids()[0], DS_EOS);
}

TEST(ChatTemplateInitTest, PhiSuccess) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::PHI, tok));
    EXPECT_EQ(tpl.family(), ChatTemplateFamily::PHI);
    ASSERT_EQ(tpl.stop_token_ids().size(), 1u);
    EXPECT_EQ(tpl.stop_token_ids()[0], PHI_END);
}

// ---- apply: ChatML ----

TEST(ChatTemplateApplyTest, ChatMLBasicStructure) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    ASSERT_FALSE(ids.empty());
    // Should start with BOS (add_bos=true and bos!=im_start)
    EXPECT_EQ(ids[0], BOS);
    // Should contain im_start for user message + im_start for assistant prefix
    int im_start_count = std::count(ids.begin(), ids.end(), IM_START);
    EXPECT_EQ(im_start_count, 2);  // user msg + assistant prefix
    // Should contain im_end for user message
    EXPECT_TRUE(contains(ids, IM_END));
    // Should end with encoded "assistant\n" tokens (after final im_start)
    int last_im_start = -1;
    for (int i = static_cast<int>(ids.size()) - 1; i >= 0; i--) {
        if (ids[i] == IM_START) { last_im_start = i; break; }
    }
    EXPECT_GT(last_im_start, 0);
}

TEST(ChatTemplateApplyTest, ChatMLNoBOSWhenDisabled) {
    Tokenizer tok = make_chat_tokenizer();
    tok.set_add_bos(false);

    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    ASSERT_FALSE(ids.empty());
    // First token should be im_start, not BOS
    EXPECT_EQ(ids[0], IM_START);
}

TEST(ChatTemplateApplyTest, ChatMLMultiTurn) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);

    std::vector<ChatMessage> msgs = {
        {"system", "You help."},
        {"user", "Hi"},
        {"assistant", "Hey"},
        {"user", "Bye"},
    };
    auto ids = tpl.apply(tok, msgs);

    // 4 messages + 1 assistant prefix = 5 im_start tokens
    int im_start_count = std::count(ids.begin(), ids.end(), IM_START);
    EXPECT_EQ(im_start_count, 5);
    // 4 im_end tokens (one per message, not for assistant prefix)
    int im_end_count = std::count(ids.begin(), ids.end(), IM_END);
    EXPECT_EQ(im_end_count, 4);
}

// ---- apply: Llama3 ----

TEST(ChatTemplateApplyTest, Llama3BasicStructure) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::LLAMA3, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids[0], BOS);
    // User message: start_header, ..role.., end_header, ..content.., eot
    EXPECT_TRUE(contains(ids, START_HEADER));
    EXPECT_TRUE(contains(ids, END_HEADER));
    EXPECT_TRUE(contains(ids, EOT));
    // 2 start_header: user + assistant prefix
    EXPECT_EQ(std::count(ids.begin(), ids.end(), START_HEADER), 2);
}

// ---- apply: Llama2 ----

TEST(ChatTemplateApplyTest, Llama2BasicStructure) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::LLAMA2, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids[0], BOS);
    EXPECT_TRUE(contains(ids, INST_START));
    EXPECT_TRUE(contains(ids, INST_END));
}

TEST(ChatTemplateApplyTest, Llama2SystemMessage) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::LLAMA2, tok);

    std::vector<ChatMessage> msgs = {
        {"system", "Be helpful."},
        {"user", "Hi"},
    };
    auto ids = tpl.apply(tok, msgs);

    // System message should be embedded within the first [INST] block
    // Only 1 [INST] (system is not a separate block, it's merged into first user)
    EXPECT_EQ(std::count(ids.begin(), ids.end(), INST_START), 1);
    EXPECT_EQ(std::count(ids.begin(), ids.end(), INST_END), 1);
}

// ---- apply: Nemotron ----

TEST(ChatTemplateApplyTest, NemotronBasicStructure) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::NEMOTRON, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids[0], BOS);
    // User message: extra_id_0 ... extra_id_1
    EXPECT_TRUE(contains(ids, EXTRA_0));
    EXPECT_TRUE(contains(ids, EXTRA_1));
    // 2 extra_id_0: user msg + assistant prefix
    EXPECT_EQ(std::count(ids.begin(), ids.end(), EXTRA_0), 2);
    // 1 extra_id_1: user msg close (assistant prefix doesn't close)
    EXPECT_EQ(std::count(ids.begin(), ids.end(), EXTRA_1), 1);
}

// ---- apply: Gemma ----

TEST(ChatTemplateApplyTest, GemmaBasicStructure) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::GEMMA, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids[0], BOS);
    EXPECT_TRUE(contains(ids, START_TURN));
    EXPECT_TRUE(contains(ids, END_TURN));
    // 2 start_of_turn: user + model prefix
    EXPECT_EQ(std::count(ids.begin(), ids.end(), START_TURN), 2);
    // 1 end_of_turn: user msg (model prefix doesn't close)
    EXPECT_EQ(std::count(ids.begin(), ids.end(), END_TURN), 1);
}

TEST(ChatTemplateApplyTest, GemmaAssistantRoleMappedToModel) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::GEMMA, tok);

    std::vector<ChatMessage> msgs = {
        {"user", "Hi"},
        {"assistant", "Hey"},
    };
    auto ids = tpl.apply(tok, msgs);
    // 3 start_of_turn: user + assistant(=model) + model prefix
    EXPECT_EQ(std::count(ids.begin(), ids.end(), START_TURN), 3);
}

// ---- apply: DeepSeek R1 ----

TEST(ChatTemplateApplyTest, DeepSeekR1BasicStructure) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::DEEPSEEK_R1, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    ASSERT_FALSE(ids.empty());
    // DeepSeek R1 always starts with BOS
    EXPECT_EQ(ids[0], BOS);
    EXPECT_TRUE(contains(ids, DS_USER));
    // Assistant prefix at the end
    EXPECT_TRUE(contains(ids, DS_ASSISTANT));
}

// ---- apply: Phi ----

TEST(ChatTemplateApplyTest, PhiBasicStructure) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::PHI, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids[0], BOS);
    EXPECT_TRUE(contains(ids, PHI_USER));
    EXPECT_TRUE(contains(ids, PHI_END));
    // 1 phi_user + 1 phi_assistant prefix
    EXPECT_TRUE(contains(ids, PHI_ASSISTANT));
}

// ---- apply: RAW returns empty ----

TEST(ChatTemplateApplyTest, RawReturnsEmpty) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::RAW, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);
    EXPECT_TRUE(ids.empty());
}

// ---- Default system message extraction ----

TEST(ChatTemplateDefaultSysTest, ExtractFromJinja) {
    Tokenizer tok = make_chat_tokenizer();
    // Set a Jinja template string containing a default system message
    tok.set_chat_template_str(
        "{% for m in messages %}"
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        "{% endfor %}"
    );

    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);
    EXPECT_EQ(tpl.default_system_message(), "You are a helpful assistant.");
}

TEST(ChatTemplateDefaultSysTest, SkipsJinjaVariables) {
    Tokenizer tok = make_chat_tokenizer();
    // Template with dynamic system content (has "messages" reference) — should be skipped
    tok.set_chat_template_str(
        "<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n"
    );

    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);
    EXPECT_TRUE(tpl.default_system_message().empty());
}

TEST(ChatTemplateDefaultSysTest, NoTemplateNoDefault) {
    Tokenizer tok = make_chat_tokenizer();
    // No chat_template_str set

    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);
    EXPECT_TRUE(tpl.default_system_message().empty());
}

TEST(ChatTemplateDefaultSysTest, InjectDefaultWhenNoUserSystem) {
    Tokenizer tok = make_chat_tokenizer();
    tok.set_chat_template_str(
        "<|im_start|>system\nDefault system.<|im_end|>\n"
    );

    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);
    EXPECT_EQ(tpl.default_system_message(), "Default system.");

    // Apply without system message -> default system should be injected
    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto ids = tpl.apply(tok, msgs);

    // 3 im_start: default system + user + assistant prefix
    int im_start_count = std::count(ids.begin(), ids.end(), IM_START);
    EXPECT_EQ(im_start_count, 3);
}

TEST(ChatTemplateDefaultSysTest, NoInjectionWhenUserProvidesSystem) {
    Tokenizer tok = make_chat_tokenizer();
    tok.set_chat_template_str(
        "<|im_start|>system\nDefault system.<|im_end|>\n"
    );

    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);

    // Apply with explicit system message -> no double injection
    std::vector<ChatMessage> msgs = {
        {"system", "Custom system."},
        {"user", "Hi"},
    };
    auto ids = tpl.apply(tok, msgs);

    // 3 im_start: user-provided system + user + assistant prefix (no default)
    int im_start_count = std::count(ids.begin(), ids.end(), IM_START);
    EXPECT_EQ(im_start_count, 3);
}

// ---- apply_with_image (Gemma vision) ----

TEST(ChatTemplateVisionTest, GemmaImageTokens) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::GEMMA, tok);

    std::vector<ChatMessage> msgs = {{"user", "What is this?"}};
    auto ids = tpl.apply_with_image(tok, msgs, /*n_image_tokens=*/4);

    ASSERT_FALSE(ids.empty());
    // Should contain BOI and EOI
    EXPECT_TRUE(contains(ids, BOI));
    EXPECT_TRUE(contains(ids, EOI));
    // Should contain exactly 4 image soft tokens
    EXPECT_EQ(std::count(ids.begin(), ids.end(), IMG_SOFT), 4);
    // BOI should come before image tokens, which come before EOI
    int boi_pos = find_pos(ids, BOI);
    int eoi_pos = find_pos(ids, EOI);
    EXPECT_GT(boi_pos, -1);
    EXPECT_GT(eoi_pos, boi_pos);
}

TEST(ChatTemplateVisionTest, NonGemmaFallsBackToTextOnly) {
    Tokenizer tok = make_chat_tokenizer();
    ChatTemplate tpl;
    tpl.init(ChatTemplateFamily::CHATML, tok);

    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto with_image = tpl.apply_with_image(tok, msgs, 4);
    auto without_image = tpl.apply(tok, msgs);

    // For non-Gemma, apply_with_image should produce same output as apply
    EXPECT_EQ(with_image, without_image);
}

} // namespace
} // namespace imp
