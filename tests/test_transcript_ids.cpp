// TranscriptIdStore + ChatTemplate::tokenize_rendered: a resent reply keeps the model's own
// BPE split (roadmap row 15). The fixture vocabulary makes "prepref" canonical as
// [prep][ref] while the "model" wrote [pre][pref].
#include <gtest/gtest.h>

#include "model/chat_template.h"
#include "model/tokenizer.h"
#include "model/transcript_ids.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace imp;

namespace {

constexpr int kBos = 1;
constexpr int kImEnd = 260;

// Byte fallback + ChatML specials + a merge chain: "re" 6 > "pr" 5 = "ref" 5 > "pre" 4 >
// "prep" 3 > "pref" 2. Merge order on "prepref": re, re, ref, pre, prep -> [prep][ref].
Tokenizer make_tokenizer() {
    std::vector<std::string> tokens = {"<unk>", "<s>", "</s>"};
    std::vector<float> scores = {0.0f, 0.0f, 0.0f};
    for (int b = 0; b < 256; b++) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.push_back(buf);
        scores.push_back(-10.0f);
    }
    tokens.push_back("<|im_start|>");  // 259
    scores.push_back(0.0f);
    tokens.push_back("<|im_end|>");  // 260
    scores.push_back(0.0f);
    const std::pair<const char*, float> merges[] = {{"pr", 5.0f},   {"re", 6.0f},  {"pre", 4.0f},
                                                    {"prep", 3.0f}, {"ref", 5.0f}, {"pref", 2.0f}};
    for (const auto& [text, score] : merges) {
        tokens.push_back(text);
        scores.push_back(score);
    }
    Tokenizer tok;
    tok.load_vocab(tokens, scores, kBos, /*eos_id=*/2);
    // GGUF token types: UNKNOWN=2, CONTROL=3, BYTE=6, NORMAL=1. Control tokens are what
    // tokenize_rendered splits the render on.
    std::vector<int32_t> types(tokens.size(), 1);
    types[0] = 2;
    types[1] = types[2] = types[kImEnd - 1] = types[kImEnd] = 3;
    for (int b = 0; b < 256; b++)
        types[3 + b] = 6;
    tok.load_token_types(types);
    tok.set_type("spm");
    tok.set_add_bos(true);
    tok.set_add_space_prefix(false);
    return tok;
}

const char* kTemplate =
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

std::vector<int32_t> ids_of(const Tokenizer& tok, std::initializer_list<const char*> pieces) {
    std::vector<int32_t> out;
    for (const char* p : pieces) {
        const int32_t id = tok.find_token(p);
        EXPECT_GE(id, 0) << p;
        out.push_back(id);
    }
    return out;
}

bool starts_with(const std::vector<int32_t>& v, const std::vector<int32_t>& prefix) {
    return v.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), v.begin());
}

}  // namespace

TEST(TranscriptIdStoreTest, LongestPrefixWinsAndCapIsFifo) {
    TranscriptIdStore store(/*max_entries=*/2);
    store.remember("ab", {1, 2});
    store.remember("abcd", {1, 2, 3, 4});
    std::vector<int32_t> ids;
    EXPECT_EQ(store.lookup("abcdef", ids), 4u);
    EXPECT_EQ(ids, (std::vector<int32_t>{1, 2, 3, 4}));
    ids.clear();
    EXPECT_EQ(store.lookup("abx", ids), 2u);  // the shorter transcript still serves a branch
    EXPECT_EQ(ids, (std::vector<int32_t>{1, 2}));
    ids.clear();
    EXPECT_EQ(store.lookup("zz", ids), 0u);
    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(store.lookup("a", ids), 0u);  // shorter than every entry
    store.remember("abcd", {1, 2, 3, 4});   // same text twice: one entry
    EXPECT_EQ(store.size(), 2u);
    store.remember("q", {9});  // cap 2: the oldest ("ab") leaves
    EXPECT_EQ(store.size(), 2u);
    ids.clear();
    EXPECT_EQ(store.lookup("abx", ids), 0u);
    EXPECT_EQ(store.lookup("q!", ids), 1u);
    store.remember("", {});  // empty transcripts are not entries
    EXPECT_EQ(store.size(), 2u);
}

TEST(TranscriptIdStoreTest, ResentReplyKeepsTheModelSplit) {
    Tokenizer tok = make_tokenizer();
    const auto canonical = ids_of(tok, {"prep", "ref"});
    const auto model_split = ids_of(tok, {"pre", "pref"});
    ASSERT_EQ(tok.encode("prepref", /*no_prefix=*/true), canonical);
    ASSERT_EQ(tok.decode(model_split), "prepref");

    ChatTemplate tpl;
    ASSERT_TRUE(tpl.init(ChatTemplateFamily::CHATML, tok, kTemplate));
    auto store = std::make_shared<TranscriptIdStore>();
    tpl.set_transcript_store(store);

    // Turn 1: the prompt as tokenized (BOS pushed by tokenize_rendered, no literal in the
    // render), then the reply as the model forwarded it.
    std::vector<ChatMessage> turn1 = {{"user", "hi"}};
    std::vector<int32_t> transcript = tpl.apply(tok, turn1);
    ASSERT_FALSE(transcript.empty());
    EXPECT_EQ(transcript[0], kBos);
    transcript.insert(transcript.end(), model_split.begin(), model_split.end());
    // The engine's span ends in the stop ids the template renders itself: dropped.
    std::vector<int32_t> forwarded = transcript;
    forwarded.push_back(kImEnd);
    forwarded.push_back(/*eos=*/2);
    store->remember_forwarded(tok, tpl.stop_token_ids(), forwarded);

    // Turn 2 resends the reply verbatim: the ids continue the model's split, then the tail.
    std::vector<ChatMessage> turn2 = {{"user", "hi"}, {"assistant", "prepref"}, {"user", "more"}};
    const std::vector<int32_t> spliced = tpl.apply(tok, turn2);
    EXPECT_TRUE(starts_with(spliced, transcript));
    EXPECT_EQ(spliced[transcript.size()], kImEnd);
    // Same text as the canonical tokenization, so the model reads the same prompt.
    ChatTemplate plain;
    ASSERT_TRUE(plain.init(ChatTemplateFamily::CHATML, tok, kTemplate));
    const std::vector<int32_t> canonical_ids = plain.apply(tok, turn2);
    EXPECT_NE(spliced, canonical_ids);
    EXPECT_EQ(TranscriptIdStore::text_of(tok, spliced), TranscriptIdStore::text_of(tok, canonical_ids));
    EXPECT_EQ(spliced.size(), canonical_ids.size());
    // The canonical arm carries [prep][ref] where the spliced one carries [pre][pref].
    auto find_pair = [](const std::vector<int32_t>& v, const std::vector<int32_t>& p) {
        return std::search(v.begin(), v.end(), p.begin(), p.end()) != v.end();
    };
    EXPECT_TRUE(find_pair(canonical_ids, canonical));
    EXPECT_FALSE(find_pair(canonical_ids, model_split));
    EXPECT_TRUE(find_pair(spliced, model_split));

    // A branch that changes the reply gets the canonical tokens (no stored prefix matches).
    std::vector<ChatMessage> other = {{"user", "hi"}, {"assistant", "prepXref"}, {"user", "more"}};
    EXPECT_EQ(tpl.apply(tok, other), plain.apply(tok, other));
    // A reply that EXTENDS the stored text (finish=length drops the final token from the
    // span) still splices: the tail is tokenized from the seam, the text stays identical.
    std::vector<ChatMessage> longer = {{"user", "hi"}, {"assistant", "prepref!"}, {"user", "more"}};
    const std::vector<int32_t> longer_ids = tpl.apply(tok, longer);
    EXPECT_TRUE(starts_with(longer_ids, transcript));
    EXPECT_EQ(TranscriptIdStore::text_of(tok, longer_ids),
              TranscriptIdStore::text_of(tok, plain.apply(tok, longer)));
    // A different first message shares nothing.
    std::vector<ChatMessage> fresh = {{"user", "ho"}, {"assistant", "prepref"}};
    EXPECT_EQ(tpl.apply(tok, fresh), plain.apply(tok, fresh));
}
