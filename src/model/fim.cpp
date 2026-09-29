#include "model/fim.h"

#include "model/tokenizer.h"

#include <initializer_list>

namespace imp {

namespace {

// Known FIM marker texts per role (llama.cpp llama-vocab.cpp list): Qwen2.5/3-Coder, StarCoder
// and Granite, DeepSeek-Coder, CodeLlama (SPM with and without the U+2581 prefix), GLM-4.
// clang-format off
const std::initializer_list<const char*> kPreTexts = {"<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>",
    "<｜fim▁begin｜>", "<PRE>", "▁<PRE>", "<|code_prefix|>"};
const std::initializer_list<const char*> kSufTexts = {"<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>",
    "<｜fim▁hole｜>", "<SUF>", "▁<SUF>", "<|code_suffix|>"};
const std::initializer_list<const char*> kMidTexts = {"<|fim_middle|>", "<fim-middle>", "<fim_middle>",
    "<｜fim▁end｜>", "<MID>", "▁<MID>", "<|code_middle|>"};
const std::initializer_list<const char*> kPadTexts = {"<|fim_pad|>", "<fim-pad>", "<fim_pad>"};
const std::initializer_list<const char*> kRepTexts = {"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<reponame>"};
const std::initializer_list<const char*> kSepTexts = {"<|file_sep|>", "<file_sep>"};
// clang-format on

bool is_marker(const Tokenizer& tok, int32_t id) {
    return id >= 0 && id < tok.vocab_size() && (tok.is_added_token(id) || tok.is_special_token(id));
}

int32_t resolve(const Tokenizer& tok, int role, std::initializer_list<const char*> texts) {
    const int32_t meta = tok.fim_meta_id(role);
    if (meta >= 0 && meta < tok.vocab_size())
        return meta;
    for (const char* t : texts) {
        const int32_t id = tok.find_token(t);
        if (is_marker(tok, id))
            return id;
    }
    return -1;
}

void append(std::vector<int32_t>& out, const std::vector<int32_t>& ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

}  // namespace

FimTokens find_fim_tokens(const Tokenizer& tok) {
    FimTokens f;
    f.pre = resolve(tok, kFimPre, kPreTexts);
    f.suf = resolve(tok, kFimSuf, kSufTexts);
    f.mid = resolve(tok, kFimMid, kMidTexts);
    f.pad = resolve(tok, kFimPad, kPadTexts);
    f.rep = resolve(tok, kFimRep, kRepTexts);
    f.sep = resolve(tok, kFimSep, kSepTexts);
    return f;
}

std::vector<int32_t> build_fim_prompt(const Tokenizer& tok, const FimTokens& fim, const FimInput& in) {
    const auto enc = [&tok](const std::string& s) {
        return s.empty() ? std::vector<int32_t>{} : tok.encode(s, /*no_prefix=*/true);
    };
    std::vector<int32_t> out;
    if (tok.add_bos() && tok.bos_id() >= 0)
        out.push_back(tok.bos_id());
    const bool repo_level = !in.extra.empty();
    if (repo_level && fim.rep >= 0) {
        out.push_back(fim.rep);
        append(out, enc("myproject\n"));
    }
    for (const auto& chunk : in.extra) {
        if (fim.sep >= 0) {
            out.push_back(fim.sep);
            append(out, enc((chunk.filename.empty() ? std::string("tmp") : chunk.filename) + "\n"));
        } else {
            append(out, enc("\n\n--- snippet ---\n\n"));
        }
        append(out, enc(chunk.text));
    }
    if (repo_level && fim.sep >= 0) {
        out.push_back(fim.sep);
        append(out, enc("filename\n"));
    }
    out.push_back(fim.pre);
    append(out, enc(in.prefix));
    append(out, enc(in.prompt));
    out.push_back(fim.suf);
    append(out, enc(in.suffix));
    out.push_back(fim.mid);
    return out;
}

std::vector<std::string> fim_stop_texts(const Tokenizer& tok, const FimTokens& fim) {
    std::vector<std::string> out;
    for (const int32_t id : {fim.pre, fim.suf, fim.mid, fim.pad, fim.rep, fim.sep}) {
        if (id < 0)
            continue;
        std::string text = tok.decode_token(id);
        if (!text.empty())
            out.push_back(std::move(text));
    }
    return out;
}

}  // namespace imp
