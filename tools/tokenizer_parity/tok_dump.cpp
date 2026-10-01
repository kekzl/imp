// imp token ids for the parity harness (run.sh): one line per corpus record, "<roundtrip> <ids...>".
// Corpus records are separated by 0x1E. CHAT=1: ChatTemplate::apply on chat_convs() instead.
// CPU only: load_gguf / load_safetensors read metadata and the tokenizer, no device upload.
#include "model/chat_template.h"
#include "model/gguf_loader.h"
#include "model/model.h"
#include "model/safetensors_loader.h"
#include "model/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Same conversations as CONVS in parity.py.
std::vector<std::vector<imp::ChatMessage>> chat_convs() {
    return {
        {{"user", "Hello!"}},
        {{"system", "You are helpful."},
         {"user", "Hi \xf0\x9f\x98\x80,\n what's 2+2?"},
         {"assistant", "It is 4."},
         {"user", "  and\n\n5+5? "}},
        {{"user",
          "\xc3\x9c"
          "bersetze: na\xc3\xafve caf\xc3\xa9 \xe2\x80\x94 \xe2\x80\x98quotes\xe2\x80\x99"}},
    };
}

// "@@ " marks result lines; the library logs to stdout too.
void print_ids(bool roundtrip, const std::vector<int32_t>& ids) {
    printf("@@ %d", roundtrip ? 1 : 0);
    for (int32_t id : ids)
        printf(" %d", id);
    printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: tok_dump <model.gguf|model_dir> <corpus.bin>\n");
        return 2;
    }
    const std::string path = argv[1];
    const bool gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
    std::unique_ptr<imp::Model> m = gguf ? imp::load_gguf(path) : imp::load_safetensors(path);
    if (!m || !m->tokenizer()) {
        fprintf(stderr, "load failed: %s\n", path.c_str());
        return 1;
    }
    const imp::Tokenizer& tok = *m->tokenizer();

    if (getenv("EOS")) {  // "@@EOS <eos_id> <eos_ids...>": the stop set a server would use (#2377)
        printf("@@EOS %d", tok.eos_id());
        for (int32_t id : tok.eos_ids())
            printf(" %d", id);
        printf("\n");
        return 0;
    }

    if (getenv("CHAT")) {
        auto family = imp::ChatTemplate::detect_family(tok.chat_template_str());
        if (family == imp::ChatTemplateFamily::RAW)
            family = imp::ChatTemplate::default_family_for_arch(m->config().arch);
        imp::ChatTemplate tpl;
        if (!tpl.init(family, tok, tok.chat_template_str())) {
            fprintf(stderr, "chat template init failed\n");
            return 1;
        }
        for (const auto& conv : chat_convs())
            print_ids(true, tpl.apply(tok, conv));
        return 0;
    }

    std::ifstream f(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string all = ss.str();
    for (size_t pos = 0, end; (end = all.find('\x1e', pos)) != std::string::npos; pos = end + 1) {
        const std::string text = all.substr(pos, end - pos);
        auto ids = tok.encode(text, /*no_prefix=*/false);
        if (!ids.empty() && ids[0] == tok.bos_id() && tok.add_bos())
            ids.erase(ids.begin());  // HF add_special_tokens=False
        print_ids(tok.decode(ids) == text, ids);
    }
    return 0;
}
