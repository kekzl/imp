#include "runtime/spec_corpus.h"

#include "core/logging.h"
#include "model/tokenizer.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>
#include <vector>

namespace imp {

namespace {

bool read_text_file(const std::filesystem::path& p, size_t budget, std::string& out) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(p, ec);
    if (ec || size == 0 || size > budget)
        return false;
    std::ifstream f(p, std::ios::binary);
    if (!f)
        return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return out.find('\0') == std::string::npos;
}

}  // namespace

int load_ngram_corpus(const std::string& path, const Tokenizer& tok, SuffixDraftIndex& idx,
                      size_t max_bytes) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    std::error_code ec;
    if (fs::is_directory(path, ec)) {
        for (const auto& e :
             fs::recursive_directory_iterator(path, fs::directory_options::skip_permission_denied, ec))
            if (e.is_regular_file(ec))
                files.push_back(e.path());
        std::sort(files.begin(), files.end());  // directory order is not stable across filesystems
    } else if (fs::is_regular_file(path, ec)) {
        files.emplace_back(path);
    }
    size_t bytes = 0;
    int n_files = 0;
    const int32_t sep = -1;
    std::string text;
    for (const auto& p : files) {
        if (bytes >= max_bytes)
            break;
        if (!read_text_file(p, max_bytes - bytes, text))
            continue;
        const std::vector<int32_t> ids = tok.encode(text, /*no_prefix=*/true);
        idx.append(ids);
        idx.append({&sep, 1});
        bytes += text.size();
        ++n_files;
    }
    IMP_LOG_INFO("speculative.ngram_corpus: %d of %zu files, %zu bytes, %d tokens from %s", n_files,
                 files.size(), bytes, idx.size(), path.c_str());
    return n_files;
}

void install_ngram_corpus(const std::string& path, const Tokenizer* tok, int min_match, int max_match,
                          std::unordered_map<int, SuffixDraftIndex>& indexes) {
    if (path.empty() || tok == nullptr)
        return;
    auto& idx = indexes.try_emplace(kNgramCorpusKey, std::max(1, min_match), max_match).first->second;
    load_ngram_corpus(path, *tok, idx, size_t{64} << 20);
}

}  // namespace imp
