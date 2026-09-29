// hf:// pure half: URI parsing, URL building, /api/models parsing, file selection.
// No I/O; the network and disk half is hf_fetch.cpp.
#include "model/hf_fetch.h"
#include "model/json_util.h"

#include <algorithm>
#include <string>
#include <vector>

namespace imp::hf {

namespace {

constexpr const char* kScheme = "hf://";

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool valid_repo_part(const std::string& p) {
    if (p.empty() || p.front() == '.' || p.front() == '-' || p.find("..") != std::string::npos)
        return false;
    return std::all_of(p.begin(), p.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
               c == '_' || c == '.';
    });
}

std::string join_names(const std::vector<std::string>& names, size_t limit = 20) {
    std::string out;
    for (size_t i = 0; i < names.size() && i < limit; ++i) {
        if (i)
            out += ", ";
        out += names[i];
    }
    if (names.size() > limit)
        out += ", ... (" + std::to_string(names.size()) + " total)";
    return out.empty() ? "(none)" : out;
}

bool is_safetensors_set_member(const std::string& name) {
    if (name.find('/') != std::string::npos)
        return false;
    return ends_with(name, ".safetensors") || ends_with(name, ".json") || ends_with(name, ".jinja") ||
           name == "tokenizer.model" || name == "merges.txt" || name == "vocab.txt";
}

}  // namespace

bool is_commit_sha(const std::string& s) {
    return s.size() == 40 && std::all_of(s.begin(), s.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

// A repo file name becomes a path under the snapshot dir: no absolute path, no "..".
bool is_safe_repo_path(const std::string& f) {
    if (f.empty() || f.front() == '/' || f.find('\\') != std::string::npos ||
        f.find('\0') != std::string::npos)
        return false;
    size_t start = 0;
    while (start <= f.size()) {
        size_t end = f.find('/', start);
        if (end == std::string::npos)
            end = f.size();
        const std::string seg = f.substr(start, end - start);
        if (seg.empty() || seg == "." || seg == "..")
            return false;
        start = end + 1;
    }
    return true;
}

bool is_hf_uri(const std::string& s) { return s.rfind(kScheme, 0) == 0; }

bool parse_hf_uri(const std::string& s, HfUri& out, std::string& err) {
    if (!is_hf_uri(s)) {
        err = "not an hf:// URI: " + s;
        return false;
    }
    std::string rest = s.substr(std::char_traits<char>::length(kScheme));
    std::string file;
    const size_t colon = rest.find(':');
    if (colon != std::string::npos) {
        file = rest.substr(colon + 1);
        rest = rest.substr(0, colon);
        if (!is_safe_repo_path(file)) {
            err = "invalid file in " + s + ": expected hf://<org>/<repo>:<file>";
            return false;
        }
    }
    const size_t slash = rest.find('/');
    if (slash == std::string::npos || rest.find('/', slash + 1) != std::string::npos ||
        !valid_repo_part(rest.substr(0, slash)) || !valid_repo_part(rest.substr(slash + 1))) {
        err = "invalid repo in " + s + ": expected hf://<org>/<repo>[:<file>]";
        return false;
    }
    out.repo = rest;
    out.file = file;
    return true;
}

std::string url_encode(const std::string& s, bool keep_slash) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        const bool unreserved = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                                c == '-' || c == '.' || c == '_' || c == '~' || (keep_slash && c == '/');
        if (unreserved) {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

std::string api_url(const std::string& endpoint, const std::string& repo, const std::string& rev) {
    return endpoint + "/api/models/" + repo + "/revision/" + url_encode(rev, false) + "?blobs=true";
}

std::string resolve_url(const std::string& endpoint, const std::string& repo, const std::string& rev,
                        const std::string& file) {
    return endpoint + "/" + repo + "/resolve/" + url_encode(rev, false) + "/" + url_encode(file, true);
}

bool parse_repo_info(const std::string& json, RepoInfo& out, std::string& err) {
    JsonParser p(json);
    JValue root = p.parse();
    if (!p.ok() || root.type != JType::OBJECT) {
        err = "malformed model info JSON";
        return false;
    }
    out = RepoInfo{};
    if (!jobj_get_string(root, "sha", out.commit) || !is_commit_sha(out.commit)) {
        err = "model info has no commit sha";
        return false;
    }
    if (const JValue* g = jobj_find(root, "gated"))
        out.gated = (g->type == JType::STRING && !g->str_val.empty()) ||
                    (g->type == JType::NUMBER && g->num_val != 0.0);
    const JValue* sib = jobj_find(root, "siblings");
    if (!sib || sib->type != JType::ARRAY) {
        err = "model info has no siblings list";
        return false;
    }
    for (const JValue& s : sib->arr) {
        if (s.type != JType::OBJECT)
            continue;
        RepoFile f;
        if (!jobj_get_string(s, "rfilename", f.name) || !is_safe_repo_path(f.name))
            continue;
        int64_t size = 0;
        if (jobj_get_int(s, "size", size) && size > 0)
            f.size = static_cast<uint64_t>(size);
        if (const JValue* lfs = jobj_find(s, "lfs"); lfs && lfs->type == JType::OBJECT) {
            std::string sha;
            if (jobj_get_string(*lfs, "sha256", sha) && sha.size() == 64)
                f.sha256 = sha;
            if (jobj_get_int(*lfs, "size", size) && size > 0)
                f.size = static_cast<uint64_t>(size);
        }
        out.files.push_back(std::move(f));
    }
    return true;
}

bool select_files(const RepoInfo& info, const std::string& selector, std::vector<RepoFile>& out,
                  std::string& load_rel, std::string& err) {
    out.clear();
    load_rel.clear();
    std::vector<std::string> ggufs, all;
    for (const auto& f : info.files) {
        all.push_back(f.name);
        if (ends_with(f.name, ".gguf"))
            ggufs.push_back(f.name);
    }
    if (!selector.empty()) {
        if (!ends_with(selector, ".gguf")) {
            err = "':" + selector + "' must name a .gguf file; omit ':<file>' for a SafeTensors repo";
            return false;
        }
        for (const auto& f : info.files) {
            if (f.name == selector) {
                out.push_back(f);
                load_rel = f.name;
                return true;
            }
        }
        err = "'" + selector + "' is not in the repo; .gguf files: " + join_names(ggufs);
        return false;
    }
    if (ggufs.size() > 1) {
        err = "repo has " + std::to_string(ggufs.size()) +
              " .gguf files, pick one with hf://<org>/<repo>:<file>: " + join_names(ggufs, 64);
        return false;
    }
    if (ggufs.size() == 1) {
        for (const auto& f : info.files)
            if (f.name == ggufs[0])
                out.push_back(f);
        load_rel = ggufs[0];
        return true;
    }
    bool has_weights = false;
    for (const auto& f : info.files) {
        if (!is_safetensors_set_member(f.name))
            continue;
        has_weights |= ends_with(f.name, ".safetensors");
        out.push_back(f);
    }
    if (!has_weights) {
        out.clear();
        err = "repo has no .gguf and no top-level .safetensors files; files: " + join_names(all);
        return false;
    }
    return true;
}

std::string repo_cache_dir(const std::string& cache_dir, const std::string& repo) {
    std::string name = "models--" + repo;
    const size_t slash = name.find('/');
    if (slash != std::string::npos)
        name.replace(slash, 1, "--");
    return cache_dir + "/" + name;
}
}  // namespace imp::hf
