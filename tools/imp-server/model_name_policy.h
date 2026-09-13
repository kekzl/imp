#pragma once

// Request `model` field allowlist (AUDIT_arch_2026 F2-1=F1-4): a name never reaches the
// filesystem as a path. Basename - looked up only among --models-dir entries. HfRepoId -
// "org/repo" (exactly one '/', no leading '/'/'.'/'~', no file extension), resolved only from
// the HF cache and must resolve inside it. Everything else (paths, "..", multi-segment) is
// rejected (404). Pure, tested in tests/test_model_name_policy.cpp.

#include <filesystem>
#include <string>

namespace imp_server {

enum class ModelNameKind { Basename, HfRepoId, Rejected };

inline bool has_model_file_extension(const std::string& s) {
    return s.find(".gguf") != std::string::npos || s.find(".safetensors") != std::string::npos;
}

inline ModelNameKind classify_model_name(const std::string& name) {
    if (name.empty() || name.front() == '/' || name.front() == '.' || name.front() == '~')
        return ModelNameKind::Rejected;
    if (name.find("..") != std::string::npos || name.find('\\') != std::string::npos)
        return ModelNameKind::Rejected;
    const auto slash = name.find('/');
    if (slash == std::string::npos)
        return ModelNameKind::Basename;
    if (name.find('/', slash + 1) != std::string::npos)
        return ModelNameKind::Rejected;  // more than one segment: a path, not a repo id
    if (slash == 0 || slash + 1 == name.size())
        return ModelNameKind::Rejected;
    if (has_model_file_extension(name))
        return ModelNameKind::Rejected;
    return ModelNameKind::HfRepoId;
}

// True iff `candidate` lies inside `base` after both are made canonical as far
// as they exist (weakly_canonical), compared element by element so "/models"
// does not contain "/models2/x".
inline bool path_within(const std::filesystem::path& base, const std::filesystem::path& candidate) {
    std::error_code ec;
    auto b = std::filesystem::weakly_canonical(base, ec);
    if (ec || b.empty())
        return false;
    auto c = std::filesystem::weakly_canonical(candidate, ec);
    if (ec || c.empty())
        return false;
    // A trailing separator ("/models/") normalises to an empty last element,
    // which would never equal the candidate's next element.
    if (!b.has_filename())
        b = b.parent_path();
    if (!c.has_filename())
        c = c.parent_path();
    auto bi = b.begin();
    auto ci = c.begin();
    for (; bi != b.end(); ++bi, ++ci) {
        if (ci == c.end() || *bi != *ci)
            return false;
    }
    return true;
}

}  // namespace imp_server
