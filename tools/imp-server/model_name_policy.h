#pragma once

// What a request's `model` field may name (AUDIT_arch_2026 F2-1 = F1-4).
//
// Until 2026-09-05 `find_model_path` fell through to resolve_model_auto() for
// any name containing '/', and that resolver's first step is `fs::exists`:
// `{"model": "/any/readable/x.gguf"}` loaded that file, tearing down the
// resident model (server.model_swap is on by default), with the comment above
// the function promising the opposite. A request string never reaches the
// filesystem as a path now. Two shapes are allowed:
//
//   Basename  - "Qwen3-8B-Q8_0.gguf": looked up among the entries of
//               --models-dir and nowhere else.
//   HfRepoId  - "org/repo": exactly one '/', neither side empty, no leading
//               '/', '.' or '~', no model-file extension (the same shape
//               src/model/hf_hub.cpp accepts); resolved from the HuggingFace
//               cache only, and the resolved path must lie inside that cache.
//
// Everything else (absolute or relative paths, "..", multi-segment names) is
// Rejected and answers 404 like an unknown name. Pure functions, no I/O, so
// tests/test_model_name_policy.cpp pins them in the CPU lane.

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
