#pragma once

// --models-dir listing behind /v1/models, /props and model-less load: every .gguf file and
// SafeTensors directory, (display name, canonical path), sorted. Never throws: an unreadable,
// looping or vanishing entry is skipped and the rest is still listed (#2884: one mode-700 dir
// turned the whole /v1/models into a 500). Symlinks resolving outside the base are dropped.
// Tested in tests/test_model_name_policy.cpp.

#include "model/hf_hub.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace imp_server {

inline std::vector<std::pair<std::string, std::string>> scan_model_files(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, std::string>> results;
    if (dir.empty())
        return results;
    std::error_code ec;
    const auto base = fs::canonical(dir, ec);
    if (ec)
        return results;
    const std::string base_prefix = base.string() + "/";
    auto inside_base = [&](const fs::path& p, std::string& out) {
        std::error_code ec2;
        const auto real = fs::canonical(p, ec2);
        if (ec2)
            return false;
        out = real.string();
        return out.starts_with(base_prefix);
    };

    // skip_permission_denied: a directory the server user cannot open is not descended into.
    fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
    for (; !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        const auto& entry = *it;
        const auto& path = entry.path();
        std::error_code ec2;
        const bool is_dir = entry.is_directory(ec2) && !ec2;
        // Hidden entries (".flash-next-lm-only", ".cache") are scratch, not models.
        if (!path.filename().empty() && path.filename().string()[0] == '.') {
            if (is_dir)
                it.disable_recursion_pending();
            continue;
        }
        std::string real;
        if (path.extension() == ".gguf" && !path.string().contains(".no_exist")) {
            const bool is_file = (entry.is_regular_file(ec2) && !ec2) || (entry.is_symlink(ec2) && !ec2);
            if (is_file && inside_base(path, real))
                results.emplace_back(path.filename().string(), real);
        }
        if (is_dir && imp::is_safetensors_dir(path.string()) && inside_base(path, real))
            results.emplace_back(path.filename().string(), real);
    }
    std::sort(results.begin(), results.end());
    return results;
}

}  // namespace imp_server
