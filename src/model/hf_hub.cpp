#include "model/hf_hub.h"
#include "core/logging.h"
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <array>

namespace imp {
namespace fs = std::filesystem;

static std::string exec_cmd(const std::string& cmd) {
    std::array<char, 4096> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
        result += buffer.data();
    pclose(pipe);
    // Trim trailing whitespace
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

bool hf_cli_available() {
    return system("which huggingface-cli > /dev/null 2>&1") == 0;
}

// Convert HF repo ID to cache directory name: org/model -> models--org--model
static std::string repo_to_cache_name(const std::string& repo_id) {
    std::string result = "models--";
    size_t slash = repo_id.find('/');
    if (slash != std::string::npos) {
        result += repo_id.substr(0, slash) + "--" + repo_id.substr(slash + 1);
    } else {
        result += repo_id;
    }
    return result;
}

static std::string resolve_hf_cache_dir() {
    if (const char* v = std::getenv("HUGGINGFACE_HUB_CACHE")) return v;
    if (const char* v = std::getenv("HF_HOME"))
        return std::string(v) + "/hub";
    if (const char* v = std::getenv("HOME"))
        return std::string(v) + "/.cache/huggingface/hub";
    return "";
}

std::string resolve_model_path(const std::string& model_id,
                                const std::string& revision) {
    // 1. If it's already a valid local path, return it
    if (fs::exists(model_id)) {
        return model_id;
    }

    // 2. If it doesn't look like a HF repo ID (no '/'), it's a bad path
    if (model_id.find('/') == std::string::npos) {
        IMP_LOG_ERROR("Model path does not exist: %s", model_id.c_str());
        return "";
    }

    // 3. Check HF cache first
    std::string cache_dir = resolve_hf_cache_dir();
    if (!cache_dir.empty()) {
        std::string cache_name = repo_to_cache_name(model_id);
        std::string model_cache = cache_dir + "/" + cache_name;
        if (fs::is_directory(model_cache)) {
            std::string snapshots = model_cache + "/snapshots";
            if (fs::is_directory(snapshots)) {
                // Use the latest snapshot (by modification time)
                std::string latest;
                std::filesystem::file_time_type latest_time{};
                for (const auto& entry : fs::directory_iterator(snapshots)) {
                    if (entry.is_directory()) {
                        auto t = entry.last_write_time();
                        if (latest.empty() || t > latest_time) {
                            latest = entry.path().string();
                            latest_time = t;
                        }
                    }
                }
                if (!latest.empty()) {
                    IMP_LOG_INFO("Found cached model: %s -> %s",
                                 model_id.c_str(), latest.c_str());
                    return latest;
                }
            }
        }
    }

    // 4. Try to download via huggingface-cli
    if (!hf_cli_available()) {
        IMP_LOG_ERROR("huggingface-cli not found. Install with: pip install huggingface-hub");
        IMP_LOG_ERROR("Or download the model manually: huggingface-cli download %s",
                      model_id.c_str());
        return "";
    }

    IMP_LOG_INFO("Downloading model from HuggingFace: %s", model_id.c_str());

    std::string cmd = "huggingface-cli download " + model_id;
    if (!revision.empty()) {
        cmd += " --revision " + revision;
    }
    // huggingface-cli download prints the snapshot path to stdout
    std::string result = exec_cmd(cmd + " 2>/dev/null");

    if (result.empty() || !fs::exists(result)) {
        IMP_LOG_ERROR("Failed to download model: %s", model_id.c_str());
        return "";
    }

    IMP_LOG_INFO("Downloaded model to: %s", result.c_str());
    return result;
}

std::string find_gguf_in_dir(const std::string& dir) {
    if (!fs::is_directory(dir)) return "";

    std::string best;
    std::uintmax_t best_size = 0;

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file() && !entry.is_symlink()) continue;
        auto path = entry.path();
        if (path.extension() != ".gguf") continue;
        // Skip mmproj files (vision encoder weights, not the main model)
        if (path.filename().string().find("mmproj") != std::string::npos) continue;
        auto sz = entry.file_size(ec);
        if (ec) continue;
        if (best.empty() || sz > best_size) {
            best = path.string();
            best_size = sz;
        }
    }
    return best;
}

std::string resolve_model_gguf(const std::string& model_id,
                                const std::string& revision) {
    // If it already ends with .gguf and exists, use directly
    if (model_id.size() > 5 &&
        model_id.substr(model_id.size() - 5) == ".gguf" &&
        fs::exists(model_id)) {
        return model_id;
    }

    std::string resolved = resolve_model_path(model_id, revision);
    if (resolved.empty()) return "";

    // If resolved path is a file, return it
    if (fs::is_regular_file(resolved)) return resolved;

    // If it's a directory, find the GGUF inside
    if (fs::is_directory(resolved)) {
        std::string gguf = find_gguf_in_dir(resolved);
        if (gguf.empty()) {
            IMP_LOG_ERROR("No .gguf files found in: %s", resolved.c_str());
        } else {
            IMP_LOG_INFO("Found GGUF: %s", gguf.c_str());
        }
        return gguf;
    }

    return resolved;
}

} // namespace imp
