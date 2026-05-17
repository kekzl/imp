#include "model/hf_hub.h"
#include "core/logging.h"
#include <cstdlib>
#include <filesystem>

namespace imp {
namespace fs = std::filesystem;

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
    if (const char* v = std::getenv("HUGGINGFACE_HUB_CACHE"))
        return v;
    if (const char* v = std::getenv("HF_HOME"))
        return std::string(v) + "/hub";
    if (const char* v = std::getenv("HOME"))
        return std::string(v) + "/.cache/huggingface/hub";
    return "";
}

std::string resolve_model_path(const std::string& model_id, const std::string& revision) {
    (void)revision;  // accepted for API compat; selecting non-default revisions
                     // requires a pre-populated cache (this resolver doesn't fetch).
    // 1. If it's already a valid local path, return it.
    if (fs::exists(model_id)) {
        return model_id;
    }

    // 2. If it doesn't look like a HF repo ID (no '/'), it's a bad path.
    if (model_id.find('/') == std::string::npos) {
        IMP_LOG_ERROR("Model path does not exist: %s", model_id.c_str());
        return "";
    }

    // 3. Check the HF cache.
    std::string cache_dir = resolve_hf_cache_dir();
    if (!cache_dir.empty()) {
        std::string cache_name = repo_to_cache_name(model_id);
        std::string model_cache = cache_dir + "/" + cache_name;
        if (fs::is_directory(model_cache)) {
            std::string snapshots = model_cache + "/snapshots";
            if (fs::is_directory(snapshots)) {
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
                    IMP_LOG_INFO("Found cached model: %s -> %s", model_id.c_str(), latest.c_str());
                    return latest;
                }
            }
        }
    }

    // 4. No fetcher: imp's host policy keeps Python tooling off the host and
    //    container. If the model isn't cached, the user pre-stages it.
    IMP_LOG_ERROR(
        "Model %s not found locally and not in the HF cache (%s). Stage it manually, e.g.:",
        model_id.c_str(), cache_dir.empty() ? "(no HF cache dir)" : cache_dir.c_str());
    IMP_LOG_ERROR("  git clone https://huggingface.co/%s <local-dir>", model_id.c_str());
    IMP_LOG_ERROR("  (or use the HF hub CLI on another machine and copy the cache)");
    return "";
}

std::string find_gguf_in_dir(const std::string& dir) {
    if (!fs::is_directory(dir))
        return "";

    std::string best;
    std::uintmax_t best_size = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            std::uintmax_t size = entry.file_size();
            if (size > best_size) {
                best = entry.path().string();
                best_size = size;
            }
        }
    }
    return best;
}

std::string resolve_model_gguf(const std::string& model_id, const std::string& revision) {
    std::string path = resolve_model_path(model_id, revision);
    if (path.empty())
        return "";

    if (fs::is_regular_file(path)) {
        // Direct file path (e.g. /path/to/model.gguf)
        return path;
    }

    if (fs::is_directory(path)) {
        std::string gguf = find_gguf_in_dir(path);
        if (gguf.empty()) {
            IMP_LOG_ERROR("No .gguf file found in directory: %s", path.c_str());
            return "";
        }
        IMP_LOG_INFO("Resolved %s to: %s", model_id.c_str(), gguf.c_str());
        return gguf;
    }

    IMP_LOG_ERROR("Path is neither a file nor a directory: %s", path.c_str());
    return "";
}

bool is_safetensors_dir(const std::string& dir) {
    if (!fs::is_directory(dir))
        return false;

    if (fs::exists(dir + "/model.safetensors") ||
        fs::exists(dir + "/model.safetensors.index.json")) {
        return true;
    }
    return false;
}

std::string resolve_model_auto(const std::string& model_id, ImpModelFormat& out_format,
                               const std::string& revision) {
    std::string path = resolve_model_path(model_id, revision);
    if (path.empty())
        return "";

    if (fs::is_regular_file(path)) {
        if (path.size() >= 5 && path.substr(path.size() - 5) == ".gguf") {
            out_format = IMP_FORMAT_GGUF;
            return path;
        }
        IMP_LOG_ERROR("Unsupported file extension: %s", path.c_str());
        return "";
    }

    if (fs::is_directory(path)) {
        if (is_safetensors_dir(path)) {
            IMP_LOG_INFO("Detected SafeTensors directory: %s", path.c_str());
            out_format = IMP_FORMAT_SAFETENSORS;
            return path;
        }

        std::string gguf = find_gguf_in_dir(path);
        if (!gguf.empty()) {
            IMP_LOG_INFO("Resolved %s to GGUF: %s", model_id.c_str(), gguf.c_str());
            out_format = IMP_FORMAT_GGUF;
            return gguf;
        }

        IMP_LOG_ERROR("Directory has neither .safetensors nor .gguf: %s", path.c_str());
        return "";
    }

    IMP_LOG_ERROR("Path is neither a file nor a directory: %s", path.c_str());
    return "";
}

}  // namespace imp
