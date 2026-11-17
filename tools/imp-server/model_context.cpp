#include "model_context.h"

#include "runtime/engine_config.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <utility>

namespace imp_server {

int advertised_context_tokens(int declared, int configured) {
    if (configured > 0)
        return configured;
    return declared > 0 ? std::min(declared, imp::kAutoMaxSeqLenCap) : 0;
}

namespace {

// Same object the loader reads (HFConfigLoader::load_config): text_config when it is an object.
int read_max_position_embeddings(const std::filesystem::path& config) {
    std::ifstream in(config);
    if (!in)
        return 0;
    const auto root = nlohmann::ordered_json::parse(in, nullptr, /*allow_exceptions=*/false);
    if (!root.is_object())
        return 0;
    const auto text = root.find("text_config");
    const auto& eff = (text != root.end() && text->is_object()) ? *text : root;
    const auto it = eff.find("max_position_embeddings");
    if (it == eff.end() || !it->is_number_integer())
        return 0;
    const auto v = it->get<long long>();
    return v > 0 && v <= 1LL << 30 ? static_cast<int>(v) : 0;
}

}  // namespace

int declared_context_tokens(const std::string& model_path) {
    namespace fs = std::filesystem;
    const fs::path config = fs::path(model_path) / "config.json";
    std::error_code ec;
    const auto mtime = fs::last_write_time(config, ec);
    if (ec)
        return 0;  // GGUF file or a dir without config.json
    static std::mutex mu;
    static std::map<std::string, std::pair<fs::file_time_type, int>> cache;
    std::lock_guard lock(mu);
    if (const auto hit = cache.find(model_path); hit != cache.end() && hit->second.first == mtime)
        return hit->second.second;
    const int tokens = read_max_position_embeddings(config);
    cache[model_path] = {mtime, tokens};
    return tokens;
}

}  // namespace imp_server
