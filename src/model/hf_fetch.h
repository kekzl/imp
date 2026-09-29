#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Hugging Face fetcher for `--model hf://<org>/<repo>[:<file>]`. Runs inside the imp
// container (libcurl), writes the HF cache layout hf_hub.cpp reads, so a second start
// resolves from disk with no network call.
namespace imp::hf {

struct HfUri {
    std::string repo;  // "org/repo"
    std::string file;  // optional ":<file>" selector, "" = auto-select
};

// True for any string starting with "hf://"; parse_hf_uri() then validates it.
bool is_hf_uri(const std::string& s);
[[nodiscard]] bool parse_hf_uri(const std::string& s, HfUri& out, std::string& err);

struct RepoFile {
    std::string name;    // path inside the repo ("rfilename")
    uint64_t size = 0;   // bytes, 0 = unknown
    std::string sha256;  // LFS oid (lowercase hex), "" for non-LFS files
};

struct RepoInfo {
    std::string commit;  // resolved revision sha
    bool gated = false;
    std::vector<RepoFile> files;
};

// 40 lowercase hex chars (a commit sha, used as snapshot dir name).
bool is_commit_sha(const std::string& s);
// Relative path with no empty, "." or ".." segment: safe to join under the snapshot dir.
bool is_safe_repo_path(const std::string& p);

// URL-encodes every byte outside [A-Za-z0-9-._~]; `keep_slash` keeps '/' (file paths).
std::string url_encode(const std::string& s, bool keep_slash);
// <endpoint>/api/models/<repo>/revision/<rev>?blobs=true
std::string api_url(const std::string& endpoint, const std::string& repo, const std::string& rev);
// <endpoint>/<repo>/resolve/<rev>/<file>
std::string resolve_url(const std::string& endpoint, const std::string& repo, const std::string& rev,
                        const std::string& file);

// Parses the /api/models response (with ?blobs=true). False + err on malformed JSON.
[[nodiscard]] bool parse_repo_info(const std::string& json, RepoInfo& out, std::string& err);

// Files to download. `selector` names one .gguf; empty: the only .gguf, else the
// SafeTensors set (top-level *.safetensors, *.json, *.jinja, tokenizer.model, merges/vocab).
// `load_rel` is the path to load, relative to the snapshot ("" = the snapshot dir).
[[nodiscard]] bool select_files(const RepoInfo& info, const std::string& selector, std::vector<RepoFile>& out,
                  std::string& load_rel, std::string& err);

// <cache>/models--<org>--<repo>, the layout hf_hub.cpp resolves.
std::string repo_cache_dir(const std::string& cache_dir, const std::string& repo);

struct FetchOptions {
    std::string endpoint = "https://huggingface.co";
    std::string cache_dir;  // HF hub cache root, e.g. hf_cache_dir()
    std::string token;      // HF_TOKEN, sent as Bearer to the endpoint only
    std::string revision = "main";
    int retries = 3;                // resumed retries per file after a transfer error
    long low_speed_timeout_s = 60;  // abort when < 1 KiB/s for this long
};

struct FetchResult {
    bool ok = false;
    std::string path;        // file (GGUF) or snapshot dir (SafeTensors) to load
    bool cache_hit = false;  // true: served from disk, no network request made
    uint64_t bytes_downloaded = 0;
    std::string error;
};

FetchResult fetch(const HfUri& uri, const FetchOptions& opt);

// SHA-256 of a file as lowercase hex, "" on I/O error.
std::string sha256_file(const std::string& path);

// CLI entry: parses `uri`, reads HF_TOKEN / HF_ENDPOINT, fetches into hf_cache_dir().
// Returns the local path, or "" after logging the error.
std::string fetch_model(const std::string& uri, const std::string& revision);

}  // namespace imp::hf
