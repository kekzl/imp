#include "model/hf_fetch.h"
#include "core/logging.h"
#include "model/hf_hub.h"
#include "model/json_util.h"

#include <curl/curl.h>
#include <openssl/evp.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>

namespace imp::hf {
namespace fs = std::filesystem;

namespace {

constexpr const char* kScheme = "hf://";
constexpr const char* kCompleteMarker = ".imp_fetch_complete";
constexpr size_t kMaxErrorBody = 4096;
constexpr size_t kMaxApiBody = size_t{64} << 20;

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_hex40(const std::string& s) {
    return s.size() == 40 && std::all_of(s.begin(), s.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

bool valid_repo_part(const std::string& p) {
    if (p.empty() || p.front() == '.' || p.front() == '-' || p.find("..") != std::string::npos)
        return false;
    return std::all_of(p.begin(), p.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
               c == '_' || c == '.';
    });
}

// A repo file name becomes a path under the snapshot dir: no absolute path, no "..".
bool valid_file_name(const std::string& f) {
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

void ensure_curl_global() {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

struct CurlDeleter {
    void operator()(CURL* c) const { curl_easy_cleanup(c); }
};
struct SlistDeleter {
    void operator()(curl_slist* s) const { curl_slist_free_all(s); }
};
using CurlPtr = std::unique_ptr<CURL, CurlDeleter>;
using SlistPtr = std::unique_ptr<curl_slist, SlistDeleter>;

// Per-response HF error headers; reset at every status line (redirect hops).
struct HeaderCtx {
    std::string error_code;
    std::string error_message;
};

size_t header_cb(char* buf, size_t size, size_t n, void* ud) {
    auto* h = static_cast<HeaderCtx*>(ud);
    std::string line(buf, size * n);
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
        line.pop_back();
    if (line.rfind("HTTP/", 0) == 0) {
        h->error_code.clear();
        h->error_message.clear();
        return size * n;
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos)
        return size * n;
    std::string key = line.substr(0, colon);
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
    std::string val = line.substr(colon + 1);
    val.erase(0, val.find_first_not_of(" \t"));
    if (key == "x-error-code")
        h->error_code = val;
    else if (key == "x-error-message")
        h->error_message = val;
    return size * n;
}

CurlPtr make_handle(const std::string& url, curl_slist* headers, HeaderCtx* hctx, long low_speed_s) {
    ensure_curl_global();
    CurlPtr c(curl_easy_init());
    if (!c)
        return c;
    curl_easy_setopt(c.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(c.get(), CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c.get(), CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(c.get(), CURLOPT_USERAGENT, "imp-hf-fetch/1");
    curl_easy_setopt(c.get(), CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(c.get(), CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(c.get(), CURLOPT_LOW_SPEED_TIME, low_speed_s);
    curl_easy_setopt(c.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c.get(), CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(c.get(), CURLOPT_HEADERFUNCTION, header_cb);
    curl_easy_setopt(c.get(), CURLOPT_HEADERDATA, hctx);
    return c;
}

// curl drops a custom Authorization header on a cross-host redirect (LFS CDN), since 7.58.
SlistPtr auth_headers(const std::string& token) {
    curl_slist* s = nullptr;
    if (!token.empty())
        s = curl_slist_append(s, ("Authorization: Bearer " + token).c_str());
    return SlistPtr(s);
}

bool transient_curl(CURLcode rc) {
    switch (rc) {
        case CURLE_PARTIAL_FILE:
        case CURLE_OPERATION_TIMEDOUT:
        case CURLE_RECV_ERROR:
        case CURLE_SEND_ERROR:
        case CURLE_COULDNT_CONNECT:
        case CURLE_COULDNT_RESOLVE_HOST:
        case CURLE_GOT_NOTHING:
        case CURLE_HTTP2:
        case CURLE_HTTP2_STREAM:
        case CURLE_SSL_CONNECT_ERROR:
            return true;
        default:
            return false;
    }
}

std::string http_error_text(long status, const HeaderCtx& h, const std::string& body, bool have_token) {
    std::string msg = "HTTP " + std::to_string(status);
    if (!h.error_code.empty())
        msg += " " + h.error_code;
    if (!h.error_message.empty())
        msg += ": " + h.error_message;
    else if (!body.empty())
        msg += ": " + body.substr(0, 300);
    if (status == 401 || status == 403 || h.error_code == "GatedRepo" || h.error_code == "RepoNotFound")
        msg += have_token ? " (HF_TOKEN was sent: check it has access to this repo)"
                          : " (HF_TOKEN is not set: private and gated repos need it)";
    return msg;
}

struct GetCtx {
    std::string body;
};

size_t get_write_cb(char* p, size_t size, size_t n, void* ud) {
    auto* g = static_cast<GetCtx*>(ud);
    if (g->body.size() + size * n > kMaxApiBody)
        return 0;
    g->body.append(p, size * n);
    return size * n;
}

bool http_get(const std::string& url, const FetchOptions& opt, std::string& body, std::string& err) {
    HeaderCtx hctx;
    SlistPtr headers = auth_headers(opt.token);
    CurlPtr c = make_handle(url, headers.get(), &hctx, opt.low_speed_timeout_s);
    if (!c) {
        err = "curl_easy_init failed";
        return false;
    }
    GetCtx g;
    curl_easy_setopt(c.get(), CURLOPT_WRITEFUNCTION, get_write_cb);
    curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &g);
    CURLcode rc = curl_easy_perform(c.get());
    if (rc != CURLE_OK) {
        err = std::string("GET ") + url + ": " + curl_easy_strerror(rc);
        return false;
    }
    long status = 0;
    curl_easy_getinfo(c.get(), CURLINFO_RESPONSE_CODE, &status);
    if (status < 200 || status >= 300) {
        err = "GET " + url + ": " + http_error_text(status, hctx, g.body, !opt.token.empty());
        return false;
    }
    body = std::move(g.body);
    return true;
}

// Body goes to the file only once the final response is 200/206; an error body is kept
// in memory for the message, so a failed request never creates or grows `.part`.
struct DlCtx {
    CURL* curl = nullptr;
    std::string part_path;
    uint64_t offset = 0;
    FILE* f = nullptr;
    bool decided = false;
    bool write_to_file = false;
    std::string err_body;
    uint64_t written = 0;
    uint64_t total = 0;
    int last_decile = -1;
    std::string name;
};

size_t dl_write_cb(char* p, size_t size, size_t n, void* ud) {
    auto* d = static_cast<DlCtx*>(ud);
    const size_t len = size * n;
    if (!d->decided) {
        d->decided = true;
        long status = 0;
        curl_easy_getinfo(d->curl, CURLINFO_RESPONSE_CODE, &status);
        if (status == 206) {
            d->f = std::fopen(d->part_path.c_str(), d->offset > 0 ? "ab" : "wb");
        } else if (status == 200) {
            d->offset = 0;  // server ignored the Range: start over
            d->f = std::fopen(d->part_path.c_str(), "wb");
        }
        d->write_to_file = (d->f != nullptr);
        if ((status == 200 || status == 206) && !d->f)
            return 0;
    }
    if (!d->write_to_file) {
        if (d->err_body.size() < kMaxErrorBody)
            d->err_body.append(p, std::min(len, kMaxErrorBody - d->err_body.size()));
        return len;
    }
    if (std::fwrite(p, 1, len, d->f) != len)
        return 0;
    d->written += len;
    if (d->total > 0) {
        const int decile = static_cast<int>((d->offset + d->written) * 10 / d->total);
        if (decile != d->last_decile) {
            d->last_decile = decile;
            IMP_LOG_INFO("hf-fetch: %s %d%% (%.1f / %.1f MiB)", d->name.c_str(), decile * 10,
                         (d->offset + d->written) / 1048576.0, d->total / 1048576.0);
        }
    }
    return len;
}

uint64_t file_size_or_zero(const std::string& p) {
    std::error_code ec;
    auto s = fs::file_size(p, ec);
    return ec ? 0 : static_cast<uint64_t>(s);
}

enum class DlStatus { Ok, Transient, Fatal };

// One attempt: resumes from the size of `part` with a Range request.
DlStatus download_once(const std::string& url, const std::string& part, const RepoFile& rf,
                       const FetchOptions& opt, uint64_t& bytes, std::string& err) {
    uint64_t offset = file_size_or_zero(part);
    if (rf.size > 0 && offset > rf.size) {
        fs::remove(part);
        offset = 0;
    }
    if (rf.size > 0 && offset == rf.size)
        return DlStatus::Ok;  // complete from an earlier run, verified by the caller
    if (offset > 0)
        IMP_LOG_INFO("hf-fetch: resuming %s at byte %llu", rf.name.c_str(), (unsigned long long)offset);

    HeaderCtx hctx;
    SlistPtr headers = auth_headers(opt.token);
    CurlPtr c = make_handle(url, headers.get(), &hctx, opt.low_speed_timeout_s);
    if (!c) {
        err = "curl_easy_init failed";
        return DlStatus::Fatal;
    }
    DlCtx d;
    d.curl = c.get();
    d.part_path = part;
    d.offset = offset;
    d.total = rf.size;
    d.name = rf.name;
    const std::string range = std::to_string(offset) + "-";
    if (offset > 0)
        curl_easy_setopt(c.get(), CURLOPT_RANGE, range.c_str());
    curl_easy_setopt(c.get(), CURLOPT_WRITEFUNCTION, dl_write_cb);
    curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &d);
    const CURLcode rc = curl_easy_perform(c.get());
    if (d.f)
        std::fclose(d.f);
    bytes += d.written;
    long status = 0;
    curl_easy_getinfo(c.get(), CURLINFO_RESPONSE_CODE, &status);

    if (rc != CURLE_OK) {
        err = rf.name + ": " + curl_easy_strerror(rc);
        return transient_curl(rc) ? DlStatus::Transient : DlStatus::Fatal;
    }
    if (status != 200 && status != 206) {
        err = rf.name + ": " + http_error_text(status, hctx, d.err_body, !opt.token.empty());
        return (status >= 500 || status == 429) ? DlStatus::Transient : DlStatus::Fatal;
    }
    if (!d.decided) {  // empty body: an empty file is valid, create it
        FILE* f = std::fopen(part.c_str(), offset > 0 ? "ab" : "wb");
        if (!f) {
            err = "cannot write " + part;
            return DlStatus::Fatal;
        }
        std::fclose(f);
    }
    return DlStatus::Ok;
}

bool write_text_atomic(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        if (!o)
            return false;
        o << text;
        if (!o.flush())
            return false;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    return !ec;
}

std::string read_first_line(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    if (!in || !std::getline(in, line))
        return "";
    while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
        line.pop_back();
    return line;
}

bool is_safetensors_set_member(const std::string& name) {
    if (name.find('/') != std::string::npos)
        return false;
    return ends_with(name, ".safetensors") || ends_with(name, ".json") || ends_with(name, ".jinja") ||
           name == "tokenizer.model" || name == "merges.txt" || name == "vocab.txt";
}

}  // namespace

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
        if (!valid_file_name(file)) {
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
    if (!jobj_get_string(root, "sha", out.commit) || !is_hex40(out.commit)) {
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
        if (!jobj_get_string(s, "rfilename", f.name) || !valid_file_name(f.name))
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

std::string sha256_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return "";
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) {
        std::fclose(f);
        return "";
    }
    std::vector<unsigned char> buf(size_t{8} << 20);
    size_t n;
    bool ok = true;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0)
        ok &= EVP_DigestUpdate(ctx.get(), buf.data(), n) == 1;
    ok &= !std::ferror(f);
    std::fclose(f);
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    if (!ok || EVP_DigestFinal_ex(ctx.get(), md, &len) != 1)
        return "";
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (unsigned int i = 0; i < len; ++i) {
        out += hex[md[i] >> 4];
        out += hex[md[i] & 15];
    }
    return out;
}

FetchResult fetch(const HfUri& uri, const FetchOptions& opt) {
    FetchResult r;
    if (opt.cache_dir.empty()) {
        r.error = "no HF cache dir (set HF_HOME or HUGGINGFACE_HUB_CACHE)";
        return r;
    }
    const std::string rev = opt.revision.empty() ? "main" : opt.revision;
    if (!valid_file_name(rev)) {
        r.error = "invalid revision: " + rev;
        return r;
    }
    const std::string repo_dir = repo_cache_dir(opt.cache_dir, uri.repo);
    const std::string ref_file = repo_dir + "/refs/" + rev;
    const std::string what = uri.repo + "@" + rev;

    // Offline first: a completed fetch of this selection answers with no network request.
    std::string commit = is_hex40(rev) ? rev : read_first_line(ref_file);
    if (is_hex40(commit)) {
        const std::string snap = repo_dir + "/snapshots/" + commit;
        std::string hit;
        if (!uri.file.empty()) {
            if (fs::is_regular_file(snap + "/" + uri.file))
                hit = snap + "/" + uri.file;
        } else if (fs::is_regular_file(snap + "/" + kCompleteMarker)) {
            const std::string rel = read_first_line(snap + "/" + kCompleteMarker);
            hit = (rel.empty() || rel == ".") ? snap : snap + "/" + rel;
        }
        if (!hit.empty() && fs::exists(hit)) {
            IMP_LOG_INFO("hf-fetch: cache hit %s (%s), no download: %s", what.c_str(), commit.c_str(),
                         hit.c_str());
            r.ok = true;
            r.cache_hit = true;
            r.path = hit;
            return r;
        }
    }

    std::string body;
    std::string err;
    if (!http_get(api_url(opt.endpoint, uri.repo, rev), opt, body, err)) {
        r.error = "model info for " + what + ": " + err;
        return r;
    }
    RepoInfo info;
    if (!parse_repo_info(body, info, err)) {
        r.error = what + ": " + err;
        return r;
    }
    if (info.gated && opt.token.empty()) {
        r.error = uri.repo + " is gated: set HF_TOKEN to a token that has accepted its terms (" +
                  opt.endpoint + "/" + uri.repo + ")";
        return r;
    }
    std::vector<RepoFile> files;
    std::string load_rel;
    if (!select_files(info, uri.file, files, load_rel, err)) {
        r.error = what + ": " + err;
        return r;
    }

    const std::string snap = repo_dir + "/snapshots/" + info.commit;
    std::error_code ec;
    fs::create_directories(snap, ec);
    fs::create_directories(fs::path(ref_file).parent_path(), ec);
    if (ec) {
        r.error = "cannot create " + snap + ": " + ec.message() + " (process uid " +
                  std::to_string(getuid()) +
                  "; a bind-mounted model dir needs docker run --user $(id -u):$(id -g))";
        return r;
    }

    for (const auto& rf : files) {
        const std::string dest = snap + "/" + rf.name;
        if (fs::is_regular_file(dest) && (rf.size == 0 || file_size_or_zero(dest) == rf.size))
            continue;
        fs::create_directories(fs::path(dest).parent_path(), ec);
        const std::string part = dest + ".part";
        const std::string url = resolve_url(opt.endpoint, uri.repo, info.commit, rf.name);
        IMP_LOG_INFO("hf-fetch: downloading %s/%s (%.1f MiB)", uri.repo.c_str(), rf.name.c_str(),
                     rf.size / 1048576.0);
        DlStatus st = DlStatus::Fatal;
        for (int attempt = 0; attempt <= std::max(0, opt.retries); ++attempt) {
            st = download_once(url, part, rf, opt, r.bytes_downloaded, err);
            if (st != DlStatus::Transient)
                break;
            IMP_LOG_WARN("hf-fetch: %s (attempt %d), will resume", err.c_str(), attempt + 1);
        }
        if (st == DlStatus::Transient) {
            r.error = err + "; kept " + std::to_string(file_size_or_zero(part)) + " bytes in " + part +
                      " to resume on the next start";
            return r;
        }
        if (st == DlStatus::Fatal) {
            fs::remove(part, ec);
            r.error = err;
            return r;
        }
        const uint64_t got = file_size_or_zero(part);
        if (rf.size > 0 && got != rf.size) {
            fs::remove(part, ec);
            r.error = rf.name + ": size " + std::to_string(got) + " != expected " + std::to_string(rf.size) +
                      "; partial file removed";
            return r;
        }
        if (!rf.sha256.empty()) {
            const std::string sha = sha256_file(part);
            if (sha != rf.sha256) {
                fs::remove(part, ec);
                r.error = rf.name + ": sha256 mismatch, expected " + rf.sha256 + " got " +
                          (sha.empty() ? "(unreadable)" : sha) + "; partial file removed";
                return r;
            }
            IMP_LOG_INFO("hf-fetch: sha256 ok %s %s", rf.name.c_str(), sha.c_str());
        }
        fs::rename(part, dest, ec);
        if (ec) {
            r.error = "rename " + part + ": " + ec.message();
            return r;
        }
    }

    if (rev != info.commit && !write_text_atomic(ref_file, info.commit + "\n")) {
        r.error = "cannot write " + ref_file;
        return r;
    }
    if (uri.file.empty() &&
        !write_text_atomic(snap + "/" + kCompleteMarker, (load_rel.empty() ? "." : load_rel) + "\n")) {
        r.error = "cannot write the completion marker in " + snap;
        return r;
    }
    r.ok = true;
    r.path = load_rel.empty() ? snap : snap + "/" + load_rel;
    IMP_LOG_INFO("hf-fetch: fetched %s (%s): %llu bytes downloaded -> %s", what.c_str(), info.commit.c_str(),
                 (unsigned long long)r.bytes_downloaded, r.path.c_str());
    return r;
}

std::string fetch_model(const std::string& uri_str, const std::string& revision) {
    HfUri uri;
    std::string err;
    if (!parse_hf_uri(uri_str, uri, err)) {
        IMP_LOG_ERROR("hf-fetch: %s", err.c_str());
        return "";
    }
    FetchOptions opt;
    opt.cache_dir = hf_cache_dir();
    if (!revision.empty())
        opt.revision = revision;
    if (const char* t = std::getenv("HF_TOKEN"); t && *t)
        opt.token = t;
    if (const char* e = std::getenv("HF_ENDPOINT"); e && *e) {
        opt.endpoint = e;
        while (!opt.endpoint.empty() && opt.endpoint.back() == '/')
            opt.endpoint.pop_back();
    }
    FetchResult r = fetch(uri, opt);
    if (!r.ok) {
        IMP_LOG_ERROR("hf-fetch: %s", r.error.c_str());
        return "";
    }
    return r.path;
}

}  // namespace imp::hf
