// hf:// fetcher (#2200) against an in-process cpp-httplib mock of the HF API and resolve
// endpoints on 127.0.0.1: URL building, file selection, Range resume, sha256 mismatch,
// gated-repo errors, cache hit with zero requests. CPU lane, no external network.

#include <gtest/gtest.h>

#include <httplib.h>

#include "model/hf_fetch.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using namespace imp::hf;

constexpr const char* kCommit = "0123456789abcdef0123456789abcdef01234567";

std::string make_blob(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::string s(n, '\0');
    for (auto& c : s)
        c = static_cast<char>(rng() & 0xff);
    return s;
}

std::string slurp(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void spit(const std::string& p, const std::string& data) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o << data;
}

struct MockFile {
    std::string data;
    std::string sha256;  // advertised LFS oid, "" = non-LFS
};

// Mock HF endpoint. `cut_next` truncates the next file response after N body bytes.
class MockHub {
public:
    std::map<std::string, MockFile> files;
    std::string gated = "false";
    std::string require_token;  // non-empty: resolve answers 401 without this bearer
    std::atomic<bool> ignore_range{false};
    std::atomic<long> cut_next{-1};
    std::atomic<int> api_requests{0};
    std::atomic<int> file_requests{0};
    std::mutex mu;
    std::vector<std::string> ranges;  // Range header of each file request ("" = none)
    std::vector<std::string> auths;

    MockHub() {
        svr_.Get(R"(/api/models/([^/]+)/([^/]+)/revision/([^/?]+))",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     ++api_requests;
                     if (req.get_param_value("blobs") != "true") {
                         res.status = 400;
                         return;
                     }
                     std::string sib;
                     for (const auto& [name, f] : files) {
                         if (!sib.empty())
                             sib += ",";
                         sib += "{\"rfilename\":\"" + name + "\",\"size\":" + std::to_string(f.data.size());
                         if (!f.sha256.empty())
                             sib += ",\"lfs\":{\"sha256\":\"" + f.sha256 +
                                    "\",\"size\":" + std::to_string(f.data.size()) + "}";
                         sib += "}";
                     }
                     res.set_content("{\"id\":\"" + req.matches[1].str() + "/" + req.matches[2].str() +
                                         "\",\"sha\":\"" + kCommit + "\",\"gated\":" + gated +
                                         ",\"siblings\":[" + sib + "]}",
                                     "application/json");
                 });
        svr_.Get(R"(/([^/]+)/([^/]+)/resolve/([^/]+)/(.+))", [this](const httplib::Request& req,
                                                                    httplib::Response& res) {
            ++file_requests;
            {
                std::lock_guard<std::mutex> lk(mu);
                ranges.push_back(req.get_header_value("Range"));
                auths.push_back(req.get_header_value("Authorization"));
            }
            if (!require_token.empty() &&
                req.get_header_value("Authorization") != "Bearer " + require_token) {
                res.status = 401;
                res.set_header("X-Error-Code", "GatedRepo");
                res.set_header("X-Error-Message", "Access to model is restricted.");
                res.set_content("{\"error\":\"restricted\"}", "application/json");
                return;
            }
            if (req.matches[3].str() != kCommit) {
                res.status = 404;
                return;
            }
            auto it = files.find(req.matches[4].str());
            if (it == files.end()) {
                res.status = 404;
                return;
            }
            auto body = std::make_shared<std::string>(it->second.data);
            if (ignore_range) {  // chunked: httplib applies no Range, the answer is a full 200
                res.set_chunked_content_provider("application/octet-stream",
                                                 [body](size_t off, httplib::DataSink& sink) {
                                                     if (off < body->size())
                                                         sink.write(body->data() + off, body->size() - off);
                                                     else
                                                         sink.done();
                                                     return true;
                                                 });
                return;
            }
            // httplib turns this into a 206 with Content-Range when the request has a Range.
            const long cut = cut_next.exchange(-1);
            res.set_content_provider(body->size(), "application/octet-stream",
                                     [body, cut](size_t off, size_t n, httplib::DataSink& sink) {
                                         size_t end = off + n;
                                         if (cut >= 0 && end > static_cast<size_t>(cut)) {
                                             if (off < static_cast<size_t>(cut))
                                                 sink.write(body->data() + off,
                                                            static_cast<size_t>(cut) - off);
                                             return false;  // drop the connection mid-body
                                         }
                                         sink.write(body->data() + off, end - off);
                                         return true;
                                     });
        });
        port_ = svr_.bind_to_any_port("127.0.0.1");
        th_ = std::thread([this] { svr_.listen_after_bind(); });
        svr_.wait_until_ready();
    }
    ~MockHub() {
        svr_.stop();
        th_.join();
    }
    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port_); }
    void add(const std::string& name, const std::string& data, bool lfs = true) {
        MockFile f{data, ""};
        if (lfs) {
            const std::string tmp = fs::temp_directory_path() / ("imp_hf_sha_" + std::to_string(::getpid()));
            spit(tmp, data);
            f.sha256 = sha256_file(tmp);
            fs::remove(tmp);
        }
        files[name] = f;
    }

private:
    httplib::Server svr_;
    int port_ = 0;
    std::thread th_;
};

class HfFetchMock : public ::testing::Test {
protected:
    void SetUp() override {
        cache_ = fs::temp_directory_path() /
                 ("imp_hf_fetch_" + std::to_string(::getpid()) + "_" +
                  ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(cache_);
        fs::create_directories(cache_);
    }
    void TearDown() override { fs::remove_all(cache_); }
    FetchOptions opts(int retries = 0) const {
        FetchOptions o;
        o.endpoint = hub.endpoint();
        o.cache_dir = cache_;
        o.retries = retries;
        o.low_speed_timeout_s = 10;
        return o;
    }
    std::string snap() const { return repo_cache_dir(cache_, "org/repo") + "/snapshots/" + kCommit; }
    // Every regular file under the cache, to assert that nothing partial is left behind.
    std::vector<std::string> cache_files() const {
        std::vector<std::string> out;
        for (const auto& e : fs::recursive_directory_iterator(cache_))
            if (e.is_regular_file())
                out.push_back(fs::relative(e.path(), cache_).string());
        return out;
    }

    MockHub hub;
    std::string cache_;
};

// ---- pure functions ----

TEST(HfFetchParse, AcceptsRepoAndFile) {
    HfUri u;
    std::string err;
    ASSERT_TRUE(parse_hf_uri("hf://Qwen/Qwen3-0.6B-GGUF", u, err)) << err;
    EXPECT_EQ(u.repo, "Qwen/Qwen3-0.6B-GGUF");
    EXPECT_EQ(u.file, "");
    ASSERT_TRUE(parse_hf_uri("hf://ggml-org/tiny-llamas:stories260K.gguf", u, err)) << err;
    EXPECT_EQ(u.repo, "ggml-org/tiny-llamas");
    EXPECT_EQ(u.file, "stories260K.gguf");
    ASSERT_TRUE(parse_hf_uri("hf://org/repo:Q4_K_M/model-Q4_K_M.gguf", u, err)) << err;
    EXPECT_EQ(u.file, "Q4_K_M/model-Q4_K_M.gguf");
    EXPECT_TRUE(is_hf_uri("hf://x"));
    EXPECT_FALSE(is_hf_uri("/models/hf://x"));
}

TEST(HfFetchParse, RejectsMalformed) {
    HfUri u;
    std::string err;
    for (const char* bad : {"hf://", "hf://org", "hf://org/", "hf:///repo", "hf://a/b/c", "hf://../repo",
                            "hf://org/..", "hf://org/re po", "hf://org/repo:", "hf://org/repo:../x.gguf",
                            "hf://org/repo:/abs.gguf", "hf://org/repo:a//b.gguf", "org/repo"}) {
        EXPECT_FALSE(parse_hf_uri(bad, u, err)) << bad;
        EXPECT_FALSE(err.empty()) << bad;
    }
}

TEST(HfFetchUrl, BuildsApiAndResolveUrls) {
    EXPECT_EQ(api_url("https://huggingface.co", "org/repo", "main"),
              "https://huggingface.co/api/models/org/repo/revision/main?blobs=true");
    EXPECT_EQ(api_url("https://huggingface.co", "org/repo", "refs/pr/1"),
              "https://huggingface.co/api/models/org/repo/revision/refs%2Fpr%2F1?blobs=true");
    EXPECT_EQ(resolve_url("https://huggingface.co", "org/repo", kCommit, "sub dir/a+b.gguf"),
              std::string("https://huggingface.co/org/repo/resolve/") + kCommit + "/sub%20dir/a%2Bb.gguf");
    EXPECT_EQ(repo_cache_dir("/c", "org/repo"), "/c/models--org--repo");
}

TEST(HfFetchInfo, ParsesSiblingsShaAndGated) {
    RepoInfo info;
    std::string err;
    const std::string sha(64, 'a');
    ASSERT_TRUE(parse_repo_info(std::string("{\"sha\":\"") + kCommit +
                                    "\",\"gated\":\"manual\",\"siblings\":[" +
                                    "{\"rfilename\":\"m.gguf\",\"size\":10,\"lfs\":{\"sha256\":\"" + sha +
                                    "\",\"size\":12}},{\"rfilename\":\"README.md\",\"size\":3},"
                                    "{\"rfilename\":\"../evil.gguf\",\"size\":1}]}",
                                info, err))
        << err;
    EXPECT_EQ(info.commit, kCommit);
    EXPECT_TRUE(info.gated);
    ASSERT_EQ(info.files.size(), 2u);  // "../evil.gguf" dropped
    EXPECT_EQ(info.files[0].sha256, sha);
    EXPECT_EQ(info.files[0].size, 12u);
    EXPECT_EQ(info.files[1].sha256, "");
    ASSERT_TRUE(parse_repo_info(std::string("{\"sha\":\"") + kCommit + "\",\"gated\":false,\"siblings\":[]}",
                                info, err));
    EXPECT_FALSE(info.gated);
    EXPECT_FALSE(parse_repo_info("{\"siblings\":[]}", info, err));
    EXPECT_FALSE(parse_repo_info("not json", info, err));
}

RepoInfo info_of(std::initializer_list<const char*> names) {
    RepoInfo i;
    i.commit = kCommit;
    for (const char* n : names)
        i.files.push_back(RepoFile{n, 1, ""});
    return i;
}

TEST(HfFetchSelect, SingleGgufIsTakenWithoutSelector) {
    std::vector<RepoFile> out;
    std::string rel, err;
    ASSERT_TRUE(select_files(info_of({".gitattributes", "README.md", "m-Q8_0.gguf"}), "", out, rel, err))
        << err;
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].name, "m-Q8_0.gguf");
    EXPECT_EQ(rel, "m-Q8_0.gguf");
}

TEST(HfFetchSelect, SeveralGgufsNeedSelectorAndListThem) {
    std::vector<RepoFile> out;
    std::string rel, err;
    const auto info = info_of({"a-Q4.gguf", "a-Q8.gguf", "README.md"});
    EXPECT_FALSE(select_files(info, "", out, rel, err));
    EXPECT_NE(err.find("a-Q4.gguf"), std::string::npos) << err;
    EXPECT_NE(err.find("a-Q8.gguf"), std::string::npos) << err;
    ASSERT_TRUE(select_files(info, "a-Q8.gguf", out, rel, err)) << err;
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(rel, "a-Q8.gguf");
    EXPECT_FALSE(select_files(info, "missing.gguf", out, rel, err));
    EXPECT_NE(err.find("a-Q4.gguf"), std::string::npos) << err;
    EXPECT_FALSE(select_files(info, "README.md", out, rel, err));
    EXPECT_NE(err.find(".gguf"), std::string::npos) << err;
}

TEST(HfFetchSelect, SafetensorsRepoTakesWeightsConfigTokenizer) {
    std::vector<RepoFile> out;
    std::string rel, err;
    ASSERT_TRUE(select_files(info_of({".gitattributes", "README.md", "config.json", "generation_config.json",
                                      "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                                      "chat_template.jinja", "model-00001-of-00002.safetensors",
                                      "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                                      "original/consolidated.safetensors", "pytorch_model.bin", "LICENSE"}),
                             "", out, rel, err))
        << err;
    std::vector<std::string> names;
    for (const auto& f : out)
        names.push_back(f.name);
    EXPECT_EQ(names,
              (std::vector<std::string>{"config.json", "generation_config.json", "tokenizer.json",
                                        "tokenizer_config.json", "tokenizer.model", "chat_template.jinja",
                                        "model-00001-of-00002.safetensors",
                                        "model-00002-of-00002.safetensors", "model.safetensors.index.json"}));
    EXPECT_EQ(rel, "");
    EXPECT_FALSE(select_files(info_of({"README.md", "pytorch_model.bin", "config.json"}), "", out, rel, err));
    EXPECT_NE(err.find("pytorch_model.bin"), std::string::npos) << err;
}

TEST(HfFetchSha, KnownVectors) {
    const std::string p = fs::temp_directory_path() / ("imp_hf_sha_vec_" + std::to_string(::getpid()));
    spit(p, "");
    EXPECT_EQ(sha256_file(p), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    spit(p, "abc");
    EXPECT_EQ(sha256_file(p), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    fs::remove(p);
    EXPECT_EQ(sha256_file(p), "");
}

// ---- against the mock hub ----

TEST_F(HfFetchMock, DownloadsVerifiesThenServesFromCacheWithoutRequests) {
    const std::string blob = make_blob(300000, 1);
    hub.add("m-Q8_0.gguf", blob);
    hub.add("README.md", "hi", false);
    FetchResult r = fetch(HfUri{"org/repo", ""}, opts());
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_FALSE(r.cache_hit);
    EXPECT_EQ(r.path, snap() + "/m-Q8_0.gguf");
    EXPECT_EQ(slurp(r.path), blob);
    EXPECT_EQ(r.bytes_downloaded, blob.size());
    EXPECT_EQ(slurp(repo_cache_dir(cache_, "org/repo") + "/refs/main"), std::string(kCommit) + "\n");
    EXPECT_FALSE(fs::exists(r.path + ".part"));
    EXPECT_EQ(hub.api_requests.load(), 1);
    EXPECT_EQ(hub.file_requests.load(), 1);  // README.md is not part of a GGUF selection

    FetchResult again = fetch(HfUri{"org/repo", ""}, opts());
    ASSERT_TRUE(again.ok) << again.error;
    EXPECT_TRUE(again.cache_hit);
    EXPECT_EQ(again.path, r.path);
    EXPECT_EQ(again.bytes_downloaded, 0u);
    EXPECT_EQ(hub.api_requests.load(), 1);  // no network at all on the second start
    EXPECT_EQ(hub.file_requests.load(), 1);

    FetchResult named = fetch(HfUri{"org/repo", "m-Q8_0.gguf"}, opts());
    ASSERT_TRUE(named.ok) << named.error;
    EXPECT_TRUE(named.cache_hit);
    EXPECT_EQ(hub.api_requests.load(), 1);
}

TEST_F(HfFetchMock, InterruptedDownloadResumesWithRangeOnNextCall) {
    const std::string blob = make_blob(500000, 2);
    hub.add("m.gguf", blob);
    hub.cut_next = 200000;
    FetchResult r = fetch(HfUri{"org/repo", "m.gguf"}, opts(/*retries=*/0));
    ASSERT_FALSE(r.ok);
    EXPECT_NE(r.error.find("resume"), std::string::npos) << r.error;
    const std::string part = snap() + "/m.gguf.part";
    ASSERT_TRUE(fs::exists(part));
    const uint64_t kept = fs::file_size(part);
    EXPECT_GT(kept, 0u);
    EXPECT_LE(kept, 200000u);
    EXPECT_FALSE(fs::exists(snap() + "/m.gguf"));

    FetchResult r2 = fetch(HfUri{"org/repo", "m.gguf"}, opts(0));
    ASSERT_TRUE(r2.ok) << r2.error;
    EXPECT_EQ(slurp(r2.path), blob);
    EXPECT_EQ(r2.bytes_downloaded, blob.size() - kept);
    ASSERT_EQ(hub.ranges.size(), 2u);
    EXPECT_EQ(hub.ranges[0], "");
    EXPECT_EQ(hub.ranges[1], "bytes=" + std::to_string(kept) + "-");
    EXPECT_FALSE(fs::exists(part));
}

TEST_F(HfFetchMock, TransferErrorIsRetriedWithRangeInsideOneCall) {
    const std::string blob = make_blob(400000, 3);
    hub.add("m.gguf", blob);
    hub.cut_next = 100000;
    FetchResult r = fetch(HfUri{"org/repo", ""}, opts(/*retries=*/2));
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_EQ(slurp(r.path), blob);
    ASSERT_EQ(hub.ranges.size(), 2u);
    EXPECT_EQ(hub.ranges[1].rfind("bytes=", 0), 0u);
    EXPECT_NE(hub.ranges[1], "bytes=0-");
}

TEST_F(HfFetchMock, ServerIgnoringRangeRestartsTheFile) {
    const std::string blob = make_blob(300000, 4);
    hub.add("m.gguf", blob);
    fs::create_directories(snap());
    spit(snap() + "/m.gguf.part", blob.substr(0, 1000));
    hub.ignore_range = true;
    FetchResult r = fetch(HfUri{"org/repo", "m.gguf"}, opts());
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_EQ(slurp(r.path), blob);
    EXPECT_EQ(hub.ranges.at(0), "bytes=1000-");
    EXPECT_EQ(r.bytes_downloaded, blob.size());  // the 200 rewrote the part from byte 0
}

TEST_F(HfFetchMock, ShaMismatchFailsAndLeavesNoFile) {
    hub.add("m.gguf", make_blob(200000, 5));
    hub.files["m.gguf"].sha256 = std::string(64, '0');
    FetchResult r = fetch(HfUri{"org/repo", "m.gguf"}, opts(2));
    ASSERT_FALSE(r.ok);
    EXPECT_NE(r.error.find("sha256 mismatch"), std::string::npos) << r.error;
    EXPECT_FALSE(fs::exists(snap() + "/m.gguf"));
    EXPECT_FALSE(fs::exists(snap() + "/m.gguf.part"));
    EXPECT_EQ(cache_files(), std::vector<std::string>{});
    EXPECT_EQ(hub.file_requests.load(), 1);  // a mismatch is not retried
}

TEST_F(HfFetchMock, GatedRepoWithoutTokenIsAClearErrorBeforeAnyDownload) {
    hub.add("m.gguf", make_blob(1000, 6));
    hub.gated = "\"manual\"";
    FetchResult r = fetch(HfUri{"org/repo", ""}, opts());
    ASSERT_FALSE(r.ok);
    EXPECT_NE(r.error.find("gated"), std::string::npos) << r.error;
    EXPECT_NE(r.error.find("HF_TOKEN"), std::string::npos) << r.error;
    EXPECT_EQ(hub.file_requests.load(), 0);
    EXPECT_EQ(cache_files(), std::vector<std::string>{});
}

TEST_F(HfFetchMock, UnauthorizedDownloadNamesTokenAndLeavesNoPart) {
    const std::string blob = make_blob(50000, 7);
    hub.add("m.gguf", blob);
    hub.require_token = "tok123";
    FetchResult r = fetch(HfUri{"org/repo", ""}, opts(2));
    ASSERT_FALSE(r.ok);
    EXPECT_NE(r.error.find("401"), std::string::npos) << r.error;
    EXPECT_NE(r.error.find("HF_TOKEN is not set"), std::string::npos) << r.error;
    EXPECT_NE(r.error.find("restricted"), std::string::npos) << r.error;
    EXPECT_EQ(cache_files(), std::vector<std::string>{});
    EXPECT_EQ(hub.file_requests.load(), 1);  // a 401 is not retried

    FetchOptions o = opts();
    o.token = "tok123";
    FetchResult ok = fetch(HfUri{"org/repo", ""}, o);
    ASSERT_TRUE(ok.ok) << ok.error;
    EXPECT_EQ(slurp(ok.path), blob);
    EXPECT_EQ(hub.auths.back(), "Bearer tok123");
}

TEST_F(HfFetchMock, SafetensorsRepoFetchesTheSetAndReturnsTheSnapshotDir) {
    hub.add("config.json", "{\"model_type\":\"llama\"}", false);
    hub.add("tokenizer.json", "{}", false);
    hub.add("model.safetensors", make_blob(120000, 8));
    hub.add("pytorch_model.bin", make_blob(1000, 9));
    FetchResult r = fetch(HfUri{"org/repo", ""}, opts());
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_EQ(r.path, snap());
    EXPECT_TRUE(fs::exists(snap() + "/model.safetensors"));
    EXPECT_TRUE(fs::exists(snap() + "/config.json"));
    EXPECT_FALSE(fs::exists(snap() + "/pytorch_model.bin"));
    EXPECT_EQ(hub.file_requests.load(), 3);
    FetchResult again = fetch(HfUri{"org/repo", ""}, opts());
    ASSERT_TRUE(again.ok) << again.error;
    EXPECT_TRUE(again.cache_hit);
    EXPECT_EQ(again.path, snap());
    EXPECT_EQ(hub.file_requests.load(), 3);
}

TEST_F(HfFetchMock, SeveralGgufsWithoutSelectorFailWithTheList) {
    hub.add("a-Q4.gguf", make_blob(10, 10));
    hub.add("a-Q8.gguf", make_blob(10, 11));
    FetchResult r = fetch(HfUri{"org/repo", ""}, opts());
    ASSERT_FALSE(r.ok);
    EXPECT_NE(r.error.find("a-Q4.gguf, a-Q8.gguf"), std::string::npos) << r.error;
    EXPECT_EQ(hub.file_requests.load(), 0);
}

TEST_F(HfFetchMock, MissingRepoIsAnError) {
    FetchOptions o = opts();
    o.endpoint = hub.endpoint() + "/nothing-here";
    FetchResult r = fetch(HfUri{"org/repo", ""}, o);
    ASSERT_FALSE(r.ok);
    EXPECT_NE(r.error.find("404"), std::string::npos) << r.error;
}

}  // namespace
