// Request `model` field is a name, never a path (AUDIT_ARCH_2026 F2-1); find_model_path()
// used to fall through to resolve_model_auto() for any name containing '/', so a request
// naming a readable .gguf loaded it and evicted the resident model.
// Mutation: drop the leading-'/' rule -> AbsolutePathIsRejected fails; drop trailing-slash
// normalization -> TrailingSlashBase fails.

#include "model_name_policy.h"
#include "model_scan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using imp_server::classify_model_name;
using imp_server::ModelNameKind;
using imp_server::path_within;

TEST(ModelNamePolicy, BasenameIsLookedUpInModelsDir) {
    EXPECT_EQ(classify_model_name("Qwen3-8B-Q8_0.gguf"), ModelNameKind::Basename);
    EXPECT_EQ(classify_model_name("Qwen3-8B-NVFP4-cortecs"), ModelNameKind::Basename);
    EXPECT_EQ(classify_model_name("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"), ModelNameKind::Basename);
}

TEST(ModelNamePolicy, HfRepoIdShape) {
    EXPECT_EQ(classify_model_name("Qwen/Qwen3-8B"), ModelNameKind::HfRepoId);
    EXPECT_EQ(classify_model_name("kekzle/Qwen3.8-27B-NVFP4-vllm"), ModelNameKind::HfRepoId);
}

TEST(ModelNamePolicy, AbsolutePathIsRejected) {
    EXPECT_EQ(classify_model_name("/tmp/x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("/etc/passwd"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("/models/Qwen3-8B-Q8_0.gguf"), ModelNameKind::Rejected);
}

TEST(ModelNamePolicy, RelativeAndTraversalPathsAreRejected) {
    EXPECT_EQ(classify_model_name("./x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("../x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("org/../../x"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("~/models/x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("a/b/c"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("org\\repo"), ModelNameKind::Rejected);
}

TEST(ModelNamePolicy, RepoIdWithModelFileExtensionIsAPathNotAnId) {
    // "org/x.gguf" is what a relative path to a file looks like; hf_hub.cpp
    // itself refuses to treat it as a repo id.
    EXPECT_EQ(classify_model_name("org/x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("org/model.safetensors"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("org/"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name(""), ModelNameKind::Rejected);
}

class PathWithin : public ::testing::Test {
protected:
    void SetUp() override {
        base_ = std::filesystem::temp_directory_path() / "imp_policy_test";
        std::filesystem::create_directories(base_ / "sub");
        std::ofstream(base_ / "sub" / "m.gguf") << "x";
        std::filesystem::create_directories(base_.string() + "2");
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(base_, ec);
        std::filesystem::remove_all(base_.string() + "2", ec);
    }
    std::filesystem::path base_;
};

TEST_F(PathWithin, InsideAndOutside) {
    EXPECT_TRUE(path_within(base_, base_ / "sub" / "m.gguf"));
    EXPECT_TRUE(path_within(base_, base_));
    EXPECT_FALSE(path_within(base_, base_.parent_path()));
    EXPECT_FALSE(path_within(base_, "/etc/passwd"));
}

TEST_F(PathWithin, SiblingWithCommonPrefixIsOutside) {
    // "/x/imp_policy_test" must not contain "/x/imp_policy_test2/..."
    EXPECT_FALSE(path_within(base_, std::filesystem::path(base_.string() + "2") / "m.gguf"));
}

TEST_F(PathWithin, TraversalIsResolvedBeforeTheCompare) {
    EXPECT_FALSE(path_within(base_, base_ / "sub" / ".." / ".." / "m.gguf"));
    EXPECT_TRUE(path_within(base_, base_ / "sub" / ".." / "sub" / "m.gguf"));
}

TEST_F(PathWithin, TrailingSlashBase) {
    EXPECT_TRUE(path_within(base_.string() + "/", base_ / "sub" / "m.gguf"));
}

// #2884: one bad entry under --models-dir threw filesystem_error out of scan_model_files and
// /v1/models answered 500. Mutation: drop the error_code in is_safetensors_dir or in the
// entry's is_directory() -> LoopingSymlinkDoesNotHideTheRest throws.
class ModelScan : public ::testing::Test {
protected:
    void SetUp() override {
        base_ = std::filesystem::temp_directory_path() / "imp_scan_test";
        std::filesystem::remove_all(base_);
        std::filesystem::create_directories(base_ / "a-model");
        std::ofstream(base_ / "a-model" / "model.safetensors") << "x";
        std::filesystem::create_directories(base_ / "z-model");
        std::ofstream(base_ / "z-model" / "model.safetensors.index.json") << "{}";
        std::ofstream(base_ / "m.gguf") << "x";
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::permissions(base_ / "locked", std::filesystem::perms::owner_all, ec);
        std::filesystem::remove_all(base_, ec);
    }
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const auto& [name, path] : imp_server::scan_model_files(base_.string()))
            out.push_back(name);
        return out;
    }
    std::filesystem::path base_;
};

TEST_F(ModelScan, ListsGgufAndSafetensorsDirs) {
    EXPECT_EQ(names(), (std::vector<std::string>{"a-model", "m.gguf", "z-model"}));
}

TEST_F(ModelScan, LoopingSymlinkDoesNotHideTheRest) {
    // ELOOP on stat, independent of the user (root bypasses permission bits, not loops).
    std::filesystem::create_symlink(base_ / "loop", base_ / "loop");
    EXPECT_FALSE(imp::is_safetensors_dir((base_ / "loop").string()));
    EXPECT_EQ(names(), (std::vector<std::string>{"a-model", "m.gguf", "z-model"}));
}

TEST_F(ModelScan, UnreadableDirectoryDoesNotHideTheRest) {
    // As root the directory stays readable and the list is the same: the assertion holds either way.
    std::filesystem::create_directories(base_ / "locked" / "inner");
    std::ofstream(base_ / "locked" / "inner" / "model.safetensors") << "x";
    std::filesystem::permissions(base_ / "locked", std::filesystem::perms::none);
    const auto got = names();
    for (const char* want : {"a-model", "m.gguf", "z-model"})
        EXPECT_NE(std::find(got.begin(), got.end(), want), got.end()) << want;
}

}  // namespace
