#include "model/hf_config_loader.h"
#include "model/model_arch.h"
#include "model/model_config.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using imp::HFConfigLoader;
using imp::ModelArch;

namespace {

struct IdentityRow {
    std::string entry;     // gguf | hf_class | model_type
    std::string spelling;  // string a checkpoint carries
    std::string arch;      // model_arch_name() of the expected ModelArch
};

std::vector<IdentityRow> load_main_rows() {
    std::ifstream f(std::string(IMP_TEST_FIXTURES_DIR) + "/arch_identity_main.tsv");
    std::vector<IdentityRow> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ls(line);
        IdentityRow r;
        std::getline(ls, r.entry, '\t');
        std::getline(ls, r.spelling, '\t');
        std::getline(ls, r.arch, '\t');
        rows.push_back(r);
    }
    return rows;
}

// model_type is only reachable through load_config: a config.json without `architectures`.
ModelArch arch_from_model_type(const std::string& model_type) {
    const auto dir = std::filesystem::temp_directory_path() / ("imp_test_archid_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    {
        std::ofstream f(dir / "config.json");
        f << R"({"model_type": ")" << model_type
          << R"(", "hidden_size": 4096, "num_attention_heads": 32, "num_hidden_layers": 32})";
    }
    imp::ModelConfig cfg;
    const bool ok = HFConfigLoader::load_config(dir.string(), cfg);
    std::filesystem::remove_all(dir);
    EXPECT_TRUE(ok) << model_type;
    return cfg.arch;
}

// Every string main accepted (117: 56 gguf, 35 hf_class, 26 model_type) keeps its ModelArch.
TEST(ArchIdentity, EveryStringMainAcceptedKeepsItsArch) {
    const auto rows = load_main_rows();
    std::map<std::string, int> per_entry;
    for (const auto& r : rows) {
        ModelArch got = ModelArch::GENERIC;
        if (r.entry == "gguf")
            got = imp::parse_model_arch(r.spelling);
        else if (r.entry == "hf_class")
            got = HFConfigLoader::map_architecture(r.spelling);
        else if (r.entry == "model_type")
            got = arch_from_model_type(r.spelling);
        else
            ADD_FAILURE() << "unknown entry point " << r.entry;
        EXPECT_STREQ(imp::model_arch_name(got), r.arch.c_str()) << r.entry << " '" << r.spelling << "'";
        per_entry[r.entry]++;
    }
    EXPECT_EQ(rows.size(), 117u);
    EXPECT_EQ(per_entry["gguf"], 56);
    EXPECT_EQ(per_entry["hf_class"], 35);
    EXPECT_EQ(per_entry["model_type"], 26);
}

}  // namespace
