// Every imp.conf.example key binds to the RuntimeConfig field of the same dotted name: setting it
// moves that field and no other. The key list is generated from the example at configure time
// (cmake/config_key_table.cmake); tools/check_config_keys.py keeps example and binder equal.

#include "runtime/config.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace imp {
namespace {

std::string show(bool v) { return v ? "true" : "false"; }
std::string show(int v) { return std::to_string(v); }
std::string show(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return buf;
}
std::string show(const std::string& v) { return v; }

// Values that differ from `v` and parse back as that type; a string key with a fixed spelling set
// takes the first candidate it accepts as a change.
std::vector<std::string> probes(bool v) { return {show(!v)}; }
std::vector<std::string> probes(int v) { return {show(v + 3)}; }
std::vector<std::string> probes(float v) { return {show(v + 0.25f)}; }
std::vector<std::string> probes(const std::string& v) {
    std::vector<std::string> out;
    for (const char* c : {"binding_probe", "on", "off", "auto", "true", "false"})
        if (v != c)
            out.emplace_back(c);
    return out;
}

struct KeyField {
    std::string key;
    std::string (*get)(const RuntimeConfig&);
    std::vector<std::string> (*candidates)(const RuntimeConfig&);
};

#define IMP_CONFIG_KEY(section, name)                                                         \
    KeyField{#section "." #name, [](const RuntimeConfig& c) { return show(c.section.name); }, \
             [](const RuntimeConfig& c) { return probes(c.section.name); }},
const std::vector<KeyField> kKeys = {
#include "config_example_keys.inc"
};
#undef IMP_CONFIG_KEY

// Keys whose binder also sets another key by design (config.cpp: deterministic implies the GEMM half).
const std::map<std::string, std::set<std::string>> kCoupled = {
    {"runtime.deterministic", {"runtime.deterministic_gemm"}},
};

std::map<std::string, std::string> snapshot(const RuntimeConfig& c) {
    std::map<std::string, std::string> out;
    for (const KeyField& k : kKeys)
        out[k.key] = k.get(c);
    return out;
}

TEST(ConfigBindings, EveryExampleKeyMovesOnlyItsOwnField) {
    ASSERT_GT(kKeys.size(), 200u);
    const RuntimeConfig defaults;
    const auto before = snapshot(defaults);
    size_t bound = 0;
    for (const KeyField& k : kKeys) {
        bool moved = false;
        for (const std::string& value : k.candidates(defaults)) {
            RuntimeConfig cfg;
            // Rejected = no binder or a value outside the key's spelling set: try the next one.
            if (!cfg.apply_overrides({k.key + "=" + value}).empty())
                continue;
            const auto after = snapshot(cfg);
            if (after.at(k.key) == before.at(k.key))
                continue;
            moved = true;
            const auto coupled = kCoupled.find(k.key);
            for (const auto& [other, v] : after) {
                if (other != k.key && v != before.at(other)) {
                    EXPECT_TRUE(coupled != kCoupled.end() && coupled->second.contains(other))
                        << k.key << "=" << value << " also moved " << other << " (" << before.at(other)
                        << " -> " << v << ")";
                }
            }
            break;
        }
        EXPECT_TRUE(moved) << k.key << " has no binder or does not move its own field";
        bound += moved;
    }
    std::printf("ConfigBindings: %zu of %zu example keys move their own field\n", bound, kKeys.size());
}

}  // namespace
}  // namespace imp
