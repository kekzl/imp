// =============================================================================
// Unit tests for imp-server CLI flag parsing (parse_server_args) — closes the
// "--vram-budget flag parse and the server-limit flags are untested" gap from
// issue #896. The planner logic behind --vram-budget is well covered elsewhere
// (test_vram_budget_reserve / test_weight_registry_preservation); this asserts
// the literal CLI parse that feeds it, on the CPU, in CI (the real server main
// never runs there).
// =============================================================================

#include <gtest/gtest.h>
#include "args.h"
#include "common/args_common.h"

#include <string>
#include <vector>

namespace {

// parse_server_args takes (argc, char** argv). Build a mutable argv from
// strings (argv[0] is the program name; parsing starts at index 1).
ServerArgs parse(std::vector<std::string> args) {
    std::vector<std::string> argv_storage;
    argv_storage.reserve(args.size() + 1);
    argv_storage.emplace_back("imp-server");
    for (auto& a : args)
        argv_storage.push_back(std::move(a));
    std::vector<char*> argv;
    argv.reserve(argv_storage.size());
    for (auto& s : argv_storage)
        argv.push_back(s.data());
    return parse_server_args(static_cast<int>(argv.size()), argv.data());
}

}  // namespace

TEST(ServerArgs, DefaultsWhenNoFlags) {
    ServerArgs a = parse({});
    EXPECT_EQ(a.vram_budget_mb, 0);   // 0 = uncapped
    EXPECT_EQ(a.max_concurrent, 64);  // guard active out of the box
    EXPECT_EQ(a.request_timeout, 300);
    EXPECT_EQ(a.rate_limit, 0);        // 0 = unlimited
    EXPECT_EQ(a.max_input_tokens, 0);  // 0 = unlimited
    EXPECT_EQ(a.port, 8080);
}

TEST(ServerArgs, VramBudgetParsed) {
    EXPECT_EQ(parse({"--vram-budget", "4096"}).vram_budget_mb, 4096);
    EXPECT_EQ(parse({"--vram-budget", "0"}).vram_budget_mb, 0);
}

TEST(ServerArgs, ServerLimitFlagsParsed) {
    ServerArgs a = parse({"--max-concurrent", "8", "--request-timeout", "120", "--rate-limit", "30",
                          "--max-input-tokens", "2000"});
    EXPECT_EQ(a.max_concurrent, 8);
    EXPECT_EQ(a.request_timeout, 120);
    EXPECT_EQ(a.rate_limit, 30);
    EXPECT_EQ(a.max_input_tokens, 2000);
}

TEST(ServerArgs, ZeroDisablesLimits) {
    // 0 is the documented "unlimited/uncapped" sentinel for every limit flag.
    ServerArgs a = parse({"--max-concurrent", "0", "--rate-limit", "0", "--request-timeout", "0"});
    EXPECT_EQ(a.max_concurrent, 0);
    EXPECT_EQ(a.rate_limit, 0);
    EXPECT_EQ(a.request_timeout, 0);
}

TEST(ServerArgs, UnrelatedFlagsLeaveLimitsAtDefault) {
    // A --vram-budget request must not perturb the other limits.
    ServerArgs a = parse({"--vram-budget", "8192", "--port", "9099"});
    EXPECT_EQ(a.vram_budget_mb, 8192);
    EXPECT_EQ(a.port, 9099);
    EXPECT_EQ(a.max_concurrent, 64);
    EXPECT_EQ(a.max_input_tokens, 0);
}

// ---------------------------------------------------------------------------
// resolve_calibration_out - `[calibration] out_path` was parsed, documented in
// config.h and offered in imp.conf.example, and read by nothing: setting it
// produced no file and no warning, because imp_calibration_write() takes the
// path as an argument and imp-cli passed --calibrate straight through.
// Debt ledger item 7.
// ---------------------------------------------------------------------------

TEST(CalibrationOutPath, TheFlagWins) {
    EXPECT_EQ(resolve_calibration_out("/from/flag.json", "/from/conf.json"), "/from/flag.json");
}

TEST(CalibrationOutPath, TheConfigKeyIsUsedWhenTheFlagCarriesNoPath) {
    // This is the case that silently did nothing before.
    EXPECT_EQ(resolve_calibration_out("", "/from/conf.json"), "/from/conf.json");
}

TEST(CalibrationOutPath, NeitherMeansNoCalibrationRun) { EXPECT_EQ(resolve_calibration_out("", ""), ""); }

TEST(CalibrationOutPath, AnEmptyConfigKeyDoesNotClobberTheFlag) {
    // imp.conf.example ships `out_path = ""`, so the default value of the key
    // is the empty string on every run that does not set it.
    EXPECT_EQ(resolve_calibration_out("/from/flag.json", ""), "/from/flag.json");
}
