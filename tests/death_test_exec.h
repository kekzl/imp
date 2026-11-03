// Threadsafe death tests re-exec the binary with execv(argv[0]), no PATH search: under
// `docker run --entrypoint test-quant` argv[0] is a bare name and every such test died with
// "execv(test-quant, ...) failed". The child is re-execed by its absolute path instead.
#ifndef IMP_TESTS_DEATH_TEST_EXEC_H
#define IMP_TESTS_DEATH_TEST_EXEC_H

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace imp_test {

// Call at the top of a test instead of GTEST_FLAG_SET(death_test_style, "threadsafe").
inline void threadsafe_death_tests() {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    std::vector<std::string> args = ::testing::internal::GetArgvs();
    std::error_code ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!args.empty() && !ec)
        args[0] = self.string();
    ::testing::internal::SetInjectableArgvs(args);
}

}  // namespace imp_test

#endif  // IMP_TESTS_DEATH_TEST_EXEC_H
