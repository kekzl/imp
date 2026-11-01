#include "compute/lt_route_trace.h"
#include "core/logging.h"
#include "core/process_diag.h"

#include <gtest/gtest.h>

#include <string>

namespace imp {
namespace {

TEST(LtRouteTrace, GemmLabelPrecedence) {
    EXPECT_STREQ(lt_route_gemm_label(true, true, true), "gemm:int8");
    EXPECT_STREQ(lt_route_gemm_label(true, true, false), "gemm:f16_acc16");
    EXPECT_STREQ(lt_route_gemm_label(false, true, false), "gemm:f16_acc32");
    EXPECT_STREQ(lt_route_gemm_label(false, false, false), "gemm:other");
}

TEST(LtRouteTrace, Fp8Label) {
    EXPECT_STREQ(lt_route_fp8_label(true, true), "gemm_cublaslt:fp8xfp8");
    EXPECT_STREQ(lt_route_fp8_label(false, true), "gemm_cublaslt:f16xfp8");
    EXPECT_STREQ(lt_route_fp8_label(true, false), "gemm_cublaslt:other");
}

// Off by default: note is a no-op, dump has nothing to log. On: the table logs and clears.
TEST(LtRouteTrace, NoteAndDumpFollowTheFlag) {
    const ProcessDiag saved = process_diag_current();
    const LogLevel saved_level = log_get_level();
    log_set_level(LogLevel::INFO);
    ProcessDiag d = saved;
    d.log_gemm_algo = false;
    process_diag_set(d);
    lt_route_note("gemm:f16_acc16", 512, 128, 2048, 1, std::source_location::current());
    testing::internal::CaptureStderr();
    testing::internal::CaptureStdout();
    lt_route_dump();
    std::string off = testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
    EXPECT_EQ(off.find("[lt-route]"), std::string::npos);

    d.log_gemm_algo = true;
    process_diag_set(d);
    lt_route_note("gemm:f16_acc16", 512, 128, 2048, 1, std::source_location::current());
    lt_route_note("gemm:f16_acc16", 512, 128, 2048, 1, std::source_location::current());
    testing::internal::CaptureStderr();
    testing::internal::CaptureStdout();
    lt_route_dump();
    std::string on = testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
    process_diag_set(saved);
    log_set_level(saved_level);
    EXPECT_NE(on.find("tests/test_lt_route_trace.cpp"), std::string::npos) << on;
    EXPECT_NE(on.find("m=512 n=128 k=2048 batch=1: 1"), std::string::npos) << on;
}

}  // namespace
}  // namespace imp
