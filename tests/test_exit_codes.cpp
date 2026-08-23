// The C API taxonomy reaching a process exit code (#1585).
//
// Before this the four binaries collapsed every failure onto 1, so a
// supervisor had to parse English to tell "no such file" from "out of VRAM".
// Exactly one of those is worth retrying.

#include "common/exit_codes.h"

#include <gtest/gtest.h>

#include <set>

namespace {

TEST(ExitCodes, SuccessIsZero) { EXPECT_EQ(imp::tools::exit_code_for(IMP_SUCCESS), 0); }

// The mapping is the taxonomy's own order made positive, because an exit
// status is an unsigned byte. One table, not two.
TEST(ExitCodes, TheTaxonomyMapsOneToOne) {
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG), 1);
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_OUT_OF_MEMORY), 2);
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_CUDA), 3);
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_FILE_NOT_FOUND), 4);
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_INVALID_MODEL), 5);
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_UNSUPPORTED), 6);
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_INTERNAL), 7);
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_CANCELLED), 8);
    EXPECT_EQ(imp::tools::exit_code_for(IMP_ERROR_CAPACITY), 9);
}

// Every code is distinct and fits a byte, or the shell truncates it into
// another meaning.
TEST(ExitCodes, EveryCodeIsDistinctAndFitsAByte) {
    const ImpError all[] = {IMP_ERROR_INVALID_ARG,    IMP_ERROR_OUT_OF_MEMORY, IMP_ERROR_CUDA,
                            IMP_ERROR_FILE_NOT_FOUND, IMP_ERROR_INVALID_MODEL, IMP_ERROR_UNSUPPORTED,
                            IMP_ERROR_INTERNAL,       IMP_ERROR_CANCELLED,     IMP_ERROR_CAPACITY};
    std::set<int> seen;
    for (ImpError e : all) {
        const int c = imp::tools::exit_code_for(e);
        EXPECT_GT(c, 0);
        EXPECT_LE(c, 255);
        EXPECT_TRUE(seen.insert(c).second) << "duplicate exit code " << c;
    }
    EXPECT_EQ(seen.size(), 9u);
}

// A value added to the enum without touching the table must degrade to the old
// behaviour rather than invent a meaning.
TEST(ExitCodes, AnUnknownErrorDegradesToOne) {
    EXPECT_EQ(imp::tools::exit_code_for(static_cast<ImpError>(-42)), 1);
    EXPECT_EQ(imp::tools::exit_code_for(static_cast<ImpError>(-10)), 1);
    EXPECT_EQ(imp::tools::exit_code_for(static_cast<ImpError>(7)), 1);
}

}  // namespace
