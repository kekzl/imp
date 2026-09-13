#pragma once

// Process exit codes for imp binaries (#1585): the C API's ten-value error taxonomy
// (include/imp/error.h) stopped at the boundary, so every binary printed prose and returned 1.
// Mapping is the taxonomy's own order, made positive: IMP_ERROR_INVALID_ARG(-1)->1, down to
// IMP_ERROR_CAPACITY(-9)->9. 1 stays the code every existing script already handles.
// Codes above 9 are free for a binary's own conditions. Documented in docs/DEPLOYMENT.md.

#include "imp/error.h"

namespace imp::tools {

// 0 = success, 1..9 = the taxonomy, 1 = anything unrecognised (so a new ImpError degrades to
// old behaviour rather than inventing a meaning).
inline int exit_code_for(ImpError err) {
    if (err == IMP_SUCCESS)
        return 0;
    const int code = -static_cast<int>(err);
    if (code < 1 || code > 9)
        return 1;
    return code;
}

}  // namespace imp::tools
