#pragma once

// Process exit codes for the imp binaries (#1585).
//
// The C API carries a ten-value error taxonomy (`include/imp/error.h`) that
// stopped at the API boundary: every binary printed `imp_error_string(err)` and
// returned 1, so a caller had to parse English prose to tell "this model file
// does not exist" from "this GPU is out of memory". Only one of those is worth
// retrying, and only one of them means the caller passed something wrong.
//
// The mapping is the taxonomy's own order, made positive, because a process
// exit status is an unsigned byte: IMP_ERROR_INVALID_ARG (-1) is exit 1, and so
// on down to IMP_ERROR_CAPACITY (-9) at exit 9. That keeps one table instead of
// two, and it keeps 1 as the code every existing script already handles, since
// INVALID_ARG is what a usage error was reported as.
//
// imp-quantize used 2 for usage errors, undocumented. It is 1 now, the same as
// every other invalid argument, because two codes for one condition is what
// made this worth filing. That is a behaviour change for anyone who tested
// `-eq 2`; the code stays non-zero, so `if ! cmd` is unaffected. Anything above
// 9 is free for a binary's own conditions; none use it.
//
// Documented for readers in `docs/DEPLOYMENT.md`.

#include "imp/error.h"

namespace imp::tools {

// 0 for success, 1..9 for the taxonomy, 1 for anything unrecognised.
//
// Unrecognised maps to 1 rather than to a new code, so an ImpError added
// without touching this table degrades to the old behaviour instead of
// inventing a meaning.
inline int exit_code_for(ImpError err) {
    if (err == IMP_SUCCESS)
        return 0;
    const int code = -static_cast<int>(err);
    if (code < 1 || code > 9)
        return 1;
    return code;
}

}  // namespace imp::tools
