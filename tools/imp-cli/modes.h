#pragma once

// One imp-cli mode per function. main() is a 800-line hard-review ceiling with
// five modes inside it (perplexity, bench, interactive, one-shot, raw); this
// file is where they move as they are touched, rather than in one sweep that
// would rewrite lines nobody changed.
//
// Each returns a process exit code. Freeing ctx/model stays with main, which
// owns them.

#include "api/imp_internal.h"
#include "args.h"

#include <cstdint>
#include <string>
#include <vector>

namespace imp_cli {

// --perplexity <file> [--calibrate-out <file>]: teacher-forced NLL over the
// tokenised corpus, plus the optional calibration write.
int run_perplexity(ImpContext ctx, const CliArgs& args, const std::vector<int32_t>& ppl_tokens,
                   const std::string& resolved_model);

}  // namespace imp_cli
