#pragma once

// One imp-cli mode per function: main() used to hold all of them (737 LOC, the fifth-largest
// function body in the repo), conflating config install, model detection, flag override,
// benchmark, chat-template resolve, REPL, /image, multi-turn and the think-tag filter.
// Moved out in #1906 (move-verbatim); main() is now load + resolve + one dispatch, 246 LOC.
// Each mode returns a process exit code; freeing ctx/model stays with main, which owns them.

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

// --bench: synthetic pp/tg benchmark on a token ramp, llama-bench methodology.
int run_bench(ImpContext ctx, ImpModel model, const CliArgs& args, const std::string& resolved_model);

// -i / --interactive: multi-turn REPL over the token API, with /image, the chat
// template and the think-tag stream filter. Takes `params` by value: the mode
// raises max_tokens to 16384 when the user did not set it.
int run_interactive(ImpContext ctx, ImpModel model, const CliArgs& args, ImpGenerateParams params);

// --prompt / stdin: one prompt, one completion, then exit.
int run_oneshot(ImpContext ctx, ImpModel model, const CliArgs& args, ImpGenerateParams params,
                const std::string& resolved_model);

}  // namespace imp_cli
