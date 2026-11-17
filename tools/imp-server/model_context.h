#pragma once

// Context window /v1/models advertises for a model that is not loaded: runtime.max_seq_len when
// set, else the checkpoint's declared window capped at kAutoMaxSeqLenCap. An upper bound: the load
// may size lower to fit VRAM, and the loaded entry then reports the servable value.

#include <string>

namespace imp_server {

// configured > 0 wins; else min(declared, kAutoMaxSeqLenCap); 0 = unknown (field omitted).
int advertised_context_tokens(int declared, int configured);

// max_position_embeddings of a SafeTensors dir (HF config.json, text_config included), cached
// per path and config.json mtime. 0 for a GGUF file or an unreadable config.
int declared_context_tokens(const std::string& model_path);

}  // namespace imp_server
