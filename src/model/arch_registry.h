#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "model/model_arch.h"

namespace imp {

// Where a checkpoint names its architecture. One spelling can mean different archs per
// source: GGUF "qwen2" is LLAMA, HF model_type "qwen2" is QWEN3.
enum class ArchSource : uint8_t {
    GGUF,           // GGUF general.architecture
    HF_CLASS,       // config.json architectures[0]; the GGUF path accepts these too
    HF_MODEL_TYPE,  // config.json model_type, read when architectures is absent
};

struct ArchSpelling {
    ArchSource source;
    std::string_view spelling;
    ModelArch arch;
};

// The one table of checkpoint spellings -> ModelArch (#2457). A new arch adds rows here only.
[[nodiscard]] std::span<const ArchSpelling> arch_spellings();

// Exact match in `source`'s spellings; nullopt when the table has none.
[[nodiscard]] std::optional<ModelArch> find_arch(ArchSource source, std::string_view s);

}  // namespace imp
