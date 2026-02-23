#pragma once

#include <string>

namespace imp {

enum class ModelArch {
    LLAMA,
    MISTRAL,
    MIXTRAL,
    DEEPSEEK,
    NEMOTRON_H_MOE,
    GENERIC,
};

const char* model_arch_name(ModelArch arch);

} // namespace imp
