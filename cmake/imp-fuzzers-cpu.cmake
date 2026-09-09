# CPU-only libFuzzer build: the parser targets and the four translation units
# they need, no CUDA language, no dependency fetched.
#
# The full build (IMP_FUZZERS=ON at the end of CMakeLists.txt) links every
# target against imp and adds the file-format and tool-stream targets, which
# need the loader and httplib. Commands: fuzz/README.md.

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "IMP_FUZZERS_CPU_ONLY=ON needs clang (-fsanitize=fuzzer); "
                        "got ${CMAKE_CXX_COMPILER_ID}. See fuzz/README.md.")
endif()

set(IMP_FUZZ_SANITIZE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer -g)

add_library(imp_fuzz_parsers OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compute/json_schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compute/gbnf_grammar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compute/gbnf_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/logging.cpp)
target_include_directories(imp_fuzz_parsers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# fuzzer-no-link instruments the library for coverage feedback without pulling
# libFuzzer's main into it.
target_compile_options(imp_fuzz_parsers PRIVATE
    -fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -g)

set(IMP_FUZZ_CPU_TARGETS json_schema regex gbnf)
foreach(t ${IMP_FUZZ_CPU_TARGETS})
    add_executable(fuzz_${t} ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/fuzz_${t}.cpp)
    target_include_directories(fuzz_${t} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/fuzz)
    target_compile_options(fuzz_${t} PRIVATE ${IMP_FUZZ_SANITIZE})
    target_link_options(fuzz_${t} PRIVATE ${IMP_FUZZ_SANITIZE})
    target_link_libraries(fuzz_${t} PRIVATE imp_fuzz_parsers)
endforeach()

message(STATUS "IMP_FUZZERS_CPU_ONLY: ${IMP_FUZZ_CPU_TARGETS}, no CUDA language")
