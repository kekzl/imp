# CMake generated Testfile for 
# Source directory: /src
# Build directory: /src/build-docker
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/src/build-docker/imp-tests[1]_include.cmake")
add_test([=[unit_tests]=] "/src/build-docker/imp-tests" "--gtest_filter=TensorTest.*:GgufLoaderTest.*:TokenizerSPMTest.*:TokenizerGPT2Test.*:TokenizerDispatchTest.*:ChatTemplate*:BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub")
set_tests_properties([=[unit_tests]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/src/CMakeLists.txt;403;add_test;/src/CMakeLists.txt;0;")
add_test([=[gpu_tests]=] "/src/build-docker/imp-tests" "--gtest_filter=-TensorTest.*:GgufLoaderTest.*:TokenizerSPMTest.*:TokenizerGPT2Test.*:TokenizerDispatchTest.*:ChatTemplate*:BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*")
set_tests_properties([=[gpu_tests]=] PROPERTIES  LABELS "gpu" _BACKTRACE_TRIPLES "/src/CMakeLists.txt;407;add_test;/src/CMakeLists.txt;0;")
add_test([=[perf_tests]=] "/src/build-docker/imp-tests" "--gtest_filter=*Perf*:*Bench*:*Throughput*")
set_tests_properties([=[perf_tests]=] PROPERTIES  LABELS "perf" _BACKTRACE_TRIPLES "/src/CMakeLists.txt;411;add_test;/src/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
subdirs("_deps/cutlass-build")
subdirs("_deps/httplib-build")
subdirs("_deps/nlohmann_json-build")
