# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

if(EXISTS "/src/build-docker/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/cutlass-populate-gitclone-lastrun.txt" AND EXISTS "/src/build-docker/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/cutlass-populate-gitinfo.txt" AND
  "/src/build-docker/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/cutlass-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/src/build-docker/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/cutlass-populate-gitinfo.txt")
  message(VERBOSE
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/src/build-docker/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/cutlass-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

# Even at VERBOSE level, we don't want to see the commands executed, but
# enabling them to be shown for DEBUG may be useful to help diagnose problems.
cmake_language(GET_MESSAGE_LOG_LEVEL active_log_level)
if(active_log_level MATCHES "DEBUG|TRACE")
  set(maybe_show_command COMMAND_ECHO STDOUT)
else()
  set(maybe_show_command "")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/src/build-docker/_deps/cutlass-src"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/src/build-docker/_deps/cutlass-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --depth 1 --no-single-branch --config "advice.detachedHead=false" "https://github.com/NVIDIA/cutlass.git" "cutlass-src"
    WORKING_DIRECTORY "/src/build-docker/_deps"
    RESULT_VARIABLE error_code
    ${maybe_show_command}
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(NOTICE "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/NVIDIA/cutlass.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "v4.4.2" --
  WORKING_DIRECTORY "/src/build-docker/_deps/cutlass-src"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v4.4.2'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/src/build-docker/_deps/cutlass-src"
    RESULT_VARIABLE error_code
    ${maybe_show_command}
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/src/build-docker/_deps/cutlass-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/src/build-docker/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/cutlass-populate-gitinfo.txt" "/src/build-docker/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/cutlass-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/src/build-docker/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/cutlass-populate-gitclone-lastrun.txt'")
endif()
