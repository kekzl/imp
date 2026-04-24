# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/src/build-docker/_deps/httplib-src")
  file(MAKE_DIRECTORY "/src/build-docker/_deps/httplib-src")
endif()
file(MAKE_DIRECTORY
  "/src/build-docker/_deps/httplib-build"
  "/src/build-docker/_deps/httplib-subbuild/httplib-populate-prefix"
  "/src/build-docker/_deps/httplib-subbuild/httplib-populate-prefix/tmp"
  "/src/build-docker/_deps/httplib-subbuild/httplib-populate-prefix/src/httplib-populate-stamp"
  "/src/build-docker/_deps/httplib-subbuild/httplib-populate-prefix/src"
  "/src/build-docker/_deps/httplib-subbuild/httplib-populate-prefix/src/httplib-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/src/build-docker/_deps/httplib-subbuild/httplib-populate-prefix/src/httplib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/src/build-docker/_deps/httplib-subbuild/httplib-populate-prefix/src/httplib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
