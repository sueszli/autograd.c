# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-src")
  file(MAKE_DIRECTORY "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-src")
endif()
file(MAKE_DIRECTORY
  "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-build"
  "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-subbuild/unity-populate-prefix"
  "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-subbuild/unity-populate-prefix/tmp"
  "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp"
  "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-subbuild/unity-populate-prefix/src"
  "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/sueszli/dev/autograd.c/debug_build/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
