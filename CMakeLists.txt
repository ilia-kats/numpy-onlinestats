cmake_minimum_required(VERSION 3.26)

project(numpy_onlinestats LANGUAGES CXX)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

if(APPLE)
  # https://github.com/Homebrew/homebrew-core/issues/112107 and
  # https://gitlab.kitware.com/cmake/cmake/-/issues/24097
  execute_process(COMMAND brew --prefix --installed libomp OUTPUT_VARIABLE OpenMP_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

find_package(OpenMP REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(CMAKE_BUILD_TYPE Release)

include_directories(digestible/include)
nanobind_add_module(_numpy_onlinestats_impl STABLE_ABI NB_STATIC src/onlinestats.cpp src/RunningStats.cpp)
set_property(TARGET _numpy_onlinestats_impl PROPERTY CXX_STANDARD 20)

# we can't use MinGW because of https://github.com/wjakob/nanobind/issues/47
# and CMake doesn't support the -openmp:llvm flag
if(MSVC)
  if(${MSVC_VERSION} LESS 1930)
    message(FATAL_ERROR "Your version of Visual Studio is too old.")
  endif()
  target_compile_options(_numpy_onlinestats_impl PRIVATE -openmp:llvm)
  target_link_options(_numpy_onlinestats_impl PRIVATE -openmp:llvm)
else()
  target_link_libraries(_numpy_onlinestats_impl PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS _numpy_onlinestats_impl LIBRARY DESTINATION numpy_onlinestats)
