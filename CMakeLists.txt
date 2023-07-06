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

find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_BUILD_TYPE Release)

include_directories(digestible/include)
nanobind_add_module(_numpy_onlinestats_impl STABLE_ABI NB_STATIC src/onlinestats.cpp)
set_property(TARGET _numpy_onlinestats_impl PROPERTY CXX_STANDARD 20)
target_compile_options(_numpy_onlinestats_impl PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_options(_numpy_onlinestats_impl PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(_numpy_onlinestats_impl PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS _numpy_onlinestats_impl LIBRARY DESTINATION numpy_onlinestats)
