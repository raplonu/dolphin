cmake_minimum_required(VERSION 3.6)

if(NOT TARGET CudaInterface::Cu)
    if(NOT DEFINED CUDAINTERFACE_ROOT)
        set(CUDAINTERFACE_ROOT $ENV{CUDAINTERFACE_ROOT})
    endif()
    if(IS_DIRECTORY ${CUDAINTERFACE_ROOT})
        add_subdirectory(${CUDAINTERFACE_ROOT} cudainterface_dir)
    else()
        message( FATAL_ERROR "CudaInterface path not found, specify it by set CUDAINTERFACE_ROOT to the right path" )
    endif()
endif()


if(NOT TARGET CUB)
    project(CUB LANGUAGES CUDA)
	add_library(CUB INTERFACE)
	target_include_directories(CUB INTERFACE tplib/cub-1.8.0)
endif()

if(NOT TARGET half)
	project(half LANGUAGES CXX)
	add_library(half INTERFACE)
	target_include_directories(half INTERFACE tplib/half)
endif()

project(Dolphin LANGUAGES CXX CUDA)

add_library(Dolphin INTERFACE)

target_compile_features(Dolphin INTERFACE cxx_std_14)

target_link_libraries(Dolphin INTERFACE CudaInterface::Cxx CudaInterface::Cu CUB half)

target_include_directories(Dolphin INTERFACE inc)

########################################################################
#
# Google Test.
#
# The tests are not built by default.  To build them, set the
# gtest_build_tests option to ON.  You can do it by running ccmake
# or specifying the -Dgtest_build_tests=ON flag when running cmake.

if (gtest_build_tests)
  # This must be set in the root directory for the tests to be run by
  # 'make test' or ctest.
  enable_testing()
  add_subdirectory(test)
endif()

if(benchmark_build)
    add_subdirectory(benchmark)
endif()