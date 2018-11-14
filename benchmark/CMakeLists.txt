project(DolphinBench LANGUAGES CXX CUDA)

find_package(benchmark REQUIRED)

add_executable(DolphinBench
    src/benchmark.cpp
    src/memoryBench.cu
    src/mvm/MVMBench.cu
    )

target_link_libraries(DolphinBench PRIVATE Dolphin benchmark pthread  cublas)
