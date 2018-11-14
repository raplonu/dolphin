project(DolphinTest)

find_package(GTest REQUIRED)

add_executable(DolphinTest
    src/memoryTest.cu
    src/reduceTest.cu
    src/VectorTypeTest.cu
    src/mvm/MVMTest.cu
)

target_link_libraries(DolphinTest
    PRIVATE GTest::GTest GTest::Main Dolphin cublas
)

gtest_add_tests(TARGET DolphinTest
TEST_SUFFIX .noArgs
TEST_LIST   noArgsTests
)

set_tests_properties(${noArgsTests}   PROPERTIES TIMEOUT 10)