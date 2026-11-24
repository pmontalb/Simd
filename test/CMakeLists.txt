add_subdirectory(${CMAKE_SOURCE_DIR}/submodules/gtest ${CMAKE_BINARY_DIR}/Simd/gtest EXCLUDE_FROM_ALL)

create_test(
    NAME
        SimdTests
    SOURCES
        Double2Tests.cpp
        Float4Tests.cpp
        Double4Tests.cpp
        Float8Tests.cpp
        PackTests.cpp
        main.cpp
    PUBLIC_INCLUDE_DIRECTORIES
        ${GTEST_INCLUDE_DIR}
    DEPENDENCIES
        Simd
    SYSTEM_DEPENDENCIES
        gtest pthread
)