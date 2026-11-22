include_guard()

cmake_minimum_required(VERSION 3.14)
project(Simd)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)
endif()

add_library(Simd INTERFACE)
target_include_directories(Simd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(Simd INTERFACE cxx_std_20)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	add_subdirectory(gtest ${CMAKE_BINARY_DIR}/Simd/gtest EXCLUDE_FROM_ALL)
	create_test(
		NAME
			SimdTests
		SOURCES
			UnitTests/Double2Tests.cpp
			UnitTests/Float4Tests.cpp
			UnitTests/Double4Tests.cpp
			UnitTests/Float8Tests.cpp
			UnitTests/PackTests.cpp
			UnitTests/main.cpp
		PUBLIC_INCLUDE_DIRECTORIES
			${GTEST_INCLUDE_DIR}
		DEPENDENCIES
			Simd
		PRIVATE_INCLUDE_DIRECTORIES
			.
		SYSTEM_DEPENDENCIES
			gtest pthread
	)
endif()
