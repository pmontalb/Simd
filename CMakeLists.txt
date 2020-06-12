include_guard()

cmake_minimum_required(VERSION 3.14)
project(Simd)

include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)

create_library(
	NAME
		Simd
	SOURCES
		Simd.h AlignedAllocator.h
)
set_target_properties(Simd PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(gtest ${CMAKE_BINARY_DIR}/Simd/gtest EXCLUDE_FROM_ALL)
create_test(
	NAME
		SimdTests
	SOURCES
		UnitTests/Double2Tests.cpp
		UnitTests/Float4Tests.cpp
		UnitTests/Double4Tests.cpp
		UnitTests/Float8Tests.cpp
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

add_subdirectory(Profiler ${CMAKE_BINARY_DIR}/Profiler EXCLUDE_FROM_ALL)

create_executable(
	NAME
		Benchmark
	SOURCES
		benchmark.cpp
	DEPENDENCIES
		Simd Profiler
)

create_executable(
	NAME
		BenchmarkVector
	SOURCES
		benchmarkVector.cpp
	DEPENDENCIES
		Simd Profiler
)