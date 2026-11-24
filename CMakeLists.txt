include_guard()

cmake_minimum_required(VERSION 3.14)
project(Simd)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)
endif()

add_library(Simd INTERFACE)
target_include_directories(Simd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(Simd INTERFACE cxx_std_20)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND CMAKE_TESTING_ENABLED)

	add_subdirectory(test)

endif()
