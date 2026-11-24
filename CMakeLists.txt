include_guard()

cmake_minimum_required(VERSION 3.28)
project(Simd)

if (PROJECT_IS_TOP_LEVEL)
	include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/All.cmake)
endif()

add_library(Simd INTERFACE)
target_include_directories(Simd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(Simd INTERFACE cxx_std_20)

if (PROJECT_IS_TOP_LEVEL AND CMAKE_TESTING_ENABLED)
	add_subdirectory(test)
endif()
