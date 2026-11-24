if (NOT CMAKE_BUILD_TYPE STREQUAL "Msan")
	find_package(GTest REQUIRED)
	find_package(Threads REQUIRED)

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
		DEPENDENCIES
			Simd
		SYSTEM_DEPENDENCIES
			GTest::gtest Threads::Threads
	)
endif()
