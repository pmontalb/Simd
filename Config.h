#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

// ============================================================================
// ISA & Compiler Detection
// ============================================================================

#ifndef SIMD_HAS_AVX512F
	#if defined(__AVX512F__)
		#define SIMD_HAS_AVX512F 1
	#else
		#define SIMD_HAS_AVX512F 0
	#endif
#endif

#ifndef SIMD_HAS_AVX512DQ
	#if defined(__AVX512DQ__)
		#define SIMD_HAS_AVX512DQ 1
	#else
		#define SIMD_HAS_AVX512DQ 0
	#endif
#endif

#ifndef SIMD_HAS_AVX2
	#if defined(__AVX2__)
		#define SIMD_HAS_AVX2 1
	#else
		#define SIMD_HAS_AVX2 0
	#endif
#endif

#if defined(__AVX__)
	#define SIMD_HAS_AVX 1
#else
	#define SIMD_HAS_AVX 0
#endif

#if defined(__FMA__) || (defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__)))
	#define SIMD_HAS_FMA 1
#else
	#define SIMD_HAS_FMA 0
#endif

#if defined(__SSE4_1__) || (defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__)))
	#define SIMD_HAS_SSE4_1 1
#else
	#define SIMD_HAS_SSE4_1 0
#endif

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
	#define SIMD_HAS_SSE2 1
#else
	#define SIMD_HAS_SSE2 0
#endif

#if SIMD_HAS_AVX512F
	#define SIMD_ALIGNMENT_BYTES 64
#elif SIMD_HAS_AVX || SIMD_HAS_AVX2
	#define SIMD_ALIGNMENT_BYTES 32
#elif SIMD_HAS_SSE2
	#define SIMD_ALIGNMENT_BYTES 16
#else
	#define SIMD_ALIGNMENT_BYTES 8
#endif

// ============================================================================
// Warning-Safe Intrinsics Include
// ============================================================================

#include "IgnoreWarning.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
	IGNORE_WARNING_PUSH()
	#if defined(__clang__)
		IGNORE_WARNING("-Weverything")
	#elif defined(__GNUC__)
		IGNORE_WARNING("-Wold-style-cast")
		IGNORE_WARNING("-Wcast-qual")
		IGNORE_WARNING("-Wconversion")
		IGNORE_WARNING("-Wsign-conversion")
	#endif
	#if defined(_MSC_VER)
		#include <intrin.h>
	#endif
	#include <immintrin.h>
	IGNORE_WARNING_POP()
#endif

namespace simd
{
	template<typename T>
	concept SupportedElement =
		std::is_same_v<T, float> ||
		std::is_same_v<T, double> ||
		std::is_same_v<T, std::int32_t> ||
		std::is_same_v<T, std::int64_t>;

	inline constexpr std::size_t MaxAlignmentBytes = SIMD_ALIGNMENT_BYTES;

	template<SupportedElement T>
	[[nodiscard]] consteval std::size_t LaneCount() noexcept
	{
#if SIMD_HAS_AVX512F
		return 64 / sizeof(T);
#elif SIMD_HAS_AVX2
		return 32 / sizeof(T);
#elif SIMD_HAS_AVX
		if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
			return 32 / sizeof(T);
		else
			return 16 / sizeof(T);
#elif SIMD_HAS_SSE2
		return 16 / sizeof(T);
#else
		return 1;
#endif
	}

	template<SupportedElement T>
	[[nodiscard]] consteval std::size_t AlignmentBytes() noexcept
	{
		const std::size_t bytes = LaneCount<T>() * sizeof(T);
		return bytes > alignof(T) ? bytes : alignof(T);
	}

	template<SupportedElement T>
	[[nodiscard]] constexpr T* AssumeAligned(T* const ptr) noexcept
	{
		return std::assume_aligned<AlignmentBytes<T>()>(ptr);
	}

	template<SupportedElement T>
	[[nodiscard]] constexpr const T* AssumeAligned(const T* const ptr) noexcept
	{
		return std::assume_aligned<AlignmentBytes<T>()>(ptr);
	}

	namespace detail
	{
		[[nodiscard]] inline bool IsAligned(const void* ptr, const std::size_t alignment) noexcept
		{
			const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
			return (addr & (alignment - 1)) == 0;
		}
	}	 // namespace detail
}	 // namespace simd
