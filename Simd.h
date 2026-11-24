#pragma once

#include "Config.h"
#include "Pack.h"
#include "Math.h"
#include "Algorithms.h"

#include <cstdint>

namespace simd
{
	// ========================================================================
	// Convenience Pack Aliases
	// ========================================================================

#if SIMD_HAS_SSE2
	using Double2 = Pack<double, 2>;
	using Float4  = Pack<float, 4>;
	using Int4    = Pack<std::int32_t, 4>;
	using Long2   = Pack<std::int64_t, 2>;
#endif

#if SIMD_HAS_AVX
	using Double4 = Pack<double, 4>;
	using Float8  = Pack<float, 8>;
#endif

#if SIMD_HAS_AVX2
	using Int8    = Pack<std::int32_t, 8>;
	using Long4   = Pack<std::int64_t, 4>;
#endif

#if SIMD_HAS_AVX512F
	using Double8 = Pack<double, 8>;
	using Float16 = Pack<float, 16>;
	using Int16   = Pack<std::int32_t, 16>;
	using Long8   = Pack<std::int64_t, 8>;
#endif

	// ========================================================================
	// Convenience Lowercase Math Aliases
	// ========================================================================

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> max(const Pack<T, Lanes> x, const Pack<T, Lanes> y) noexcept
	{
		return Max(x, y);
	}

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> min(const Pack<T, Lanes> x, const Pack<T, Lanes> y) noexcept
	{
		return Min(x, y);
	}

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> abs(const Pack<T, Lanes> x) noexcept
	{
		return Abs(x);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] SIMD_ALWAYS_INLINE Pack<T, Lanes> sqrt(const Pack<T, Lanes> x) noexcept
	{
		return Sqrt(x);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> exp(const Pack<T, Lanes> x) noexcept
	{
		return Exp(x);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> log(const Pack<T, Lanes> x) noexcept
	{
		return Log(x);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> sin(const Pack<T, Lanes> x) noexcept
	{
		return Sin(x);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> cos(const Pack<T, Lanes> x) noexcept
	{
		return Cos(x);
	}
}	 // namespace simd
