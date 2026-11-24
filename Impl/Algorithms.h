#pragma once

#include "Pack.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace simd
{
	// ========================================================================
	// Unrolled SIMD Loop Dispatcher
	// ========================================================================

	template<SupportedElement T, std::size_t Unroll = 4, typename UnrolledFn, typename SingleFn, typename TailFn>
	SIMD_ALWAYS_INLINE constexpr void ForEachUnrolled(const std::size_t count, UnrolledFn&& unrolledBody, SingleFn&& singleBody, TailFn&& tailBody) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || LaneCount<T>() == 1)
		{
			tailBody(std::size_t { 0 }, count);
			return;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t lanes = LaneCount<T>();
		constexpr std::size_t step = Unroll * lanes;

		std::size_t i = 0;
		for (; i + step <= count; i += step)
			unrolledBody(i);

		for (; i + lanes <= count; i += lanes)
			singleBody(i);

		if (i < count)
			tailBody(i, count - i);
	}

	// ========================================================================
	// Common Linear-Algebra Building Blocks
	// ========================================================================

	namespace detail
	{
		template<std::floating_point T>
		[[nodiscard]] SIMD_ALWAYS_INLINE T SubWidthDot(const T* const __restrict x, const T* const __restrict y, const std::size_t rem) noexcept
		{
#if SIMD_HAS_AVX512F || SIMD_HAS_AVX2
			if constexpr (std::same_as<T, double>)
			{
				if (rem < 2)
					return rem == 0 ? T { 0 } : x[0] * y[0];

				static constexpr bool hasAvx512 = LaneCount<double>() >= 8;
				__m128d folded;
				if (hasAvx512 && (rem & 4U) != 0U)
				{
					const auto acc = _mm256_mul_pd(_mm256_loadu_pd(x), _mm256_loadu_pd(y));
					folded = _mm_add_pd(_mm256_castpd256_pd128(acc), _mm256_extractf128_pd(acc, 1));
					if ((rem & 2U) != 0U)
						folded = _mm_add_pd(folded, _mm_mul_pd(_mm_loadu_pd(x + 4), _mm_loadu_pd(y + 4)));
				}
				else
					folded = _mm_mul_pd(_mm_loadu_pd(x), _mm_loadu_pd(y));

				const auto pairSum = _mm_add_sd(folded, _mm_unpackhi_pd(folded, folded));
				if ((rem & 1U) != 0U)
					return _mm_cvtsd_f64(_mm_add_sd(pairSum, _mm_mul_sd(_mm_load_sd(x + rem - 1), _mm_load_sd(y + rem - 1))));
				return _mm_cvtsd_f64(pairSum);
			}
			else if constexpr (std::same_as<T, float>)
			{
				if (rem < 4)
				{
					T sum = T { 0 };
					for (std::size_t j = 0; j < rem; ++j)
						sum += x[j] * y[j];
					return sum;
				}
				__m128 folded = _mm_mul_ps(_mm_loadu_ps(x), _mm_loadu_ps(y));
				if ((rem & 8U) != 0U)
				{
					const auto hi = _mm_mul_ps(_mm_loadu_ps(x + 4), _mm_loadu_ps(y + 4));
					folded = _mm_add_ps(folded, hi);
				}
				const auto shuf = _mm_movehdup_ps(folded);
				auto sums = _mm_add_ps(folded, shuf);
				sums = _mm_add_ss(sums, _mm_movehl_ps(sums, sums));
				T sum = _mm_cvtss_f32(sums);
				for (std::size_t j = rem & ~std::size_t { 3 }; j < rem; ++j)
					sum += x[j] * y[j];
				return sum;
			}
#endif
			T out = T { 0 };
			for (std::size_t i = 0; i < rem; ++i)
				out += x[i] * y[i];
			return out;
		}

		template<std::floating_point T>
		[[nodiscard]] SIMD_ALWAYS_INLINE T SubWidthEuclideanNormSquared(const T* const __restrict x, const std::size_t rem) noexcept
		{
			return SubWidthDot(x, x, rem);
		}

		template<std::floating_point T>
		[[nodiscard]] SIMD_ALWAYS_INLINE T SubWidthAbsSum(const T* const __restrict x, const std::size_t rem) noexcept
		{
#if SIMD_HAS_AVX512F || SIMD_HAS_AVX2
			if constexpr (std::same_as<T, double>)
			{
				if (rem < 2)
					return rem == 0 ? T { 0 } : std::abs(x[0]);

				static constexpr bool hasAvx512 = LaneCount<double>() >= 8;
				const auto signMask = _mm_set1_pd(-0.0);
				__m128d folded;
				if (hasAvx512 && (rem & 4U) != 0U)
				{
					const auto acc = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd(x));
					folded = _mm_add_pd(_mm256_castpd256_pd128(acc), _mm256_extractf128_pd(acc, 1));
					if ((rem & 2U) != 0U)
						folded = _mm_add_pd(folded, _mm_andnot_pd(signMask, _mm_loadu_pd(x + 4)));
				}
				else
					folded = _mm_andnot_pd(signMask, _mm_loadu_pd(x));

				const auto pairSum = _mm_add_sd(folded, _mm_unpackhi_pd(folded, folded));
				if ((rem & 1U) != 0U)
					return _mm_cvtsd_f64(_mm_add_sd(pairSum, _mm_andnot_pd(signMask, _mm_load_sd(x + rem - 1))));
				return _mm_cvtsd_f64(pairSum);
			}
#endif
			T out = T { 0 };
			for (std::size_t i = 0; i < rem; ++i)
				out += (x[i] < T { 0 } ? -x[i] : x[i]);
			return out;
		}

		template<std::floating_point T>
		[[nodiscard]] SIMD_ALWAYS_INLINE T SubWidthAbsMax(const T* const __restrict x, const std::size_t rem) noexcept
		{
#if SIMD_HAS_AVX512F || SIMD_HAS_AVX2
			if constexpr (std::same_as<T, double>)
			{
				if (rem < 2)
					return rem == 0 ? T { 0 } : std::abs(x[0]);

				static constexpr bool hasAvx512 = LaneCount<double>() >= 8;
				const auto signMask = _mm_set1_pd(-0.0);
				__m128d folded;
				if (hasAvx512 && (rem & 4U) != 0U)
				{
					const auto acc = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd(x));
					folded = _mm_max_pd(_mm256_castpd256_pd128(acc), _mm256_extractf128_pd(acc, 1));
					if ((rem & 2U) != 0U)
						folded = _mm_max_pd(folded, _mm_andnot_pd(signMask, _mm_loadu_pd(x + 4)));
				}
				else
					folded = _mm_andnot_pd(signMask, _mm_loadu_pd(x));

				const auto pairMax = _mm_max_sd(folded, _mm_unpackhi_pd(folded, folded));
				if ((rem & 1U) != 0U)
					return _mm_cvtsd_f64(_mm_max_sd(pairMax, _mm_andnot_pd(signMask, _mm_load_sd(x + rem - 1))));
				return _mm_cvtsd_f64(pairMax);
			}
#endif
			T out = T { 0 };
			for (std::size_t i = 0; i < rem; ++i)
			{
				const T v = x[i] < T { 0 } ? -x[i] : x[i];
				out = v > out ? v : out;
			}
			return out;
		}

#if SIMD_HAS_AVX512F
		template<std::floating_point T, std::size_t Unroll = 1>
		[[nodiscard]] SIMD_ALWAYS_INLINE T HalfWidthDotProduct(const T* const __restrict x, const T* const __restrict y, const std::size_t count) noexcept
		{
			constexpr std::size_t halfWidth = LaneCount<T>() / 2;
			Pack<T, halfWidth> acc[Unroll] {};
			for (std::size_t k = 0; k < Unroll; ++k)
				acc[k] = Pack<T, halfWidth>::Zero();

			std::size_t i = 0;
			for (; i + Unroll * halfWidth <= count; i += Unroll * halfWidth)
			{
				for (std::size_t k = 0; k < Unroll; ++k)
					acc[k] = Fma(LoadUnaligned<T, halfWidth>(x + i + k * halfWidth), LoadUnaligned<T, halfWidth>(y + i + k * halfWidth), acc[k]);
			}
			for (; i + halfWidth <= count; i += halfWidth)
				acc[0] = Fma(LoadUnaligned<T, halfWidth>(x + i), LoadUnaligned<T, halfWidth>(y + i), acc[0]);
			for (std::size_t k = 1; k < Unroll; ++k)
				acc[0] += acc[k];

			T out = HorizontalSum(acc[0]);
			if (i < count)
				out += SubWidthDot(x + i, y + i, count - i);
			return out;
		}

		template<std::floating_point T, std::size_t Unroll = 1>
		[[nodiscard]] SIMD_ALWAYS_INLINE T HalfWidthEuclideanNormSquared(const T* const __restrict x, const std::size_t count) noexcept
		{
			constexpr std::size_t halfWidth = LaneCount<T>() / 2;
			Pack<T, halfWidth> acc[Unroll] {};
			for (std::size_t k = 0; k < Unroll; ++k)
				acc[k] = Pack<T, halfWidth>::Zero();

			std::size_t i = 0;
			for (; i + Unroll * halfWidth <= count; i += Unroll * halfWidth)
			{
				for (std::size_t k = 0; k < Unroll; ++k)
				{
					const auto vx = LoadUnaligned<T, halfWidth>(x + i + k * halfWidth);
					acc[k] = Fma(vx, vx, acc[k]);
				}
			}
			for (; i + halfWidth <= count; i += halfWidth)
			{
				const auto vx = LoadUnaligned<T, halfWidth>(x + i);
				acc[0] = Fma(vx, vx, acc[0]);
			}
			for (std::size_t k = 1; k < Unroll; ++k)
				acc[0] += acc[k];

			T out = HorizontalSum(acc[0]);
			if (i < count)
				out += SubWidthEuclideanNormSquared(x + i, count - i);
			return out;
		}

		template<std::floating_point T, std::size_t Unroll = 1>
		[[nodiscard]] SIMD_ALWAYS_INLINE T HalfWidthReduceAbsSum(const T* const __restrict x, const std::size_t count) noexcept
		{
			constexpr std::size_t halfWidth = LaneCount<T>() / 2;
			Pack<T, halfWidth> acc[Unroll] {};
			for (std::size_t k = 0; k < Unroll; ++k)
				acc[k] = Pack<T, halfWidth>::Zero();

			std::size_t i = 0;
			for (; i + Unroll * halfWidth <= count; i += Unroll * halfWidth)
			{
				for (std::size_t k = 0; k < Unroll; ++k)
					acc[k] += Abs(LoadUnaligned<T, halfWidth>(x + i + k * halfWidth));
			}
			for (; i + halfWidth <= count; i += halfWidth)
				acc[0] += Abs(LoadUnaligned<T, halfWidth>(x + i));
			for (std::size_t k = 1; k < Unroll; ++k)
				acc[0] += acc[k];

			T out = HorizontalSum(acc[0]);
			if (i < count)
				out += SubWidthAbsSum(x + i, count - i);
			return out;
		}

		template<std::floating_point T, std::size_t Unroll = 1>
		[[nodiscard]] SIMD_ALWAYS_INLINE T HalfWidthReduceSum(const T* const __restrict x, const std::size_t count) noexcept
		{
			constexpr std::size_t halfWidth = LaneCount<T>() / 2;
			Pack<T, halfWidth> acc[Unroll] {};
			for (std::size_t k = 0; k < Unroll; ++k)
				acc[k] = Pack<T, halfWidth>::Zero();

			std::size_t i = 0;
			for (; i + Unroll * halfWidth <= count; i += Unroll * halfWidth)
			{
				for (std::size_t k = 0; k < Unroll; ++k)
					acc[k] += LoadUnaligned<T, halfWidth>(x + i + k * halfWidth);
			}
			for (; i + halfWidth <= count; i += halfWidth)
				acc[0] += LoadUnaligned<T, halfWidth>(x + i);
			for (std::size_t k = 1; k < Unroll; ++k)
				acc[0] += acc[k];

			T out = HorizontalSum(acc[0]);
			for (; i < count; ++i)
				out += x[i];
			return out;
		}

		template<std::floating_point T, std::size_t Unroll = 1>
		[[nodiscard]] SIMD_ALWAYS_INLINE T HalfWidthReduceAbsMax(const T* const __restrict x, const std::size_t count) noexcept
		{
			constexpr std::size_t halfWidth = LaneCount<T>() / 2;
			Pack<T, halfWidth> acc[Unroll] {};
			for (std::size_t k = 0; k < Unroll; ++k)
				acc[k] = Pack<T, halfWidth>::Zero();

			std::size_t i = 0;
			for (; i + Unroll * halfWidth <= count; i += Unroll * halfWidth)
			{
				for (std::size_t k = 0; k < Unroll; ++k)
					acc[k] = Max(acc[k], Abs(LoadUnaligned<T, halfWidth>(x + i + k * halfWidth)));
			}
			for (; i + halfWidth <= count; i += halfWidth)
				acc[0] = Max(acc[0], Abs(LoadUnaligned<T, halfWidth>(x + i)));
			for (std::size_t k = 1; k < Unroll; ++k)
				acc[0] = Max(acc[0], acc[k]);

			T out = HorizontalMax(acc[0]);
			if (i < count)
			{
				const T tail = SubWidthAbsMax(x + i, count - i);
				out = tail > out ? tail : out;
			}
			return out;
		}
#endif
	}	 // namespace detail

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T DotProduct(const T* const __restrict x, const T* const __restrict y, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || LaneCount<T>() == 1)
		{
			T acc = T { 0 };
			for (std::size_t i = 0; i < count; ++i)
				acc += x[i] * y[i];
			return acc;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t lanes = LaneCount<T>();
		constexpr std::size_t step = Unroll * lanes;

#if SIMD_HAS_AVX512F
		if constexpr (Unroll == 1 && lanes >= 8)
		{
			if (count < 2 * lanes)
			{
				constexpr std::size_t half = lanes / 2;
				auto hTotal = Pack<T, half>::Zero();
				std::size_t j = 0;
				for (; j + half <= count; j += half)
					hTotal = Fma(LoadUnaligned<T, half>(x + j), LoadUnaligned<T, half>(y + j), hTotal);
				T sum = HorizontalSum(hTotal);
				if (j < count)
					sum += detail::SubWidthDot(x + j, y + j, count - j);
				return sum;
			}
		}
#endif

#if !SIMD_HAS_AVX512F
		if constexpr (Unroll == 1)
		{
			auto total = LoadUnaligned<T>(x) * LoadUnaligned<T>(y);
			std::size_t i = lanes;
			for (; i + lanes <= count; i += lanes)
				total = Fma(LoadUnaligned<T>(x + i), LoadUnaligned<T>(y + i), total);
			T sum = HorizontalSum(total);
			if (i < count)
				sum += detail::SubWidthDot(x + i, y + i, count - i);
			return sum;
		}
#endif

		std::size_t i = 0;
		auto total = Pack<T>::Zero();
		if constexpr (Unroll == 4)
		{
			auto acc0 = Pack<T>::Zero();
			auto acc1 = Pack<T>::Zero();
			auto acc2 = Pack<T>::Zero();
			auto acc3 = Pack<T>::Zero();
			for (; i + step <= count; i += step)
			{
				acc0 = Fma(LoadUnaligned<T>(x + i), LoadUnaligned<T>(y + i), acc0);
				acc1 = Fma(LoadUnaligned<T>(x + i + lanes), LoadUnaligned<T>(y + i + lanes), acc1);
				acc2 = Fma(LoadUnaligned<T>(x + i + 2 * lanes), LoadUnaligned<T>(y + i + 2 * lanes), acc2);
				acc3 = Fma(LoadUnaligned<T>(x + i + 3 * lanes), LoadUnaligned<T>(y + i + 3 * lanes), acc3);
			}
			total = (acc0 + acc1) + (acc2 + acc3);
		}
		else
		{
			Pack<T> acc[Unroll] {};
			for (; i + step <= count; i += step)
			{
				for (std::size_t u = 0; u < Unroll; ++u)
					acc[u] = Fma(LoadUnaligned<T>(x + i + u * lanes), LoadUnaligned<T>(y + i + u * lanes), acc[u]);
			}
			for (std::size_t u = 0; u < Unroll; ++u)
				total += acc[u];
		}

		for (; i + lanes <= count; i += lanes)
			total = Fma(LoadUnaligned<T>(x + i), LoadUnaligned<T>(y + i), total);

#if SIMD_HAS_AVX512F
		if (i < count)
		{
			const std::size_t rem = count - i;
			total = Fma(LoadPartial<T>(x + i, rem), LoadPartial<T>(y + i, rem), total);
		}
		return HorizontalSum(total);
#else
		T sum = HorizontalSum(total);
		if (i < count)
			sum += detail::SubWidthDot(x + i, y + i, count - i);
		return sum;
#endif
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T ReduceSum(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || LaneCount<T>() == 1)
		{
			T acc = T { 0 };
			for (std::size_t i = 0; i < count; ++i)
				acc += x[i];
			return acc;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t lanes = LaneCount<T>();
		constexpr std::size_t step = Unroll * lanes;

		std::size_t i = 0;
		auto total = Pack<T>::Zero();
		if constexpr (Unroll == 4)
		{
			auto acc0 = Pack<T>::Zero();
			auto acc1 = Pack<T>::Zero();
			auto acc2 = Pack<T>::Zero();
			auto acc3 = Pack<T>::Zero();
			for (; i + step <= count; i += step)
			{
				acc0 += LoadUnaligned<T>(x + i);
				acc1 += LoadUnaligned<T>(x + i + lanes);
				acc2 += LoadUnaligned<T>(x + i + 2 * lanes);
				acc3 += LoadUnaligned<T>(x + i + 3 * lanes);
			}
			total = (acc0 + acc1) + (acc2 + acc3);
		}
		else
		{
			Pack<T> acc[Unroll] {};
			for (; i + step <= count; i += step)
			{
				for (std::size_t u = 0; u < Unroll; ++u)
					acc[u] += LoadUnaligned<T>(x + i + u * lanes);
			}
			for (std::size_t u = 0; u < Unroll; ++u)
				total += acc[u];
		}

		for (; i + lanes <= count; i += lanes)
			total += LoadUnaligned<T>(x + i);

#if SIMD_HAS_AVX512F
		if (i < count)
			total += LoadPartial<T>(x + i, count - i);
		return HorizontalSum(total);
#else
		T sum = HorizontalSum(total);
		for (; i < count; ++i)
			sum += x[i];
		return sum;
#endif
	}

	template<std::floating_point T>
	[[nodiscard]] constexpr std::size_t ArgMaxAbs(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || count < LaneCount<T>() || LaneCount<T>() == 1)
		{
			std::size_t bestIdx = 0;
			T bestVal = x[0] < T { 0 } ? -x[0] : x[0];
			for (std::size_t i = 1; i < count; ++i)
			{
				const T v = x[i] < T { 0 } ? -x[i] : x[i];
				if (v > bestVal)
				{
					bestVal = v;
					bestIdx = i;
				}
			}
			return bestIdx;
		}
		// LCOV_EXCL_STOP

		using IndexT = IndexTypeForT<T>;
		constexpr std::size_t lanes = LaneCount<T>();

		auto bestVals = Abs(LoadUnaligned<T>(x));
		auto bestIdxs = Iota<IndexT, lanes>(0, 1);
		const auto stepIdx = Broadcast<IndexT, lanes>(static_cast<IndexT>(lanes));
		auto curIdxs = bestIdxs + stepIdx;

		std::size_t i = lanes;
		for (; i + lanes <= count; i += lanes)
		{
			const auto vals = Abs(LoadUnaligned<T>(x + i));
			const auto mask = CmpGreater(vals, bestVals);
			bestVals = Blend(mask, vals, bestVals);
			bestIdxs = Blend(mask, curIdxs, bestIdxs);
			curIdxs += stepIdx;
		}

		const auto valArr = bestVals.ToArray();
		const auto idxArr = bestIdxs.ToArray();
		T bestVal = valArr[0];
		std::size_t bestIdx = static_cast<std::size_t>(idxArr[0]);
		for (std::size_t k = 1; k < lanes; ++k)
		{
			const auto kIdx = static_cast<std::size_t>(idxArr[k]);
			if (valArr[k] > bestVal || (valArr[k] >= bestVal && kIdx < bestIdx))
			{
				bestVal = valArr[k];
				bestIdx = kIdx;
			}
		}

		for (; i < count; ++i)
		{
			const T v = x[i] < T { 0 } ? -x[i] : x[i];
			if (v > bestVal)
			{
				bestVal = v;
				bestIdx = i;
			}
		}

		return bestIdx;
	}

	template<std::floating_point T>
	[[nodiscard]] constexpr std::size_t ArgMinAbs(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || count < LaneCount<T>() || LaneCount<T>() == 1)
		{
			std::size_t bestIdx = 0;
			T bestVal = x[0] < T { 0 } ? -x[0] : x[0];
			for (std::size_t i = 1; i < count; ++i)
			{
				const T v = x[i] < T { 0 } ? -x[i] : x[i];
				if (v < bestVal)
				{
					bestVal = v;
					bestIdx = i;
				}
			}
			return bestIdx;
		}
		// LCOV_EXCL_STOP

		using IndexT = IndexTypeForT<T>;
		constexpr std::size_t lanes = LaneCount<T>();

		auto bestVals = Abs(LoadUnaligned<T>(x));
		auto bestIdxs = Iota<IndexT, lanes>(0, 1);
		const auto stepIdx = Broadcast<IndexT, lanes>(static_cast<IndexT>(lanes));
		auto curIdxs = bestIdxs + stepIdx;

		std::size_t i = lanes;
		for (; i + lanes <= count; i += lanes)
		{
			const auto vals = Abs(LoadUnaligned<T>(x + i));
			const auto mask = CmpLess(vals, bestVals);
			bestVals = Blend(mask, vals, bestVals);
			bestIdxs = Blend(mask, curIdxs, bestIdxs);
			curIdxs += stepIdx;
		}

		const auto valArr = bestVals.ToArray();
		const auto idxArr = bestIdxs.ToArray();
		T bestVal = valArr[0];
		std::size_t bestIdx = static_cast<std::size_t>(idxArr[0]);
		for (std::size_t k = 1; k < lanes; ++k)
		{
			const auto kIdx = static_cast<std::size_t>(idxArr[k]);
			if (valArr[k] < bestVal || (valArr[k] <= bestVal && kIdx < bestIdx))
			{
				bestVal = valArr[k];
				bestIdx = kIdx;
			}
		}

		for (; i < count; ++i)
		{
			const T v = x[i] < T { 0 } ? -x[i] : x[i];
			if (v < bestVal)
			{
				bestVal = v;
				bestIdx = i;
			}
		}

		return bestIdx;
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	SIMD_ALWAYS_INLINE constexpr void Axpy(T* const __restrict y, const T* const __restrict x, const T alpha, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || LaneCount<T>() == 1)
		{
			for (std::size_t i = 0; i < count; ++i)
				y[i] += alpha * x[i];
			return;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t lanes = LaneCount<T>();
		constexpr std::size_t step = Unroll * lanes;

#if !SIMD_HAS_AVX512F
		if (count < lanes)
		{
			for (std::size_t j = 0; j < count; ++j)
				y[j] += alpha * x[j];
			return;
		}
#endif

		const auto vAlpha = Broadcast<T>(alpha);

		std::size_t i = 0;
#if !SIMD_HAS_AVX512F
		if (count < step)
		{
			for (; i + lanes <= count; i += lanes)
				StoreUnaligned(y + i, Fma(vAlpha, LoadUnaligned<T>(x + i), LoadUnaligned<T>(y + i)));
			for (; i < count; ++i)
				y[i] += alpha * x[i];
			return;
		}
		if (count <= std::size_t { 8192 })
		{
			for (; i + step <= count; i += step)
			{
				for (std::size_t u = 0; u < Unroll; ++u)
				{
					const std::size_t idx = i + u * lanes;
					const auto vx = LoadUnaligned<T>(x + idx);
					const auto vy = LoadUnaligned<T>(y + idx);
					StoreUnaligned(y + idx, Fma(vAlpha, vx, vy));
				}
			}
		}
		else
#endif
		{
			for (; i + step <= count; i += step)
			{
#if !SIMD_HAS_AVX512F
				__builtin_prefetch(x + i + 64, 0, 1);
				__builtin_prefetch(y + i + 64, 1, 1);
#endif
				for (std::size_t u = 0; u < Unroll; ++u)
				{
					const std::size_t idx = i + u * lanes;
					const auto vx = LoadUnaligned<T>(x + idx);
					const auto vy = LoadUnaligned<T>(y + idx);
					StoreUnaligned(y + idx, Fma(vAlpha, vx, vy));
				}
			}
		}

		for (; i + lanes <= count; i += lanes)
			StoreUnaligned(y + i, Fma(vAlpha, LoadUnaligned<T>(x + i), LoadUnaligned<T>(y + i)));

#if SIMD_HAS_AVX512F
		if (i < count)
		{
			const std::size_t rem = count - i;
			const auto vx = LoadPartial<T>(x + i, rem);
			const auto vy = LoadPartial<T>(y + i, rem);
			StorePartial(y + i, Fma(vAlpha, vx, vy), rem);
		}
#else
		for (; i < count; ++i)
			y[i] += alpha * x[i];
#endif
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T EuclideanNormSquared(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || LaneCount<T>() == 1)
		{
			T acc = T { 0 };
			for (std::size_t i = 0; i < count; ++i)
				acc += x[i] * x[i];
			return acc;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t lanes = LaneCount<T>();
		constexpr std::size_t step = Unroll * lanes;

#if !SIMD_HAS_AVX512F
		if constexpr (Unroll == 1)
		{
			const auto vx0 = LoadUnaligned<T>(x);
			auto total = vx0 * vx0;
			std::size_t i = lanes;
			for (; i + lanes <= count; i += lanes)
			{
				const auto vx = LoadUnaligned<T>(x + i);
				total = Fma(vx, vx, total);
			}
			T sum = HorizontalSum(total);
			if (i < count)
				sum += detail::SubWidthEuclideanNormSquared(x + i, count - i);
			return sum;
		}
#endif

		std::size_t i = 0;
		auto total = Pack<T>::Zero();
		if constexpr (Unroll == 4)
		{
			auto acc0 = Pack<T>::Zero();
			auto acc1 = Pack<T>::Zero();
			auto acc2 = Pack<T>::Zero();
			auto acc3 = Pack<T>::Zero();
			for (; i + step <= count; i += step)
			{
				const auto v0 = LoadUnaligned<T>(x + i);
				const auto v1 = LoadUnaligned<T>(x + i + lanes);
				const auto v2 = LoadUnaligned<T>(x + i + 2 * lanes);
				const auto v3 = LoadUnaligned<T>(x + i + 3 * lanes);
				acc0 = Fma(v0, v0, acc0);
				acc1 = Fma(v1, v1, acc1);
				acc2 = Fma(v2, v2, acc2);
				acc3 = Fma(v3, v3, acc3);
			}
			total = (acc0 + acc1) + (acc2 + acc3);
		}
		else
		{
			Pack<T> acc[Unroll] {};
			for (; i + step <= count; i += step)
			{
				for (std::size_t u = 0; u < Unroll; ++u)
				{
					const auto vx = LoadUnaligned<T>(x + i + u * lanes);
					acc[u] = Fma(vx, vx, acc[u]);
				}
			}
			for (std::size_t u = 0; u < Unroll; ++u)
				total += acc[u];
		}

		for (; i + lanes <= count; i += lanes)
		{
			const auto vx = LoadUnaligned<T>(x + i);
			total = Fma(vx, vx, total);
		}

#if SIMD_HAS_AVX512F
		if (i < count)
		{
			const auto vx = LoadPartial<T>(x + i, count - i);
			total = Fma(vx, vx, total);
		}
		return HorizontalSum(total);
#else
		T sum = HorizontalSum(total);
		if (i < count)
			sum += detail::SubWidthEuclideanNormSquared(x + i, count - i);
		return sum;
#endif
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T ReduceAbsSum(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || LaneCount<T>() == 1)
		{
			T acc = T { 0 };
			for (std::size_t i = 0; i < count; ++i)
				acc += (x[i] < T { 0 } ? -x[i] : x[i]);
			return acc;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t lanes = LaneCount<T>();
		constexpr std::size_t step = Unroll * lanes;

#if !SIMD_HAS_AVX512F
		if constexpr (Unroll == 1)
		{
			auto total = Abs(LoadUnaligned<T>(x));
			std::size_t i = lanes;
			for (; i + lanes <= count; i += lanes)
				total += Abs(LoadUnaligned<T>(x + i));
			T sum = HorizontalSum(total);
			if (i < count)
				sum += detail::SubWidthAbsSum(x + i, count - i);
			return sum;
		}
#endif

		std::size_t i = 0;
		auto total = Pack<T>::Zero();
		if constexpr (Unroll == 4)
		{
			auto acc0 = Pack<T>::Zero();
			auto acc1 = Pack<T>::Zero();
			auto acc2 = Pack<T>::Zero();
			auto acc3 = Pack<T>::Zero();
			for (; i + step <= count; i += step)
			{
				acc0 += Abs(LoadUnaligned<T>(x + i));
				acc1 += Abs(LoadUnaligned<T>(x + i + lanes));
				acc2 += Abs(LoadUnaligned<T>(x + i + 2 * lanes));
				acc3 += Abs(LoadUnaligned<T>(x + i + 3 * lanes));
			}
			total = (acc0 + acc1) + (acc2 + acc3);
		}
		else
		{
			Pack<T> acc[Unroll] {};
			for (; i + step <= count; i += step)
			{
				for (std::size_t u = 0; u < Unroll; ++u)
					acc[u] += Abs(LoadUnaligned<T>(x + i + u * lanes));
			}
			for (std::size_t u = 0; u < Unroll; ++u)
				total += acc[u];
		}

		for (; i + lanes <= count; i += lanes)
			total += Abs(LoadUnaligned<T>(x + i));

#if SIMD_HAS_AVX512F
		if (i < count)
			total += Abs(LoadPartial<T>(x + i, count - i));
		return HorizontalSum(total);
#else
		T sum = HorizontalSum(total);
		if (i < count)
			sum += detail::SubWidthAbsSum(x + i, count - i);
		return sum;
#endif
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T ReduceMin(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || count < lanes || lanes == 1)
		{
			T best = x[0];
			for (std::size_t i = 1; i < count; ++i)
				best = x[i] < best ? x[i] : best;
			return best;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t step = Unroll * lanes;
		std::size_t i = 0;
		Pack<T> bestPack;

		if (count >= step)
		{
			if constexpr (Unroll == 4)
			{
				auto acc0 = LoadUnaligned<T>(x);
				auto acc1 = LoadUnaligned<T>(x + lanes);
				auto acc2 = LoadUnaligned<T>(x + 2 * lanes);
				auto acc3 = LoadUnaligned<T>(x + 3 * lanes);
				i = step;
				for (; i + step <= count; i += step)
				{
					acc0 = Min(acc0, LoadUnaligned<T>(x + i));
					acc1 = Min(acc1, LoadUnaligned<T>(x + i + lanes));
					acc2 = Min(acc2, LoadUnaligned<T>(x + i + 2 * lanes));
					acc3 = Min(acc3, LoadUnaligned<T>(x + i + 3 * lanes));
				}
				bestPack = Min(Min(acc0, acc1), Min(acc2, acc3));
			}
			else
			{
				Pack<T> acc[Unroll];
				for (std::size_t u = 0; u < Unroll; ++u)
					acc[u] = LoadUnaligned<T>(x + u * lanes);
				i = step;
				for (; i + step <= count; i += step)
				{
					for (std::size_t u = 0; u < Unroll; ++u)
						acc[u] = Min(acc[u], LoadUnaligned<T>(x + i + u * lanes));
				}
				for (std::size_t u = 1; u < Unroll; ++u)
					acc[0] = Min(acc[0], acc[u]);
				bestPack = acc[0];
			}
		}
		else
		{
			bestPack = LoadUnaligned<T>(x);
			i = lanes;
		}

		for (; i + lanes <= count; i += lanes)
			bestPack = Min(bestPack, LoadUnaligned<T>(x + i));

		T best = HorizontalMin(bestPack);
		for (; i < count; ++i)
			best = x[i] < best ? x[i] : best;
		return best;
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T ReduceMax(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || count < lanes || lanes == 1)
		{
			T best = x[0];
			for (std::size_t i = 1; i < count; ++i)
				best = x[i] > best ? x[i] : best;
			return best;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t step = Unroll * lanes;
		std::size_t i = 0;
		Pack<T> bestPack;

		if (count >= step)
		{
			if constexpr (Unroll == 4)
			{
				auto acc0 = LoadUnaligned<T>(x);
				auto acc1 = LoadUnaligned<T>(x + lanes);
				auto acc2 = LoadUnaligned<T>(x + 2 * lanes);
				auto acc3 = LoadUnaligned<T>(x + 3 * lanes);
				i = step;
				for (; i + step <= count; i += step)
				{
					acc0 = Max(acc0, LoadUnaligned<T>(x + i));
					acc1 = Max(acc1, LoadUnaligned<T>(x + i + lanes));
					acc2 = Max(acc2, LoadUnaligned<T>(x + i + 2 * lanes));
					acc3 = Max(acc3, LoadUnaligned<T>(x + i + 3 * lanes));
				}
				bestPack = Max(Max(acc0, acc1), Max(acc2, acc3));
			}
			else
			{
				Pack<T> acc[Unroll];
				for (std::size_t u = 0; u < Unroll; ++u)
					acc[u] = LoadUnaligned<T>(x + u * lanes);
				i = step;
				for (; i + step <= count; i += step)
				{
					for (std::size_t u = 0; u < Unroll; ++u)
						acc[u] = Max(acc[u], LoadUnaligned<T>(x + i + u * lanes));
				}
				for (std::size_t u = 1; u < Unroll; ++u)
					acc[0] = Max(acc[0], acc[u]);
				bestPack = acc[0];
			}
		}
		else
		{
			bestPack = LoadUnaligned<T>(x);
			i = lanes;
		}

		for (; i + lanes <= count; i += lanes)
			bestPack = Max(bestPack, LoadUnaligned<T>(x + i));

		T best = HorizontalMax(bestPack);
		for (; i < count; ++i)
			best = x[i] > best ? x[i] : best;
		return best;
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T ReduceAbsMin(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || count < lanes || lanes == 1)
		{
			T best = x[0] < T { 0 } ? -x[0] : x[0];
			for (std::size_t i = 1; i < count; ++i)
			{
				const T v = x[i] < T { 0 } ? -x[i] : x[i];
				best = v < best ? v : best;
			}
			return best;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t step = Unroll * lanes;
		std::size_t i = 0;
		Pack<T> bestPack;

		if (count >= step)
		{
			if constexpr (Unroll == 4)
			{
				auto acc0 = Abs(LoadUnaligned<T>(x));
				auto acc1 = Abs(LoadUnaligned<T>(x + lanes));
				auto acc2 = Abs(LoadUnaligned<T>(x + 2 * lanes));
				auto acc3 = Abs(LoadUnaligned<T>(x + 3 * lanes));
				i = step;
				for (; i + step <= count; i += step)
				{
					acc0 = Min(acc0, Abs(LoadUnaligned<T>(x + i)));
					acc1 = Min(acc1, Abs(LoadUnaligned<T>(x + i + lanes)));
					acc2 = Min(acc2, Abs(LoadUnaligned<T>(x + i + 2 * lanes)));
					acc3 = Min(acc3, Abs(LoadUnaligned<T>(x + i + 3 * lanes)));
				}
				bestPack = Min(Min(acc0, acc1), Min(acc2, acc3));
			}
			else
			{
				Pack<T> acc[Unroll];
				for (std::size_t u = 0; u < Unroll; ++u)
					acc[u] = Abs(LoadUnaligned<T>(x + u * lanes));
				i = step;
				for (; i + step <= count; i += step)
				{
					for (std::size_t u = 0; u < Unroll; ++u)
						acc[u] = Min(acc[u], Abs(LoadUnaligned<T>(x + i + u * lanes)));
				}
				for (std::size_t u = 1; u < Unroll; ++u)
					acc[0] = Min(acc[0], acc[u]);
				bestPack = acc[0];
			}
		}
		else
		{
			bestPack = Abs(LoadUnaligned<T>(x));
			i = lanes;
		}

		for (; i + lanes <= count; i += lanes)
			bestPack = Min(bestPack, Abs(LoadUnaligned<T>(x + i)));

		T best = HorizontalMin(bestPack);
		for (; i < count; ++i)
		{
			const T v = x[i] < T { 0 } ? -x[i] : x[i];
			best = v < best ? v : best;
		}
		return best;
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T ReduceAbsMax(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || count < lanes || lanes == 1)
		{
			T best = x[0] < T { 0 } ? -x[0] : x[0];
			for (std::size_t i = 1; i < count; ++i)
			{
				const T v = x[i] < T { 0 } ? -x[i] : x[i];
				best = v > best ? v : best;
			}
			return best;
		}
		// LCOV_EXCL_STOP

		constexpr std::size_t step = Unroll * lanes;
		std::size_t i = 0;
		Pack<T> bestPack;

		if (count >= step)
		{
			if constexpr (Unroll == 4)
			{
				auto acc0 = Abs(LoadUnaligned<T>(x));
				auto acc1 = Abs(LoadUnaligned<T>(x + lanes));
				auto acc2 = Abs(LoadUnaligned<T>(x + 2 * lanes));
				auto acc3 = Abs(LoadUnaligned<T>(x + 3 * lanes));
				i = step;
				for (; i + step <= count; i += step)
				{
					acc0 = Max(acc0, Abs(LoadUnaligned<T>(x + i)));
					acc1 = Max(acc1, Abs(LoadUnaligned<T>(x + i + lanes)));
					acc2 = Max(acc2, Abs(LoadUnaligned<T>(x + i + 2 * lanes)));
					acc3 = Max(acc3, Abs(LoadUnaligned<T>(x + i + 3 * lanes)));
				}
				bestPack = Max(Max(acc0, acc1), Max(acc2, acc3));
			}
			else
			{
				Pack<T> acc[Unroll];
				for (std::size_t u = 0; u < Unroll; ++u)
					acc[u] = Abs(LoadUnaligned<T>(x + u * lanes));
				i = step;
				for (; i + step <= count; i += step)
				{
					for (std::size_t u = 0; u < Unroll; ++u)
						acc[u] = Max(acc[u], Abs(LoadUnaligned<T>(x + i + u * lanes)));
				}
				for (std::size_t u = 1; u < Unroll; ++u)
					acc[0] = Max(acc[0], acc[u]);
				bestPack = acc[0];
			}
		}
		else
		{
			bestPack = Abs(LoadUnaligned<T>(x));
			i = lanes;
		}

		for (; i + lanes <= count; i += lanes)
			bestPack = Max(bestPack, Abs(LoadUnaligned<T>(x + i)));

		T best = HorizontalMax(bestPack);
		for (; i < count; ++i)
		{
			const T v = x[i] < T { 0 } ? -x[i] : x[i];
			best = v > best ? v : best;
		}
		return best;
	}

	// ========================================================================
	// Strided SIMD Algorithms
	// ========================================================================

	template<std::floating_point T>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T DotProduct(const T* const __restrict x, const std::size_t incX, const T* const __restrict y, const std::size_t incY,
																   const std::size_t count) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		auto acc0 = Pack<T>::Zero();
		auto acc1 = Pack<T>::Zero();
		auto acc2 = Pack<T>::Zero();
		auto acc3 = Pack<T>::Zero();
		T tailSum = T { 0 };

		ForEachUnrolled<T>(
			count,
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				acc0 = Fma(LoadStrided(x + i * incX, incX), LoadStrided(y + i * incY, incY), acc0);
				acc1 = Fma(LoadStrided(x + (i + lanes) * incX, incX), LoadStrided(y + (i + lanes) * incY, incY), acc1);
				acc2 = Fma(LoadStrided(x + (i + 2 * lanes) * incX, incX), LoadStrided(y + (i + 2 * lanes) * incY, incY), acc2);
				acc3 = Fma(LoadStrided(x + (i + 3 * lanes) * incX, incX), LoadStrided(y + (i + 3 * lanes) * incY, incY), acc3);
			},
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				acc0 = Fma(LoadStrided(x + i * incX, incX), LoadStrided(y + i * incY, incY), acc0);
			},
			[&](const std::size_t start, const std::size_t rem) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				for (std::size_t j = 0; j < rem; ++j)
					tailSum += x[(start + j) * incX] * y[(start + j) * incY];
			});

		return HorizontalSum((acc0 + acc1) + (acc2 + acc3)) + tailSum;
	}

	template<std::floating_point T>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T EuclideanNormSquared(const T* const __restrict x, const std::size_t count, const std::size_t incX) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		auto acc0 = Pack<T>::Zero();
		auto acc1 = Pack<T>::Zero();
		auto acc2 = Pack<T>::Zero();
		auto acc3 = Pack<T>::Zero();
		T tailSum = T { 0 };

		ForEachUnrolled<T>(
			count,
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				const auto p0 = LoadStrided(x + i * incX, incX);
				const auto p1 = LoadStrided(x + (i + lanes) * incX, incX);
				const auto p2 = LoadStrided(x + (i + 2 * lanes) * incX, incX);
				const auto p3 = LoadStrided(x + (i + 3 * lanes) * incX, incX);
				acc0 = Fma(p0, p0, acc0);
				acc1 = Fma(p1, p1, acc1);
				acc2 = Fma(p2, p2, acc2);
				acc3 = Fma(p3, p3, acc3);
			},
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				const auto p = LoadStrided(x + i * incX, incX);
				acc0 = Fma(p, p, acc0);
			},
			[&](const std::size_t start, const std::size_t rem) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				for (std::size_t j = 0; j < rem; ++j)
				{
					const T v = x[(start + j) * incX];
					tailSum += v * v;
				}
			});

		return HorizontalSum((acc0 + acc1) + (acc2 + acc3)) + tailSum;
	}

	template<std::floating_point T>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T ReduceAbsSum(const T* const __restrict x, const std::size_t count, const std::size_t incX) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		auto acc0 = Pack<T>::Zero();
		auto acc1 = Pack<T>::Zero();
		auto acc2 = Pack<T>::Zero();
		auto acc3 = Pack<T>::Zero();
		T tailSum = T { 0 };

		ForEachUnrolled<T>(
			count,
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				acc0 += Abs(LoadStrided(x + i * incX, incX));
				acc1 += Abs(LoadStrided(x + (i + lanes) * incX, incX));
				acc2 += Abs(LoadStrided(x + (i + 2 * lanes) * incX, incX));
				acc3 += Abs(LoadStrided(x + (i + 3 * lanes) * incX, incX));
			},
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				acc0 += Abs(LoadStrided(x + i * incX, incX));
			},
			[&](const std::size_t start, const std::size_t rem) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				for (std::size_t j = 0; j < rem; ++j)
					tailSum += std::abs(x[(start + j) * incX]);
			});

		return HorizontalSum((acc0 + acc1) + (acc2 + acc3)) + tailSum;
	}

	template<std::floating_point T>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr T ReduceSum(const T* const __restrict x, const std::size_t count, const std::size_t incX) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		auto acc0 = Pack<T>::Zero();
		auto acc1 = Pack<T>::Zero();
		auto acc2 = Pack<T>::Zero();
		auto acc3 = Pack<T>::Zero();
		T tailSum = T { 0 };

		ForEachUnrolled<T>(
			count,
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				acc0 += LoadStrided(x + i * incX, incX);
				acc1 += LoadStrided(x + (i + lanes) * incX, incX);
				acc2 += LoadStrided(x + (i + 2 * lanes) * incX, incX);
				acc3 += LoadStrided(x + (i + 3 * lanes) * incX, incX);
			},
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				acc0 += LoadStrided(x + i * incX, incX);
			},
			[&](const std::size_t start, const std::size_t rem) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				for (std::size_t j = 0; j < rem; ++j)
					tailSum += x[(start + j) * incX];
			});

		return HorizontalSum((acc0 + acc1) + (acc2 + acc3)) + tailSum;
	}

	template<std::floating_point T>
	SIMD_ALWAYS_INLINE constexpr void ScaledMultiplyAdd(T* const __restrict x, const T* const __restrict y, const T alpha, const std::size_t count,
															const std::size_t incX, const std::size_t incY) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		const auto pAlpha = Pack<T>::Broadcast(alpha);

		ForEachUnrolled<T>(
			count,
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				StoreStrided(x + i * incX, incX, pAlpha * LoadStrided(y + i * incY, incY));
				StoreStrided(x + (i + lanes) * incX, incX, pAlpha * LoadStrided(y + (i + lanes) * incY, incY));
				StoreStrided(x + (i + 2 * lanes) * incX, incX, pAlpha * LoadStrided(y + (i + 2 * lanes) * incY, incY));
				StoreStrided(x + (i + 3 * lanes) * incX, incX, pAlpha * LoadStrided(y + (i + 3 * lanes) * incY, incY));
			},
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				StoreStrided(x + i * incX, incX, pAlpha * LoadStrided(y + i * incY, incY));
			},
			[&](const std::size_t start, const std::size_t rem) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				for (std::size_t j = 0; j < rem; ++j)
					x[(start + j) * incX] = alpha * y[(start + j) * incY];
			});
	}

	template<std::floating_point T>
	SIMD_ALWAYS_INLINE constexpr void ScaledMultiplyAdd(T* const __restrict x, const T* const __restrict y, const T alpha, const T beta,
																const std::size_t count, const std::size_t incX, const std::size_t incY) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		const auto pAlpha = Pack<T>::Broadcast(alpha);
		const auto pBeta = Pack<T>::Broadcast(beta);

		ForEachUnrolled<T>(
			count,
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				StoreStrided(x + i * incX, incX, Fma(pAlpha, LoadStrided(y + i * incY, incY), pBeta * LoadStrided(x + i * incX, incX)));
				StoreStrided(x + (i + lanes) * incX, incX, Fma(pAlpha, LoadStrided(y + (i + lanes) * incY, incY), pBeta * LoadStrided(x + (i + lanes) * incX, incX)));
				StoreStrided(x + (i + 2 * lanes) * incX, incX, Fma(pAlpha, LoadStrided(y + (i + 2 * lanes) * incY, incY), pBeta * LoadStrided(x + (i + 2 * lanes) * incX, incX)));
				StoreStrided(x + (i + 3 * lanes) * incX, incX, Fma(pAlpha, LoadStrided(y + (i + 3 * lanes) * incY, incY), pBeta * LoadStrided(x + (i + 3 * lanes) * incX, incX)));
			},
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				StoreStrided(x + i * incX, incX, Fma(pAlpha, LoadStrided(y + i * incY, incY), pBeta * LoadStrided(x + i * incX, incX)));
			},
			[&](const std::size_t start, const std::size_t rem) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				for (std::size_t j = 0; j < rem; ++j)
					x[(start + j) * incX] = beta * x[(start + j) * incX] + alpha * y[(start + j) * incY];
			});
	}

	template<std::floating_point T>
	SIMD_ALWAYS_INLINE constexpr void Scale(T* const __restrict x, const T alpha, const std::size_t count, const std::size_t incX) noexcept
	{
		assert(count > 0);
		SIMD_ASSUME(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		const auto pAlpha = Pack<T>::Broadcast(alpha);

		ForEachUnrolled<T>(
			count,
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				StoreStrided(x + i * incX, incX, pAlpha * LoadStrided(x + i * incX, incX));
				StoreStrided(x + (i + lanes) * incX, incX, pAlpha * LoadStrided(x + (i + lanes) * incX, incX));
				StoreStrided(x + (i + 2 * lanes) * incX, incX, pAlpha * LoadStrided(x + (i + 2 * lanes) * incX, incX));
				StoreStrided(x + (i + 3 * lanes) * incX, incX, pAlpha * LoadStrided(x + (i + 3 * lanes) * incX, incX));
			},
			[&](const std::size_t i) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				StoreStrided(x + i * incX, incX, pAlpha * LoadStrided(x + i * incX, incX));
			},
			[&](const std::size_t start, const std::size_t rem) SIMD_NOEXCEPT_ALWAYS_INLINE
			{
				for (std::size_t j = 0; j < rem; ++j)
					x[(start + j) * incX] *= alpha;
			});
	}

	// ========================================================================
	// In-Register Matrix Transpose
	// ========================================================================

	template<std::floating_point T, std::size_t Lanes = LaneCount<T>()>
	SIMD_ALWAYS_INLINE constexpr void Transpose(std::array<Pack<T, Lanes>, Lanes>& rows) noexcept
	{
		if (!std::is_constant_evaluated())
		{
#if SIMD_HAS_AVX2
			if constexpr (std::same_as<T, float> && Lanes == 8)
			{
				const __m256 t0 = _mm256_unpacklo_ps(rows[0].Native(), rows[1].Native());
				const __m256 t1 = _mm256_unpackhi_ps(rows[0].Native(), rows[1].Native());
				const __m256 t2 = _mm256_unpacklo_ps(rows[2].Native(), rows[3].Native());
				const __m256 t3 = _mm256_unpackhi_ps(rows[2].Native(), rows[3].Native());
				const __m256 t4 = _mm256_unpacklo_ps(rows[4].Native(), rows[5].Native());
				const __m256 t5 = _mm256_unpackhi_ps(rows[4].Native(), rows[5].Native());
				const __m256 t6 = _mm256_unpacklo_ps(rows[6].Native(), rows[7].Native());
				const __m256 t7 = _mm256_unpackhi_ps(rows[6].Native(), rows[7].Native());

				const __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
				const __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
				const __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
				const __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
				const __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
				const __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
				const __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
				const __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

				rows[0] = Pack<float, 8>::FromNative(_mm256_permute2f128_ps(tt0, tt4, 0x20));
				rows[1] = Pack<float, 8>::FromNative(_mm256_permute2f128_ps(tt1, tt5, 0x20));
				rows[2] = Pack<float, 8>::FromNative(_mm256_permute2f128_ps(tt2, tt6, 0x20));
				rows[3] = Pack<float, 8>::FromNative(_mm256_permute2f128_ps(tt3, tt7, 0x20));
				rows[4] = Pack<float, 8>::FromNative(_mm256_permute2f128_ps(tt0, tt4, 0x31));
				rows[5] = Pack<float, 8>::FromNative(_mm256_permute2f128_ps(tt1, tt5, 0x31));
				rows[6] = Pack<float, 8>::FromNative(_mm256_permute2f128_ps(tt2, tt6, 0x31));
				rows[7] = Pack<float, 8>::FromNative(_mm256_permute2f128_ps(tt3, tt7, 0x31));
				return;
			}
			else if constexpr (std::same_as<T, double> && Lanes == 4)
			{
				const __m256d t0 = _mm256_unpacklo_pd(rows[0].Native(), rows[1].Native());
				const __m256d t1 = _mm256_unpackhi_pd(rows[0].Native(), rows[1].Native());
				const __m256d t2 = _mm256_unpacklo_pd(rows[2].Native(), rows[3].Native());
				const __m256d t3 = _mm256_unpackhi_pd(rows[2].Native(), rows[3].Native());

				rows[0] = Pack<double, 4>::FromNative(_mm256_permute2f128_pd(t0, t2, 0x20));
				rows[1] = Pack<double, 4>::FromNative(_mm256_permute2f128_pd(t1, t3, 0x20));
				rows[2] = Pack<double, 4>::FromNative(_mm256_permute2f128_pd(t0, t2, 0x31));
				rows[3] = Pack<double, 4>::FromNative(_mm256_permute2f128_pd(t1, t3, 0x31));
				return;
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (std::same_as<T, float> && Lanes == 4)
			{
				const __m128 t0 = _mm_unpacklo_ps(rows[0].Native(), rows[1].Native());
				const __m128 t1 = _mm_unpackhi_ps(rows[0].Native(), rows[1].Native());
				const __m128 t2 = _mm_unpacklo_ps(rows[2].Native(), rows[3].Native());
				const __m128 t3 = _mm_unpackhi_ps(rows[2].Native(), rows[3].Native());

				rows[0] = Pack<float, 4>::FromNative(_mm_movelh_ps(t0, t2));
				rows[1] = Pack<float, 4>::FromNative(_mm_movehl_ps(t2, t0));
				rows[2] = Pack<float, 4>::FromNative(_mm_movelh_ps(t1, t3));
				rows[3] = Pack<float, 4>::FromNative(_mm_movehl_ps(t3, t1));
				return;
			}
			else if constexpr (std::same_as<T, double> && Lanes == 2)
			{
				const __m128d r0 = rows[0].Native();
				const __m128d r1 = rows[1].Native();
				rows[0] = Pack<double, 2>::FromNative(_mm_unpacklo_pd(r0, r1));
				rows[1] = Pack<double, 2>::FromNative(_mm_unpackhi_pd(r0, r1));
				return;
			}
#endif
		}

		std::array<typename Pack<T, Lanes>::AlignedArray, Lanes> tmp {};
		for (std::size_t r = 0; r < Lanes; ++r)
			StoreAligned(tmp[r].data(), rows[r]);

		for (std::size_t r = 0; r < Lanes; ++r)
		{
			typename Pack<T, Lanes>::AlignedArray col {};
			for (std::size_t c = 0; c < Lanes; ++c)
				col[c] = tmp[c][r];
			rows[r] = Pack<T, Lanes>(col);
		}
	}
}	 // namespace simd
