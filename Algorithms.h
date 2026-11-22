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
	constexpr void ForEachUnrolled(const std::size_t count, UnrolledFn&& unrolledBody, SingleFn&& singleBody, TailFn&& tailBody) noexcept
	{
		assert(count > 0);

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

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] constexpr T DotProduct(const T* const __restrict x, const T* const __restrict y, const std::size_t count) noexcept
	{
		assert(count > 0);

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
		for (; i < count; ++i)
			sum += x[i] * y[i];
		return sum;
#endif
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] constexpr T ReduceSum(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

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
	constexpr void Axpy(T* const __restrict y, const T* const __restrict x, const T alpha, const std::size_t count) noexcept
	{
		assert(count > 0);

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
		const auto vAlpha = Broadcast<T>(alpha);

		std::size_t i = 0;
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
	[[nodiscard]] constexpr T SumOfSquares(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

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
		for (; i < count; ++i)
			sum += x[i] * x[i];
		return sum;
#endif
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] constexpr T ReduceAbsSum(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

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
		for (; i < count; ++i)
			sum += (x[i] < T { 0 } ? -x[i] : x[i]);
		return sum;
#endif
	}

	template<std::floating_point T, std::size_t Unroll = 4>
	[[nodiscard]] constexpr T ReduceMin(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		if (std::is_constant_evaluated() || count < lanes || lanes == 1)
		{
			T best = x[0];
			for (std::size_t i = 1; i < count; ++i)
				best = x[i] < best ? x[i] : best;
			return best;
		}

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
	[[nodiscard]] constexpr T ReduceMax(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
		if (std::is_constant_evaluated() || count < lanes || lanes == 1)
		{
			T best = x[0];
			for (std::size_t i = 1; i < count; ++i)
				best = x[i] > best ? x[i] : best;
			return best;
		}

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
	[[nodiscard]] constexpr T ReduceAbsMin(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
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
	[[nodiscard]] constexpr T ReduceAbsMax(const T* const __restrict x, const std::size_t count) noexcept
	{
		assert(count > 0);

		constexpr std::size_t lanes = LaneCount<T>();
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
}	 // namespace simd
