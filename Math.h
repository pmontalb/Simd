#pragma once

#include "Pack.h"

#include <cmath>
#include <concepts>
#include <cstdint>

namespace simd
{
	// ========================================================================
	// Vectorized Transcendental & Trigonometric Math (SSE / AVX2 / AVX-512)
	// ========================================================================

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Exp(const Pack<T, Lanes> x) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto arr = x.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = std::exp(arr[i]);
			return Pack<T, Lanes>::FromScalar(arr);
		}
		// LCOV_EXCL_STOP

		using IntT = IndexTypeForT<T>;
		if constexpr (std::is_same_v<T, float>)
		{
			const auto clamped = Min(Max(x, Broadcast<float, Lanes>(-87.336544f)), Broadcast<float, Lanes>(88.722839f));
			const auto fx = RoundToNearest(clamped * 1.44269504088896341f);
			auto r = Fnma(fx, 0.693359375f, clamped);
			r = Fnma(fx, -2.12194440e-4f, r);

			auto y = Broadcast<float, Lanes>(1.9875691500e-4f);
			y = Fma(y, r, Broadcast<float, Lanes>(1.3981999507e-3f));
			y = Fma(y, r, Broadcast<float, Lanes>(8.3334519073e-3f));
			y = Fma(y, r, Broadcast<float, Lanes>(4.1665795894e-2f));
			y = Fma(y, r, Broadcast<float, Lanes>(1.6666665459e-1f));
			y = Fma(y, r, Broadcast<float, Lanes>(5.0000001201e-1f));
			y = Fma(y, r * r, r + 1.0f);

			const auto k = ConvertToInt(fx) + Broadcast<IntT, Lanes>(127);
			const auto pow2k = BitCast<float>(k << 23);
			return y * pow2k;
		}
		else
		{
			const auto clamped = Min(Max(x, Broadcast<double, Lanes>(-708.3964185322641)), Broadcast<double, Lanes>(709.782712893384));
			const auto fx = RoundToNearest(clamped * 1.44269504088896340736);
			auto r = Fnma(fx, 0.693145751953125, clamped);
			r = Fnma(fx, 1.42860682030941723212e-6, r);

			auto y = Broadcast<double, Lanes>(2.08767569878680989792e-9);
			y = Fma(y, r, Broadcast<double, Lanes>(2.50521083854417187750e-8));
			y = Fma(y, r, Broadcast<double, Lanes>(2.75573192239858906526e-7));
			y = Fma(y, r, Broadcast<double, Lanes>(2.75573192239858906526e-6));
			y = Fma(y, r, Broadcast<double, Lanes>(2.48015873015873015873e-5));
			y = Fma(y, r, Broadcast<double, Lanes>(1.98412698412698412698e-4));
			y = Fma(y, r, Broadcast<double, Lanes>(1.38888888888888888889e-3));
			y = Fma(y, r, Broadcast<double, Lanes>(8.33333333333333333333e-3));
			y = Fma(y, r, Broadcast<double, Lanes>(4.16666666666666666667e-2));
			y = Fma(y, r, Broadcast<double, Lanes>(1.66666666666666666667e-1));
			y = Fma(y, r, Broadcast<double, Lanes>(5.00000000000000000000e-1));
			y = Fma(y, r * r, r + 1.0);

			const auto k = ConvertToInt(fx) + Broadcast<IntT, Lanes>(1023);
			const auto pow2k = BitCast<double>(k << 52);
			return y * pow2k;
		}
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Log(const Pack<T, Lanes> x) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto arr = x.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = std::log(arr[i]);
			return Pack<T, Lanes>::FromScalar(arr);
		}
		// LCOV_EXCL_STOP

		using IntT = IndexTypeForT<T>;
		if constexpr (std::is_same_v<T, float>)
		{
			const auto ix = BitCast<IntT>(x);
			const auto expBits = (ix >> 23) - Broadcast<IntT, Lanes>(127);
			auto e = ConvertToFloat<float>(expBits);
			auto m = BitCast<float>((ix & Broadcast<IntT, Lanes>(0x007FFFFF)) | Broadcast<IntT, Lanes>(0x3F800000));

			const auto needsHalf = CmpLess(m, Broadcast<float, Lanes>(0.707106781186547524f));
			const auto needsDouble = CmpGreater(m, Broadcast<float, Lanes>(1.414213562373095048f));
			m = Blend(needsHalf, m * 2.0f, m);
			e = Blend(needsHalf, e - 1.0f, e);
			m = Blend(needsDouble, m * 0.5f, m);
			e = Blend(needsDouble, e + 1.0f, e);

			const auto f = m - 1.0f;
			const auto f2 = f * f;

			auto y = Broadcast<float, Lanes>(7.0376836292e-2f);
			y = Fma(y, f, Broadcast<float, Lanes>(-1.1514610310e-1f));
			y = Fma(y, f, Broadcast<float, Lanes>(1.1676998740e-1f));
			y = Fma(y, f, Broadcast<float, Lanes>(-1.2420140846e-1f));
			y = Fma(y, f, Broadcast<float, Lanes>(1.4249322787e-1f));
			y = Fma(y, f, Broadcast<float, Lanes>(-1.6668057665e-1f));
			y = Fma(y, f, Broadcast<float, Lanes>(2.0000714765e-1f));
			y = Fma(y, f, Broadcast<float, Lanes>(-2.4999993993e-1f));
			y = Fma(y, f, Broadcast<float, Lanes>(3.3333331174e-1f));
			y = y * (f * f2);

			y = Fma(e, -2.12194440e-4f, y);
			y = Fma(f2, -0.5f, y);
			return Fma(e, 0.693359375f, f + y);
		}
		else
		{
			const auto ix = BitCast<IntT>(x);
			const auto expBits = (ix >> 52) - Broadcast<IntT, Lanes>(1023);
			auto e = ConvertToFloat<double>(expBits);
			auto m = BitCast<double>((ix & Broadcast<IntT, Lanes>(0x000FFFFFFFFFFFFFLL)) | Broadcast<IntT, Lanes>(0x3FF0000000000000LL));

			const auto needsDouble = CmpGreater(m, Broadcast<double, Lanes>(1.4142135623730950488));
			m = Blend(needsDouble, m * 0.5, m);
			e = Blend(needsDouble, e + 1.0, e);

			const auto u = (m - 1.0) / (m + 1.0);
			const auto s = u * u;

			auto p = Broadcast<double, Lanes>(0.15384615384615384615);
			p = Fma(p, s, Broadcast<double, Lanes>(0.18181818181818181818));
			p = Fma(p, s, Broadcast<double, Lanes>(0.22222222222222222222));
			p = Fma(p, s, Broadcast<double, Lanes>(0.28571428571428571429));
			p = Fma(p, s, Broadcast<double, Lanes>(0.40000000000000000000));
			p = Fma(p, s, Broadcast<double, Lanes>(0.66666666666666666667));
			const auto logM = Fma(u * s, p, u * 2.0);

			const auto res = Fma(e, 1.42860682030941723212e-6, logM);
			return Fma(e, 0.693145751953125, res);
		}
	}

	namespace detail
	{
		template<std::floating_point T, std::size_t Lanes>
		[[nodiscard]] constexpr Pack<T, Lanes> SinCosCore(const Pack<T, Lanes> x, const std::int32_t quadrantOffset) noexcept
		{
			using IntT = IndexTypeForT<T>;
			if constexpr (std::is_same_v<T, float>)
			{
				const auto qf = RoundToNearest(x * 0.63661977236758134308f);
				auto r = Fnma(qf, 1.5707962512969970703125f, x);
				r = Fnma(qf, 7.54978941586159635335e-8f, r);

				const auto r2 = r * r;
				auto s = Broadcast<float, Lanes>(-1.9515295891e-4f);
				s = Fma(s, r2, Broadcast<float, Lanes>(8.3321608736e-3f));
				s = Fma(s, r2, Broadcast<float, Lanes>(-1.6666654611e-1f));
				s = Fma(s * r2, r, r);

				auto c = Broadcast<float, Lanes>(2.443315711809948e-5f);
				c = Fma(c, r2, Broadcast<float, Lanes>(-1.388731625493765e-3f));
				c = Fma(c, r2, Broadcast<float, Lanes>(4.166664568298827e-2f));
				c = Fma(c * r2, r2, Fma(r2, -0.5f, Broadcast<float, Lanes>(1.0f)));

				const auto q = ConvertToInt(qf) + Broadcast<IntT, Lanes>(quadrantOffset);
				const auto useCos = CmpGreater(ConvertToFloat<float>(q & Broadcast<IntT, Lanes>(1)), Broadcast<float, Lanes>(0.5f));
				const auto mag = Blend(useCos, c, s);
				const auto signBit = (q & Broadcast<IntT, Lanes>(2)) << 30;
				return BitCast<float>(BitCast<IntT>(mag) ^ signBit);
			}
			else
			{
				const auto qf = RoundToNearest(x * 0.63661977236758134308);
				auto r = Fnma(qf, 1.57079632673412561417, x);
				r = Fnma(qf, 6.07710050650619224932e-11, r);

				const auto r2 = r * r;
				auto s = Broadcast<double, Lanes>(1.58969099521155010221e-10);
				s = Fma(s, r2, Broadcast<double, Lanes>(-2.50507602534068634195e-8));
				s = Fma(s, r2, Broadcast<double, Lanes>(2.75573137070700676789e-6));
				s = Fma(s, r2, Broadcast<double, Lanes>(-1.98412698298579493134e-4));
				s = Fma(s, r2, Broadcast<double, Lanes>(8.33333333332248946124e-3));
				s = Fma(s, r2, Broadcast<double, Lanes>(-1.66666666666666324348e-1));
				s = Fma(s * r2, r, r);

				auto c = Broadcast<double, Lanes>(-1.13585365213876817300e-11);
				c = Fma(c, r2, Broadcast<double, Lanes>(2.08757232110174057913e-9));
				c = Fma(c, r2, Broadcast<double, Lanes>(-2.75573143513906633035e-7));
				c = Fma(c, r2, Broadcast<double, Lanes>(2.48015872894767294178e-5));
				c = Fma(c, r2, Broadcast<double, Lanes>(-1.38888888888741095749e-3));
				c = Fma(c, r2, Broadcast<double, Lanes>(4.16666666666666019037e-2));
				c = Fma(c * r2, r2, Fma(r2, -0.5, Broadcast<double, Lanes>(1.0)));

				const auto q = ConvertToInt(qf) + Broadcast<IntT, Lanes>(quadrantOffset);
				const auto useCos = CmpGreater(ConvertToFloat<double>(q & Broadcast<IntT, Lanes>(1)), Broadcast<double, Lanes>(0.5));
				const auto mag = Blend(useCos, c, s);
				const auto signBit = (q & Broadcast<IntT, Lanes>(2)) << 62;
				return BitCast<double>(BitCast<IntT>(mag) ^ signBit);
			}
		}
	}	 // namespace detail

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Sin(const Pack<T, Lanes> x) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto arr = x.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = std::sin(arr[i]);
			return Pack<T, Lanes>::FromScalar(arr);
		}
		// LCOV_EXCL_STOP
		return detail::SinCosCore(x, 0);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Cos(const Pack<T, Lanes> x) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto arr = x.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = std::cos(arr[i]);
			return Pack<T, Lanes>::FromScalar(arr);
		}
		// LCOV_EXCL_STOP
		return detail::SinCosCore(x, 1);
	}
}	 // namespace simd
