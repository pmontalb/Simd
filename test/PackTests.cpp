#include <gtest/gtest.h>

#include <Simd.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace simdt
{
	namespace
	{
		template<typename T>
		constexpr T ConstexprDotTest() noexcept
		{
			constexpr std::array<T, 7> a { T { 1 }, T { 2 }, T { 3 }, T { 4 }, T { 5 }, T { 6 }, T { 7 } };
			constexpr std::array<T, 7> b { T { 2 }, T { 3 }, T { 4 }, T { 5 }, T { 6 }, T { 7 }, T { 8 } };
			return simd::DotProduct(a.data(), b.data(), a.size());
		}

		template<typename T>
		constexpr T ConstexprPackFmaTest() noexcept
		{
			const auto a = simd::Pack<T, 4>::Iota(T { 1 }, T { 1 });
			const auto b = simd::Broadcast<T, 4>(T { 2 });
			const auto c = simd::Broadcast<T, 4>(T { 3 });
			return simd::HorizontalSum(simd::Fma(a, b, c));
		}

		static_assert(ConstexprDotTest<double>() >= 167.9999 && ConstexprDotTest<double>() <= 168.0001);
		static_assert(ConstexprDotTest<float>() >= 167.9999f && ConstexprDotTest<float>() <= 168.0001f);
		static_assert(ConstexprPackFmaTest<double>() >= 31.9999 && ConstexprPackFmaTest<double>() <= 32.0001);
		static_assert(ConstexprPackFmaTest<float>() >= 31.9999f && ConstexprPackFmaTest<float>() <= 32.0001f);

		constexpr bool ConstexprAlgorithmsTest() noexcept
		{
			std::array<double, 5> x { -2.0, 4.0, -1.0, 3.0, -5.0 };
			std::array<double, 5> y { 1.0, 1.0, 1.0, 1.0, 1.0 };
			simd::Axpy(y.data(), x.data(), 2.0, x.size());
			const double sq = simd::EuclideanNormSquared(x.data(), x.size());
			const double asum = simd::ReduceAbsSum(x.data(), x.size());
			const double mn = simd::ReduceMin(x.data(), x.size());
			const double mx = simd::ReduceMax(x.data(), x.size());
			const double amn = simd::ReduceAbsMin(x.data(), x.size());
			const double amx = simd::ReduceAbsMax(x.data(), x.size());
			const std::size_t imx = simd::ArgMaxAbs(x.data(), x.size());
			const std::size_t imn = simd::ArgMinAbs(x.data(), x.size());
			return y[4] <= -8.999 && y[4] >= -9.001 && sq >= 54.999 && sq <= 55.001 && asum >= 14.999 && asum <= 15.001 &&
				   mn <= -4.999 && mx >= 3.999 && amn >= 0.999 && amn <= 1.001 && amx >= 4.999 && imx == 4 && imn == 2;
		}
		static_assert(ConstexprAlgorithmsTest());

		static_assert(std::is_trivial_v<simd::Pack<float>> && std::is_standard_layout_v<simd::Pack<float>>);
		static_assert(std::is_trivial_v<simd::Pack<double>> && std::is_standard_layout_v<simd::Pack<double>>);
#if SIMD_HAS_SSE2
		static_assert(std::is_trivial_v<simd::Double2> && std::is_standard_layout_v<simd::Double2>);
		static_assert(std::is_trivial_v<simd::Float4> && std::is_standard_layout_v<simd::Float4>);
#endif
#if SIMD_HAS_AVX
		static_assert(std::is_trivial_v<simd::Double4> && std::is_standard_layout_v<simd::Double4>);
		static_assert(std::is_trivial_v<simd::Float8> && std::is_standard_layout_v<simd::Float8>);
#endif
	}	 // namespace

	TEST(PackTests, CompileTimeQueries)
	{
		EXPECT_GE(simd::LaneCount<float>(), 1U);
		EXPECT_GE(simd::LaneCount<double>(), 1U);
		EXPECT_GE(simd::AlignmentBytes<float>(), alignof(float));
		EXPECT_GE(simd::AlignmentBytes<double>(), alignof(double));
		EXPECT_EQ(simd::MaxAlignmentBytes, static_cast<std::size_t>(SIMD_ALIGNMENT_BYTES));
	}

	TEST(PackTests, ArithmeticAndFma)
	{
		const auto a = simd::Iota<double>(1.0, 1.0);
		const auto b = simd::Broadcast<double>(2.0);
		const auto c = simd::Broadcast<double>(3.0);

		const auto fma = simd::Fma(a, b, c);
		const auto fms = simd::Fms(a, b, c);
		const auto fnma = simd::Fnma(a, b, c);

		for (std::size_t i = 0; i < simd::LaneCount<double>(); ++i)
		{
			const double ai = static_cast<double>(i + 1);
			EXPECT_DOUBLE_EQ(fma[i], ai * 2.0 + 3.0);
			EXPECT_DOUBLE_EQ(fms[i], ai * 2.0 - 3.0);
			EXPECT_DOUBLE_EQ(fnma[i], -ai * 2.0 + 3.0);
		}
	}

	TEST(PackTests, LoadStorePartial)
	{
		constexpr std::size_t lanes = simd::LaneCount<float>();
		std::vector<float> src(lanes + 3);
		std::iota(src.begin(), src.end(), 1.0f);

		const std::size_t valid = lanes > 1 ? lanes - 1 : 1;
		const auto p = simd::LoadPartial<float>(src.data() + 1, valid);
		for (std::size_t i = 0; i < valid; ++i)
			EXPECT_FLOAT_EQ(p[i], src[i + 1]);
		for (std::size_t i = valid; i < lanes; ++i)
			EXPECT_FLOAT_EQ(p[i], 0.0f);

		std::vector<float> dst(lanes + 3, -99.0f);
		simd::StorePartial(dst.data() + 1, p, valid);
		EXPECT_FLOAT_EQ(dst[0], -99.0f);
		for (std::size_t i = 0; i < valid; ++i)
			EXPECT_FLOAT_EQ(dst[i + 1], src[i + 1]);
		EXPECT_FLOAT_EQ(dst[valid + 1], -99.0f);
	}

	TEST(PackTests, HorizontalReductions)
	{
		const auto pf = simd::Iota<float>(1.0f, 1.0f);
		constexpr std::size_t nf = simd::LaneCount<float>();
		EXPECT_FLOAT_EQ(simd::HorizontalSum(pf), static_cast<float>(nf * (nf + 1) / 2));
		EXPECT_FLOAT_EQ(simd::HorizontalMin(pf), 1.0f);
		EXPECT_FLOAT_EQ(simd::HorizontalMax(pf), static_cast<float>(nf));

		const auto pd = simd::Iota<double>(1.0, 1.0);
		constexpr std::size_t nd = simd::LaneCount<double>();
		EXPECT_DOUBLE_EQ(simd::HorizontalSum(pd), static_cast<double>(nd * (nd + 1) / 2));
		EXPECT_DOUBLE_EQ(simd::HorizontalMin(pd), 1.0);
		EXPECT_DOUBLE_EQ(simd::HorizontalMax(pd), static_cast<double>(nd));
	}

	TEST(PackTests, DotProductAndReduceSum)
	{
		for (const std::size_t n : { std::size_t { 1 }, std::size_t { 3 }, std::size_t { 7 }, std::size_t { 33 }, std::size_t { 129 } })
		{
			std::vector<double> x(n), y(n);
			for (std::size_t i = 0; i < n; ++i)
			{
				x[i] = static_cast<double>(i + 1) * 0.5;
				y[i] = static_cast<double>(n - i) * 0.25;
			}
			double expectedDot = 0.0;
			double expectedSum = 0.0;
			for (std::size_t i = 0; i < n; ++i)
			{
				expectedDot += x[i] * y[i];
				expectedSum += x[i];
			}
			EXPECT_NEAR(simd::DotProduct(x.data(), y.data(), n), expectedDot, 1e-12);
			EXPECT_NEAR(simd::ReduceSum(x.data(), n), expectedSum, 1e-12);
		}
	}

	TEST(PackTests, ArgMinMaxAbs)
	{
		std::vector<float> v { -3.0f, 1.0f, -15.0f, 7.0f, 0.5f, -0.25f, 15.0f, -2.0f, 4.0f, -0.25f, 9.0f };
		EXPECT_EQ(simd::ArgMaxAbs(v.data(), v.size()), 2U);	 // first |-15.0f| at index 2
		EXPECT_EQ(simd::ArgMinAbs(v.data(), v.size()), 5U);	 // first |-0.25f| at index 5

		std::vector<double> d { -3.0, 1.0, -15.0, 7.0, 0.5, -0.25, 15.0, -2.0, 4.0, -0.25, 9.0 };
		EXPECT_EQ(simd::ArgMaxAbs(d.data(), d.size()), 2U);
		EXPECT_EQ(simd::ArgMinAbs(d.data(), d.size()), 5U);
	}

	TEST(PackTests, AxpySumOfSquaresAndReductions)
	{
		for (const std::size_t n : { std::size_t { 1 }, std::size_t { 3 }, std::size_t { 7 }, std::size_t { 33 }, std::size_t { 67 } })
		{
			std::vector<double> x(n), y(n), expectedY(n);
			for (std::size_t i = 0; i < n; ++i)
			{
				x[i] = (i % 2 == 0 ? 1.0 : -1.0) * (static_cast<double>(i) + 1.5);
				y[i] = static_cast<double>(i) * 0.5;
				expectedY[i] = y[i] + 2.5 * x[i];
			}

			simd::Axpy(y.data(), x.data(), 2.5, n);
			for (std::size_t i = 0; i < n; ++i)
				EXPECT_DOUBLE_EQ(y[i], expectedY[i]);

			double expectedSq = 0.0;
			double expectedAbsSum = 0.0;
			double expectedMin = x[0];
			double expectedMax = x[0];
			double expectedAbsMin = std::abs(x[0]);
			double expectedAbsMax = std::abs(x[0]);
			for (std::size_t i = 0; i < n; ++i)
			{
				expectedSq += x[i] * x[i];
				expectedAbsSum += std::abs(x[i]);
				expectedMin = std::min(expectedMin, x[i]);
				expectedMax = std::max(expectedMax, x[i]);
				expectedAbsMin = std::min(expectedAbsMin, std::abs(x[i]));
				expectedAbsMax = std::max(expectedAbsMax, std::abs(x[i]));
			}

			EXPECT_NEAR(simd::EuclideanNormSquared(x.data(), n), expectedSq, 1e-10);
			EXPECT_NEAR(simd::ReduceAbsSum(x.data(), n), expectedAbsSum, 1e-10);
			EXPECT_DOUBLE_EQ(simd::ReduceMin(x.data(), n), expectedMin);
			EXPECT_DOUBLE_EQ(simd::ReduceMax(x.data(), n), expectedMax);
			EXPECT_DOUBLE_EQ(simd::ReduceAbsMin(x.data(), n), expectedAbsMin);
			EXPECT_DOUBLE_EQ(simd::ReduceAbsMax(x.data(), n), expectedAbsMax);
		}
	}

	TEST(PackTests, TranscendentalAndTrigonometric)
	{
		const auto pf = simd::Iota<float>(0.25f, 0.4f);
		const auto expF = simd::Exp(pf);
		const auto logF = simd::Log(pf);
		const auto sinF = simd::Sin(pf - 1.5f);
		const auto cosF = simd::Cos(pf - 1.5f);
		for (std::size_t i = 0; i < simd::LaneCount<float>(); ++i)
		{
			EXPECT_FLOAT_EQ(expF[i], std::exp(pf[i]));
			EXPECT_FLOAT_EQ(logF[i], std::log(pf[i]));
			EXPECT_FLOAT_EQ(sinF[i], std::sin(pf[i] - 1.5f));
			EXPECT_FLOAT_EQ(cosF[i], std::cos(pf[i] - 1.5f));
		}

		const auto pd = simd::Iota<double>(0.25, 0.75);
		const auto expD = simd::Exp(pd);
		const auto logD = simd::Log(pd);
		const auto sinD = simd::Sin(pd - 1.5);
		const auto cosD = simd::Cos(pd - 1.5);
		for (std::size_t i = 0; i < simd::LaneCount<double>(); ++i)
		{
			EXPECT_NEAR(expD[i], std::exp(pd[i]), 1e-14 * std::exp(pd[i]));
			EXPECT_NEAR(logD[i], std::log(pd[i]), 1e-14);
			EXPECT_NEAR(sinD[i], std::sin(pd[i] - 1.5), 1e-14);
			EXPECT_NEAR(cosD[i], std::cos(pd[i] - 1.5), 1e-14);
		}
	}

	TEST(PackTests, OperatorsAlignedForEachAndIntegerPacks)
	{
		// Unary -, Reciprocal, scalar-pack operators, scalar FMA/FMS/FNMA
		auto p = simd::Iota<double>(1.0, 1.0);
		const auto negP = -p;
		const auto recP = simd::Reciprocal(p);
		const auto sAdd = (p + 2.0) + (2.0 + p);
		const auto sSub = (p - 1.0) - (1.0 - p);
		const auto sMul = (p * 3.0) * (2.0 * p);
		const auto sDiv = (p / 2.0) + (4.0 / p);
		const auto fmaS = simd::Fma(p, 2.0, simd::Broadcast<double>(1.0));
		const auto fmsS = simd::Fms(2.0, p, simd::Broadcast<double>(1.0));
		const auto fnmaS = simd::Fnma(p, 2.0, simd::Broadcast<double>(10.0));
		p += 1.0;
		p -= 1.0;
		p *= 2.0;
		p /= 2.0;
		for (std::size_t i = 0; i < simd::LaneCount<double>(); ++i)
		{
			const double v = static_cast<double>(i + 1);
			EXPECT_DOUBLE_EQ(negP[i], -v);
			EXPECT_DOUBLE_EQ(recP[i], 1.0 / v);
			EXPECT_DOUBLE_EQ(sAdd[i], 2.0 * v + 4.0);
			EXPECT_DOUBLE_EQ(sSub[i], 2.0 * v - 2.0);
			EXPECT_DOUBLE_EQ(sMul[i], 6.0 * v * v);
			EXPECT_DOUBLE_EQ(sDiv[i], v / 2.0 + 4.0 / v);
			EXPECT_DOUBLE_EQ(fmaS[i], 2.0 * v + 1.0);
			EXPECT_DOUBLE_EQ(fmsS[i], 2.0 * v - 1.0);
			EXPECT_DOUBLE_EQ(fnmaS[i], 10.0 - 2.0 * v);
			EXPECT_DOUBLE_EQ(p[i], v);
		}

		// AssumeAligned & LoadPartial/StorePartial for double
		alignas(simd::MaxAlignmentBytes) std::array<double, simd::LaneCount<double>() * 2> alignedBuf {};
		for (std::size_t i = 0; i < alignedBuf.size(); ++i)
			alignedBuf[i] = static_cast<double>(i + 1);
		double* const mutPtr = simd::AssumeAligned<double>(alignedBuf.data());
		const double* const constPtr = simd::AssumeAligned<double>(static_cast<const double*>(alignedBuf.data()));
		const auto loaded = simd::LoadAligned<double>(constPtr);
		simd::StoreAligned(mutPtr, loaded + 1.0);
		EXPECT_DOUBLE_EQ(alignedBuf[0], 2.0);

		const std::size_t dLanes = simd::LaneCount<double>();
		const std::size_t validD = dLanes > 1 ? dLanes - 1 : 1;
		const auto dp = simd::LoadPartial<double>(alignedBuf.data(), validD);
		std::array<double, simd::LaneCount<double>()> dDst {};
		dDst.fill(-7.0);
		simd::StorePartial(dDst.data(), dp, validD);
		simd::StorePartial(dDst.data(), dp, 0);
		EXPECT_DOUBLE_EQ(dDst[0], alignedBuf[0]);
		if (dLanes > 1)
		{
			EXPECT_DOUBLE_EQ(dDst[dLanes - 1], -7.0);
		}

		// ForEachUnrolled (Unroll = 4 and Unroll = 2) + Unroll = 2 reductions
		std::vector<double> data(35, 2.0);
		simd::ForEachUnrolled<double, 4>(
			data.size(),
			[&](const std::size_t i)
			{
				for (std::size_t u = 0; u < 4; ++u)
					simd::StoreUnaligned(data.data() + i + u * dLanes, simd::LoadUnaligned<double>(data.data() + i + u * dLanes) * 2.0);
			},
			[&](const std::size_t i)
			{
				simd::StoreUnaligned(data.data() + i, simd::LoadUnaligned<double>(data.data() + i) * 2.0);
			},
			[&](const std::size_t i, const std::size_t rem)
			{
				for (std::size_t j = 0; j < rem; ++j)
					data[i + j] *= 2.0;
			});
		EXPECT_DOUBLE_EQ((simd::DotProduct<double, 2>(data.data(), data.data(), data.size())), 35.0 * 16.0);
		EXPECT_DOUBLE_EQ((simd::ReduceSum<double, 2>(data.data(), data.size())), 35.0 * 4.0);
		EXPECT_DOUBLE_EQ((simd::EuclideanNormSquared<double, 2>(data.data(), data.size())), 35.0 * 16.0);
		EXPECT_DOUBLE_EQ((simd::ReduceAbsSum<double, 2>(data.data(), data.size())), 35.0 * 4.0);
		EXPECT_DOUBLE_EQ((simd::ReduceMin<double, 2>(data.data(), data.size())), 4.0);
		EXPECT_DOUBLE_EQ((simd::ReduceMax<double, 2>(data.data(), data.size())), 4.0);
		EXPECT_DOUBLE_EQ((simd::ReduceAbsMin<double, 2>(data.data(), data.size())), 4.0);
		EXPECT_DOUBLE_EQ((simd::ReduceAbsMax<double, 2>(data.data(), data.size())), 4.0);

		// Integer Pack<int32_t> & Pack<int64_t>
		const auto i32a = simd::Iota<std::int32_t>(1, 1);
		const auto i32b = simd::Broadcast<std::int32_t>(2);
		const auto i32res = ((i32a + i32b) - i32b) * i32b;
		EXPECT_EQ(simd::HorizontalMin(i32res), 2);
		EXPECT_EQ(simd::HorizontalMax(i32res), static_cast<std::int32_t>(2 * simd::LaneCount<std::int32_t>()));
		EXPECT_EQ(simd::Abs(-i32a)[0], 1);
		EXPECT_EQ((i32a | i32b)[0], 3);
	}

	TEST(PackTests, AxpyLargePrefetchAndTail)
	{
		std::vector<double> ax(8205, 1.5);
		std::vector<double> ay(8205, 2.0);
		simd::Axpy(ay.data(), ax.data(), 2.0, ay.size());
		EXPECT_DOUBLE_EQ(ay[0], 5.0);
		EXPECT_DOUBLE_EQ(ay[8204], 5.0);
	}

	TEST(PackTests, SubWidthEvenRemainder)
	{
		const std::array<double, 2> d2a { 3.0, -4.0 };
		const std::array<double, 2> d2b { 2.0, 5.0 };
		EXPECT_DOUBLE_EQ(simd::DotProduct(d2a.data(), d2b.data(), 2), -14.0);
		EXPECT_DOUBLE_EQ(simd::ReduceAbsSum(d2a.data(), 2), 7.0);
	}

	TEST(PackTests, ArgMinMaxAbsTail)
	{
		const std::array<double, 7> extArr { 2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -20.0 };
		EXPECT_EQ(simd::ArgMaxAbs(extArr.data(), extArr.size()), 6u);
		const std::array<double, 7> minArr { -10.0, 9.0, -8.0, 7.0, -6.0, 5.0, -0.25 };
		EXPECT_EQ(simd::ArgMinAbs(minArr.data(), minArr.size()), 6u);
	}

	TEST(PackTests, ReduceExtremumSubWidthAndMiddle)
	{
		const std::array<double, 2> d2 { 3.0, -4.0 };
		EXPECT_DOUBLE_EQ(simd::ReduceMin(d2.data(), 2), -4.0);
		EXPECT_DOUBLE_EQ(simd::ReduceMax(d2.data(), 2), 3.0);
		EXPECT_DOUBLE_EQ(simd::ReduceAbsMin(d2.data(), 2), 3.0);
		EXPECT_DOUBLE_EQ(simd::ReduceAbsMax(d2.data(), 2), 4.0);

		const std::array<double, 8> d8 { -3.0, 5.0, -1.0, 4.0, 2.0, -7.0, 6.0, 0.5 };
		EXPECT_DOUBLE_EQ((simd::ReduceMin<double, 4>(d8.data(), d8.size())), -7.0);
		EXPECT_DOUBLE_EQ((simd::ReduceMax<double, 4>(d8.data(), d8.size())), 6.0);
		EXPECT_DOUBLE_EQ((simd::ReduceAbsMin<double, 4>(d8.data(), d8.size())), 0.5);
		EXPECT_DOUBLE_EQ((simd::ReduceAbsMax<double, 4>(d8.data(), d8.size())), 7.0);
	}

	TEST(PackTests, ForEachUnrolledSingleStep)
	{
		std::array<double, 4> fe { 1.0, 2.0, 3.0, 4.0 };
		simd::ForEachUnrolled<double, 4>(
			fe.size(),
			[&](const std::size_t) {},
			[&](const std::size_t i) { fe[i] += 1.0; },
			[&](const std::size_t, const std::size_t) {});
		EXPECT_DOUBLE_EQ(fe[0], 2.0);
	}

	TEST(PackTests, MaskAndPackFromScalarRuntime)
	{
		std::array<bool, 4> mb { true, false, true, false };
		const auto mRuntime = simd::Mask<double, 4>::FromScalar(mb);
		EXPECT_TRUE(mRuntime.Scalar()[0]);
		EXPECT_FALSE(mRuntime.Scalar()[1]);

		std::array<double, 4> pb { 1.0, 2.0, 3.0, 4.0 };
		const auto pRuntime = simd::Pack<double, 4>::FromScalar(pb);
		EXPECT_DOUBLE_EQ(pRuntime[2], 3.0);
	}

	TEST(PackTests, BlendFloatMaskToInt32)
	{
		const auto fMask = simd::CmpLess(simd::Iota<float>(0.0f, 1.0f), simd::Broadcast<float>(4.0f));
		const auto i32Sel = simd::Blend(fMask, simd::Broadcast<std::int32_t>(10), simd::Broadcast<std::int32_t>(20));
		EXPECT_EQ(i32Sel[0], 10);
		EXPECT_EQ(i32Sel[4], 20);
	}

	TEST(PackTests, LoadStorePartialFullWidthAndInt128)
	{
		const std::array<double, 4> d4 { -3.0, 5.0, -1.0, 4.0 };
		const auto lpFull = simd::LoadPartial<double, 4>(d4.data(), 4);
		EXPECT_DOUBLE_EQ(lpFull[3], 4.0);
		std::array<double, 4> spFull {};
		simd::StorePartial(spFull.data(), lpFull, 4);
		EXPECT_DOUBLE_EQ(spFull[3], 4.0);

		std::array<std::int32_t, 4> i128Dst {};
		simd::StoreUnaligned(i128Dst.data(), simd::Pack<std::int32_t, 4>::Broadcast(42));
		EXPECT_EQ(i128Dst[3], 42);
	}
}	 // namespace simdt
