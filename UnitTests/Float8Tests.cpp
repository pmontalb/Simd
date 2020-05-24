
#include <gtest/gtest.h>

#include <Simd.h>

#include <cmath>

#ifdef __AVX__

namespace simdt
{
	class Float8Tests : public ::testing::Test
	{
	};

	TEST_F(Float8Tests, LoadAndGet)
	{
		simd::Float8::AlignedArray xsIn;
		xsIn[0] = std::log(2.0f);
		xsIn[1] = std::log(4.0f);
		xsIn[2] = std::exp(1.0f);
		xsIn[3] = std::exp(2.0f);
		xsIn[4] = std::log(4.0f);
		xsIn[5] = std::log(8.0f);
		xsIn[6] = std::exp(4.0f);
		xsIn[7] = std::exp(8.0f);
		simd::Float8 x(std::move(xsIn));

		simd::Float8::AlignedArray xs;
		x.Get(xs);

		for (size_t i = 0; i < xsIn.size(); ++i)
			ASSERT_FLOAT_EQ(xs[i], xsIn[i]);
	}

	TEST_F(Float8Tests, Add)
	{
		simd::Float8::AlignedArray xs;
		for (size_t i = 0; i < xs.size(); ++i)
			xs[i] = std::log(2.0f * static_cast<float>(i + 1));
		simd::Float8 x(std::move(xs));
		simd::Float8::AlignedArray ys;
		for (size_t i = 0; i < ys.size(); ++i)
			ys[i] = std::exp(2.0f * static_cast<float>(i + 1));
		simd::Float8 y(std::move(ys));
		auto z = x + y;

		simd::Float8::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], xs[i] + ys[i]);
	}

	TEST_F(Float8Tests, Subtract)
	{
		simd::Float8::AlignedArray xs;
		for (size_t i = 0; i < xs.size(); ++i)
			xs[i] = std::log(2.0f * static_cast<float>(i + 1));
		simd::Float8 x(std::move(xs));
		simd::Float8::AlignedArray ys;
		for (size_t i = 0; i < ys.size(); ++i)
			ys[i] = std::exp(2.0f * static_cast<float>(i + 1));
		simd::Float8 y(std::move(ys));
		auto z = x - y;

		simd::Float8::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], xs[i] - ys[i]);
	}

	TEST_F(Float8Tests, Multiply)
	{
		simd::Float8::AlignedArray xs;
		for (size_t i = 0; i < xs.size(); ++i)
			xs[i] = std::log(2.0f * static_cast<float>(i + 1));
		simd::Float8 x(std::move(xs));
		simd::Float8::AlignedArray ys;
		for (size_t i = 0; i < ys.size(); ++i)
			ys[i] = std::exp(2.0f * static_cast<float>(i + 1));
		simd::Float8 y(std::move(ys));
		auto z = x * y;

		simd::Float8::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], xs[i] * ys[i]);
	}

	TEST_F(Float8Tests, Divide)
	{
		simd::Float8::AlignedArray xs;
		for (size_t i = 0; i < xs.size(); ++i)
			xs[i] = std::log(2.0f * static_cast<float>(i + 1));
		simd::Float8 x(std::move(xs));
		simd::Float8::AlignedArray ys;
		for (size_t i = 0; i < ys.size(); ++i)
			ys[i] = std::exp(2.0f * static_cast<float>(i + 1));
		simd::Float8 y(std::move(ys));
		auto z = x / y;

		simd::Float8::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], xs[i] / ys[i]);
	}

	TEST_F(Float8Tests, Max)
	{
		simd::Float8::AlignedArray xs;
		for (size_t i = 0; i < xs.size(); ++i)
			xs[i] = std::log(2.0f * static_cast<float>(i + 1));
		simd::Float8 x(std::move(xs));
		simd::Float8::AlignedArray ys;
		for (size_t i = 0; i < ys.size(); ++i)
			ys[i] = std::exp(2.0f * static_cast<float>(i + 1));
		simd::Float8 y(std::move(ys));
		auto z = simd::max(x, y);

		simd::Float8::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], std::max(xs[i], ys[i]));
	}

	TEST_F(Float8Tests, Min)
	{
		simd::Float8::AlignedArray xs;
		for (size_t i = 0; i < xs.size(); ++i)
			xs[i] = std::log(2.0f * static_cast<float>(i + 1));
		simd::Float8 x(std::move(xs));
		simd::Float8::AlignedArray ys;
		for (size_t i = 0; i < ys.size(); ++i)
			ys[i] = std::exp(2.0f * static_cast<float>(i + 1));
		simd::Float8 y(std::move(ys));
		auto z = simd::min(x, y);

		simd::Float8::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], std::min(xs[i], ys[i]));
	}

	TEST_F(Float8Tests, Abs)
	{
		simd::Float8::AlignedArray xs;
		for (size_t i = 0; i < xs.size(); ++i)
			xs[i] = static_cast<float>(std::pow(-1, i + 1) * std::fabs(std::log(2.0 * static_cast<double>(i + 1))));
		simd::Float8 x(std::move(xs));
		auto z = simd::abs(x);

		simd::Float8::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], std::abs(xs[i]));
	}

	TEST_F(Float8Tests, Sqrt)
	{
		simd::Float8::AlignedArray xs;
		for (size_t i = 0; i < xs.size(); ++i)
			xs[i] = static_cast<float>(std::fabs(std::log(2.0 * static_cast<double>(i + 1))));
		simd::Float8 x(std::move(xs));
		auto z = simd::sqrt(x);

		simd::Float8::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], std::sqrt(xs[i]));
	}
}

#endif
