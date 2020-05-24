
#include <gtest/gtest.h>

#include <Simd.h>

#include <cmath>

namespace simdt
{
	class Float4Tests : public ::testing::Test
	{
	};

	TEST_F(Float4Tests, LoadAndGet)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));

		float x1 = std::numeric_limits<float>::quiet_NaN();
		float x2 = std::numeric_limits<float>::quiet_NaN();
		float x3 = std::numeric_limits<float>::quiet_NaN();
		float x4 = std::numeric_limits<float>::quiet_NaN();
		x.Get(x1, x2, x3, x4);

		ASSERT_FLOAT_EQ(x1, std::log(2.0f));
		ASSERT_FLOAT_EQ(x2, std::log(4.0f));
		ASSERT_FLOAT_EQ(x3, std::exp(1.0f));
		ASSERT_FLOAT_EQ(x4, std::exp(2.0f));

		simd::Float4::AlignedArray z;
		z[0] = std::log(2.0f);
		z[1] = std::log(4.0f);
		z[2] = std::exp(1.0f);
		z[3] = std::exp(2.0f);
		simd::Float4 y(std::move(z));

		simd::Float4::AlignedArray t;
		y.Get(t);

		ASSERT_FLOAT_EQ(t[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(t[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(t[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(t[3], std::exp(2.0f));
	}

	TEST_F(Float4Tests, Add)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		simd::Float4 y(std::log(4.0f), std::log(8.0f), std::exp(2.0f), std::exp(4.0f));
		auto z = x + y;

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		ASSERT_FLOAT_EQ(ys[0], std::log(4.0f));
		ASSERT_FLOAT_EQ(ys[1], std::log(8.0f));
		ASSERT_FLOAT_EQ(ys[2], std::exp(2.0f));
		ASSERT_FLOAT_EQ(ys[3], std::exp(4.0f));

		simd::Float4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], xs[i] + ys[i]);
	}

	TEST_F(Float4Tests, Subtract)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		simd::Float4 y(std::log(4.0f), std::log(8.0f), std::exp(2.0f), std::exp(4.0f));
		auto z = x - y;

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		ASSERT_FLOAT_EQ(ys[0], std::log(4.0f));
		ASSERT_FLOAT_EQ(ys[1], std::log(8.0f));
		ASSERT_FLOAT_EQ(ys[2], std::exp(2.0f));
		ASSERT_FLOAT_EQ(ys[3], std::exp(4.0f));

		simd::Float4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], xs[i] - ys[i]);
	}

	TEST_F(Float4Tests, Multiply)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		simd::Float4 y(std::log(4.0f), std::log(8.0f), std::exp(2.0f), std::exp(4.0f));
		auto z = x * y;

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		ASSERT_FLOAT_EQ(ys[0], std::log(4.0f));
		ASSERT_FLOAT_EQ(ys[1], std::log(8.0f));
		ASSERT_FLOAT_EQ(ys[2], std::exp(2.0f));
		ASSERT_FLOAT_EQ(ys[3], std::exp(4.0f));

		simd::Float4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], xs[i] * ys[i]);
	}

	TEST_F(Float4Tests, Divide)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		simd::Float4 y(std::log(4.0f), std::log(8.0f), std::exp(2.0f), std::exp(4.0f));
		auto z = x / y;

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		ASSERT_FLOAT_EQ(ys[0], std::log(4.0f));
		ASSERT_FLOAT_EQ(ys[1], std::log(8.0f));
		ASSERT_FLOAT_EQ(ys[2], std::exp(2.0f));
		ASSERT_FLOAT_EQ(ys[3], std::exp(4.0f));

		simd::Float4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], xs[i] / ys[i]);
	}

	TEST_F(Float4Tests, Max)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		simd::Float4 y(std::log(4.0f), std::log(8.0f), std::exp(2.0f), std::exp(4.0f));
		auto z = simd::max(x, y);

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		ASSERT_FLOAT_EQ(ys[0], std::log(4.0f));
		ASSERT_FLOAT_EQ(ys[1], std::log(8.0f));
		ASSERT_FLOAT_EQ(ys[2], std::exp(2.0f));
		ASSERT_FLOAT_EQ(ys[3], std::exp(4.0f));

		simd::Float4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], std::max(xs[i], ys[i]));
	}

	TEST_F(Float4Tests, Min)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		simd::Float4 y(std::log(4.0f), std::log(8.0f), std::exp(2.0f), std::exp(4.0f));
		auto z = simd::min(x, y);

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		ASSERT_FLOAT_EQ(ys[0], std::log(4.0f));
		ASSERT_FLOAT_EQ(ys[1], std::log(8.0f));
		ASSERT_FLOAT_EQ(ys[2], std::exp(2.0f));
		ASSERT_FLOAT_EQ(ys[3], std::exp(4.0f));

		simd::Float4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_FLOAT_EQ(zs[i], std::min(xs[i], ys[i]));
	}

	TEST_F(Float4Tests, Abs)
	{
		simd::Float4 x(std::log(2.0f), -std::log(4.0f), -std::exp(1.0f), std::exp(2.0f));
		auto y = simd::abs(x);

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], -std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], -std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		for (size_t i = 0; i < ys.size(); ++i)
			ASSERT_FLOAT_EQ(ys[i], std::abs(xs[i]));
	}

	TEST_F(Float4Tests, Sqrt)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		auto y = simd::sqrt(x);

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		for (size_t i = 0; i < ys.size(); ++i)
			ASSERT_FLOAT_EQ(ys[i], std::sqrt(xs[i]));
	}

	TEST_F(Float4Tests, Log)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		auto y = simd::log(x);

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		for (size_t i = 0; i < ys.size(); ++i)
			ASSERT_FLOAT_EQ(ys[i], std::log(xs[i]));
	}

	TEST_F(Float4Tests, Exp)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		auto y = simd::exp(x);

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		for (size_t i = 0; i < ys.size(); ++i)
			ASSERT_FLOAT_EQ(ys[i], std::exp(xs[i]));
	}

	TEST_F(Float4Tests, Sin)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		auto y = simd::sin(x);

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		for (size_t i = 0; i < ys.size(); ++i)
			ASSERT_FLOAT_EQ(ys[i], std::sin(xs[i]));
	}

	TEST_F(Float4Tests, Cos)
	{
		simd::Float4 x(std::log(2.0f), std::log(4.0f), std::exp(1.0f), std::exp(2.0f));
		auto y = simd::cos(x);

		simd::Float4::AlignedArray xs;
		x.Get(xs);

		ASSERT_FLOAT_EQ(xs[0], std::log(2.0f));
		ASSERT_FLOAT_EQ(xs[1], std::log(4.0f));
		ASSERT_FLOAT_EQ(xs[2], std::exp(1.0f));
		ASSERT_FLOAT_EQ(xs[3], std::exp(2.0f));

		simd::Float4::AlignedArray ys;
		y.Get(ys);

		for (size_t i = 0; i < ys.size(); ++i)
			ASSERT_FLOAT_EQ(ys[i], std::cos(xs[i]));
	}
}
