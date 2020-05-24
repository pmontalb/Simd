
#include <gtest/gtest.h>

#include <Simd.h>

#include <cmath>

#ifdef __AVX2__

namespace simdt
{
	class Double4Tests : public ::testing::Test
	{
	};

	TEST_F(Double4Tests, LoadAndGet)
	{
		simd::Double4::AlignedArray xsIn;
		xsIn[0] = std::log(2.0);
		xsIn[1] = std::log(4.0);
		xsIn[2] = std::exp(1.0);
		xsIn[3] = std::exp(2.0);
		simd::Double4 x(std::move(xsIn));

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray z;
		z[0] = std::log(2.0);
		z[1] = std::log(4.0);
		z[2] = std::exp(1.0);
		z[3] = std::exp(2.0);
		simd::Double4 y(std::move(z));

		simd::Double4::AlignedArray t;
		y.Get(t);

		ASSERT_DOUBLE_EQ(t[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(t[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(t[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(t[3], std::exp(2.0));
	}

	TEST_F(Double4Tests, Add)
	{
		simd::Double4 x(std::log(2.0), std::log(4.0), std::exp(1.0), std::exp(2.0));
		simd::Double4 y(std::log(4.0), std::log(8.0), std::exp(2.0), std::exp(4.0));
		auto z = x + y;

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray ys;
		y.Get(ys);

		ASSERT_DOUBLE_EQ(ys[0], std::log(4.0));
		ASSERT_DOUBLE_EQ(ys[1], std::log(8.0));
		ASSERT_DOUBLE_EQ(ys[2], std::exp(2.0));
		ASSERT_DOUBLE_EQ(ys[3], std::exp(4.0));

		simd::Double4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_DOUBLE_EQ(zs[i], xs[i] + ys[i]);
	}

	TEST_F(Double4Tests, Subtract)
	{
		simd::Double4 x(std::log(2.0), std::log(4.0), std::exp(1.0), std::exp(2.0));
		simd::Double4 y(std::log(4.0), std::log(8.0), std::exp(2.0), std::exp(4.0));
		auto z = x - y;

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray ys;
		y.Get(ys);

		ASSERT_DOUBLE_EQ(ys[0], std::log(4.0));
		ASSERT_DOUBLE_EQ(ys[1], std::log(8.0));
		ASSERT_DOUBLE_EQ(ys[2], std::exp(2.0));
		ASSERT_DOUBLE_EQ(ys[3], std::exp(4.0));

		simd::Double4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_DOUBLE_EQ(zs[i], xs[i] - ys[i]);
	}

	TEST_F(Double4Tests, Multiply)
	{
		simd::Double4 x(std::log(2.0), std::log(4.0), std::exp(1.0), std::exp(2.0));
		simd::Double4 y(std::log(4.0), std::log(8.0), std::exp(2.0), std::exp(4.0));
		auto z = x * y;

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray ys;
		y.Get(ys);

		ASSERT_DOUBLE_EQ(ys[0], std::log(4.0));
		ASSERT_DOUBLE_EQ(ys[1], std::log(8.0));
		ASSERT_DOUBLE_EQ(ys[2], std::exp(2.0));
		ASSERT_DOUBLE_EQ(ys[3], std::exp(4.0));

		simd::Double4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_DOUBLE_EQ(zs[i], xs[i] * ys[i]);
	}

	TEST_F(Double4Tests, Divide)
	{
		simd::Double4 x(std::log(2.0), std::log(4.0), std::exp(1.0), std::exp(2.0));
		simd::Double4 y(std::log(4.0), std::log(8.0), std::exp(2.0), std::exp(4.0));
		auto z = x / y;

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray ys;
		y.Get(ys);

		ASSERT_DOUBLE_EQ(ys[0], std::log(4.0));
		ASSERT_DOUBLE_EQ(ys[1], std::log(8.0));
		ASSERT_DOUBLE_EQ(ys[2], std::exp(2.0));
		ASSERT_DOUBLE_EQ(ys[3], std::exp(4.0));

		simd::Double4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_DOUBLE_EQ(zs[i], xs[i] / ys[i]);
	}

	TEST_F(Double4Tests, Max)
	{
		simd::Double4 x(std::log(2.0), std::log(4.0), std::exp(1.0), std::exp(2.0));
		simd::Double4 y(std::log(4.0), std::log(8.0), std::exp(2.0), std::exp(4.0));
		auto z = simd::max(x, y);

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray ys;
		y.Get(ys);

		ASSERT_DOUBLE_EQ(ys[0], std::log(4.0));
		ASSERT_DOUBLE_EQ(ys[1], std::log(8.0));
		ASSERT_DOUBLE_EQ(ys[2], std::exp(2.0));
		ASSERT_DOUBLE_EQ(ys[3], std::exp(4.0));

		simd::Double4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_DOUBLE_EQ(zs[i], std::max(xs[i], ys[i]));
	}

	TEST_F(Double4Tests, Min)
	{
		simd::Double4 x(std::log(2.0), std::log(4.0), std::exp(1.0), std::exp(2.0));
		simd::Double4 y(std::log(4.0), std::log(8.0), std::exp(2.0), std::exp(4.0));
		auto z = simd::min(x, y);

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray ys;
		y.Get(ys);

		ASSERT_DOUBLE_EQ(ys[0], std::log(4.0));
		ASSERT_DOUBLE_EQ(ys[1], std::log(8.0));
		ASSERT_DOUBLE_EQ(ys[2], std::exp(2.0));
		ASSERT_DOUBLE_EQ(ys[3], std::exp(4.0));

		simd::Double4::AlignedArray zs;
		z.Get(zs);

		for (size_t i = 0; i < zs.size(); ++i)
			ASSERT_DOUBLE_EQ(zs[i], std::min(xs[i], ys[i]));
	}

	TEST_F(Double4Tests, Abs)
	{
		simd::Double4 x(std::log(2.0), -std::log(4.0), -std::exp(1.0), std::exp(2.0));
		auto y = simd::abs(x);

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], -std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], -std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray ys;
		y.Get(ys);

		for (size_t i = 0; i < ys.size(); ++i)
			ASSERT_DOUBLE_EQ(ys[i], std::abs(xs[i]));
	}

	TEST_F(Double4Tests, Sqrt)
	{
		simd::Double4 x(std::log(2.0), std::log(4.0), std::exp(1.0), std::exp(2.0));
		auto y = simd::sqrt(x);

		simd::Double4::AlignedArray xs;
		x.Get(xs);

		ASSERT_DOUBLE_EQ(xs[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(xs[1], std::log(4.0));
		ASSERT_DOUBLE_EQ(xs[2], std::exp(1.0));
		ASSERT_DOUBLE_EQ(xs[3], std::exp(2.0));

		simd::Double4::AlignedArray ys;
		y.Get(ys);

		for (size_t i = 0; i < ys.size(); ++i)
			ASSERT_DOUBLE_EQ(ys[i], std::sqrt(xs[i]));
	}
}

#endif
