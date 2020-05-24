
#include <gtest/gtest.h>

#include <Simd.h>

#include <cmath>

namespace simdt
{
	class Double2Tests : public ::testing::Test
	{
	};

	TEST_F(Double2Tests, LoadAndGet)
	{
		simd::Double2 x(std::log(2.0), std::exp(1.0));

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, std::exp(1.0));

		simd::Double2::AlignedArray z;
		z[0] = std::log(2.0);
		z[1] = std::exp(1.0);
		simd::Double2 y(std::move(z));

		simd::Double2::AlignedArray t;
		x.Get(t);

		ASSERT_DOUBLE_EQ(t[0], std::log(2.0));
		ASSERT_DOUBLE_EQ(t[1], std::exp(1.0));
	}

	TEST_F(Double2Tests, Add)
	{
		simd::Double2 x(std::log(2.0), std::exp(1.0));
		simd::Double2 y(std::log(4.0), std::exp(2.0));
		auto z = x + y;

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, std::exp(1.0));

		double y1 = std::numeric_limits<double>::quiet_NaN();
		double y2 = std::numeric_limits<double>::quiet_NaN();
		y.Get(y1, y2);

		ASSERT_DOUBLE_EQ(y1, std::log(4.0));
		ASSERT_DOUBLE_EQ(y2, std::exp(2.0));

		double z1 = std::numeric_limits<double>::quiet_NaN();
		double z2 = std::numeric_limits<double>::quiet_NaN();
		z.Get(z1, z2);

		ASSERT_DOUBLE_EQ(z1, std::log(2.0) + std::log(4.0));
		ASSERT_DOUBLE_EQ(z2, std::exp(1.0) + std::exp(2.0));
	}

	TEST_F(Double2Tests, Subtract)
	{
		simd::Double2 x(std::log(2.0), std::exp(1.0));
		simd::Double2 y(std::log(4.0), std::exp(2.0));
		auto z = x - y;

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, std::exp(1.0));

		double y1 = std::numeric_limits<double>::quiet_NaN();
		double y2 = std::numeric_limits<double>::quiet_NaN();
		y.Get(y1, y2);

		ASSERT_DOUBLE_EQ(y1, std::log(4.0));
		ASSERT_DOUBLE_EQ(y2, std::exp(2.0));

		double z1 = std::numeric_limits<double>::quiet_NaN();
		double z2 = std::numeric_limits<double>::quiet_NaN();
		z.Get(z1, z2);

		ASSERT_DOUBLE_EQ(z1, std::log(2.0) - std::log(4.0));
		ASSERT_DOUBLE_EQ(z2, std::exp(1.0) - std::exp(2.0));
	}

	TEST_F(Double2Tests, Multiply)
	{
		simd::Double2 x(std::log(2.0), std::exp(1.0));
		simd::Double2 y(std::log(4.0), std::exp(2.0));
		auto z = x * y;

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, std::exp(1.0));

		double y1 = std::numeric_limits<double>::quiet_NaN();
		double y2 = std::numeric_limits<double>::quiet_NaN();
		y.Get(y1, y2);

		ASSERT_DOUBLE_EQ(y1, std::log(4.0));
		ASSERT_DOUBLE_EQ(y2, std::exp(2.0));

		double z1 = std::numeric_limits<double>::quiet_NaN();
		double z2 = std::numeric_limits<double>::quiet_NaN();
		z.Get(z1, z2);

		ASSERT_DOUBLE_EQ(z1, std::log(2.0) * std::log(4.0));
		ASSERT_DOUBLE_EQ(z2, std::exp(1.0) * std::exp(2.0));
	}

	TEST_F(Double2Tests, Divide)
	{
		simd::Double2 x(std::log(2.0), std::exp(1.0));
		simd::Double2 y(std::log(4.0), std::exp(2.0));
		auto z = x / y;

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, std::exp(1.0));

		double y1 = std::numeric_limits<double>::quiet_NaN();
		double y2 = std::numeric_limits<double>::quiet_NaN();
		y.Get(y1, y2);

		ASSERT_DOUBLE_EQ(y1, std::log(4.0));
		ASSERT_DOUBLE_EQ(y2, std::exp(2.0));

		double z1 = std::numeric_limits<double>::quiet_NaN();
		double z2 = std::numeric_limits<double>::quiet_NaN();
		z.Get(z1, z2);

		ASSERT_DOUBLE_EQ(z1, std::log(2.0) / std::log(4.0));
		ASSERT_DOUBLE_EQ(z2, std::exp(1.0) / std::exp(2.0));
	}

	TEST_F(Double2Tests, Max)
	{
		simd::Double2 x(std::log(2.0), std::exp(1.0));
		simd::Double2 y(std::log(4.0), std::exp(2.0));
		auto z = simd::max(x, y);

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, std::exp(1.0));

		double y1 = std::numeric_limits<double>::quiet_NaN();
		double y2 = std::numeric_limits<double>::quiet_NaN();
		y.Get(y1, y2);

		ASSERT_DOUBLE_EQ(y1, std::log(4.0));
		ASSERT_DOUBLE_EQ(y2, std::exp(2.0));

		double z1 = std::numeric_limits<double>::quiet_NaN();
		double z2 = std::numeric_limits<double>::quiet_NaN();
		z.Get(z1, z2);

		ASSERT_DOUBLE_EQ(z1, std::max(std::log(2.0), std::log(4.0)));
		ASSERT_DOUBLE_EQ(z2, std::max(std::exp(1.0), std::exp(2.0)));
	}

	TEST_F(Double2Tests, Min)
	{
		simd::Double2 x(std::log(2.0), std::exp(1.0));
		simd::Double2 y(std::log(4.0), std::exp(2.0));
		auto z = simd::min(x, y);

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, std::exp(1.0));

		double y1 = std::numeric_limits<double>::quiet_NaN();
		double y2 = std::numeric_limits<double>::quiet_NaN();
		y.Get(y1, y2);

		ASSERT_DOUBLE_EQ(y1, std::log(4.0));
		ASSERT_DOUBLE_EQ(y2, std::exp(2.0));

		double z1 = std::numeric_limits<double>::quiet_NaN();
		double z2 = std::numeric_limits<double>::quiet_NaN();
		z.Get(z1, z2);

		ASSERT_DOUBLE_EQ(z1, std::min(std::log(2.0), std::log(4.0)));
		ASSERT_DOUBLE_EQ(z2, std::min(std::exp(1.0), std::exp(2.0)));
	}

	TEST_F(Double2Tests, Abs)
	{
		simd::Double2 x(std::log(2.0), -std::exp(1.0));

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, -std::exp(1.0));

		auto y = simd::abs(x);
		double y1 = std::numeric_limits<double>::quiet_NaN();
		double y2 = std::numeric_limits<double>::quiet_NaN();
		y.Get(y1, y2);

		ASSERT_DOUBLE_EQ(y1, std::log(2.0));
		ASSERT_DOUBLE_EQ(y2, std::exp(1.0));
	}

	TEST_F(Double2Tests, Sqrt)
	{
		simd::Double2 x(std::log(2.0), std::exp(1.0));

		double x1 = std::numeric_limits<double>::quiet_NaN();
		double x2 = std::numeric_limits<double>::quiet_NaN();
		x.Get(x1, x2);

		ASSERT_DOUBLE_EQ(x1, std::log(2.0));
		ASSERT_DOUBLE_EQ(x2, std::exp(1.0));

		auto y = simd::sqrt(x);
		double y1 = std::numeric_limits<double>::quiet_NaN();
		double y2 = std::numeric_limits<double>::quiet_NaN();
		y.Get(y1, y2);

		ASSERT_DOUBLE_EQ(y1, std::sqrt(std::log(2.0)));
		ASSERT_DOUBLE_EQ(y2, std::sqrt(std::exp(1.0)));
	}
}
