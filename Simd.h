#pragma once

#define USE_SSE2

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wimplicit-float-conversion"
	#pragma clang diagnostic ignored "-Wold-style-cast"
	#pragma clang diagnostic ignored "-Wcast-qual"
	#include "sse_mathfun.h"
	#pragma clang diagnostic pop
#elif defined(__GNUC__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wold-style-cast"
	#pragma GCC diagnostic ignored "-Wcast-qual"
	#pragma GCC diagnostic ignored "-Wcast-align"
	#pragma GCC diagnostic ignored "-Wfloat-conversion"
	#pragma GCC diagnostic ignored "-Wmissing-declarations"
	#include "sse_mathfun.h"
	#pragma GCC diagnostic pop
#endif

#include <type_traits>
#include <array>
#include <cassert>

#include <x86intrin.h>

namespace simd
{
	class Double2;
	class Float4;
	class Int4;
	
	class Double4;
	class Float8;
	class Int8;

	namespace detail
	{
		template<typename T>
		struct IsSupported { static constexpr bool value = false; };
		
		template<>
		struct IsSupported<Double2> { static constexpr bool value = true; };

		template<>
		struct IsSupported<Float4> { static constexpr bool value = true; };

		template<>
		struct IsSupported<Int4> { static constexpr bool value = true; };
		
		template<>
		struct IsSupported<Double4> { static constexpr bool value = true; };

		template<>
		struct IsSupported<Float8> { static constexpr bool value = true; };

		template<>
		struct IsSupported<Int8> { static constexpr bool value = true; };

		template<typename TypeImpl>
		class IArithmeticType
		{
			static_assert(detail::IsSupported<TypeImpl>::value, "Unsupported type");
		public:
			inline TypeImpl& operator+=(const TypeImpl& rhs) noexcept { return this->AddEqual(rhs); }
			inline TypeImpl& operator-=(const TypeImpl& rhs) noexcept { return this->SubtractEqual(rhs); }
			inline TypeImpl& operator*=(const TypeImpl& rhs) noexcept { return this->MultiplyEqual(rhs); }
			inline TypeImpl& operator/=(const TypeImpl& rhs) noexcept { return this->DivideEqual(rhs); }

			inline TypeImpl operator+(const TypeImpl& rhs) const noexcept { return this->Add(rhs); }
			inline TypeImpl operator-(const TypeImpl& rhs) const noexcept { return this->Subtract(rhs); }
			inline TypeImpl operator*(const TypeImpl& rhs) const noexcept { return this->Multiply(rhs); }
			inline TypeImpl operator/(const TypeImpl& rhs) const noexcept { return this->Divide(rhs); }

			inline TypeImpl& AddEqual(const TypeImpl& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->AddEqualImpl(rhs);
			}
			inline TypeImpl Add(const TypeImpl& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs += rhs;

				return lhs;
			}

			inline TypeImpl& SubtractEqual(const TypeImpl& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->SubtractEqualImpl(rhs);
			}
			inline TypeImpl Subtract(const TypeImpl& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs -= rhs;

				return lhs;
			}

			inline TypeImpl& MultiplyEqual(const TypeImpl& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->MultiplyEqualImpl(rhs);
			}
			inline TypeImpl Multiply(const TypeImpl& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs *= rhs;

				return lhs;
			}

			inline TypeImpl& DivideEqual(const TypeImpl& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->DivideEqualImpl(rhs);
			}
			inline TypeImpl Divide(const TypeImpl& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs /= rhs;

				return lhs;
			}

			inline TypeImpl AbsoluteValue() const noexcept
			{
				return static_cast<const TypeImpl*>(this)->AbsoluteValueImpl();
			}

			inline TypeImpl Max(const TypeImpl& rhs) const noexcept
			{
				return static_cast<const TypeImpl*>(this)->MaxImpl(rhs);
			}

			inline TypeImpl Min(const TypeImpl& rhs) const noexcept
			{
				return static_cast<const TypeImpl*>(this)->MinImpl(rhs);
			}
		};

		static inline bool isAligned(const void* ptr, const size_t alignment = 16) noexcept
		{
			auto uintPtr = reinterpret_cast<std::uintptr_t>(ptr);
			return !(uintPtr % alignment);
		}
	}

	#if defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__)
		class Double2: public detail::IArithmeticType<Double2>
		{
			friend class detail::IArithmeticType<Double2>;
		public:
			struct __attribute__((aligned(16))) AlignedArray : public std::array<double, 2> { using std::array<double, 2>::array; };

			Double2() noexcept = default;
			explicit Double2(const double x, const double y) noexcept
			{
				_value = _mm_set_pd(y, x);  // that's how the values are stored!
			}
			explicit Double2(AlignedArray&& xy) noexcept
				: Double2(xy.data())
			{
			}
			explicit Double2(double* xy) noexcept
			{
				assert(detail::isAligned(xy));
				_value = _mm_load_pd(xy);
			}

			void Get(double& x, double& y) const noexcept
			{
				//_mm_store1_pd(&x, _value);
				x = _value[0];
				y = _value[1];
			}
			void Get(AlignedArray& xy) const noexcept
			{
				_mm_store_pd(xy.data(), _value);
			}

			inline Double2 SquareRoot() const noexcept
			{
				Double2 ret {};
				ret._value = _mm_sqrt_pd(_value);
				return ret;
			}

		private:
			inline Double2& AddEqualImpl(const Double2& rhs) noexcept
			{
				_value = _mm_add_pd(_value, rhs._value);
				return *this;
			}

			inline Double2& SubtractEqualImpl(const Double2& rhs) noexcept
			{
				_value = _mm_sub_pd(_value, rhs._value);
				return *this;
			}

			inline Double2& MultiplyEqualImpl(const Double2& rhs) noexcept
			{
				_value = _mm_mul_pd(_value, rhs._value);
				return *this;
			}

			inline Double2& DivideEqualImpl(const Double2& rhs) noexcept
			{
				_value = _mm_div_pd(_value, rhs._value);
				return *this;
			}

			inline Double2 AbsoluteValueImpl() const noexcept
			{
				static const __m128d signmask = _mm_set1_pd(-0.0); // 0x80000000

				Double2 ret {};
				ret._value = _mm_andnot_pd(signmask, _value);
				return ret;
			}

			inline Double2 MaxImpl(const Double2& rhs) const noexcept
			{
				Double2 ret {};
				ret._value = _mm_max_pd(rhs._value, _value);
				return ret;
			}

			inline Double2 MinImpl(const Double2& rhs) const noexcept
			{
				Double2 ret {};
				ret._value = _mm_min_pd(rhs._value, _value);
				return ret;
			}

		private:
			__m128d _value {};
		};

		static inline Double2 max(const Double2& x, const Double2& y) noexcept { return x.Max(y); }
		static inline Double2 min(const Double2& x, const Double2& y) noexcept { return x.Min(y); }
		static inline Double2 abs(const Double2& x) noexcept { return x.AbsoluteValue(); }
		static inline Double2 sqrt(const Double2& x) noexcept { return x.SquareRoot(); }

		class Float4: public detail::IArithmeticType<Float4>
		{
			friend class detail::IArithmeticType<Float4>;
		public:
			struct __attribute__((aligned(16))) AlignedArray : public std::array<float, 4> {};

			Float4() noexcept = default;
			explicit Float4(const float x, const float y, const float z, const float t) noexcept
			{
				_value = _mm_set_ps(t, z, y, x);
			}
			explicit Float4(AlignedArray&& xy) noexcept
				: Float4(xy.data())
			{
			}
			explicit Float4(float* xy) noexcept
			{
				assert(detail::isAligned(xy));
				_value = _mm_load_ps(xy);
			}

			void Get(float& x, float& y, float& z, float& t) const noexcept
			{
				x = _value[0];
				y = _value[1];
				z = _value[2];
				t = _value[3];
			}
			void Get(AlignedArray& xy) const noexcept
			{
				_mm_store_ps(xy.data(), _value);
			}

			// using sse_mathfun
			inline Float4 Log() const noexcept
			{
				Float4 ret {};
				ret._value = log_ps(_value);
				return ret;
			}
			inline Float4 Exp() const noexcept
			{
				Float4 ret {};
				ret._value = exp_ps(_value);
				return ret;
			}
			inline Float4 Sin() const noexcept
			{
				Float4 ret {};
				ret._value = sin_ps(_value);
				return ret;
			}
			inline Float4 Cos() const noexcept
			{
				Float4 ret {};
				ret._value = cos_ps(_value);
				return ret;
			}

			inline Float4 SquareRoot() const noexcept
			{
				Float4 ret {};
				ret._value = _mm_sqrt_ps(_value);
				return ret;
			}
		private:
			inline Float4& AddEqualImpl(const Float4& rhs) noexcept
			{
				_value = _mm_add_ps(_value, rhs._value);
				return *this;
			}

			inline Float4& SubtractEqualImpl(const Float4& rhs) noexcept
			{
				_value = _mm_sub_ps(_value, rhs._value);
				return *this;
			}

			inline Float4& MultiplyEqualImpl(const Float4& rhs) noexcept
			{
				_value = _mm_mul_ps(_value, rhs._value);
				return *this;
			}

			inline Float4& DivideEqualImpl(const Float4& rhs) noexcept
			{
				_value = _mm_div_ps(_value, rhs._value);
				return *this;
			}

			inline Float4 AbsoluteValueImpl() const noexcept
			{
				static const __m128 signmask = _mm_set1_ps(-0.0f); // 0x80000000

				Float4 ret {};
				ret._value = _mm_andnot_ps(signmask, _value);
				return ret;
			}

			inline Float4 MaxImpl(const Float4& rhs) const noexcept
			{
				Float4 ret {};
				ret._value = _mm_max_ps(rhs._value, _value);
				return ret;
			}

			inline Float4 MinImpl(const Float4& rhs) const noexcept
			{
				Float4 ret {};
				ret._value = _mm_min_ps(rhs._value, _value);
				return ret;
			}

		private:
			__m128 _value {};
		};

		static inline Float4 max(const Float4& x, const Float4& y) noexcept { return x.Max(y); }
		static inline Float4 min(const Float4& x, const Float4& y) noexcept { return x.Min(y); }
		static inline Float4 abs(const Float4& x) noexcept { return x.AbsoluteValue(); }
		static inline Float4 sqrt(const Float4& x) noexcept { return x.SquareRoot(); }
		static inline Float4 exp(const Float4& x) noexcept { return x.Exp(); }
		static inline Float4 log(const Float4& x) noexcept { return x.Log(); }
		static inline Float4 sin(const Float4& x) noexcept { return x.Sin(); }
		static inline Float4 cos(const Float4& x) noexcept { return x.Cos(); }
	#endif

	#if defined(__AVX2__)
		class Double4: public detail::IArithmeticType<Double4>
		{
			friend class detail::IArithmeticType<Double4>;
		public:
			struct __attribute__((aligned(16))) AlignedArray : public std::array<double, 4> { using std::array<double, 4>::array; };

			Double4() noexcept = default;
			explicit Double4(const double x, const double y, const double z, const double t) noexcept
			{
				_value = _mm256_set_pd(t, z, y, x);  // that's how the values are stored!
			}
			explicit Double4(AlignedArray&& xy) noexcept
				: Double4(xy.data())
			{
			}
			explicit Double4(double* xy) noexcept
			{
				assert(detail::isAligned(xy));
				_value = _mm256_load_pd(xy);
			}

			void Get(double& x, double& y, double& z, double& t) const noexcept
			{
				//_mm_store1_pd(&x, _value);
				x = _value[0];
				y = _value[1];
				z = _value[2];
				t = _value[3];
			}
			void Get(AlignedArray& xy) const noexcept
			{
				_mm256_store_pd(xy.data(), _value);
			}

			inline Double4 SquareRoot() const noexcept
			{
				Double4 ret {};
				ret._value = _mm256_sqrt_pd(_value);
				return ret;
			}

		private:
			inline Double4& AddEqualImpl(const Double4& rhs) noexcept
			{
				_value = _mm256_add_pd(_value, rhs._value);
				return *this;
			}

			inline Double4& SubtractEqualImpl(const Double4& rhs) noexcept
			{
				_value = _mm256_sub_pd(_value, rhs._value);
				return *this;
			}

			inline Double4& MultiplyEqualImpl(const Double4& rhs) noexcept
			{
				_value = _mm256_mul_pd(_value, rhs._value);
				return *this;
			}

			inline Double4& DivideEqualImpl(const Double4& rhs) noexcept
			{
				_value = _mm256_div_pd(_value, rhs._value);
				return *this;
			}

			inline Double4 AbsoluteValueImpl() const noexcept
			{
				static const auto minus1 = _mm256_set1_epi64x(-1);
				static const auto mask = _mm256_castsi256_pd(_mm256_srli_epi64(minus1, 1));

				Double4 ret {};
				ret._value = _mm256_and_pd(mask, _value);
				return ret;
			}

			inline Double4 MaxImpl(const Double4& rhs) const noexcept
			{
				Double4 ret {};
				ret._value = _mm256_max_pd(rhs._value, _value);
				return ret;
			}

			inline Double4 MinImpl(const Double4& rhs) const noexcept
			{
				Double4 ret {};
				ret._value = _mm256_min_pd(rhs._value, _value);
				return ret;
			}

		private:
			__m256d _value {};
		};

		static inline Double4 max(const Double4& x, const Double4& y) noexcept { return x.Max(y); }
		static inline Double4 min(const Double4& x, const Double4& y) noexcept { return x.Min(y); }
		static inline Double4 abs(const Double4& x) noexcept { return x.AbsoluteValue(); }
		static inline Double4 sqrt(const Double4& x) noexcept { return x.SquareRoot(); }
	#endif

	#if defined(__AVX__)
		class Float8: public detail::IArithmeticType<Float8>
		{
			friend class detail::IArithmeticType<Float8>;
		public:
			struct __attribute__((aligned(16))) AlignedArray : public std::array<float, 8> {};

			Float8() noexcept = default;
			explicit Float8(AlignedArray&& xy) noexcept
					: Float8(xy.data())
			{
			}
			explicit Float8(float* xy) noexcept
			{
				assert(detail::isAligned(xy));
				_value = _mm256_load_ps(xy);
			}
			
			void Get(AlignedArray& xy) const noexcept
			{
				_mm256_store_ps(xy.data(), _value);
			}
			
			inline Float8 SquareRoot() const noexcept
			{
				Float8 ret {};
				ret._value = _mm256_sqrt_ps(_value);
				return ret;
			}
		private:
			inline Float8& AddEqualImpl(const Float8& rhs) noexcept
			{
				_value = _mm256_add_ps(_value, rhs._value);
				return *this;
			}
	
			inline Float8& SubtractEqualImpl(const Float8& rhs) noexcept
			{
				_value = _mm256_sub_ps(_value, rhs._value);
				return *this;
			}
	
			inline Float8& MultiplyEqualImpl(const Float8& rhs) noexcept
			{
				_value = _mm256_mul_ps(_value, rhs._value);
				return *this;
			}
	
			inline Float8& DivideEqualImpl(const Float8& rhs) noexcept
			{
				_value = _mm256_div_ps(_value, rhs._value);
				return *this;
			}
	
			inline Float8 AbsoluteValueImpl() const noexcept
			{
				static const __m256 signmask = _mm256_set1_ps(-0.0f); // 0x80000000

				Float8 ret {};
				ret._value = _mm256_andnot_ps(signmask, _value);
				return ret;
			}
	
			inline Float8 MaxImpl(const Float8& rhs) const noexcept
			{
				Float8 ret {};
				ret._value = _mm256_max_ps(rhs._value, _value);
				return ret;
			}
	
			inline Float8 MinImpl(const Float8& rhs) const noexcept
			{
				Float8 ret {};
				ret._value = _mm256_min_ps(rhs._value, _value);
				return ret;
			}
	
		private:
			__m256 _value {};
		};
	
		static inline Float8 max(const Float8& x, const Float8& y) noexcept { return x.Max(y); }
		static inline Float8 min(const Float8& x, const Float8& y) noexcept { return x.Min(y); }
		static inline Float8 abs(const Float8& x) noexcept { return x.AbsoluteValue(); }
		static inline Float8 sqrt(const Float8& x) noexcept { return x.SquareRoot(); }
	#endif
}
