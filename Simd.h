#pragma once

#include "AlignedAllocator.h"
#define USE_SSE2

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wimplicit-float-conversion"
	#pragma clang diagnostic ignored "-Wold-style-cast"
	#pragma clang diagnostic ignored "-Wcast-qual"
	#include "sse_mathfun.h"
	#pragma clang diagnostic pop
#elif defined(__INTEL_COMPILER)
	#include "sse_mathfun.h"
#elif defined(__GNUC__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wold-style-cast"
	#pragma GCC diagnostic ignored "-Wcast-qual"
	#pragma GCC diagnostic ignored "-Wcast-align"
	#pragma GCC diagnostic ignored "-Wfloat-conversion"
	#pragma GCC diagnostic ignored "-Wmissing-declarations"
	#include "sse_mathfun.h"
	#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
	#pragma warning( push )
	#pragma warning( disable : 4305)
	#include "sse_mathfun.h"
	#pragma warning( pop )
#endif

#include <type_traits>
#include <array>
#include <vector>
#include <cassert>

#ifndef _MSC_VER
	#include <x86intrin.h>
	#define ALIGNAS(ALIGNMENT) __attribute__((aligned(ALIGNMENT)))
#else
	#include <intrin.h>

	#if (defined(_M_AMD64) || defined(_M_X64))
		#define __SSE2__
	#endif

	#define ALIGNAS(ALIGNMENT) __declspec(align(ALIGNMENT))
#endif

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
			inline TypeImpl& operator+=(TypeImpl&& rhs) noexcept { return this->AddEqual(std::move(rhs)); }
			inline TypeImpl& operator-=(const TypeImpl& rhs) noexcept { return this->SubtractEqual(rhs); }
			inline TypeImpl& operator-=(TypeImpl&& rhs) noexcept { return this->SubtractEqual(std::move(rhs)); }
			inline TypeImpl& operator*=(const TypeImpl& rhs) noexcept { return this->MultiplyEqual(rhs); }
			inline TypeImpl& operator*=(TypeImpl&& rhs) noexcept { return this->MultiplyEqual(std::move(rhs)); }
			inline TypeImpl& operator/=(const TypeImpl& rhs) noexcept { return this->DivideEqual(rhs); }
			inline TypeImpl& operator/=(TypeImpl&& rhs) noexcept { return this->DivideEqual(std::move(rhs)); }

			inline TypeImpl operator+(const TypeImpl& rhs) const noexcept { return this->Add(rhs); }
			inline TypeImpl operator+(TypeImpl&& rhs) const noexcept { return this->Add(std::move(rhs)); }
			inline TypeImpl operator-(const TypeImpl& rhs) const noexcept { return this->Subtract(rhs); }
			inline TypeImpl operator-(TypeImpl&& rhs) const noexcept { return this->Subtract(std::move(rhs)); }
			inline TypeImpl operator*(const TypeImpl& rhs) const noexcept { return this->Multiply(rhs); }
			inline TypeImpl operator*(TypeImpl&& rhs) const noexcept { return this->Multiply(std::move(rhs)); }
			inline TypeImpl operator/(const TypeImpl& rhs) const noexcept { return this->Divide(rhs); }
			inline TypeImpl operator/(TypeImpl&& rhs) const noexcept { return this->Divide(std::move(rhs)); }

			inline TypeImpl& AddEqual(const TypeImpl& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->AddEqualImpl(rhs);
			}
			inline TypeImpl& AddEqual(TypeImpl&& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->AddEqualImpl(std::move(rhs));
			}
			inline TypeImpl Add(const TypeImpl& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs += rhs;

				return lhs;
			}
			inline TypeImpl Add(TypeImpl&& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs += std::move(rhs);

				return lhs;
			}

			inline TypeImpl& SubtractEqual(const TypeImpl& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->SubtractEqualImpl(rhs);
			}
			inline TypeImpl& SubtractEqual(TypeImpl&& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->SubtractEqualImpl(std::move(rhs));
			}
			inline TypeImpl Subtract(const TypeImpl& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs -= rhs;

				return lhs;
			}
			inline TypeImpl Subtract(TypeImpl&& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs -= std::move(rhs);

				return lhs;
			}

			inline TypeImpl& MultiplyEqual(const TypeImpl& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->MultiplyEqualImpl(rhs);
			}
			inline TypeImpl& MultiplyEqual(TypeImpl&& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->MultiplyEqualImpl(std::move(rhs));
			}
			inline TypeImpl Multiply(const TypeImpl& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs *= rhs;

				return lhs;
			}
			inline TypeImpl Multiply(TypeImpl&& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs *= std::move(rhs);

				return lhs;
			}

			inline TypeImpl& DivideEqual(const TypeImpl& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->DivideEqualImpl(rhs);
			}
			inline TypeImpl& DivideEqual(TypeImpl&& rhs) noexcept
			{
				return static_cast<TypeImpl*>(this)->DivideEqualImpl(std::move(rhs));
			}
			inline TypeImpl Divide(const TypeImpl& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs /= rhs;

				return lhs;
			}
			inline TypeImpl Divide(TypeImpl&& rhs) const noexcept
			{
				TypeImpl lhs(*static_cast<const TypeImpl*>(this));
				lhs /= std::move(rhs);

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
			inline TypeImpl Max(TypeImpl&& rhs) const noexcept
			{
				return static_cast<const TypeImpl*>(this)->MaxImpl(std::move(rhs));
			}

			inline TypeImpl Min(const TypeImpl& rhs) const noexcept
			{
				return static_cast<const TypeImpl*>(this)->MinImpl(rhs);
			}
			inline TypeImpl Min(TypeImpl&& rhs) const noexcept
			{
				return static_cast<const TypeImpl*>(this)->MinImpl(std::move(rhs));
			}
		};

		static inline bool isAligned(const void* ptr, const size_t alignment) noexcept
		{
			auto uintPtr = reinterpret_cast<std::uintptr_t>(ptr);
			return !(uintPtr % alignment);
		}
	}

	#if defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__)
		class Double2: public detail::IArithmeticType<Double2>
		{
			friend class detail::IArithmeticType<Double2>;
			static constexpr size_t alignment = { 16 };
		public:
			inline static constexpr size_t Alignment() noexcept { return alignment; }
			using AlignedVector = std::vector<double, AlignedAllocator<double, alignment>>;
			template<size_t N = 2> struct ALIGNAS(alignment) AlignedArray : public std::array<double, N> { using std::array<double, N>::array; };

			Double2() noexcept = default;
			inline explicit Double2(const double x) noexcept
			{
				_value = _mm_set_pd1(x);
			}
			inline explicit Double2(const double x, const double y) noexcept
			{
				_value = _mm_set_pd(y, x);  // that's how the values are stored!
			}
			inline explicit Double2(AlignedArray<2>&& xy) noexcept
				: Double2(xy.data())
			{
			}
			inline explicit Double2(double* xy) noexcept
			{
				assert(detail::isAligned(xy, alignment));
				_value = _mm_load_pd(xy);
			}

			inline void Get(double& x, double& y) const noexcept
			{
				#ifndef _MSC_VER
					x = _value[0];
					y = _value[1];
				#else
					const auto* val = reinterpret_cast<const double*>(&_value);
					x = val[0];
					y = val[1];
				#endif
			}
			inline void Get(AlignedArray<2>& xy) const noexcept { Get(xy.data()); }
			inline void Get(double* xy) const noexcept
			{
				assert(detail::isAligned(xy, alignment));
				_mm_store_pd(xy, _value);
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
			inline Double2& AddEqualImpl(Double2&& rhs) noexcept { return AddEqualImpl(rhs); }

			inline Double2& SubtractEqualImpl(const Double2& rhs) noexcept
			{
				_value = _mm_sub_pd(_value, rhs._value);
				return *this;
			}
			inline Double2& SubtractEqualImpl(Double2&& rhs) noexcept { return SubtractEqualImpl(rhs); }

			inline Double2& MultiplyEqualImpl(const Double2& rhs) noexcept
			{
				_value = _mm_mul_pd(_value, rhs._value);
				return *this;
			}
			inline Double2& MultiplyEqualImpl(Double2&& rhs) noexcept { return MultiplyEqualImpl(rhs); }

			inline Double2& DivideEqualImpl(const Double2& rhs) noexcept
			{
				_value = _mm_div_pd(_value, rhs._value);
				return *this;
			}
			inline Double2& DivideEqualImpl(Double2&& rhs) noexcept { return DivideEqualImpl(rhs); }

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
			inline Double2 MaxImpl(Double2&& rhs) noexcept { return MaxImpl(rhs); }

			inline Double2 MinImpl(const Double2& rhs) const noexcept
			{
				Double2 ret {};
				ret._value = _mm_min_pd(rhs._value, _value);
				return ret;
			}
			inline Double2 MinImpl(Double2&& rhs) noexcept { return MinImpl(rhs); }

		private:
			__m128d _value {};
		};

		static inline Double2 max(const Double2& x, const Double2& y) noexcept { return x.Max(y); }
		static inline Double2 max(Double2&& x, Double2&& y) noexcept { return x.Max(y); }
		static inline Double2 min(const Double2& x, const Double2& y) noexcept { return x.Min(y); }
		static inline Double2 min(Double2&& x, Double2&& y) noexcept { return x.Min(y); }
		static inline Double2 abs(const Double2& x) noexcept { return x.AbsoluteValue(); }
		static inline Double2 abs(Double2&& x) noexcept { return x.AbsoluteValue(); }
		static inline Double2 sqrt(const Double2& x) noexcept { return x.SquareRoot(); }
		static inline Double2 sqrt(Double2&& x) noexcept { return x.SquareRoot(); }

		class Float4: public detail::IArithmeticType<Float4>
		{
			friend class detail::IArithmeticType<Float4>;
			static constexpr size_t alignment = { 16 };
		public:
			inline static constexpr size_t Alignment() noexcept { return alignment; }
			using AlignedVector = std::vector<float, AlignedAllocator<float, alignment>>;
			template<size_t N = 4> struct ALIGNAS(alignment) AlignedArray : public std::array<float, N> { using std::array<float, N>::array; };

			Float4() noexcept = default;
			inline explicit Float4(const float x) noexcept
			{
				_value = _mm_set_ps1(x);
			}
			inline explicit Float4(const float x, const float y, const float z, const float t) noexcept
			{
				_value = _mm_set_ps(t, z, y, x);
			}
			inline explicit Float4(AlignedArray<4>&& xy) noexcept
				: Float4(xy.data())
			{
			}
			inline explicit Float4(float* xy) noexcept
			{
				assert(detail::isAligned(xy, alignment));
				_value = _mm_load_ps(xy);
			}

			inline void Get(float& x, float& y, float& z, float& t) const noexcept
			{
				#ifndef _MSC_VER
					x = _value[0];
					y = _value[1];
					z = _value[2];
					t = _value[3];
				#else
					const auto* val = reinterpret_cast<const float*>(&_value);
					x = val[0];
					y = val[1];
					z = val[2];
					t = val[3];
				#endif
			}
			inline void Get(AlignedArray<4>& xy) const noexcept { Get(xy.data()); }
			inline void Get(float* xy) const noexcept
			{
				assert(detail::isAligned(xy, alignment));
				_mm_store_ps(xy, _value);
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
			inline Float4& AddEqualImpl(Float4&& rhs) noexcept { return AddEqualImpl(rhs); }

			inline Float4& SubtractEqualImpl(const Float4& rhs) noexcept
			{
				_value = _mm_sub_ps(_value, rhs._value);
				return *this;
			}
			inline Float4& SubtractEqualImpl(Float4&& rhs) noexcept { return SubtractEqualImpl(rhs); }

			inline Float4& MultiplyEqualImpl(const Float4& rhs) noexcept
			{
				_value = _mm_mul_ps(_value, rhs._value);
				return *this;
			}
			inline Float4& MultiplyEqualImpl(Float4&& rhs) noexcept { return MultiplyEqualImpl(rhs); }

			inline Float4& DivideEqualImpl(const Float4& rhs) noexcept
			{
				_value = _mm_div_ps(_value, rhs._value);
				return *this;
			}
			inline Float4& DivideEqualImpl(Float4&& rhs) noexcept { return DivideEqualImpl(rhs); }

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
			inline Float4 MaxImpl(Float4&& rhs) noexcept { return MaxImpl(rhs); }

			inline Float4 MinImpl(const Float4& rhs) const noexcept
			{
				Float4 ret {};
				ret._value = _mm_min_ps(rhs._value, _value);
				return ret;
			}
			inline Float4 MinImpl(Float4&& rhs) noexcept { return MinImpl(rhs); }

		private:
			__m128 _value {};
		};

		static inline Float4 max(const Float4& x, const Float4& y) noexcept { return x.Max(y); }
		static inline Float4 max(Float4&& x, Float4&& y) noexcept { return x.Max(y); }
		static inline Float4 min(const Float4& x, const Float4& y) noexcept { return x.Min(y); }
		static inline Float4 min(Float4&& x, Float4&& y) noexcept { return x.Min(y); }
		static inline Float4 abs(const Float4& x) noexcept { return x.AbsoluteValue(); }
		static inline Float4 abs(Float4&& x) noexcept { return x.AbsoluteValue(); }
		static inline Float4 sqrt(const Float4& x) noexcept { return x.SquareRoot(); }
		static inline Float4 sqrt(Float4&& x) noexcept { return x.SquareRoot(); }
		static inline Float4 exp(const Float4& x) noexcept { return x.Exp(); }
		static inline Float4 exp(Float4&& x) noexcept { return x.Exp(); }
		static inline Float4 log(const Float4& x) noexcept { return x.Log(); }
		static inline Float4 log(Float4&& x) noexcept { return x.Log(); }
		static inline Float4 sin(const Float4& x) noexcept { return x.Sin(); }
		static inline Float4 sin(Float4&& x) noexcept { return x.Sin(); }
		static inline Float4 cos(const Float4& x) noexcept { return x.Cos(); }
		static inline Float4 cos(Float4&& x) noexcept { return x.Cos(); }

	#endif

	#if defined(__AVX2__)
		class Double4: public detail::IArithmeticType<Double4>
		{
			friend class detail::IArithmeticType<Double4>;
			static constexpr size_t alignment = { 32 };
		public:
			inline static constexpr size_t Alignment() noexcept { return alignment; }
			using AlignedVector = std::vector<double, AlignedAllocator<float, alignment>>;
			template<size_t N = 4> struct ALIGNAS(alignment) AlignedArray : public std::array<double, N> { using std::array<double, N>::array; };

			Double4() noexcept = default;
			inline explicit Double4(const double x, const double y, const double z, const double t) noexcept
			{
				_value = _mm256_set_pd(t, z, y, x);  // that's how the values are stored!
			}
			inline explicit Double4(AlignedArray&& xy) noexcept
				: Double4(xy.data())
			{
			}
			inline explicit Double4(double* xy) noexcept
			{
				assert(detail::isAligned(xy, alignment));
				_value = _mm256_load_pd(xy);
			}

			inline void Get(double& x, double& y, double& z, double& t) const noexcept
			{
				//_mm_store1_pd(&x, _value);
				x = _value[0];
				y = _value[1];
				z = _value[2];
				t = _value[3];
			}
			inline void Get(AlignedArray& xy) const noexcept { Get(xy.data()); }
			inline void Get(double* xy) const noexcept
			{
				_mm256_store_pd(xy, _value);
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
			inline Double4& AddEqualImpl(Double4&& rhs) noexcept { return AddEqualImpl(rhs); }

			inline Double4& SubtractEqualImpl(const Double4& rhs) noexcept
			{
				_value = _mm256_sub_pd(_value, rhs._value);
				return *this;
			}
			inline Double4& SubtractEqualImpl(Double4&& rhs) noexcept { return SubtractEqualImpl(rhs); }

			inline Double4& MultiplyEqualImpl(const Double4& rhs) noexcept
			{
				_value = _mm256_mul_pd(_value, rhs._value);
				return *this;
			}
			inline Double4& MultiplyEqualImpl(Double4&& rhs) noexcept { return MultiplyEqualImpl(rhs); }

			inline Double4& DivideEqualImpl(const Double4& rhs) noexcept
			{
				_value = _mm256_div_pd(_value, rhs._value);
				return *this;
			}
			inline Double4& DivideEqualImpl(Double4&& rhs) noexcept { return DivideEqualImpl(rhs); }

			inline Double4 AbsoluteValueImpl() const noexcept
			{
				static const __m256d signmask = _mm256_set1_pd(-0.0);

				Double4 ret {};
				ret._value = _mm256_andnot_pd(signmask, _value);
				return ret;
			}

			inline Double4 MaxImpl(const Double4& rhs) const noexcept
			{
				Double4 ret {};
				ret._value = _mm256_max_pd(rhs._value, _value);
				return ret;
			}
			inline Double4 MaxImpl(Double4&& rhs) const noexcept { return MaxImpl(rhs); }

			inline Double4 MinImpl(const Double4& rhs) const noexcept
			{
				Double4 ret {};
				ret._value = _mm256_min_pd(rhs._value, _value);
				return ret;
			}
			inline Double4 MinImpl(Double4&& rhs) const noexcept { return MinImpl(rhs); }

		private:
			__m256d _value {};
		};

		static inline Double4 max(const Double4& x, const Double4& y) noexcept { return x.Max(y); }
		static inline Double4 max(Double4&& x, Double4&& y) noexcept { return x.Max(y); }
		static inline Double4 min(const Double4& x, const Double4& y) noexcept { return x.Min(y); }
		static inline Double4 min(Double4&& x, Double4&& y) noexcept { return x.Min(y); }
		static inline Double4 abs(const Double4& x) noexcept { return x.AbsoluteValue(); }
		static inline Double4 abs(Double4&& x) noexcept { return x.AbsoluteValue(); }
		static inline Double4 sqrt(const Double4& x) noexcept { return x.SquareRoot(); }
		static inline Double4 sqrt(Double4&& x) noexcept { return x.SquareRoot(); }
	#endif

	#if defined(__AVX__)
		class Float8: public detail::IArithmeticType<Float8>
		{
			friend class detail::IArithmeticType<Float8>;
			static constexpr size_t alignment = { 32 };
		public:
			inline static constexpr size_t Alignment() noexcept { return alignment; }
			using AlignedVector = std::vector<float, AlignedAllocator<float, alignment>>;
			template<size_t N = 8> struct ALIGNAS(alignment) AlignedArray : public std::array<float, N> { using std::array<float, N>::array; };

			Float8() noexcept = default;
			inline explicit Float8(AlignedArray<8>&& xy) noexcept
					: Float8(xy.data())
			{
			}
			inline explicit Float8(float* xy) noexcept
			{
				assert(detail::isAligned(xy, alignment));
				_value = _mm256_load_ps(xy);
			}
			
			inline void Get(AlignedArray<8>& xy) const noexcept { Get(xy.data()); }
			inline void Get(float* xy) const noexcept
			{
				assert(detail::isAligned(xy, alignment));
				_mm256_store_ps(xy, _value);
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
			inline Float8& AddEqualImpl(Float8&& rhs) noexcept { return AddEqualImpl(rhs); }
	
			inline Float8& SubtractEqualImpl(const Float8& rhs) noexcept
			{
				_value = _mm256_sub_ps(_value, rhs._value);
				return *this;
			}
			inline Float8& SubtractEqualImpl(Float8&& rhs) noexcept { return SubtractEqualImpl(rhs); }
	
			inline Float8& MultiplyEqualImpl(const Float8& rhs) noexcept
			{
				_value = _mm256_mul_ps(_value, rhs._value);
				return *this;
			}
			inline Float8& MultiplyEqualImpl(Float8&& rhs) noexcept { return MultiplyEqualImpl(rhs); }
	
			inline Float8& DivideEqualImpl(const Float8& rhs) noexcept
			{
				_value = _mm256_div_ps(_value, rhs._value);
				return *this;
			}
			inline Float8& DivideEqualImpl(Float8&& rhs) noexcept { return DivideEqualImpl(rhs); }
	
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
			inline Float8 MaxImpl(Float8&& rhs) const noexcept { return MaxImpl(rhs); }
	
			inline Float8 MinImpl(const Float8& rhs) const noexcept
			{
				Float8 ret {};
				ret._value = _mm256_min_ps(rhs._value, _value);
				return ret;
			}
			inline Float8 MinImpl(Float8&& rhs) const noexcept { return MinImpl(rhs); }
	
		private:
			__m256 _value {};
		};
	
		static inline Float8 max(const Float8& x, const Float8& y) noexcept { return x.Max(y); }
		static inline Float8 max(Float8&& x, Float8&& y) noexcept { return x.Max(y); }
		static inline Float8 min(const Float8& x, const Float8& y) noexcept { return x.Min(y); }
		static inline Float8 min(Float8&& x, Float8&& y) noexcept { return x.Min(y); }
		static inline Float8 abs(const Float8& x) noexcept { return x.AbsoluteValue(); }
		static inline Float8 abs(Float8&& x) noexcept { return x.AbsoluteValue(); }
		static inline Float8 sqrt(const Float8& x) noexcept { return x.SquareRoot(); }
		static inline Float8 sqrt(Float8&& x) noexcept { return x.SquareRoot(); }
	#endif
}
