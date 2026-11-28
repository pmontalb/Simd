#pragma once

#include "Config.h"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace simd
{
	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	class Pack;

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	class Mask;

	template<SupportedElement T>
	struct IndexTypeFor
	{
		using Type = std::conditional_t<sizeof(T) == 8, std::int64_t, std::int32_t>;
	};

	template<SupportedElement T>
	using IndexTypeForT = typename IndexTypeFor<T>::Type;

	namespace detail
	{
		template<typename T, std::size_t Lanes>
		struct NativeTypes
		{
			using Vec = std::array<T, Lanes>;
			using MaskVec = std::array<bool, Lanes>;
			static constexpr bool isHardware = false;
			static constexpr std::size_t alignment = alignof(T);
		};

#if SIMD_HAS_SSE2
		template<>
		struct NativeTypes<float, 4>
		{
			using Vec = __m128;
			using MaskVec = __m128;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 16;
		};

		template<>
		struct NativeTypes<double, 2>
		{
			using Vec = __m128d;
			using MaskVec = __m128d;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 16;
		};

		template<>
		struct NativeTypes<std::int32_t, 4>
		{
			using Vec = __m128i;
			using MaskVec = __m128i;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 16;
		};

		template<>
		struct NativeTypes<std::int64_t, 2>
		{
			using Vec = __m128i;
			using MaskVec = __m128i;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 16;
		};
#endif

#if SIMD_HAS_AVX
		template<>
		struct NativeTypes<float, 8>
		{
			using Vec = __m256;
			using MaskVec = __m256;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 32;
		};

		template<>
		struct NativeTypes<double, 4>
		{
			using Vec = __m256d;
			using MaskVec = __m256d;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 32;
		};
#endif

#if SIMD_HAS_AVX2
		template<>
		struct NativeTypes<std::int32_t, 8>
		{
			using Vec = __m256i;
			using MaskVec = __m256i;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 32;
		};

		template<>
		struct NativeTypes<std::int64_t, 4>
		{
			using Vec = __m256i;
			using MaskVec = __m256i;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 32;
		};
#endif

#if SIMD_HAS_AVX512F
		template<>
		struct NativeTypes<float, 16>
		{
			using Vec = __m512;
			using MaskVec = __mmask16;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 64;
		};

		template<>
		struct NativeTypes<double, 8>
		{
			using Vec = __m512d;
			using MaskVec = __mmask8;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 64;
		};

		template<>
		struct NativeTypes<std::int32_t, 16>
		{
			using Vec = __m512i;
			using MaskVec = __mmask16;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 64;
		};

		template<>
		struct NativeTypes<std::int64_t, 8>
		{
			using Vec = __m512i;
			using MaskVec = __mmask8;
			static constexpr bool isHardware = true;
			static constexpr std::size_t alignment = 64;
		};
#endif
	}	 // namespace detail

	// ========================================================================
	// Mask<T, Lanes>
	// ========================================================================

	template<SupportedElement T, std::size_t Lanes>
	class Mask
	{
	public:
		using NativeMask = typename detail::NativeTypes<T, Lanes>::MaskVec;
		static constexpr std::size_t lanes = Lanes;

		constexpr Mask() noexcept = default;

		[[nodiscard]] static SIMD_ALWAYS_INLINE Mask FromNative(const NativeMask m) noexcept
		{
			Mask res;
			res._storage.vec = m;
			return res;
		}

		[[nodiscard]] static constexpr Mask FromScalar(const std::array<bool, Lanes>& m) noexcept
		{
			Mask res;
			res._storage.scalar = m;
			return res;
		}

		[[nodiscard]] SIMD_ALWAYS_INLINE NativeMask Native() const noexcept { return _storage.vec; }
		[[nodiscard]] constexpr std::array<bool, Lanes> Scalar() const noexcept { return _storage.scalar; }

	private:
		union Storage
		{
			std::array<bool, Lanes> scalar;
			NativeMask vec;
		} _storage;

		[[nodiscard]] static SIMD_ALWAYS_INLINE NativeMask ZeroNative() noexcept
		{
			if constexpr (!detail::NativeTypes<T, Lanes>::isHardware)
				return {};
#if SIMD_HAS_AVX512F
			else if constexpr (Lanes * sizeof(T) == 64)
				return static_cast<NativeMask>(0);
#endif
#if SIMD_HAS_AVX2 || SIMD_HAS_AVX
			else if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return _mm256_setzero_ps();
				else if constexpr (std::is_same_v<T, double>)
					return _mm256_setzero_pd();
				else
					return _mm256_setzero_si256();
			}
#endif
#if SIMD_HAS_SSE2
			else if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return _mm_setzero_ps();
				else if constexpr (std::is_same_v<T, double>)
					return _mm_setzero_pd();
				else
					return _mm_setzero_si128();
			}
#endif
			else
				return {};
		}
	};

	// ========================================================================
	// Pack<T, Lanes>
	// ========================================================================

	template<SupportedElement T, std::size_t Lanes>
	class Pack
	{
	public:
		using ValueType = T;
		using NativeVec = typename detail::NativeTypes<T, Lanes>::Vec;
		static constexpr std::size_t lanes = Lanes;
		static constexpr std::size_t alignment = detail::NativeTypes<T, Lanes>::alignment;
		static constexpr bool isHardware = detail::NativeTypes<T, Lanes>::isHardware;

		struct alignas(alignment) AlignedArray : public std::array<T, Lanes>
		{
			using std::array<T, Lanes>::array;
		};

		constexpr Pack() noexcept = default;

		constexpr explicit Pack(const T value) noexcept
		{
			*this = Broadcast(value);
		}

		template<typename Arg0, typename Arg1, typename... Rest>
			requires (Lanes > 1 && (2 + sizeof...(Rest)) == Lanes && std::convertible_to<Arg0, T> && std::convertible_to<Arg1, T> && (std::convertible_to<Rest, T> && ...))
		constexpr explicit Pack(const Arg0 arg0, const Arg1 arg1, const Rest... rest) noexcept
		{
			const std::array<T, Lanes> arr { static_cast<T>(arg0), static_cast<T>(arg1), static_cast<T>(rest)... };
			*this = std::is_constant_evaluated() ? FromScalar(arr) : LoadUnaligned(arr.data());  // LCOV_EXCL_LINE
		}

		constexpr explicit Pack(const AlignedArray& arr) noexcept
		{
			*this = LoadAligned(arr.data());
		}

		constexpr explicit Pack(const T* const __restrict ptr) noexcept
		{
			*this = LoadAligned(ptr);
		}

		[[nodiscard]] static constexpr std::size_t LanesCount() noexcept { return Lanes; }
		[[nodiscard]] static constexpr std::size_t Alignment() noexcept { return alignment; }

		[[nodiscard]] static SIMD_ALWAYS_INLINE Pack FromNative(const NativeVec v) noexcept
		{
			Pack p;
			p._storage.vec = v;
			return p;
		}

		[[nodiscard]] static constexpr Pack FromScalar(const std::array<T, Lanes>& s) noexcept
		{
			Pack p;
			p._storage.scalar = s;
			return p;
		}

		[[nodiscard]] SIMD_ALWAYS_INLINE NativeVec Native() const noexcept { return _storage.vec; }

		// --------------------------------------------------------------------
		// Construction / Load / Store
		// --------------------------------------------------------------------

		[[nodiscard]] static constexpr Pack Zero() noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
				return FromScalar(std::array<T, Lanes> {});
			// LCOV_EXCL_STOP
			return FromNative(ZeroNative());
		}

		[[nodiscard]] static constexpr Pack Broadcast(const T value) noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				arr.fill(value);
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm512_set1_ps(value));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm512_set1_pd(value));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm512_set1_epi32(value));
				else
					return FromNative(_mm512_set1_epi64(value));
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm256_set1_ps(value));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm256_set1_pd(value));
#if SIMD_HAS_AVX2
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm256_set1_epi32(value));
				else
					return FromNative(_mm256_set1_epi64x(value));
#endif
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm_set1_ps(value));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm_set1_pd(value));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm_set1_epi32(value));
				else
					return FromNative(_mm_set1_epi64x(value));
			}
#endif
			std::array<T, Lanes> arr {};
			arr.fill(value);
			return FromScalar(arr);
		}

		[[nodiscard]] static constexpr Pack Iota(const T start = T { 0 }, const T step = T { 1 }) noexcept
		{
			std::array<T, Lanes> arr {};
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = static_cast<T>(start + static_cast<T>(i) * step);
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
				return FromScalar(arr);
			// LCOV_EXCL_STOP
			return LoadUnaligned(arr.data());
		}

		[[nodiscard]] static constexpr Pack LoadAligned(const T* const __restrict ptr) noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				for (std::size_t i = 0; i < Lanes; ++i)
					arr[i] = ptr[i];
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
			assert(detail::IsAligned(ptr, alignment));
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm512_load_ps(ptr));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm512_load_pd(ptr));
				else
					return FromNative(_mm512_load_si512(static_cast<const __m512i*>(static_cast<const void*>(ptr))));
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm256_load_ps(ptr));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm256_load_pd(ptr));
#if SIMD_HAS_AVX2
				else
					return FromNative(_mm256_load_si256(static_cast<const __m256i*>(static_cast<const void*>(ptr))));
#endif
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm_load_ps(ptr));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm_load_pd(ptr));
				else
					return FromNative(_mm_load_si128(static_cast<const __m128i*>(static_cast<const void*>(ptr))));
			}
#endif
			std::array<T, Lanes> arr {};
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = ptr[i];
			return FromScalar(arr);
		}

		[[nodiscard]] static constexpr Pack LoadUnaligned(const T* const __restrict ptr) noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				for (std::size_t i = 0; i < Lanes; ++i)
					arr[i] = ptr[i];
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm512_loadu_ps(ptr));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm512_loadu_pd(ptr));
				else
					return FromNative(_mm512_loadu_si512(static_cast<const __m512i*>(static_cast<const void*>(ptr))));
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm256_loadu_ps(ptr));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm256_loadu_pd(ptr));
#if SIMD_HAS_AVX2
				else
					return FromNative(_mm256_loadu_si256(static_cast<const __m256i*>(static_cast<const void*>(ptr))));
#endif
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm_loadu_ps(ptr));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm_loadu_pd(ptr));
				else
					return FromNative(_mm_loadu_si128(static_cast<const __m128i*>(static_cast<const void*>(ptr))));
			}
#endif
			std::array<T, Lanes> arr {};
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = ptr[i];
			return FromScalar(arr);
		}

		[[nodiscard]] static constexpr Pack LoadPartial(const T* const __restrict ptr, const std::size_t validLanes) noexcept
		{
			if (validLanes >= Lanes)
				return LoadUnaligned(ptr);
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				for (std::size_t i = 0; i < validLanes; ++i)
					arr[i] = ptr[i];
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				const auto mask = static_cast<typename Mask<T, Lanes>::NativeMask>((1ULL << validLanes) - 1ULL);
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm512_maskz_loadu_ps(mask, ptr));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm512_maskz_loadu_pd(mask, ptr));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm512_maskz_loadu_epi32(mask, ptr));
				else
					return FromNative(_mm512_maskz_loadu_epi64(mask, ptr));
			}
#endif
#if SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				const __m256i mask = Avx2PartialMask(validLanes);
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm256_maskload_ps(ptr, mask));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm256_maskload_pd(ptr, mask));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm256_maskload_epi32(ptr, mask));
				else
					return FromNative(_mm256_maskload_epi64(reinterpret_cast<const long long*>(ptr), mask));
			}
#endif
			std::array<T, Lanes> tmp {};
			for (std::size_t i = 0; i < validLanes; ++i)
				tmp[i] = ptr[i];
			return LoadUnaligned(tmp.data());
		}

		constexpr void StoreAligned(T* const __restrict ptr) const noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				for (std::size_t i = 0; i < Lanes; ++i)
					ptr[i] = _storage.scalar[i];
				return;
			}
			// LCOV_EXCL_STOP
			assert(detail::IsAligned(ptr, alignment));
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					_mm512_store_ps(ptr, _storage.vec);
				else if constexpr (std::is_same_v<T, double>)
					_mm512_store_pd(ptr, _storage.vec);
				else
					_mm512_store_si512(reinterpret_cast<__m512i*>(ptr), _storage.vec);
				return;
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					_mm256_store_ps(ptr, _storage.vec);
				else if constexpr (std::is_same_v<T, double>)
					_mm256_store_pd(ptr, _storage.vec);
#if SIMD_HAS_AVX2
				else
					_mm256_store_si256(reinterpret_cast<__m256i*>(ptr), _storage.vec);
#endif
				return;
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					_mm_store_ps(ptr, _storage.vec);
				else if constexpr (std::is_same_v<T, double>)
					_mm_store_pd(ptr, _storage.vec);
				else
					_mm_store_si128(reinterpret_cast<__m128i*>(ptr), _storage.vec);
				return;
			}
#endif
			for (std::size_t i = 0; i < Lanes; ++i)
				ptr[i] = _storage.scalar[i];
		}

		constexpr void StoreUnaligned(T* const __restrict ptr) const noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				for (std::size_t i = 0; i < Lanes; ++i)
					ptr[i] = _storage.scalar[i];
				return;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					_mm512_storeu_ps(ptr, _storage.vec);
				else if constexpr (std::is_same_v<T, double>)
					_mm512_storeu_pd(ptr, _storage.vec);
				else
					_mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), _storage.vec);
				return;
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					_mm256_storeu_ps(ptr, _storage.vec);
				else if constexpr (std::is_same_v<T, double>)
					_mm256_storeu_pd(ptr, _storage.vec);
#if SIMD_HAS_AVX2
				else
					_mm256_storeu_si256(static_cast<__m256i*>(static_cast<void*>(ptr)), _storage.vec);
#endif
				return;
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					_mm_storeu_ps(ptr, _storage.vec);
				else if constexpr (std::is_same_v<T, double>)
					_mm_storeu_pd(ptr, _storage.vec);
				else
					_mm_storeu_si128(static_cast<__m128i*>(static_cast<void*>(ptr)), _storage.vec);
				return;
			}
#endif
			for (std::size_t i = 0; i < Lanes; ++i)
				ptr[i] = _storage.scalar[i];
		}

		constexpr void StorePartial(T* const __restrict ptr, const std::size_t validLanes) const noexcept
		{
			if (validLanes >= Lanes)
			{
				StoreUnaligned(ptr);
				return;
			}
			if (validLanes == 0)
				return;
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				for (std::size_t i = 0; i < validLanes; ++i)
					ptr[i] = _storage.scalar[i];
				return;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				const auto mask = static_cast<typename Mask<T, Lanes>::NativeMask>((1ULL << validLanes) - 1ULL);
				if constexpr (std::is_same_v<T, float>)
					_mm512_mask_storeu_ps(ptr, mask, _storage.vec);
				else if constexpr (std::is_same_v<T, double>)
					_mm512_mask_storeu_pd(ptr, mask, _storage.vec);
				else if constexpr (std::is_same_v<T, std::int32_t>)
					_mm512_mask_storeu_epi32(ptr, mask, _storage.vec);
				else
					_mm512_mask_storeu_epi64(ptr, mask, _storage.vec);
				return;
			}
#endif
#if SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				const __m256i mask = Avx2PartialMask(validLanes);
				if constexpr (std::is_same_v<T, float>)
					_mm256_maskstore_ps(ptr, mask, _storage.vec);
				else if constexpr (std::is_same_v<T, double>)
					_mm256_maskstore_pd(ptr, mask, _storage.vec);
				else if constexpr (std::is_same_v<T, std::int32_t>)
					_mm256_maskstore_epi32(ptr, mask, _storage.vec);
				else
					_mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr), mask, _storage.vec);
				return;
			}
#endif
			std::array<T, Lanes> tmp {};
			StoreUnaligned(tmp.data());
			for (std::size_t i = 0; i < validLanes; ++i)
				ptr[i] = tmp[i];
		}

		// --------------------------------------------------------------------
		// Element Access
		// --------------------------------------------------------------------

		constexpr void ToArray(std::array<T, Lanes>& out) const noexcept
		{
			StoreUnaligned(out.data());
		}

		constexpr void Get(AlignedArray& out) const noexcept
		{
			StoreAligned(out.data());
		}

		constexpr void Get(T* const __restrict ptr) const noexcept
		{
			StoreAligned(ptr);
		}

		template<typename... Args>
			requires (Lanes > 1 && sizeof...(Args) == Lanes && (std::same_as<Args, T> && ...))
		constexpr void Get(Args&... out) const noexcept
		{
			const auto arr = ToArray();
			std::size_t idx = 0;
			((out = arr[idx++]), ...);
		}

		[[nodiscard]] constexpr std::array<T, Lanes> ToArray() const noexcept
		{
			std::array<T, Lanes> out {};
			StoreUnaligned(out.data());
			return out;
		}

		[[nodiscard]] constexpr T operator[](const std::size_t index) const noexcept
		{
			assert(index < Lanes);
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
				return _storage.scalar[index];
			// LCOV_EXCL_STOP
			return ToArray()[index];
		}

		// --------------------------------------------------------------------
		// Arithmetic Operators (Pass-by-Value)
		// --------------------------------------------------------------------

		[[nodiscard]] friend constexpr Pack operator-(const Pack a) noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				Pack res;
				for (std::size_t i = 0; i < Lanes; ++i)
					res._storage.scalar[i] = -a._storage.scalar[i];
				return res;
			}
			// LCOV_EXCL_STOP
			return Zero() - a;
		}

		[[nodiscard]] friend constexpr Pack operator+(const Pack a, const Pack b) noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				Pack res;
				for (std::size_t i = 0; i < Lanes; ++i)
					res._storage.scalar[i] = a._storage.scalar[i] + b._storage.scalar[i];
				return res;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm512_add_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm512_add_pd(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm512_add_epi32(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm512_add_epi64(a._storage.vec, b._storage.vec));
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm256_add_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm256_add_pd(a._storage.vec, b._storage.vec));
#if SIMD_HAS_AVX2
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm256_add_epi32(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm256_add_epi64(a._storage.vec, b._storage.vec));
#endif
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm_add_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm_add_pd(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm_add_epi32(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm_add_epi64(a._storage.vec, b._storage.vec));
			}
#endif
			return a;
		}

		[[nodiscard]] friend constexpr Pack operator-(const Pack a, const Pack b) noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				Pack res;
				for (std::size_t i = 0; i < Lanes; ++i)
					res._storage.scalar[i] = a._storage.scalar[i] - b._storage.scalar[i];
				return res;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm512_sub_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm512_sub_pd(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm512_sub_epi32(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm512_sub_epi64(a._storage.vec, b._storage.vec));
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm256_sub_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm256_sub_pd(a._storage.vec, b._storage.vec));
#if SIMD_HAS_AVX2
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm256_sub_epi32(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm256_sub_epi64(a._storage.vec, b._storage.vec));
#endif
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm_sub_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm_sub_pd(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm_sub_epi32(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm_sub_epi64(a._storage.vec, b._storage.vec));
			}
#endif
			return a;
		}

		[[nodiscard]] friend constexpr Pack operator*(const Pack a, const Pack b) noexcept
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				Pack res;
				for (std::size_t i = 0; i < Lanes; ++i)
					res._storage.scalar[i] = a._storage.scalar[i] * b._storage.scalar[i];
				return res;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm512_mul_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm512_mul_pd(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm512_mullo_epi32(a._storage.vec, b._storage.vec));
#if SIMD_HAS_AVX512DQ
				else
					return FromNative(_mm512_mullo_epi64(a._storage.vec, b._storage.vec));
#endif
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm256_mul_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm256_mul_pd(a._storage.vec, b._storage.vec));
#if SIMD_HAS_AVX2
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm256_mullo_epi32(a._storage.vec, b._storage.vec));
#endif
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm_mul_ps(a._storage.vec, b._storage.vec));
				else if constexpr (std::is_same_v<T, double>)
					return FromNative(_mm_mul_pd(a._storage.vec, b._storage.vec));
#if SIMD_HAS_SSE4_1
				else if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm_mullo_epi32(a._storage.vec, b._storage.vec));
#endif
			}
#endif
			if constexpr (std::is_integral_v<T>)
			{
				auto lhs = a.ToArray();
				auto rhs = b.ToArray();
				for (std::size_t i = 0; i < Lanes; ++i)
					lhs[i] *= rhs[i];
				return LoadUnaligned(lhs.data());
			}
			else
				return a;
		}

		[[nodiscard]] friend constexpr Pack operator/(const Pack a, const Pack b) noexcept
			requires std::floating_point<T>
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				Pack res;
				for (std::size_t i = 0; i < Lanes; ++i)
					res._storage.scalar[i] = a._storage.scalar[i] / b._storage.scalar[i];
				return res;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm512_div_ps(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm512_div_pd(a._storage.vec, b._storage.vec));
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm256_div_ps(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm256_div_pd(a._storage.vec, b._storage.vec));
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return FromNative(_mm_div_ps(a._storage.vec, b._storage.vec));
				else
					return FromNative(_mm_div_pd(a._storage.vec, b._storage.vec));
			}
#endif
			return a;
		}

		[[nodiscard]] friend constexpr Pack operator+(const Pack a, const T s) noexcept { return a + Broadcast(s); }
		[[nodiscard]] friend constexpr Pack operator+(const T s, const Pack a) noexcept { return Broadcast(s) + a; }
		[[nodiscard]] friend constexpr Pack operator-(const Pack a, const T s) noexcept { return a - Broadcast(s); }
		[[nodiscard]] friend constexpr Pack operator-(const T s, const Pack a) noexcept { return Broadcast(s) - a; }
		[[nodiscard]] friend constexpr Pack operator*(const Pack a, const T s) noexcept { return a * Broadcast(s); }
		[[nodiscard]] friend constexpr Pack operator*(const T s, const Pack a) noexcept { return Broadcast(s) * a; }
		[[nodiscard]] friend constexpr Pack operator/(const Pack a, const T s) noexcept requires std::floating_point<T> { return a / Broadcast(s); }
		[[nodiscard]] friend constexpr Pack operator/(const T s, const Pack a) noexcept requires std::floating_point<T> { return Broadcast(s) / a; }

		constexpr Pack& operator+=(const Pack rhs) noexcept { *this = *this + rhs; return *this; }
		constexpr Pack& operator-=(const Pack rhs) noexcept { *this = *this - rhs; return *this; }
		constexpr Pack& operator*=(const Pack rhs) noexcept { *this = *this * rhs; return *this; }
		constexpr Pack& operator/=(const Pack rhs) noexcept requires std::floating_point<T> { *this = *this / rhs; return *this; }

		constexpr Pack& operator+=(const T s) noexcept { return (*this += Broadcast(s)); }
		constexpr Pack& operator-=(const T s) noexcept { return (*this -= Broadcast(s)); }
		constexpr Pack& operator*=(const T s) noexcept { return (*this *= Broadcast(s)); }
		constexpr Pack& operator/=(const T s) noexcept requires std::floating_point<T> { return (*this /= Broadcast(s)); }

		[[nodiscard]] friend constexpr Pack operator&(const Pack a, const Pack b) noexcept
			requires std::integral<T>
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				for (std::size_t i = 0; i < Lanes; ++i)
					arr[i] = a._storage.scalar[i] & b._storage.scalar[i];
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
				return FromNative(_mm512_and_si512(a._storage.vec, b._storage.vec));
#endif
#if SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
				return FromNative(_mm256_and_si256(a._storage.vec, b._storage.vec));
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
				return FromNative(_mm_and_si128(a._storage.vec, b._storage.vec));
#endif
			std::array<T, Lanes> arr {};
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = a._storage.scalar[i] & b._storage.scalar[i];
			return FromScalar(arr);
		}

		[[nodiscard]] friend constexpr Pack operator|(const Pack a, const Pack b) noexcept
			requires std::integral<T>
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				for (std::size_t i = 0; i < Lanes; ++i)
					arr[i] = a._storage.scalar[i] | b._storage.scalar[i];
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
				return FromNative(_mm512_or_si512(a._storage.vec, b._storage.vec));
#endif
#if SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
				return FromNative(_mm256_or_si256(a._storage.vec, b._storage.vec));
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
				return FromNative(_mm_or_si128(a._storage.vec, b._storage.vec));
#endif
			std::array<T, Lanes> arr {};
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = a._storage.scalar[i] | b._storage.scalar[i];
			return FromScalar(arr);
		}

		[[nodiscard]] friend constexpr Pack operator^(const Pack a, const Pack b) noexcept
			requires std::integral<T>
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				for (std::size_t i = 0; i < Lanes; ++i)
					arr[i] = a._storage.scalar[i] ^ b._storage.scalar[i];
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
				return FromNative(_mm512_xor_si512(a._storage.vec, b._storage.vec));
#endif
#if SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
				return FromNative(_mm256_xor_si256(a._storage.vec, b._storage.vec));
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
				return FromNative(_mm_xor_si128(a._storage.vec, b._storage.vec));
#endif
			std::array<T, Lanes> arr {};
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = a._storage.scalar[i] ^ b._storage.scalar[i];
			return FromScalar(arr);
		}

		[[nodiscard]] friend constexpr Pack operator<<(const Pack a, const int shift) noexcept
			requires std::integral<T>
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				for (std::size_t i = 0; i < Lanes; ++i)
					arr[i] = static_cast<T>(static_cast<std::make_unsigned_t<T>>(a._storage.scalar[i]) << shift);
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm512_slli_epi32(a._storage.vec, static_cast<unsigned int>(shift)));
				else
					return FromNative(_mm512_slli_epi64(a._storage.vec, static_cast<unsigned int>(shift)));
			}
#endif
#if SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm256_slli_epi32(a._storage.vec, shift));
				else
					return FromNative(_mm256_slli_epi64(a._storage.vec, shift));
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm_slli_epi32(a._storage.vec, shift));
				else
					return FromNative(_mm_slli_epi64(a._storage.vec, shift));
			}
#endif
			std::array<T, Lanes> arr {};
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = static_cast<T>(static_cast<std::make_unsigned_t<T>>(a._storage.scalar[i]) << shift);
			return FromScalar(arr);
		}

		[[nodiscard]] friend constexpr Pack operator>>(const Pack a, const int shift) noexcept
			requires std::integral<T>
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated() || !isHardware)
			{
				std::array<T, Lanes> arr {};
				for (std::size_t i = 0; i < Lanes; ++i)
					arr[i] = static_cast<T>(static_cast<std::make_unsigned_t<T>>(a._storage.scalar[i]) >> shift);
				return FromScalar(arr);
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
			if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm512_srli_epi32(a._storage.vec, static_cast<unsigned int>(shift)));
				else
					return FromNative(_mm512_srli_epi64(a._storage.vec, static_cast<unsigned int>(shift)));
			}
#endif
#if SIMD_HAS_AVX2
			if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm256_srli_epi32(a._storage.vec, shift));
				else
					return FromNative(_mm256_srli_epi64(a._storage.vec, shift));
			}
#endif
#if SIMD_HAS_SSE2
			if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, std::int32_t>)
					return FromNative(_mm_srli_epi32(a._storage.vec, shift));
				else
					return FromNative(_mm_srli_epi64(a._storage.vec, shift));
			}
#endif
			std::array<T, Lanes> arr {};
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = static_cast<T>(static_cast<std::make_unsigned_t<T>>(a._storage.scalar[i]) >> shift);
			return FromScalar(arr);
		}

	private:
		union Storage
		{
			std::array<T, Lanes> scalar;
			NativeVec vec;
		} _storage;

		[[nodiscard]] static SIMD_ALWAYS_INLINE NativeVec ZeroNative() noexcept
		{
			if constexpr (!isHardware)
				return {};
#if SIMD_HAS_AVX512F
			else if constexpr (Lanes * sizeof(T) == 64)
			{
				if constexpr (std::is_same_v<T, float>)
					return _mm512_setzero_ps();
				else if constexpr (std::is_same_v<T, double>)
					return _mm512_setzero_pd();
				else
					return _mm512_setzero_si512();
			}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
			else if constexpr (Lanes * sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, float>)
					return _mm256_setzero_ps();
				else if constexpr (std::is_same_v<T, double>)
					return _mm256_setzero_pd();
#if SIMD_HAS_AVX2
				else
					return _mm256_setzero_si256();
#endif
			}
#endif
#if SIMD_HAS_SSE2
			else if constexpr (Lanes * sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, float>)
					return _mm_setzero_ps();
				else if constexpr (std::is_same_v<T, double>)
					return _mm_setzero_pd();
				else
					return _mm_setzero_si128();
			}
#endif
			else
				return {};
		}

#if SIMD_HAS_AVX2
		[[nodiscard]] static SIMD_ALWAYS_INLINE __m256i Avx2PartialMask(const std::size_t validLanes) noexcept
		{
			if constexpr (sizeof(T) == 4)
			{
				alignas(32) static constexpr std::int32_t table[9][8] = {
					{  0,  0,  0,  0,  0,  0,  0,  0 },
					{ -1,  0,  0,  0,  0,  0,  0,  0 },
					{ -1, -1,  0,  0,  0,  0,  0,  0 },
					{ -1, -1, -1,  0,  0,  0,  0,  0 },
					{ -1, -1, -1, -1,  0,  0,  0,  0 },
					{ -1, -1, -1, -1, -1,  0,  0,  0 },
					{ -1, -1, -1, -1, -1, -1,  0,  0 },
					{ -1, -1, -1, -1, -1, -1, -1,  0 },
					{ -1, -1, -1, -1, -1, -1, -1, -1 }
				};
				return _mm256_load_si256(static_cast<const __m256i*>(static_cast<const void*>(table[validLanes < 8 ? validLanes : 8])));
			}
			else
			{
				alignas(32) static constexpr std::int64_t table[5][4] = {
					{  0,  0,  0,  0 },
					{ -1,  0,  0,  0 },
					{ -1, -1,  0,  0 },
					{ -1, -1, -1,  0 },
					{ -1, -1, -1, -1 }
				};
				return _mm256_load_si256(static_cast<const __m256i*>(static_cast<const void*>(table[validLanes < 4 ? validLanes : 4])));
			}
		}
#endif
	};

	// ========================================================================
	// Free-Function Construction / Load / Store Helpers
	// ========================================================================

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	[[nodiscard]] constexpr Pack<T, Lanes> Zero() noexcept
	{
		return Pack<T, Lanes>::Zero();
	}

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	[[nodiscard]] constexpr Pack<T, Lanes> Broadcast(const T value) noexcept
	{
		return Pack<T, Lanes>::Broadcast(value);
	}

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	[[nodiscard]] constexpr Pack<T, Lanes> Iota(const T start = T { 0 }, const T step = T { 1 }) noexcept
	{
		return Pack<T, Lanes>::Iota(start, step);
	}

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	[[nodiscard]] constexpr Pack<T, Lanes> LoadAligned(const T* const ptr) noexcept
	{
		return Pack<T, Lanes>::LoadAligned(ptr);
	}

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	[[nodiscard]] constexpr Pack<T, Lanes> LoadUnaligned(const T* const ptr) noexcept
	{
		return Pack<T, Lanes>::LoadUnaligned(ptr);
	}

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	[[nodiscard]] constexpr Pack<T, Lanes> LoadPartial(const T* const ptr, const std::size_t validLanes) noexcept
	{
		return Pack<T, Lanes>::LoadPartial(ptr, validLanes);
	}

	template<SupportedElement T, std::size_t Lanes>
	constexpr void StoreAligned(T* const ptr, const Pack<T, Lanes> p) noexcept
	{
		p.StoreAligned(ptr);
	}

	template<SupportedElement T, std::size_t Lanes>
	constexpr void StoreUnaligned(T* const ptr, const Pack<T, Lanes> p) noexcept
	{
		p.StoreUnaligned(ptr);
	}

	template<SupportedElement T, std::size_t Lanes>
	SIMD_ALWAYS_INLINE constexpr void StorePartial(T* const ptr, const Pack<T, Lanes> p, const std::size_t validLanes) noexcept
	{
		p.StorePartial(ptr, validLanes);
	}

	namespace detail
	{
		template<SupportedElement T, std::size_t Lanes>
		struct StridedIndexTraits
		{
			static constexpr std::size_t maxGatherInc = static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)() / static_cast<std::int32_t>(Lanes));
		};

#if SIMD_HAS_AVX512F
		[[nodiscard]] SIMD_ALWAYS_INLINE __m512i MakeGatherIndices32x16(const std::size_t inc) noexcept
		{
			const auto step = static_cast<std::int32_t>(inc);
			return _mm512_set_epi32(15 * step, 14 * step, 13 * step, 12 * step, 11 * step, 10 * step, 9 * step, 8 * step, 7 * step, 6 * step, 5 * step, 4 * step, 3 * step, 2 * step, step, 0);
		}
#endif

#if SIMD_HAS_AVX512F || SIMD_HAS_AVX2
		[[nodiscard]] SIMD_ALWAYS_INLINE __m256i MakeGatherIndices32x8(const std::size_t inc) noexcept
		{
			const auto step = static_cast<std::int32_t>(inc);
			return _mm256_set_epi32(7 * step, 6 * step, 5 * step, 4 * step, 3 * step, 2 * step, step, 0);
		}

		[[nodiscard]] SIMD_ALWAYS_INLINE __m128i MakeGatherIndices32x4(const std::size_t inc) noexcept
		{
			const auto step = static_cast<std::int32_t>(inc);
			return _mm_set_epi32(3 * step, 2 * step, step, 0);
		}
#endif
	}	 // namespace detail

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr Pack<T, Lanes> LoadStrided(const T* const __restrict ptr, const std::size_t inc) noexcept
	{
		if constexpr (Lanes == 1)
		{
			typename Pack<T, Lanes>::AlignedArray arr {};
			arr[0] = ptr[0];
			return Pack<T, Lanes>(arr);
		}
		else
		{
			if (!std::is_constant_evaluated())
			{
#if SIMD_HAS_AVX512F
				if (inc <= detail::StridedIndexTraits<T, Lanes>::maxGatherInc)
				{
					if constexpr (std::same_as<T, float> && Lanes == 16)
						return Pack<float, 16>::FromNative(_mm512_i32gather_ps(detail::MakeGatherIndices32x16(inc), ptr, static_cast<int>(sizeof(float))));
					else if constexpr (std::same_as<T, double> && Lanes == 8)
						return Pack<double, 8>::FromNative(_mm512_i32gather_pd(detail::MakeGatherIndices32x8(inc), ptr, static_cast<int>(sizeof(double))));
					else if constexpr (std::same_as<T, float> && Lanes == 8)
						return Pack<float, 8>::FromNative(_mm256_i32gather_ps(ptr, detail::MakeGatherIndices32x8(inc), static_cast<int>(sizeof(float))));
					else if constexpr (std::same_as<T, double> && Lanes == 4)
						return Pack<double, 4>::FromNative(_mm256_i32gather_pd(ptr, detail::MakeGatherIndices32x4(inc), static_cast<int>(sizeof(double))));
				}
#elif SIMD_HAS_AVX2
				if (inc <= detail::StridedIndexTraits<T, Lanes>::maxGatherInc)
				{
					if constexpr (std::same_as<T, float> && Lanes == 8)
						return Pack<float, 8>::FromNative(_mm256_i32gather_ps(ptr, detail::MakeGatherIndices32x8(inc), static_cast<int>(sizeof(float))));
					else if constexpr (std::same_as<T, double> && Lanes == 4)
						return Pack<double, 4>::FromNative(_mm256_i32gather_pd(ptr, detail::MakeGatherIndices32x4(inc), static_cast<int>(sizeof(double))));
				}
#endif
			}
			typename Pack<T, Lanes>::AlignedArray arr {};
			for (std::size_t k = 0; k < Lanes; ++k)
				arr[k] = ptr[k * inc];
			return Pack<T, Lanes>(arr);
		}
	}

	template<SupportedElement T, std::size_t Lanes>
	SIMD_ALWAYS_INLINE constexpr void StoreStrided(T* const __restrict ptr, const std::size_t inc, const Pack<T, Lanes> p) noexcept
	{
		if constexpr (Lanes == 1)
		{
			ptr[0] = p.ToArray()[0];
		}
		else
		{
			typename Pack<T, Lanes>::AlignedArray arr {};
			StoreAligned(arr.data(), p);
			for (std::size_t k = 0; k < Lanes; ++k)
				ptr[k * inc] = arr[k];
		}
	}

	template<SupportedElement T, std::size_t Lanes = LaneCount<T>()>
	[[nodiscard]] SIMD_ALWAYS_INLINE constexpr Pack<T, Lanes> Gather(const T* const __restrict base, const int* const __restrict indices) noexcept
	{
		if constexpr (Lanes == 1)
		{
			typename Pack<T, Lanes>::AlignedArray arr {};
			arr[0] = base[indices[0]];
			return Pack<T, Lanes>(arr);
		}
		else
		{
			if (!std::is_constant_evaluated())
			{
#if SIMD_HAS_AVX512F
				if constexpr (std::same_as<T, float> && Lanes == 16)
				{
					// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
					const __m512i vIdx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices));
					return Pack<float, 16>::FromNative(_mm512_i32gather_ps(vIdx, base, static_cast<int>(sizeof(float))));
				}
				else if constexpr (std::same_as<T, double> && Lanes == 8)
				{
					// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
					const __m256i vIdx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
					return Pack<double, 8>::FromNative(_mm512_i32gather_pd(vIdx, base, static_cast<int>(sizeof(double))));
				}
				else if constexpr (std::same_as<T, float> && Lanes == 8)
				{
					// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
					const __m256i vIdx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
					return Pack<float, 8>::FromNative(_mm256_i32gather_ps(base, vIdx, static_cast<int>(sizeof(float))));
				}
				else if constexpr (std::same_as<T, double> && Lanes == 4)
				{
					// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
					const __m128i vIdx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
					return Pack<double, 4>::FromNative(_mm256_i32gather_pd(base, vIdx, static_cast<int>(sizeof(double))));
				}
#elif SIMD_HAS_AVX2
				if constexpr (std::same_as<T, float> && Lanes == 8)
				{
					// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
					const __m256i vIdx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
					return Pack<float, 8>::FromNative(_mm256_i32gather_ps(base, vIdx, static_cast<int>(sizeof(float))));
				}
				else if constexpr (std::same_as<T, double> && Lanes == 4)
				{
					// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
					const __m128i vIdx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
					return Pack<double, 4>::FromNative(_mm256_i32gather_pd(base, vIdx, static_cast<int>(sizeof(double))));
				}
#endif
			}
			// LCOV_EXCL_START
			typename Pack<T, Lanes>::AlignedArray arr {};
			for (std::size_t k = 0; k < Lanes; ++k)
				arr[k] = base[indices[k]];
			return Pack<T, Lanes>(arr);
			// LCOV_EXCL_STOP
		}
	}

	template<SupportedElement T, std::size_t Lanes>
	SIMD_ALWAYS_INLINE constexpr void Scatter(T* const __restrict base, const int* const __restrict indices, const Pack<T, Lanes> p) noexcept
	{
		if constexpr (Lanes == 1)
		{
			base[indices[0]] = p.ToArray()[0];
		}
		else
		{
			if (!std::is_constant_evaluated())
			{
#if SIMD_HAS_AVX512F
				if constexpr (std::same_as<T, float> && Lanes == 16)
				{
					// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
					const __m512i vIdx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices));
					_mm512_i32scatter_ps(base, vIdx, p.Native(), static_cast<int>(sizeof(float)));
					return;
				}
				else if constexpr (std::same_as<T, double> && Lanes == 8)
				{
					// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
					const __m256i vIdx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
					_mm512_i32scatter_pd(base, vIdx, p.Native(), static_cast<int>(sizeof(double)));
					return;
				}
#endif
			}
			typename Pack<T, Lanes>::AlignedArray arr {};
			StoreAligned(arr.data(), p);
			for (std::size_t k = 0; k < Lanes; ++k)
				base[indices[k]] = arr[k];
		}
	}

	// ========================================================================
	// FMA, Elementwise Math & Min/Max
	// ========================================================================

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fma(const Pack<T, Lanes> a, const Pack<T, Lanes> b, const Pack<T, Lanes> c) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto sa = a.ToArray();
			const auto sb = b.ToArray();
			const auto sc = c.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				sa[i] = sa[i] * sb[i] + sc[i];
			return Pack<T, Lanes>::LoadUnaligned(sa.data());
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_FMA
	#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm512_fmadd_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm512_fmadd_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
	#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm256_fmadd_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm256_fmadd_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
	#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm_fmadd_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm_fmadd_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
#endif
		return a * b + c;
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fms(const Pack<T, Lanes> a, const Pack<T, Lanes> b, const Pack<T, Lanes> c) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto sa = a.ToArray();
			const auto sb = b.ToArray();
			const auto sc = c.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				sa[i] = sa[i] * sb[i] - sc[i];
			return Pack<T, Lanes>::LoadUnaligned(sa.data());
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_FMA
	#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm512_fmsub_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm512_fmsub_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
	#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm256_fmsub_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm256_fmsub_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
	#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm_fmsub_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm_fmsub_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
#endif
		return a * b - c;
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fnma(const Pack<T, Lanes> a, const Pack<T, Lanes> b, const Pack<T, Lanes> c) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto sa = a.ToArray();
			const auto sb = b.ToArray();
			const auto sc = c.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				sa[i] = sc[i] - sa[i] * sb[i];
			return Pack<T, Lanes>::LoadUnaligned(sa.data());
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_FMA
	#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm512_fnmadd_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm512_fnmadd_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
	#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm256_fnmadd_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm256_fnmadd_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
	#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm_fnmadd_ps(a.Native(), b.Native(), c.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm_fnmadd_pd(a.Native(), b.Native(), c.Native()));
		}
	#endif
#endif
		return c - a * b;
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Reciprocal(const Pack<T, Lanes> a) noexcept
	{
		return T { 1 } / a;
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fma(const Pack<T, Lanes> a, const T b, const Pack<T, Lanes> c) noexcept
	{
		return Fma(a, Broadcast<T, Lanes>(b), c);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fma(const T a, const Pack<T, Lanes> b, const Pack<T, Lanes> c) noexcept
	{
		return Fma(Broadcast<T, Lanes>(a), b, c);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fms(const Pack<T, Lanes> a, const T b, const Pack<T, Lanes> c) noexcept
	{
		return Fms(a, Broadcast<T, Lanes>(b), c);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fms(const T a, const Pack<T, Lanes> b, const Pack<T, Lanes> c) noexcept
	{
		return Fms(Broadcast<T, Lanes>(a), b, c);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fnma(const Pack<T, Lanes> a, const T b, const Pack<T, Lanes> c) noexcept
	{
		return Fnma(a, Broadcast<T, Lanes>(b), c);
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Fnma(const T a, const Pack<T, Lanes> b, const Pack<T, Lanes> c) noexcept
	{
		return Fnma(Broadcast<T, Lanes>(a), b, c);
	}

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Abs(const Pack<T, Lanes> a) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto sa = a.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				sa[i] = sa[i] < T { 0 } ? static_cast<T>(-sa[i]) : sa[i];
			return Pack<T, Lanes>::LoadUnaligned(sa.data());
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm512_abs_ps(a.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm512_abs_pd(a.Native()));
			else if constexpr (std::is_same_v<T, std::int32_t>)
				return Pack<T, Lanes>::FromNative(_mm512_abs_epi32(a.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm512_abs_epi64(a.Native()));
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm256_andnot_pd(_mm256_set1_pd(-0.0), a.Native()));
#if SIMD_HAS_AVX2
			else if constexpr (std::is_same_v<T, std::int32_t>)
				return Pack<T, Lanes>::FromNative(_mm256_abs_epi32(a.Native()));
#endif
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm_andnot_ps(_mm_set1_ps(-0.0f), a.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm_andnot_pd(_mm_set1_pd(-0.0), a.Native()));
		}
#endif
		auto sa = a.ToArray();
		for (std::size_t i = 0; i < Lanes; ++i)
			sa[i] = sa[i] < T { 0 } ? static_cast<T>(-sa[i]) : sa[i];
		return Pack<T, Lanes>::LoadUnaligned(sa.data());
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] SIMD_ALWAYS_INLINE Pack<T, Lanes> Sqrt(const Pack<T, Lanes> a) noexcept
	{
		if constexpr (!Pack<T, Lanes>::isHardware)
		{
			auto sa = a.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				sa[i] = std::sqrt(sa[i]);
			return Pack<T, Lanes>::LoadUnaligned(sa.data());
		}
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm512_sqrt_ps(a.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm512_sqrt_pd(a.Native()));
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm256_sqrt_ps(a.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm256_sqrt_pd(a.Native()));
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm_sqrt_ps(a.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm_sqrt_pd(a.Native()));
		}
#endif
		return a;
	}

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Min(const Pack<T, Lanes> a, const Pack<T, Lanes> b) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto sa = a.ToArray();
			const auto sb = b.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				sa[i] = sa[i] < sb[i] ? sa[i] : sb[i];
			return Pack<T, Lanes>::LoadUnaligned(sa.data());
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm512_min_ps(a.Native(), b.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm512_min_pd(a.Native(), b.Native()));
			else if constexpr (std::is_same_v<T, std::int32_t>)
				return Pack<T, Lanes>::FromNative(_mm512_min_epi32(a.Native(), b.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm512_min_epi64(a.Native(), b.Native()));
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm256_min_ps(a.Native(), b.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm256_min_pd(a.Native(), b.Native()));
#if SIMD_HAS_AVX2
			else if constexpr (std::is_same_v<T, std::int32_t>)
				return Pack<T, Lanes>::FromNative(_mm256_min_epi32(a.Native(), b.Native()));
#endif
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm_min_ps(a.Native(), b.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm_min_pd(a.Native(), b.Native()));
#if SIMD_HAS_SSE4_1
			else if constexpr (std::is_same_v<T, std::int32_t>)
				return Pack<T, Lanes>::FromNative(_mm_min_epi32(a.Native(), b.Native()));
#endif
		}
#endif
		auto sa = a.ToArray();
		const auto sb = b.ToArray();
		for (std::size_t i = 0; i < Lanes; ++i)
			sa[i] = sa[i] < sb[i] ? sa[i] : sb[i];
		return Pack<T, Lanes>::LoadUnaligned(sa.data());
	}

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> Max(const Pack<T, Lanes> a, const Pack<T, Lanes> b) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto sa = a.ToArray();
			const auto sb = b.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				sa[i] = sa[i] > sb[i] ? sa[i] : sb[i];
			return Pack<T, Lanes>::LoadUnaligned(sa.data());
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm512_max_ps(a.Native(), b.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm512_max_pd(a.Native(), b.Native()));
			else if constexpr (std::is_same_v<T, std::int32_t>)
				return Pack<T, Lanes>::FromNative(_mm512_max_epi32(a.Native(), b.Native()));
			else
				return Pack<T, Lanes>::FromNative(_mm512_max_epi64(a.Native(), b.Native()));
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm256_max_ps(a.Native(), b.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm256_max_pd(a.Native(), b.Native()));
#if SIMD_HAS_AVX2
			else if constexpr (std::is_same_v<T, std::int32_t>)
				return Pack<T, Lanes>::FromNative(_mm256_max_epi32(a.Native(), b.Native()));
#endif
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm_max_ps(a.Native(), b.Native()));
			else if constexpr (std::is_same_v<T, double>)
				return Pack<T, Lanes>::FromNative(_mm_max_pd(a.Native(), b.Native()));
#if SIMD_HAS_SSE4_1
			else if constexpr (std::is_same_v<T, std::int32_t>)
				return Pack<T, Lanes>::FromNative(_mm_max_epi32(a.Native(), b.Native()));
#endif
		}
#endif
		auto sa = a.ToArray();
		const auto sb = b.ToArray();
		for (std::size_t i = 0; i < Lanes; ++i)
			sa[i] = sa[i] > sb[i] ? sa[i] : sb[i];
		return Pack<T, Lanes>::LoadUnaligned(sa.data());
	}

	// ========================================================================
	// Horizontal Reductions
	// ========================================================================

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr T HorizontalSum(const Pack<T, Lanes> a) noexcept
	{
		if constexpr (!Pack<T, Lanes>::isHardware || std::is_integral_v<T>)
		{
			const auto sa = a.ToArray();
			T sum = T { 0 };
			for (std::size_t i = 0; i < Lanes; ++i)
				sum += sa[i];
			return sum;
		}
		else
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated())
			{
				const auto sa = a.ToArray();
				T sum = T { 0 };
				for (std::size_t i = 0; i < Lanes; ++i)
					sum += sa[i];
				return sum;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return _mm512_reduce_add_ps(a.Native());
			else
				return _mm512_reduce_add_pd(a.Native());
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
			{
				const __m128 lo = _mm256_castps256_ps128(a.Native());
				const __m128 hi = _mm256_extractf128_ps(a.Native(), 1);
				__m128 sum = _mm_add_ps(lo, hi);
				__m128 shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
				sum = _mm_add_ps(sum, shuf);
				shuf = _mm_movehl_ps(shuf, sum);
				sum = _mm_add_ss(sum, shuf);
				return _mm_cvtss_f32(sum);
			}
			else
			{
				const __m128d lo = _mm256_castpd256_pd128(a.Native());
				const __m128d hi = _mm256_extractf128_pd(a.Native(), 1);
				__m128d sum = _mm_add_pd(lo, hi);
				const __m128d hi64 = _mm_unpackhi_pd(sum, sum);
				sum = _mm_add_sd(sum, hi64);
				return _mm_cvtsd_f64(sum);
			}
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
			{
				__m128 sum = a.Native();
				__m128 shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
				sum = _mm_add_ps(sum, shuf);
				shuf = _mm_movehl_ps(shuf, sum);
				sum = _mm_add_ss(sum, shuf);
				return _mm_cvtss_f32(sum);
			}
			else
			{
				const __m128d hi64 = _mm_unpackhi_pd(a.Native(), a.Native());
				return _mm_cvtsd_f64(_mm_add_sd(a.Native(), hi64));
			}
		}
#endif
			return a[0];
		}
	}

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr T HorizontalMin(const Pack<T, Lanes> a) noexcept
	{
		if constexpr (!Pack<T, Lanes>::isHardware || std::is_integral_v<T>)
		{
			const auto sa = a.ToArray();
			T res = sa[0];
			for (std::size_t i = 1; i < Lanes; ++i)
				res = sa[i] < res ? sa[i] : res;
			return res;
		}
		else
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated())
			{
				const auto sa = a.ToArray();
				T res = sa[0];
				for (std::size_t i = 1; i < Lanes; ++i)
					res = sa[i] < res ? sa[i] : res;
				return res;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return _mm512_reduce_min_ps(a.Native());
			else
				return _mm512_reduce_min_pd(a.Native());
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
			{
				const __m128 lo = _mm256_castps256_ps128(a.Native());
				const __m128 hi = _mm256_extractf128_ps(a.Native(), 1);
				__m128 m = _mm_min_ps(lo, hi);
				__m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1));
				m = _mm_min_ps(m, shuf);
				shuf = _mm_movehl_ps(shuf, m);
				m = _mm_min_ss(m, shuf);
				return _mm_cvtss_f32(m);
			}
			else
			{
				const __m128d lo = _mm256_castpd256_pd128(a.Native());
				const __m128d hi = _mm256_extractf128_pd(a.Native(), 1);
				__m128d m = _mm_min_pd(lo, hi);
				const __m128d hi64 = _mm_unpackhi_pd(m, m);
				return _mm_cvtsd_f64(_mm_min_sd(m, hi64));
			}
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
			{
				__m128 m = a.Native();
				__m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1));
				m = _mm_min_ps(m, shuf);
				shuf = _mm_movehl_ps(shuf, m);
				m = _mm_min_ss(m, shuf);
				return _mm_cvtss_f32(m);
			}
			else
			{
				const __m128d hi64 = _mm_unpackhi_pd(a.Native(), a.Native());
				return _mm_cvtsd_f64(_mm_min_sd(a.Native(), hi64));
			}
		}
#endif
			return a[0];
		}
	}

	template<SupportedElement T, std::size_t Lanes>
	[[nodiscard]] constexpr T HorizontalMax(const Pack<T, Lanes> a) noexcept
	{
		if constexpr (!Pack<T, Lanes>::isHardware || std::is_integral_v<T>)
		{
			const auto sa = a.ToArray();
			T res = sa[0];
			for (std::size_t i = 1; i < Lanes; ++i)
				res = sa[i] > res ? sa[i] : res;
			return res;
		}
		else
		{
			// LCOV_EXCL_START
			if (std::is_constant_evaluated())
			{
				const auto sa = a.ToArray();
				T res = sa[0];
				for (std::size_t i = 1; i < Lanes; ++i)
					res = sa[i] > res ? sa[i] : res;
				return res;
			}
			// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return _mm512_reduce_max_ps(a.Native());
			else
				return _mm512_reduce_max_pd(a.Native());
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
			{
				const __m128 lo = _mm256_castps256_ps128(a.Native());
				const __m128 hi = _mm256_extractf128_ps(a.Native(), 1);
				__m128 m = _mm_max_ps(lo, hi);
				__m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1));
				m = _mm_max_ps(m, shuf);
				shuf = _mm_movehl_ps(shuf, m);
				m = _mm_max_ss(m, shuf);
				return _mm_cvtss_f32(m);
			}
			else
			{
				const __m128d lo = _mm256_castpd256_pd128(a.Native());
				const __m128d hi = _mm256_extractf128_pd(a.Native(), 1);
				__m128d m = _mm_max_pd(lo, hi);
				const __m128d hi64 = _mm_unpackhi_pd(m, m);
				return _mm_cvtsd_f64(_mm_max_sd(m, hi64));
			}
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
			{
				__m128 m = a.Native();
				__m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1));
				m = _mm_max_ps(m, shuf);
				shuf = _mm_movehl_ps(shuf, m);
				m = _mm_max_ss(m, shuf);
				return _mm_cvtss_f32(m);
			}
			else
			{
				const __m128d hi64 = _mm_unpackhi_pd(a.Native(), a.Native());
				return _mm_cvtsd_f64(_mm_max_sd(a.Native(), hi64));
			}
		}
#endif
			return a[0];
		}
	}

	// ========================================================================
	// Comparisons & Blend (ArgMin / ArgMax Support)
	// ========================================================================

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Mask<T, Lanes> CmpLess(const Pack<T, Lanes> a, const Pack<T, Lanes> b) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			std::array<bool, Lanes> m {};
			const auto sa = a.ToArray();
			const auto sb = b.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				m[i] = sa[i] < sb[i];
			return Mask<T, Lanes>::FromScalar(m);
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Mask<T, Lanes>::FromNative(_mm512_cmp_ps_mask(a.Native(), b.Native(), _CMP_LT_OQ));
			else
				return Mask<T, Lanes>::FromNative(_mm512_cmp_pd_mask(a.Native(), b.Native(), _CMP_LT_OQ));
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Mask<T, Lanes>::FromNative(_mm256_cmp_ps(a.Native(), b.Native(), _CMP_LT_OQ));
			else
				return Mask<T, Lanes>::FromNative(_mm256_cmp_pd(a.Native(), b.Native(), _CMP_LT_OQ));
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Mask<T, Lanes>::FromNative(_mm_cmplt_ps(a.Native(), b.Native()));
			else
				return Mask<T, Lanes>::FromNative(_mm_cmplt_pd(a.Native(), b.Native()));
		}
#endif
		return {};
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Mask<T, Lanes> CmpGreater(const Pack<T, Lanes> a, const Pack<T, Lanes> b) noexcept
	{
		return CmpLess(b, a);
	}

	template<SupportedElement T, SupportedElement U, std::size_t Lanes>
		requires (sizeof(T) == sizeof(U))
	[[nodiscard]] constexpr Pack<U, Lanes> Blend(const Mask<T, Lanes> mask, const Pack<U, Lanes> ifTrue, const Pack<U, Lanes> ifFalse) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware || !Pack<U, Lanes>::isHardware)
		{
			const auto& m = mask.Scalar();
			const auto st = ifTrue.ToArray();
			auto sf = ifFalse.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
			{
				if (m[i])
					sf[i] = st[i];
			}
			return Pack<U, Lanes>::LoadUnaligned(sf.data());
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<U, float>)
				return Pack<U, Lanes>::FromNative(_mm512_mask_blend_ps(mask.Native(), ifFalse.Native(), ifTrue.Native()));
			else if constexpr (std::is_same_v<U, double>)
				return Pack<U, Lanes>::FromNative(_mm512_mask_blend_pd(mask.Native(), ifFalse.Native(), ifTrue.Native()));
			else if constexpr (std::is_same_v<U, std::int32_t>)
				return Pack<U, Lanes>::FromNative(_mm512_mask_blend_epi32(mask.Native(), ifFalse.Native(), ifTrue.Native()));
			else
				return Pack<U, Lanes>::FromNative(_mm512_mask_blend_epi64(mask.Native(), ifFalse.Native(), ifTrue.Native()));
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float> && std::is_same_v<U, float>)
				return Pack<U, Lanes>::FromNative(_mm256_blendv_ps(ifFalse.Native(), ifTrue.Native(), mask.Native()));
			else if constexpr (std::is_same_v<T, double> && std::is_same_v<U, double>)
				return Pack<U, Lanes>::FromNative(_mm256_blendv_pd(ifFalse.Native(), ifTrue.Native(), mask.Native()));
#if SIMD_HAS_AVX2
			else if constexpr (std::is_same_v<T, float> && std::is_integral_v<U>)
				return Pack<U, Lanes>::FromNative(_mm256_blendv_epi8(ifFalse.Native(), ifTrue.Native(), _mm256_castps_si256(mask.Native())));
			else if constexpr (std::is_same_v<T, double> && std::is_integral_v<U>)
				return Pack<U, Lanes>::FromNative(_mm256_blendv_epi8(ifFalse.Native(), ifTrue.Native(), _mm256_castpd_si256(mask.Native())));
#endif
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float> && std::is_same_v<U, float>)
			{
#if SIMD_HAS_SSE4_1
				return Pack<U, Lanes>::FromNative(_mm_blendv_ps(ifFalse.Native(), ifTrue.Native(), mask.Native()));
#else
				return Pack<U, Lanes>::FromNative(_mm_or_ps(_mm_and_ps(mask.Native(), ifTrue.Native()), _mm_andnot_ps(mask.Native(), ifFalse.Native())));
#endif
			}
			else if constexpr (std::is_same_v<T, double> && std::is_same_v<U, double>)
			{
#if SIMD_HAS_SSE4_1
				return Pack<U, Lanes>::FromNative(_mm_blendv_pd(ifFalse.Native(), ifTrue.Native(), mask.Native()));
#else
				return Pack<U, Lanes>::FromNative(_mm_or_pd(_mm_and_pd(mask.Native(), ifTrue.Native()), _mm_andnot_pd(mask.Native(), ifFalse.Native())));
#endif
			}
			else if constexpr (std::is_same_v<T, float> && std::is_integral_v<U>)
			{
				const __m128i m = _mm_castps_si128(mask.Native());
				return Pack<U, Lanes>::FromNative(_mm_or_si128(_mm_and_si128(m, ifTrue.Native()), _mm_andnot_si128(m, ifFalse.Native())));
			}
			else if constexpr (std::is_same_v<T, double> && std::is_integral_v<U>)
			{
				const __m128i m = _mm_castpd_si128(mask.Native());
				return Pack<U, Lanes>::FromNative(_mm_or_si128(_mm_and_si128(m, ifTrue.Native()), _mm_andnot_si128(m, ifFalse.Native())));
			}
		}
#endif
		return ifFalse;
	}

	// ========================================================================
	// BitCast, Rounding & Integer/Float Conversions
	// ========================================================================

	template<SupportedElement DstT, SupportedElement SrcT, std::size_t Lanes>
		requires (sizeof(DstT) == sizeof(SrcT))
	[[nodiscard]] constexpr Pack<DstT, Lanes> BitCast(const Pack<SrcT, Lanes> a) noexcept
	{
		if constexpr (std::is_same_v<DstT, SrcT>)
			return a;
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<SrcT, Lanes>::isHardware || !Pack<DstT, Lanes>::isHardware)
		{
			const auto src = a.ToArray();
			std::array<DstT, Lanes> dst {};
			for (std::size_t i = 0; i < Lanes; ++i)
				dst[i] = std::bit_cast<DstT>(src[i]);
			return Pack<DstT, Lanes>::FromScalar(dst);
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(SrcT) == 64)
		{
			if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
				return Pack<DstT, Lanes>::FromNative(_mm512_castps_si512(a.Native()));
			else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
				return Pack<DstT, Lanes>::FromNative(_mm512_castsi512_ps(a.Native()));
			else if constexpr (std::is_same_v<SrcT, double> && std::is_same_v<DstT, std::int64_t>)
				return Pack<DstT, Lanes>::FromNative(_mm512_castpd_si512(a.Native()));
			else if constexpr (std::is_same_v<SrcT, std::int64_t> && std::is_same_v<DstT, double>)
				return Pack<DstT, Lanes>::FromNative(_mm512_castsi512_pd(a.Native()));
		}
#endif
#if SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(SrcT) == 32)
		{
			if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
				return Pack<DstT, Lanes>::FromNative(_mm256_castps_si256(a.Native()));
			else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
				return Pack<DstT, Lanes>::FromNative(_mm256_castsi256_ps(a.Native()));
			else if constexpr (std::is_same_v<SrcT, double> && std::is_same_v<DstT, std::int64_t>)
				return Pack<DstT, Lanes>::FromNative(_mm256_castpd_si256(a.Native()));
			else if constexpr (std::is_same_v<SrcT, std::int64_t> && std::is_same_v<DstT, double>)
				return Pack<DstT, Lanes>::FromNative(_mm256_castsi256_pd(a.Native()));
		}
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(SrcT) == 16)
		{
			if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
				return Pack<DstT, Lanes>::FromNative(_mm_castps_si128(a.Native()));
			else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
				return Pack<DstT, Lanes>::FromNative(_mm_castsi128_ps(a.Native()));
			else if constexpr (std::is_same_v<SrcT, double> && std::is_same_v<DstT, std::int64_t>)
				return Pack<DstT, Lanes>::FromNative(_mm_castpd_si128(a.Native()));
			else if constexpr (std::is_same_v<SrcT, std::int64_t> && std::is_same_v<DstT, double>)
				return Pack<DstT, Lanes>::FromNative(_mm_castsi128_pd(a.Native()));
		}
#endif
		return {};
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<T, Lanes> RoundToNearest(const Pack<T, Lanes> a) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware)
		{
			auto arr = a.ToArray();
			for (std::size_t i = 0; i < Lanes; ++i)
				arr[i] = std::nearbyint(arr[i]);
			return Pack<T, Lanes>::FromScalar(arr);
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm512_roundscale_ps(a.Native(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			else
				return Pack<T, Lanes>::FromNative(_mm512_roundscale_pd(a.Native(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		}
#endif
#if SIMD_HAS_AVX || SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm256_round_ps(a.Native(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			else
				return Pack<T, Lanes>::FromNative(_mm256_round_pd(a.Native(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		}
#endif
#if SIMD_HAS_SSE4_1
		if constexpr (Lanes * sizeof(T) == 16)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<T, Lanes>::FromNative(_mm_round_ps(a.Native(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			else
				return Pack<T, Lanes>::FromNative(_mm_round_pd(a.Native(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
		}
#endif
		const T magic = std::is_same_v<T, float> ? static_cast<T>(12582912.0f) : static_cast<T>(6755399441055744.0);
		return (a + magic) - magic;
	}

	template<std::floating_point T, std::size_t Lanes>
	[[nodiscard]] constexpr Pack<IndexTypeForT<T>, Lanes> ConvertToInt(const Pack<T, Lanes> a) noexcept
	{
		using IntT = IndexTypeForT<T>;
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<T, Lanes>::isHardware || !Pack<IntT, Lanes>::isHardware)
		{
			const auto src = a.ToArray();
			std::array<IntT, Lanes> dst {};
			for (std::size_t i = 0; i < Lanes; ++i)
				dst[i] = static_cast<IntT>(src[i]);
			return Pack<IntT, Lanes>::FromScalar(dst);
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(T) == 64)
		{
			if constexpr (std::is_same_v<T, float>)
				return Pack<IntT, Lanes>::FromNative(_mm512_cvtps_epi32(a.Native()));
#if SIMD_HAS_AVX512DQ
			else
				return Pack<IntT, Lanes>::FromNative(_mm512_cvtpd_epi64(a.Native()));
#endif
		}
#endif
#if SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(T) == 32 && std::is_same_v<T, float>)
			return Pack<IntT, Lanes>::FromNative(_mm256_cvtps_epi32(a.Native()));
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(T) == 16 && std::is_same_v<T, float>)
			return Pack<IntT, Lanes>::FromNative(_mm_cvtps_epi32(a.Native()));
#endif
		if constexpr (std::is_same_v<T, double>)
		{
			const auto magicF = Broadcast<double, Lanes>(6755399441055744.0);
			const auto magicI = Broadcast<std::int64_t, Lanes>(0x4338000000000000LL);
			return BitCast<std::int64_t>(a + magicF) - magicI;
		}
		return {};
	}

	template<std::floating_point DstT, std::integral SrcT, std::size_t Lanes>
		requires (sizeof(DstT) == sizeof(SrcT))
	[[nodiscard]] constexpr Pack<DstT, Lanes> ConvertToFloat(const Pack<SrcT, Lanes> a) noexcept
	{
		// LCOV_EXCL_START
		if (std::is_constant_evaluated() || !Pack<SrcT, Lanes>::isHardware || !Pack<DstT, Lanes>::isHardware)
		{
			const auto src = a.ToArray();
			std::array<DstT, Lanes> dst {};
			for (std::size_t i = 0; i < Lanes; ++i)
				dst[i] = static_cast<DstT>(src[i]);
			return Pack<DstT, Lanes>::FromScalar(dst);
		}
		// LCOV_EXCL_STOP
#if SIMD_HAS_AVX512F
		if constexpr (Lanes * sizeof(SrcT) == 64)
		{
			if constexpr (std::is_same_v<DstT, float>)
				return Pack<DstT, Lanes>::FromNative(_mm512_cvtepi32_ps(a.Native()));
#if SIMD_HAS_AVX512DQ
			else
				return Pack<DstT, Lanes>::FromNative(_mm512_cvtepi64_pd(a.Native()));
#endif
		}
#endif
#if SIMD_HAS_AVX2
		if constexpr (Lanes * sizeof(SrcT) == 32 && std::is_same_v<DstT, float>)
			return Pack<DstT, Lanes>::FromNative(_mm256_cvtepi32_ps(a.Native()));
#endif
#if SIMD_HAS_SSE2
		if constexpr (Lanes * sizeof(SrcT) == 16 && std::is_same_v<DstT, float>)
			return Pack<DstT, Lanes>::FromNative(_mm_cvtepi32_ps(a.Native()));
#endif
		if constexpr (std::is_same_v<DstT, double>)
		{
			const auto magicI = Broadcast<std::int64_t, Lanes>(0x4338000000000000LL);
			const auto magicF = Broadcast<double, Lanes>(6755399441055744.0);
			return BitCast<double>(a + magicI) - magicF;
		}
		return {};
	}
}	 // namespace simd
