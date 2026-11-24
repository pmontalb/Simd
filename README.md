# Simd

A zero-overhead, C++20 header-only SIMD library providing trivial POD register wrappers (`simd::Pack<T, Lanes>`), vectorized math (`Exp`, `Log`, `Sin`, `Cos`), and unrolled linear-algebra algorithms across **AVX-512**, **AVX2 + FMA**, **SSE2 / SSE4.1**, and `constexpr` scalar fallbacks.

## Features

- **Trivial POD Register Type (`simd::Pack<T, Lanes>`)**: Satisfies `std::is_trivial_v` and `std::is_standard_layout_v`, passed by value in SIMD registers.
- **Automatic ISA Scaling**: `simd::Pack<T>` selects the widest native register for `T` (`float`, `double`, `std::int32_t`, `std::int64_t`) at compile time.
- **`constexpr`-Friendly**: All `Pack`, `Math`, and `Algorithms` operations provide `std::is_constant_evaluated()` scalar fallbacks.
- **Zero-Overhead Algorithms**: `DotProduct`, `Axpy`, `SumOfSquares`, `Reduce*`, `ArgMinAbs`/`ArgMaxAbs`, and `ForEachUnrolled` use `__restrict` pointers, 4x register-bound accumulator unrolling, and `assert(count > 0)`.
- **Warning-Clean**: Compiles cleanly with `-Wall -Wextra -Werror` (GCC) and `-Weverything -Werror` (Clang) under C++20 with `-fno-rtti`.

## ISA Tiers & Compile-Time Queries

| ISA Tier | `float` Lanes | `double` Lanes | Alignment |
| --- | --- | --- | --- |
| **AVX-512** (`__AVX512F__`, `__AVX512DQ__`) | 16 | 8 | 64 bytes |
| **AVX / AVX2 + FMA** (`__AVX2__`, `__FMA__`) | 8 | 4 | 32 bytes |
| **SSE2 / SSE4.1** (`__SSE2__`) | 4 | 2 | 16 bytes |
| **Scalar / `constexpr`** | 1 | 1 | `alignof(T)` |

Available in `Impl/Config.h`:
- `simd::LaneCount<T>()`
- `simd::AlignmentBytes<T>()` / `simd::MaxAlignmentBytes`
- `simd::AssumeAligned<T>(ptr)`

## Convenience Type Aliases

Defined in `Simd.h`:
- **SSE2**: `simd::Double2`, `simd::Float4`, `simd::Int4`, `simd::Long2`
- **AVX / AVX2**: `simd::Double4`, `simd::Float8`, `simd::Int8`, `simd::Long4`
- **AVX-512**: `simd::Double8`, `simd::Float16`, `simd::Int16`, `simd::Long8`

## Usage Examples

### 1. Basic `Pack` Arithmetic, FMA & Math

```cpp
#include <Simd.h>

void Compute(const double* const __restrict x, const double* const __restrict y, double* const __restrict out, const std::size_t count) noexcept
{
	const auto alpha = simd::Broadcast<double>(2.5);
	constexpr std::size_t lanes = simd::LaneCount<double>();

	std::size_t i = 0;
	for (; i + lanes <= count; i += lanes)
	{
		const auto vx = simd::LoadUnaligned<double>(x + i);
		const auto vy = simd::LoadUnaligned<double>(y + i);
		const auto res = simd::Exp(simd::Fma(alpha, vx, vy));
		simd::StoreUnaligned(out + i, res);
	}
	for (; i < count; ++i)
		out[i] = std::exp(2.5 * x[i] + y[i]);
}
```

### 2. Fixed-Width Aliases (`Double4`, `Float8`, etc.)

```cpp
#include <Simd.h>

simd::Double4 Rotate(const simd::Double4 x, const simd::Double4 y, const double theta) noexcept
{
	const auto c = simd::Cos(simd::Double4(theta));
	const auto s = simd::Sin(simd::Double4(theta));
	return simd::Fms(c, x, s * y); // c * x - s * y
}
```

### 3. High-Level Algorithms (`Impl/Algorithms.h`)

```cpp
#include <Simd.h>
#include <cassert>

double WeightedNorm(const double* const __restrict x, const double* const __restrict y, const std::size_t count) noexcept
{
	assert(count > 0);
	const double dot = simd::DotProduct(x, y, count);
	const double sumSq = simd::SumOfSquares(x, count);
	const std::size_t maxIdx = simd::ArgMaxAbs(x, count);
	return dot + sumSq + std::abs(x[maxIdx]);
}
```

### 4. Custom Unrolled Loop with `ForEachUnrolled`

```cpp
#include <Simd.h>

void Axpby(double* const __restrict y, const double* const __restrict x, const double alpha, const double beta, const std::size_t count) noexcept
{
	assert(count > 0);
	constexpr std::size_t lanes = simd::LaneCount<double>();
	const auto vAlpha = simd::Broadcast<double>(alpha);
	const auto vBeta = simd::Broadcast<double>(beta);

	simd::ForEachUnrolled<double, 4>(
		count,
		[&](const std::size_t i)
		{
			for (std::size_t u = 0; u < 4; ++u)
			{
				const std::size_t idx = i + u * lanes;
				const auto vx = simd::LoadUnaligned<double>(x + idx);
				const auto vy = simd::LoadUnaligned<double>(y + idx);
				simd::StoreUnaligned(y + idx, simd::Fma(vAlpha, vx, vBeta * vy));
			}
		},
		[&](const std::size_t i)
		{
			const auto vx = simd::LoadUnaligned<double>(x + i);
			const auto vy = simd::LoadUnaligned<double>(y + i);
			simd::StoreUnaligned(y + i, simd::Fma(vAlpha, vx, vBeta * vy));
		},
		[&](const std::size_t i, const std::size_t rem)
		{
			for (std::size_t j = 0; j < rem; ++j)
				y[i + j] = beta * y[i + j] + alpha * x[i + j];
		});
}
```

## API Reference

- **Construction / Load / Store (`Impl/Pack.h`)**:
  `Zero<T>()`, `Broadcast(value)`, `Iota(start, step)`, `LoadAligned(ptr)`, `LoadUnaligned(ptr)`, `LoadPartial(ptr, validLanes)`, `StoreAligned(ptr, pack)`, `StoreUnaligned(ptr, pack)`, `StorePartial(ptr, pack, validLanes)`
- **Arithmetic & FMA (`Impl/Pack.h`)**:
  `+`, `-`, `*`, `/` (pack-pack and pack-scalar), `Fma(a, b, c)`, `Fms(a, b, c)`, `Fnma(a, b, c)`, `Reciprocal(a)`, `Abs(a)`, `Sqrt(a)`, `Min(a, b)`, `Max(a, b)`
- **Horizontal Reductions (`Impl/Pack.h`)**:
  `HorizontalSum(a)`, `HorizontalMin(a)`, `HorizontalMax(a)`
- **Comparisons, Blending & Conversions (`Impl/Pack.h`)**:
  `CmpLess(a, b)`, `CmpGreater(a, b)`, `Blend(mask, ifTrue, ifFalse)`, `BitCast<DstT>(a)`, `RoundToNearest(a)`, `ConvertToInt(a)`, `ConvertToFloat<DstT>(a)`
- **Transcendental & Trigonometric (`Impl/Math.h`)**:
  `Exp(a)`, `Log(a)`, `Sin(a)`, `Cos(a)`
- **Algorithms (`Impl/Algorithms.h`)**:
  `ForEachUnrolled<T, Unroll>`, `DotProduct`, `Axpy`, `SumOfSquares`, `ReduceSum`, `ReduceAbsSum`, `ReduceMin`, `ReduceMax`, `ReduceAbsMin`, `ReduceAbsMax`, `ArgMaxAbs`, `ArgMinAbs`

## Building & Testing

`Simd` uses `CMakePresets.json` from the `cmake` submodule:

```bash
cmake --preset gcc-release
cmake --build --preset gcc-release
ctest --preset gcc-release --output-on-failure
```

Other available presets include `clang-release`, `gcc-debug`, `clang-debug`, `gcc-coverage`, `clang-coverage`, `gcc-asan`, and `clang-asan` (`cmake --list-presets` for the full list).

## CMake Integration

`Simd` exposes an `INTERFACE` target `Simd`:

```cmake
add_subdirectory(Simd)
target_link_libraries(MyTarget PRIVATE Simd)
```
