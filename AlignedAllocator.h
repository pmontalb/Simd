#pragma once

/* Adapted from https://stackoverflow.com/questions/12942548/making-stdvector-allocate-aligned-memory */

namespace detail {
	template<size_t Align>
	static inline void* allocate_aligned_memory(size_t size) noexcept
	{
		static_assert(Align >= sizeof(void*), "Align < sizeof(void*)");
		static_assert(Align >= 0 && (Align & (Align - 1)) == 0, "Align not a power of 2");

		if (size == 0)
			return nullptr;

		void* ptr;
		const auto err = posix_memalign(&ptr, Align, size);
		if (err != 0)
			return nullptr;

		return ptr;
	}
	static inline void deallocate_aligned_memory(void* ptr) noexcept { free(ptr); }
}

template <typename T, size_t Align>
class AlignedAllocator;

template <size_t Align>
class AlignedAllocator<void, Align>
{
public:
	using pointer = void*;
	using const_pointer = const void*;
	using value_type = void;

	template <class U> struct rebind
	{
		using other = AlignedAllocator<U, Align>;
	};
};

template <typename T, size_t Align>
class AlignedAllocator
{
public:
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using propagate_on_container_move_assignment = std::true_type;

	template <class U>
	struct rebind
	{
		using other = AlignedAllocator<U, Align>;
	};

public:
	AlignedAllocator() noexcept = default;

	template <class U>
	inline AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

	inline auto max_size() const noexcept { return (std::numeric_limits<size_type>::max() - size_type(Align)) / sizeof(T); }

	inline auto address(reference x) noexcept { return std::addressof(x); }
	inline auto address(const_reference x) const noexcept { return std::addressof(x); }

	inline auto allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = nullptr) const
	{
		void* ptr = detail::allocate_aligned_memory<Align>(n * sizeof(T));
		if (!ptr)
			throw std::bad_alloc();
		return reinterpret_cast<pointer>(ptr);
	}

	void deallocate(pointer p, size_type) const noexcept { detail::deallocate_aligned_memory(p); }

	template <class U, class ...Args>
	void construct(U* p, Args&&... args) const noexcept { ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...); }

	void destroy(pointer p) const noexcept { p->~T(); }
};

template <typename T, size_t Align>
class AlignedAllocator<const T, Align>
{
public:
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using propagate_on_container_move_assignment = std::true_type;

	template <class U>
	struct rebind
	{
		using other = AlignedAllocator<U, Align>;
	};

public:
	AlignedAllocator() noexcept = default;

	template <class U>
	inline AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

	inline auto max_size() const noexcept { return (std::numeric_limits<size_type>::max() - size_type(Align)) / sizeof(T); }
	inline auto address(const_reference x) const noexcept { return std::addressof(x); }

	inline auto allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = nullptr) const
	{
		void* ptr = detail::allocate_aligned_memory<Align>(n * sizeof(T));
		if (!ptr)
			throw std::bad_alloc();
		return reinterpret_cast<pointer>(ptr);
	}

	void deallocate(pointer p, size_type) noexcept { detail::deallocate_aligned_memory(p); }

	template <class U, class ...Args>
	void
	construct(U* p, Args&&... args)
	{ ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...); }

	void destroy(pointer p) const noexcept { p->~T(); }
};

template <typename T, size_t TAlign, typename U, size_t UAlign>
inline bool operator== (const AlignedAllocator<T,TAlign>&, const AlignedAllocator<U, UAlign>&) noexcept { return TAlign == UAlign; }

template <typename T, size_t TAlign, typename U, size_t UAlign>
inline bool operator!= (const AlignedAllocator<T,TAlign>&, const AlignedAllocator<U, UAlign>&) noexcept { return TAlign != UAlign; }
