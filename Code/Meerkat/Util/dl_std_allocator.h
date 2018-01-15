#ifndef __MEERKAT_STD_ALLOCATOR_H__
#define __MEERKAT_STD_ALLOCATOR_H__

namespace DeepLearning
{
	void *_std_allocate(size_t _Count, size_t _Sz);
	void _std_deallocate(void * _Ptr, size_t _Count, size_t _Sz);

	template<typename T>
	class dl_std_allocator
	{
	public:
		static_assert(!::std::is_const<T>::value,
			"The C Standard forbids containers of const elements "
			"because allocator<const T> is ill-formed.");

		typedef T value_type;
		typedef value_type *pointer;
		typedef const value_type *const_pointer;
		typedef value_type& reference;
		typedef const value_type& const_reference;

		typedef size_t size_type;
		typedef ptrdiff_t difference_type;

		typedef std::true_type propagate_on_container_move_assignment;
		typedef std::true_type is_always_equal;

		template<typename _Other>
		struct rebind
		{    // convert this type to allocator<_Other>
			typedef dl_std_allocator<_Other> other;
		};

		pointer address(reference _Val) noexcept
		{    // return address of mutable _Val
			return &_Val;
		}

		const_pointer address(const_reference _Val) const noexcept
		{    // return address of nonmutable _Val
			return &_Val;
		}

		dl_std_allocator() noexcept
		{    // construct default allocator (do nothing)
		}

		dl_std_allocator(const dl_std_allocator<T>&) noexcept
		{    // construct by copying (do nothing)
		}

		template<class _Other>
		dl_std_allocator(const dl_std_allocator<_Other>&) noexcept
		{    // construct from a related allocator (do nothing)
		}

		template<class _Other>
		dl_std_allocator<T>& operator=(const dl_std_allocator<_Other>&)
		{    // assign from a related allocator (do nothing)
			return (*this);
		}

		void deallocate(pointer _Ptr, size_type _Count)
		{    // deallocate object at _Ptr
			_std_deallocate(_Ptr, _Count, sizeof(T));
		}

		pointer allocate(size_type _Count)
		{    // allocate array of _Count elements
			return (static_cast<pointer>(_std_allocate(_Count, sizeof(T))));
		}

		template<class _Objty,
			class... _Types>
			void construct(_Objty *_Ptr, _Types&&... _Args)
		{    // construct _Objty(_Types...) at _Ptr
			::new ((void *)_Ptr) _Objty(::std::forward<_Types>(_Args)...);
		}

		template<class _Uty>
		void destroy(_Uty *_Ptr)
		{    // destroy object at _Ptr
			_Ptr->~_Uty();
		}

		size_t max_size() const noexcept
		{    // estimate maximum array size
			return ((size_t)(-1) / sizeof(T));
		}
	};

}

#endif