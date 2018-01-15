#ifndef __MEERKAT_STL_H__
#define __MEERKAT_STL_H__

#include <vector>
#include <unordered_map>
#include <string>
#include "dl_std_allocator.h"
#include "Common/Platform.h"

namespace DeepLearning
{
	template<typename T>
	using dl_vector = ::std::vector<T, dl_std_allocator<T>>;
	
	template<class _Kty,
		class _Ty,
		class _Hasher = ::std::hash<_Kty>,
		class _Keyeq = ::std::equal_to<_Kty>,
		class _Alloc = dl_std_allocator<::std::pair<const _Kty, _Ty> > >
	using dl_unordered_map = ::std::unordered_map<_Kty, _Ty, _Hasher, _Keyeq, _Alloc>;
	
	typedef ::std::basic_string<char, ::std::char_traits<char>, dl_std_allocator<char> > dl_string;
	
	typedef dl_vector<dl_uint32> dl_tensor_shape;
}

#endif