#ifndef __MEERKAT_PLATFORM_H__
#define __MEERKAT_PLATFORM_H__

namespace Meerkat
{
#if defined(_MSC_VER)
	typedef float			dl_tensor;
	typedef unsigned int	dl_uint32;
#else
#error "unkown platform!"
#endif
}
#endif