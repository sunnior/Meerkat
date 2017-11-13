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

#define DL_PANIC_ON_FAIL(exp, msg)

#define DL_NEW(T) new T
#define DL_SAFE_DELETE(p) delete p; \
						  p = nullptr;

	enum ComputeType
	{
		ComputeType_CPU,
		ComputeType_GPU,
	};
}
#endif