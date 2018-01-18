#ifndef __MEERKAT_PLATFORM_H__
#define __MEERKAT_PLATFORM_H__

#include <stdlib.h>

namespace DeepLearning
{
#if defined(_MSC_VER)
	typedef float			dl_tensor;
	typedef unsigned int	dl_uint32;
	typedef unsigned char	dl_uint8;
	typedef size_t			dl_size;
#else
#error "unkown platform!"
#endif

#define DL_PANIC_ON_FAIL(exp, msg)

#define TODO(msg)

#define DL_NEW(T) new T

#define DL_CPU_ALLOC(s) malloc(s)

#define DL_SAFE_DELETE(p) if (p)	  \
							delete p; \
						  p = nullptr;

	enum ComputeType
	{
		ComputeType_CPU,
		ComputeType_GPU,
	};
}

#define FORCE_LINK_THIS_LAYER(x) int force_link_layer_##x = 0;
#define FORCE_LINK_THAT_LAYER(x) { extern int force_link_layer_##x; force_link_layer_##x = 1; }

#endif