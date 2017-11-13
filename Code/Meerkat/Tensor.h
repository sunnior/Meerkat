#ifndef __MEERKAT_TENSOR_H__
#define __MEERKAT_TENSOR_H__

#include "Platform.h"

namespace Meerkat
{
	class Tensor
	{
	public:
	private:
		dl_tensor* m_cpu_data{ nullptr };
		dl_tensor* m_gpu_data{ nullptr };
	};
}

#endif