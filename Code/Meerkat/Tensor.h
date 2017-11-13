#ifndef __MEERKAT_TENSOR_H__
#define __MEERKAT_TENSOR_H__

#include "Platform.h"

namespace Meerkat
{
	class Tensor
	{
	public:
		Tensor(dl_uint32* shape, dl_uint32 dimension);

		void Alloc(ComputeType type);

	private:
		static const dl_uint32 s_max_dimension = 8;

	private:
		dl_tensor* m_cpu_data{ nullptr };
		dl_tensor* m_gpu_data{ nullptr };

		dl_uint32  m_dimension{ 0 };
		dl_uint32  m_shape[s_max_dimension];

	};
}

#endif