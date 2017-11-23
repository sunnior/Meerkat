#ifndef __MEERKAT_TENSOR_H__
#define __MEERKAT_TENSOR_H__

#include <initializer_list>
#include "Platform.h"

namespace DeepLearning
{
	class Tensor
	{
	public:
		Tensor(std::initializer_list<dl_uint32> shape);

		void Alloc(ComputeType type);
		void Zeros();
		dl_uint32 GetShape(dl_uint32 idx) { return m_shape[idx]; }
		dl_tensor* GetCpuData() { return m_cpu_data; }
	private:
		static const dl_uint32 s_max_dimension = 8;

	private:
		dl_tensor* m_cpu_data{ nullptr };
		dl_tensor* m_gpu_data{ nullptr };

		dl_size	   m_size{ 0 };
		dl_uint32  m_dimension{ 0 };
		dl_uint32  m_shape[s_max_dimension];

	};
}

#endif