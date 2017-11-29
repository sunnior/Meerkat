#ifndef __MEERKAT_TENSOR_H__
#define __MEERKAT_TENSOR_H__

#include <initializer_list>
#include "Platform.h"

namespace DeepLearning
{
	class Tensor
	{
	public:
		Tensor(ComputeType type, std::initializer_list<dl_uint32> shape);

		void Zeros() { FillWith((dl_tensor)0.0f); }
		void FillWith(dl_tensor value);
		dl_uint32 GetShape(dl_uint32 idx) const { return m_shape[idx]; }
		dl_tensor* GetData() { return (m_type == ComputeType_CPU) ? m_cpu_data : m_gpu_data; }
		const dl_tensor* GetData() const { return (m_type == ComputeType_CPU) ? m_cpu_data : m_gpu_data; }
		dl_size GetSize() const { return m_size; }
		void LoadData(dl_tensor* pData);

	private:
		void _Alloc(ComputeType type);
	private:
		static const dl_uint32 s_max_dimension = 8;

	private:
		ComputeType m_type;

		dl_tensor* m_cpu_data{ nullptr };
		dl_tensor* m_gpu_data{ nullptr };

		dl_size	   m_size{ 0 };
		dl_uint32  m_dimension{ 0 };
		dl_uint32  m_shape[s_max_dimension];

	};
}

#endif