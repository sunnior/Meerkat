#ifndef __MEERKAT_TENSOR_H__
#define __MEERKAT_TENSOR_H__

#include "Util/dl_stl.h"
#include "Platform.h"

namespace DeepLearning
{
	class Tensor
	{
	public:
		Tensor(ComputeType type, const dl_tensor_shape& shape);

		void Zeros() { FillWith((dl_tensor)0.0f); }
		void FillWith(dl_tensor value);
		dl_uint32 GetShape(dl_uint32 idx) const { return m_shape[idx]; }
		const dl_tensor_shape& GetShape() const { return m_shape; }

		dl_tensor* GetData() { return (m_type == ComputeType_CPU) ? m_cpu_data : m_gpu_data; }
		const dl_tensor* GetData() const { return (m_type == ComputeType_CPU) ? m_cpu_data : m_gpu_data; }
		dl_size GetSize() const { return m_size; }
		void LoadData(dl_tensor* pData);

	private:
		void _Alloc(ComputeType type);

	private:
		ComputeType m_type;

		dl_tensor* m_cpu_data{ nullptr };
		dl_tensor* m_gpu_data{ nullptr };

		dl_size	   m_size{ 0 };
		dl_tensor_shape  m_shape;

	};
}

#endif