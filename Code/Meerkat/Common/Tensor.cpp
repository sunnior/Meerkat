#include "Tensor.h"

namespace DeepLearning
{

	Tensor::Tensor(ComputeType type, const dl_tensor_shape& shape)
		: m_type(type)
		, m_shape(shape)
		, m_size(1)
	{
		for (dl_uint32 n : m_shape)
		{
			m_size *= n;
		}

		_Alloc(m_type);
	}

	void Tensor::_Alloc(ComputeType type)
	{
		m_cpu_data = static_cast<dl_tensor*>(DL_CPU_ALLOC(m_size*sizeof(dl_tensor)));
	}

	void Tensor::FillWith(dl_tensor value)
	{
		if (m_type == ComputeType_CPU)
		{
			for (dl_size i = 0; i < m_size; ++i)
			{
				m_cpu_data[i] = value;
			}
			//std::memset(m_cpu_data, 0, sizeof(dl_tensor)*m_size);
		}
	}

	void Tensor::LoadData(dl_tensor* pData)
	{
		if (m_type == ComputeType_CPU)
		{
			std::memcpy(m_cpu_data, pData, m_size*sizeof(dl_tensor));
		}
	}

}