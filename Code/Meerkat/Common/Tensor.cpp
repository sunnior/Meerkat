#include "Tensor.h"

namespace DeepLearning
{

	Tensor::Tensor(ComputeType type, std::initializer_list<dl_uint32> shape)
		: m_type(type)
	{
		DL_PANIC_ON_FAIL(shape.size() <= s_max_dimension && dimension > 0, "invalid dimension");
		m_dimension = static_cast<dl_uint32>(shape.size());
		m_size = 1;
		dl_uint32 i = 0;
		for (dl_uint32 n : shape)
		{
			m_shape[i] = n;
			m_size *= n;
			++i;
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