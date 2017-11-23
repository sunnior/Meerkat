#include "Tensor.h"

namespace DeepLearning
{

	Tensor::Tensor(std::initializer_list<dl_uint32> shape)
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
	}

	void Tensor::Alloc(ComputeType type)
	{
		m_cpu_data = static_cast<dl_tensor*>(DL_CPU_ALLOC(m_size*sizeof(dl_tensor)));
	}

	void Tensor::Zeros()
	{
		std::memset(m_cpu_data, 0, sizeof(dl_tensor)*m_size);
	}

}