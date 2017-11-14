#include "Tensor.h"

namespace Meerkat
{

	Tensor::Tensor(std::initializer_list<dl_uint32> shape)
	{
		DL_PANIC_ON_FAIL(shape.size() <= s_max_dimension && dimension > 0, "invalid dimension");
		dl_uint32 m_dimension = static_cast<dl_uint32>(shape.size());
		dl_uint32 i = 0;
		for (dl_uint32 n : shape)
		{
			m_shape[i] = n;
			++i;
		}
	}

	void Tensor::Alloc(ComputeType type)
	{
		;
	}

	void Tensor::Zeros()
	{

	}

}