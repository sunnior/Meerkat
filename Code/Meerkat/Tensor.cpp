#include "Tensor.h"

namespace Meerkat
{
	Tensor::Tensor(dl_uint32* shape, dl_uint32 dimension)
	{
		DL_PANIC_ON_FAIL(dimension <= s_max_dimension && dimension > 0, "invalid dimension");
		for (dl_uint32 i = 0; i < dimension; ++i)
		{
			m_shape[i] = shape[i];
		}
	}

	void Tensor::Alloc(ComputeType type)
	{
		;
	}
}