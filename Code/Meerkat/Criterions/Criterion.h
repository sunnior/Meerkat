#ifndef __MEERKAT_CRITERION_H__
#define __MEERKAT_CRITERION_H__

#include "Common/Tensor.h"

namespace DeepLearning
{
	class Criterion
	{
	public:
		TODO("if output is a single scalar, use dl_tensor is enough");
		virtual void Forward(const Tensor* input, const Tensor* target, Tensor* output) = 0;
	};
}

#endif