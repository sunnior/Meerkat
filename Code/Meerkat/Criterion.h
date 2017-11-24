#ifndef __MEERKAT_CRITERION_H__
#define __MEERKAT_CRITERION_H__

#include "Tensor.h"

namespace DeepLearning
{
	class Criterion
	{
	public:
		virtual void Forward();
	};
}

#endif