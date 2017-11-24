#ifndef __MEERKAT_MIN_SQUARE_ERROR_CRITERION_H__
#define __MEERKAT_MIN_SQUARE_ERROR_CRITERION_H__

#include "Criterion.h"

namespace DeepLearning
{
	class MinSquareErrorCriterion : public Criterion
	{
	public:
		void Forward(const Tensor* input, const Tensor* target, Tensor* output) override;
	};
}

#endif