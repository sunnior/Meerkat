#ifndef __MEERKAT_SGD_OPTIMIZER_H__
#define __MEERKAT_SGD_OPTIMIZER_H__

#include "Optimizer.h"

namespace DeepLearning
{
	class SgdOptimizer : public Optimizer
	{
	public:
		void Optimize(Tensor* param, Tensor* grad_param) override;
	};
}

#endif