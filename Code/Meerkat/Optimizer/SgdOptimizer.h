#ifndef __MEERKAT_SGD_OPTIMIZER_H__
#define __MEERKAT_SGD_OPTIMIZER_H__

#include "Optimizer.h"

namespace DeepLearning
{
	class SgdOptimizer : public Optimizer
	{
	public:
		SgdOptimizer(ComputeType type, Tensor** params, Tensor** grad_params, dl_uint32 count)
			: Optimizer(type, params, grad_params, count)
		{};

		void Update() override;
	private:
		dl_uint32 m_eval{ 0 };
		dl_tensor m_learning_rate_base{ 0.001f };

	};
}

#endif