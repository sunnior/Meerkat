#ifndef __MEERKAT_SGD_OPTIMIZER_H__
#define __MEERKAT_SGD_OPTIMIZER_H__

#include "Optimizer.h"

namespace DeepLearning
{
	class SgdOptimizer : public Optimizer
	{
	public:
		SgdOptimizer(ComputeType type, dl_vector<Tensor*> params, dl_vector<Tensor*> grad_params)
			: Optimizer(type, params, grad_params)
		{};

		void Update() override;
	private:
		dl_uint32 m_eval{ 0 };
		dl_tensor m_learning_rate_base{ 0.001f };

	};
}

#endif