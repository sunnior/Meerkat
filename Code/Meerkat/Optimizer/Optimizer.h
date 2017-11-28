#ifndef __MEERKAT_OPTIMIZER_H__
#define __MEERKAT_OPTIMIZER_H__

namespace DeepLearning
{
	class Tensor;

	class Optimizer
	{
	public:
		virtual void Optimize(Tensor* param, Tensor* grad_param) = 0;
	};
}

#endif