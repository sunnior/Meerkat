#ifndef __MEERKAT_CLASSNLL_CRITERION_H__
#define __MEERKAT_CLASSNLL_CRITERION_H__

#include "Criterion.h"

namespace DeepLearning
{
	class ClassNllCriterion: public Criterion
	{
	public:
		ClassNllCriterion(ComputeType type)
			: Criterion(type)
		{};

	protected:
		void _ForwardCpu(const Tensor* input, const Tensor* target, dl_tensor* output) override;
		void _ForwardGpu(const Tensor* input, const Tensor* target, dl_tensor* output) override;

		void _BackwardCpu(const Tensor* input, const Tensor* target, Tensor* output) override;
		void _BackwardGpu(const Tensor* input, const Tensor* target, Tensor* output) override;

	};
}

#endif