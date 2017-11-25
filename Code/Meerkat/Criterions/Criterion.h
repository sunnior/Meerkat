#ifndef __MEERKAT_CRITERION_H__
#define __MEERKAT_CRITERION_H__

#include "Common/Tensor.h"

namespace DeepLearning
{
	class Criterion
	{
	public:
		virtual ~Criterion() {};

		void Forward(ComputeType type, const Tensor* input, const Tensor* target, dl_tensor* output)
		{
			(type == ComputeType_CPU) ? _ForwardCpu(input, target, output) : _ForwardGpu(input, target, output);
		}

	protected:
		virtual void _ForwardCpu(const Tensor* input, const Tensor* target, dl_tensor* output) = 0;
		virtual void _ForwardGpu(const Tensor* input, const Tensor* target, dl_tensor* output) = 0;

	};
}

#endif