#ifndef __MEERKAT_LAYER_H__
#define __MEERKAT_LAYER_H__

#include "Platform.h"
#include "Tensor.h"

namespace DeepLearning
{
	class Layer
	{
	public:
		virtual ~Layer() {};

		void Forward(ComputeType type, Tensor* input, Tensor* output)
		{
			(type == ComputeType_CPU) ? _ForwardCpu(input, output) : _ForwardGpu(input, output);
		}

	protected:
		virtual void _ForwardGpu(Tensor* input, Tensor* output) = 0;
		virtual void _ForwardCpu(Tensor* input, Tensor* output) = 0;
	};
}
#endif