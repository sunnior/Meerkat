#ifndef __MEERKAT_LOGSOFTMAX_LAYER_H__
#define __MEERKAT_LOGSOFTMAX_LAYER_H__

#include "Layer.h"

namespace DeepLearning
{
	class LogSoftMaxLayer : public Layer
	{
	public:
		LogSoftMaxLayer(ComputeType type)
			: Layer(type, false)
		{};

	protected:
		void _ForwardGpu(const Tensor* input, Tensor* output) override;
		void _ForwardCpu(const Tensor* input, Tensor* output) override;

		void _BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) override;
		void _BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) override;
	};

}



#endif