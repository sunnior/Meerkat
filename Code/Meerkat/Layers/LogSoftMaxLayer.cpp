#include "LogSoftMaxLayer.h"
#include "Common/Blas.h"
#include <cmath>

namespace DeepLearning
{
	void LogSoftMaxLayer::_ForwardGpu(const Tensor* input, Tensor* output)
	{

	}

	void LogSoftMaxLayer::_ForwardCpu(const Tensor* input, Tensor* output)
	{
		dl_uint32 param_size = input->GetShape(1);
		dl_uint32 batch_size = input->GetShape(0);

		for (dl_uint32 i = 0; i < batch_size; ++i)
		{
			dl_tensor max_value = dl_max_cpu<dl_tensor>(input->GetData() + i * param_size, param_size);
			dl_tensor logsum = 0;
			for (dl_uint32 j = 0; j < param_size; ++j)
			{
				logsum += exp(*(input->GetData() + i*param_size + j) - max_value);
			}
			logsum = max_value + log(logsum);

			for (dl_uint32 j = 0; j < param_size; ++j)
			{
				*(output->GetData() + i*param_size + j) = *(input->GetData() + i*param_size + j) - logsum;
			}
		}
	}

	void LogSoftMaxLayer::_BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output)
	{
	}

	void LogSoftMaxLayer::_BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output)
	{

	}

}