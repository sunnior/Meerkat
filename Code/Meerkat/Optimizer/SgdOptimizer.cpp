#include "SgdOptimizer.h"
#include "Common/Blas.h"
#include "Common/Tensor.h"

namespace DeepLearning
{

	void SgdOptimizer::Optimize(Tensor* param, Tensor* grad_param)
	{
		const dl_tensor learning_rate = 0.01f;
		dl_size nelement = param->GetSize();
		dl_axpy_cpu<dl_tensor>(nelement, learning_rate, grad_param->GetData(), 1, param->GetData(), 1);
	}

}