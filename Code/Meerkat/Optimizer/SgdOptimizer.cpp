#include "SgdOptimizer.h"
#include "Common/Blas.h"
#include "Common/Tensor.h"

namespace DeepLearning
{
	void SgdOptimizer::Update(const dl_vector<::std::pair<Tensor*, Tensor*>>& params)
	{
		dl_tensor learning_rate = m_learning_rate_base / (1 + m_eval*0.0001f);
		++m_eval;
		for (auto& it : params)
		{
			Tensor* param = it.first;
			Tensor* param_grad = it.second;
			dl_size nelement = param->GetSize();
			dl_axpy_cpu<dl_tensor>(nelement, -1.0f*learning_rate, param_grad->GetData(), 1, param->GetData(), 1);
		}

	}

}